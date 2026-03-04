from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import TrainingParams, TrainingState, TrainingStatus

logger = logging.getLogger(__name__)

STATE_FILE = Path(os.environ.get("DATA_DIR", "/data")) / "training_state.json"
LOG_FILE = Path(os.environ.get("DATA_DIR", "/data")) / "training.log"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))

STEP_PATTERN = re.compile(r"=== Step (\d)/5: (.+) ===")
MAX_LOG_LINES = 5000


class TrainingManager:
    """Manages a single training job at a time."""

    def __init__(self) -> None:
        self._state = TrainingState()
        self._process: Optional[subprocess.Popen] = None
        self._log_buffer: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._load_state()

    # ── State persistence ────────────────────────────────────────────────

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self._state = TrainingState(**data)
            except Exception:
                logger.exception("Failed to load state file, starting fresh")
                self._state = TrainingState()

        # Detect stale RUNNING state (container restart)
        if self._state.status == TrainingStatus.RUNNING:
            pid = self._state.pid
            if pid and self._is_pid_alive(pid):
                logger.info("Found running process pid=%d, will not touch it", pid)
            else:
                logger.warning("Stale RUNNING state detected, marking as FAILED")
                self._state.mark_failed("Training interrupted (container restart)")
                self._save_state()

        # Reload log buffer from log file if it exists
        if LOG_FILE.exists():
            try:
                lines = LOG_FILE.read_text().splitlines()
                self._log_buffer = deque(lines[-MAX_LOG_LINES:], maxlen=MAX_LOG_LINES)
            except Exception:
                pass

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(self._state.model_dump_json(indent=2))

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def state(self) -> TrainingState:
        return self._state

    def get_log_lines(self) -> list[str]:
        with self._lock:
            return list(self._log_buffer)

    def start_training(self, params: TrainingParams) -> None:
        if self._state.status == TrainingStatus.RUNNING:
            raise RuntimeError("Training is already running")

        # Clear previous logs
        with self._lock:
            self._log_buffer.clear()

        env = os.environ.copy()
        env["WAKE_WORD"] = params.wake_word
        env["N_SAMPLES"] = str(params.n_samples)
        env["N_SAMPLES_VAL"] = str(params.n_samples_val)
        env["TRAINING_STEPS"] = str(params.training_steps)
        env["LAYER_SIZE"] = str(params.layer_size)
        env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            ["/app/train.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        self._state.mark_running(params, self._process.pid)
        self._save_state()

        # Clear log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text("")

        thread = threading.Thread(target=self._monitor, daemon=True)
        thread.start()

    def cancel_training(self) -> None:
        if self._state.status != TrainingStatus.RUNNING or self._process is None:
            return

        try:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception:
            logger.exception("Error killing training process")

        self._state.mark_cancelled()
        self._save_state()
        self._push_line("[Training cancelled by user]")

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    # ── Monitor thread ───────────────────────────────────────────────────

    def _monitor(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        log_fh = open(LOG_FILE, "a")
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                self._append_line(line, log_fh)

                # Detect step transitions
                m = STEP_PATTERN.match(line)
                if m:
                    step_num = int(m.group(1))
                    step_label = m.group(2)
                    self._state.current_step = step_num
                    self._state.step_label = step_label
                    self._save_state()

            proc.wait()
            if self._state.status == TrainingStatus.RUNNING:
                if proc.returncode == 0:
                    self._state.mark_completed()
                    self._push_line("[Training completed successfully]")
                else:
                    self._state.mark_failed(f"Process exited with code {proc.returncode}")
                    self._push_line(f"[Training failed with exit code {proc.returncode}]")
                self._save_state()
        except Exception as e:
            logger.exception("Monitor thread error")
            if self._state.status == TrainingStatus.RUNNING:
                self._state.mark_failed(str(e))
                self._save_state()
        finally:
            log_fh.close()
            self._process = None

    def _append_line(self, line: str, log_fh) -> None:
        with self._lock:
            self._log_buffer.append(line)
        log_fh.write(line + "\n")
        log_fh.flush()
        self._push_line(line)

    def _push_line(self, line: str) -> None:
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(line)
            except Exception:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


# Module-level singleton
manager = TrainingManager()
