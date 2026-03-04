from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TrainingStatus(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingParams(BaseModel):
    wake_word: str = Field(..., min_length=1, description="Wake word phrase to train")
    n_samples: int = Field(10000, ge=100, le=100000)
    n_samples_val: int = Field(2000, ge=100, le=50000)
    training_steps: int = Field(50000, ge=1000, le=500000)
    layer_size: int = Field(32, ge=16, le=256)


class TrainingState(BaseModel):
    status: TrainingStatus = TrainingStatus.IDLE
    params: Optional[TrainingParams] = None
    current_step: int = 0
    total_steps: int = 5
    step_label: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None

    def mark_running(self, params: TrainingParams, pid: int) -> None:
        self.status = TrainingStatus.RUNNING
        self.params = params
        self.current_step = 0
        self.step_label = "Starting..."
        self.started_at = datetime.now().isoformat()
        self.finished_at = None
        self.error = None
        self.pid = pid

    def mark_completed(self) -> None:
        self.status = TrainingStatus.COMPLETED
        self.finished_at = datetime.now().isoformat()
        self.pid = None

    def mark_failed(self, error: str = "") -> None:
        self.status = TrainingStatus.FAILED
        self.finished_at = datetime.now().isoformat()
        self.error = error
        self.pid = None

    def mark_cancelled(self) -> None:
        self.status = TrainingStatus.CANCELLED
        self.finished_at = datetime.now().isoformat()
        self.pid = None
