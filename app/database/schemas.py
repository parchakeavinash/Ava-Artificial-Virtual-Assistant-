
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# TASK SCHEMAS
# ===========================================================================

class TaskCreate(BaseModel):
    """Payload to create a new task."""
    title: str = Field(..., min_length=1, max_length=500, description="Task title")
    description: str = Field(default="", max_length=2000, description="Optional details about the task")


class TaskUpdate(BaseModel):
    """Payload to update an existing task (all fields optional)."""
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    description: Optional[str] = Field(default=None, max_length=2000)
    status: Optional[Literal["pending", "completed", "cancelled"]] = None


class TaskResponse(BaseModel):
    """Task data returned from the API."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


# ===========================================================================
# DIARY SCHEMAS
# ===========================================================================

class DiaryCreate(BaseModel):
    """Payload to create a new diary entry."""
    title: str = Field(default="", max_length=500, description="Optional title for the entry")
    content: str = Field(..., min_length=1, description="The idea, note, or reflection to save")


class DiaryUpdate(BaseModel):
    """Payload to update an existing diary entry (all fields optional)."""
    title: Optional[str] = Field(default=None, max_length=500)
    content: Optional[str] = Field(default=None, min_length=1)


class DiaryResponse(BaseModel):
    """Diary entry data returned from the API."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    created_at: datetime
