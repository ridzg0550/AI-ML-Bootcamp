"""Pydantic request/response schemas for the DungeonBrain++ API."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ── Request Models ──────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Player's action or dialogue")


# ── Response Models ─────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    turn_count: int
    opening_message: str


class MessageResponse(BaseModel):
    response: str
    turn_count: int
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    turn_count: int
    total_memories: int
    permanent_memories: int
    transient_memories: int
    total_memory_tokens: int
    native_context_tokens: int
    effective_context_tokens: int
    context_amplification: float
    avg_salience: float
    total_links: int
    avg_link_strength: float
    memory_by_era: Dict[str, int]
    npc_count: int
    npc_relationships: Dict[str, int]
    npc_list: List[Dict[str, Any]]
    active_quests: int
    completed_quests: int
    total_quests: int
    quest_list: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    slot_memory: Dict[str, Any]
    embedding_dimension: int
    faiss_index_size: int
    dialogue_history: List[str]


class HistoryResponse(BaseModel):
    dialogue_history: List[str]
    turn_count: int


class ErrorResponse(BaseModel):
    detail: str
