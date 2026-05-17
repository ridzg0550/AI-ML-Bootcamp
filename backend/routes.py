"""API route definitions for DungeonBrain++ backend."""

import asyncio
from fastapi import APIRouter, HTTPException
from backend.schemas import (
    MessageRequest, MessageResponse, SessionResponse,
    StatsResponse, HistoryResponse
)
from backend.session_manager import SessionManager

router = APIRouter(prefix="/api")

# Session manager is injected by app.py at startup
session_manager: SessionManager = None


def set_session_manager(sm: SessionManager):
    global session_manager
    session_manager = sm


# ── Session Endpoints ───────────────────────────────────────────────

@router.post("/session/new", response_model=SessionResponse)
async def create_session():
    """Start a new dungeon campaign."""
    session_id, opening = await asyncio.to_thread(session_manager.create_session)
    brain = session_manager.get_session(session_id)
    return SessionResponse(
        session_id=session_id,
        turn_count=brain.turn_count if brain else 0,
        opening_message=opening,
    )


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a campaign and free resources."""
    if not session_manager.end_session(session_id):
        raise HTTPException(404, "Session not found")
    return {"status": "ended"}


# ── Gameplay ────────────────────────────────────────────────────────

@router.post("/session/{session_id}/message", response_model=MessageResponse)
async def send_message(session_id: str, req: MessageRequest):
    """Send a player action and get the DM response."""
    brain = session_manager.get_session(session_id)
    if brain is None:
        raise HTTPException(404, "Session not found")

    response = await asyncio.to_thread(brain.respond, req.message)
    stats = await asyncio.to_thread(brain.get_memory_stats)

    return MessageResponse(
        response=response,
        turn_count=brain.turn_count,
        stats=stats,
    )


# ── Stats & Data ────────────────────────────────────────────────────

@router.get("/session/{session_id}/stats")
async def get_stats(session_id: str):
    """Get complete memory statistics."""
    brain = session_manager.get_session(session_id)
    if brain is None:
        raise HTTPException(404, "Session not found")
    stats = await asyncio.to_thread(brain.get_memory_stats)
    return stats


@router.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get dialogue history."""
    brain = session_manager.get_session(session_id)
    if brain is None:
        raise HTTPException(404, "Session not found")
    return HistoryResponse(
        dialogue_history=brain.dialogue_history,
        turn_count=brain.turn_count,
    )


@router.post("/session/{session_id}/save")
async def save_session(session_id: str):
    """Persist session state to disk."""
    ok = await asyncio.to_thread(session_manager.save_session, session_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"status": "saved"}


@router.get("/health")
async def health_check():
    """Server health check."""
    return {
        "status": "ok",
        "active_sessions": session_manager.active_count if session_manager else 0,
    }
