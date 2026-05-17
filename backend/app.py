"""
FastAPI application for DungeonBrain++ Neuromorphic Dungeon Master.
Serves both the API and the frontend static files.
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routes import router, set_session_manager
from backend.session_manager import SessionManager

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = FastAPI(
    title="DungeonBrain++",
    description="Neuromorphic Memory-Augmented Dungeon Master",
    version="2.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    if not GROQ_API_KEY:
        print("\n⚠️  WARNING: GROQ_API_KEY not set!")
        print("Set it via: $env:GROQ_API_KEY='your_key_here'\n")
    sm = SessionManager(api_key=GROQ_API_KEY)
    set_session_manager(sm)
    print("✅ DungeonBrain++ API ready")


# API routes
app.include_router(router)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/styles.css")
    async def serve_css():
        return FileResponse(os.path.join(FRONTEND_DIR, "styles.css"))

    @app.get("/app.js")
    async def serve_js():
        return FileResponse(os.path.join(FRONTEND_DIR, "app.js"))
