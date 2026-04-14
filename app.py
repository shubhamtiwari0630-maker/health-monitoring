"""
server/app.py — FastAPI application for AI Smart Health Monitoring System.
Implements the full OpenEnv spec. All scores strictly in (0, 1).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from env import HealthEnv  # noqa: E402

app = FastAPI(
    title="AI Smart Health Monitoring System",
    description="OpenEnv-compatible RL environment for health monitoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: HealthEnv = HealthEnv()
_initialized: bool = False
_episode_rewards: list = []
_current_task: Optional[str] = None

_UI_PATH = Path(__file__).parent.parent / "static" / "index.html"


def _clamp(v: float) -> float:
    """Ensure score is strictly in (0, 1) — OpenEnv validator requirement."""
    return round(max(0.01, min(0.99, float(v))), 4)


def _correct_action(heart_rate: int, temperature: float) -> int:
    if heart_rate > 120 or temperature > 39.0:
        return 2
    elif heart_rate > 100 or temperature > 38.0:
        return 1
    return 0


# ── Models ────────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: int


class GradeRequest(BaseModel):
    heart_rate: int
    temperature: float
    action: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if _UI_PATH.exists():
        return HTMLResponse(_UI_PATH.read_text())
    return JSONResponse({"status": "ok", "name": "health-monitoring-env"})


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "ok", "name": "health-monitoring-env", "version": "1.0.0"}


@app.post("/reset")
def reset(request: ResetRequest = None) -> Dict[str, Any]:
    global _env, _initialized, _episode_rewards, _current_task
    _env = HealthEnv()
    state = _env.reset()
    _initialized = True
    _episode_rewards = []
    if request and request.task_id:
        _current_task = request.task_id
    return {"state": state, "task": _current_task}


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    global _episode_rewards
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    if request.action not in (0, 1, 2):
        raise HTTPException(status_code=422, detail=f"Invalid action {request.action}.")
    next_state, reward, done, info = _env.step(request.action)
    _episode_rewards.append(reward)
    return {"state": next_state, "reward": reward, "done": done, "info": info}


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return _env.state()


@app.get("/grade")
def grade_get() -> Dict[str, Any]:
    """GET /grade — required by inference.py. Returns average episode score."""
    global _episode_rewards, _current_task
    if not _episode_rewards:
        return {"score": 0.5, "graded": False, "task": _current_task}
    avg   = sum(_episode_rewards) / len(_episode_rewards)
    score = _clamp(avg)
    return {
        "score": score,
        "passed": score >= 0.5,
        "task": _current_task,
        "steps": len(_episode_rewards),
        "graded": True,
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "vitals_check",
                "name": "Vitals Check",
                "difficulty": "easy",
                "passing_score": 0.5,
                "description": "Detect elevated heart rate (>100 bpm) and issue at least a warning.",
                "grader_endpoint": "/grade/vitals_check",
            },
            {
                "id": "anomaly_detection",
                "name": "Anomaly Detection",
                "difficulty": "medium",
                "passing_score": 0.6,
                "description": "Respond correctly using both HR and temperature signals.",
                "grader_endpoint": "/grade/anomaly_detection",
            },
            {
                "id": "triage_report",
                "name": "Triage Report",
                "difficulty": "hard",
                "passing_score": 0.7,
                "description": "Emergency alert when HR>120 OR temp>39. Zero tolerance.",
                "grader_endpoint": "/grade/triage_report",
            },
        ]
    }


# ── Per-task graders (scores ALWAYS strictly in (0, 1)) ───────────────────────
@app.post("/grade/vitals_check")
def grade_vitals_check(request: GradeRequest) -> Dict[str, Any]:
    """Easy: reward any warning/alert when HR > 100."""
    if request.heart_rate > 100:
        # Correct is to warn or alert
        score = _clamp(0.95) if request.action >= 1 else _clamp(0.1)
    else:
        # Normal vitals — do nothing
        score = _clamp(0.95) if request.action == 0 else _clamp(0.4)
    return {"task": "vitals_check", "score": score, "passed": score >= 0.5}


@app.post("/grade/anomaly_detection")
def grade_anomaly_detection(request: GradeRequest) -> Dict[str, Any]:
    """Medium: match action to exact severity level."""
    correct = _correct_action(request.heart_rate, request.temperature)
    if request.action == correct:
        score = _clamp(0.95)
    elif abs(request.action - correct) == 1:
        score = _clamp(0.55)
    else:
        score = _clamp(0.1)
    return {"task": "anomaly_detection", "score": score, "passed": score >= 0.5}


@app.post("/grade/triage_report")
def grade_triage_report(request: GradeRequest) -> Dict[str, Any]:
    """Hard: exact match required for critical cases; partial for others."""
    correct = _correct_action(request.heart_rate, request.temperature)
    if correct == 2:
        # Critical — must send emergency_alert; no partial credit
        score = _clamp(0.95) if request.action == 2 else _clamp(0.05)
    elif correct == 1:
        score = _clamp(0.95) if request.action == 1 else (
            _clamp(0.5) if request.action == 2 else _clamp(0.1)
        )
    else:
        score = _clamp(0.95) if request.action == 0 else _clamp(0.35)
    return {"task": "triage_report", "score": score, "passed": score >= 0.5}


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
