"""
inference.py — OpenEnv inference script for AI Smart Health Monitoring System.
Runs 3 task episodes emitting [START]/[STEP]/[END] structured logs to stdout.

Required env vars:
  API_BASE_URL  — LLM API base URL  (default: https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier  (default: meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN      — HuggingFace token / API key
  ENV_URL       — Running server URL (default: http://localhost:7860)
"""

import os
import sys
import json
import requests
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "10"))

# Exactly 3 tasks — validator requires >= 3 tasks with graders
TASKS = [
    "vitals_check",
    "anomaly_detection",
    "triage_report",
]

# ── LLM client (optional, falls back to heuristic) ───────────────────────────
client = None
if HF_TOKEN:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"[DEBUG] LLM init error: {e}", flush=True)


def _clamp(v: float) -> float:
    """Score must be strictly in (0, 1) — OpenEnv validator requirement."""
    return round(max(0.01, min(0.99, float(v))), 4)


def heuristic_action(state: dict) -> int:
    """Rule-based fallback when LLM is unavailable."""
    hr   = float(state.get("heart_rate",  75))
    temp = float(state.get("temperature", 37.0))
    if hr > 120 or temp > 39.0:
        return 2
    elif hr > 100 or temp > 38.0:
        return 1
    return 0


def llm_action(state: dict, task_id: str) -> int:
    """Ask LLM for action 0/1/2; falls back to heuristic on any failure."""
    if not client:
        return heuristic_action(state)
    hr   = state.get("heart_rate",  75)
    temp = state.get("temperature", 37.0)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a health monitoring AI. "
                        "Reply with ONLY a single digit: 0, 1, or 2.\n"
                        "0 = do_nothing (normal vitals)\n"
                        "1 = send_warning (mildly concerning)\n"
                        "2 = emergency_alert (critically abnormal)"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task_id}\n"
                        f"heart_rate={hr}, temperature={temp}\n"
                        "Choose action:"
                    ),
                },
            ],
            max_tokens=5,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        action = int(text[0])
        if action not in (0, 1, 2):
            raise ValueError(f"Bad action: {action}")
        return action
    except Exception as e:
        print(f"[DEBUG] LLM error: {e} — using heuristic", flush=True)
        return heuristic_action(state)


def run_task(task_id: str) -> float:
    """
    Run one full episode for a task.
    Emits [START], [STEP]×N, [END] to stdout with flush=True.
    Returns score strictly in (0, 1).
    """
    print(f"[START] task={task_id}", flush=True)

    rewards = []
    steps_done = 0
    score = 0.5

    try:
        # ── Reset ──────────────────────────────────────────────────────────
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=15,
        )
        reset_resp.raise_for_status()
        data  = reset_resp.json()
        state = data.get("state", data.get("observation", data))
        if not isinstance(state, dict):
            state = {}

        # ── Steps ──────────────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            action = llm_action(state, task_id)

            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=15,
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            raw_reward = result.get("reward", 0.5)
            done       = bool(result.get("done", False))
            state      = result.get("state", result.get("observation", state))
            if not isinstance(state, dict):
                state = {}

            reward = _clamp(raw_reward)
            rewards.append(reward)
            steps_done = step

            print(
                f"[STEP] step={step} action={action} "
                f"reward={reward:.4f} done={str(done).lower()} error=null",
                flush=True,
            )

            if done:
                break

        # ── Grade ──────────────────────────────────────────────────────────
        try:
            grade_resp = requests.get(f"{ENV_URL}/grade", timeout=10)
            grade_data = grade_resp.json()
            raw_score  = grade_data.get("score", grade_data.get("value"))
            if raw_score is not None:
                score = _clamp(raw_score)
            else:
                score = _clamp(sum(rewards) / len(rewards)) if rewards else 0.5
        except Exception:
            score = _clamp(sum(rewards) / len(rewards)) if rewards else 0.5

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score = _clamp(sum(rewards) / len(rewards)) if rewards else 0.5

    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    # [END] must have task= and score= — required by validator
    print(
        f"[END] task={task_id} score={score:.4f} steps={steps_done} rewards={rewards_str}",
        flush=True,
    )
    return score


def main():
    all_scores = []
    for task_id in TASKS:
        print(f"\n{'='*55}", flush=True)
        s = run_task(task_id)
        all_scores.append(s)
        print(f"[TASK SCORE] {task_id} => {s:.4f}", flush=True)

    overall = sum(all_scores) / len(all_scores)
    print(f"\n[FINAL] overall_score={overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
