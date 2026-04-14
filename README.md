---
title: AI Health Monitoring
emoji: ❤️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🏥 AI Health Monitoring — OpenEnv Environment

An RL environment where an AI agent monitors patient vitals (heart rate & temperature)
and decides when to alert medical staff.

## Environment Description

The agent observes real-time vitals and must classify severity to trigger the right alert level.

## Observation Space

| Field        | Type  | Range       |
|-------------|-------|-------------|
| heart_rate  | int   | 55 – 160    |
| temperature | float | 36.0 – 40.5 |

## Action Space

| Action | Meaning         |
|--------|-----------------|
| 0      | do_nothing      |
| 1      | send_warning    |
| 2      | emergency_alert |

## Reward Function

| Condition                          | Correct Action | Reward |
|------------------------------------|---------------|--------|
| hr > 120 or temp > 39.0           | 2             | 1.0    |
| hr > 100 or temp > 38.0           | 1             | 1.0    |
| Normal                             | 0             | 1.0    |
| Off by 1 action                    | —             | 0.5    |
| Off by 2 actions                   | —             | 0.0    |

## Tasks

- **easy_task**: Detect heart rate > 100 and send a warning
- **medium_task**: Detect increasing heart rate trend and respond
- **hard_task**: Issue emergency_alert for critical vitals (hr > 120 OR temp > 39)

## API Endpoints

| Method | Endpoint              | Description           |
|--------|-----------------------|-----------------------|
| POST   | /reset                | Start new episode     |
| POST   | /step                 | Take action           |
| GET    | /state                | Current episode state |
| GET    | /tasks                | List all tasks        |
| GET    | /grade/easy_task      | Grade easy task       |
| GET    | /grade/medium_task    | Grade medium task     |
| GET    | /grade/hard_task      | Grade hard task       |

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="your_hf_token"
python inference.py
```