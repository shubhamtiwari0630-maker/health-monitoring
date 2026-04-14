"""Grader for vitals_check (easy): reward any warning when HR > 100."""


def grade(completion=None, state=None, heart_rate=75, temperature=37.0, action=0, **kwargs) -> float:
    hr   = int(heart_rate)
    temp = float(temperature)
    act  = int(action)

    if hr > 100:
        raw = 0.95 if act >= 1 else 0.1
    else:
        raw = 0.95 if act == 0 else 0.4

    return float(max(0.01, min(0.99, raw)))
