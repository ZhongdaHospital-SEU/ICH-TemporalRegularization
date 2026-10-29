import json
from pathlib import Path

data = json.loads(Path("results/metrics_core.json").read_text(encoding="utf-8"))
dev = data["development"]
for model, payload in dev.items():
    point = payload.get("point") if isinstance(payload, dict) else None
    if not isinstance(point, dict):
        continue
    print(model)
    for key, value in point.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
