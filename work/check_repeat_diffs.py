import json
from pathlib import Path

dev = json.loads(Path("results/metrics_core.json").read_text(encoding="utf-8"))["development"]
variability = dev["cv_repeat_variability"]
values = {
    key: variability[key]["macro_auc"]["values"]
    for key in ("xgb_summary", "gru", "dggru", "dggru_tr")
}
for key, series in values.items():
    print(key, [round(value, 4) for value in series])

tr = values["dggru_tr"]
for other in ("xgb_summary", "dggru", "gru"):
    diffs = [a - b for a, b in zip(tr, values[other])]
    mean = sum(diffs) / len(diffs)
    sd = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    print(
        f"TR - {other}: mean {mean:+.4f} sd {sd:.4f} min {min(diffs):+.4f} max {max(diffs):+.4f} "
        f"all_positive={all(d > 0 for d in diffs)}"
    )
