import csv
from collections import defaultdict
from pathlib import Path

rows = list(
    csv.DictReader(
        open(Path("results/tables/eicu_site_calibration_sites.csv"), encoding="utf-8")
    )
)
methods = sorted({row["method"] for row in rows})
print("methods:", methods)

values = defaultdict(dict)
for row in rows:
    key = (row["endpoint"], row["model"], row["hospitalid"])
    values[key][row["method"]] = float(row["ece"])

for endpoint in ("any_deterioration", "severe_deterioration"):
    for model in ("xgb_summary", "dggru_tr"):
        sites = sorted({k[2] for k in values if k[0] == endpoint and k[1] == model})
        for change in ("intercept_only", "slope_intercept"):
            if change not in methods:
                continue
            improved = [
                site
                for site in sites
                if values[(endpoint, model, site)][change]
                < values[(endpoint, model, site)]["uncalibrated"]
            ]
            diffs = sorted(
                values[(endpoint, model, site)][change]
                - values[(endpoint, model, site)]["uncalibrated"]
                for site in sites
            )
            median = diffs[len(diffs) // 2] if len(diffs) % 2 else (
                diffs[len(diffs) // 2 - 1] + diffs[len(diffs) // 2]
            ) / 2
            print(
                f"{endpoint:20s} {model:12s} {change:16s} improved {len(improved)}/{len(sites)} "
                f"median {median:+.3f}"
            )
