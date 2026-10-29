import sys, json
sys.path.insert(0, r"D:\TT paper\1024 TP Intracerebral Hemorrhage\work")
import build_landmark_inputs as B
from src.ich_dynamic.common import load_config
cfg = load_config()
built = B.build_landmark_dataset("mimic4", 24, channels=cfg["channels"], strict_reference=False)
print("keys:", list(built.audit.keys()))
for k,v in built.audit.items():
    if isinstance(v, dict) and ("check" in json.dumps(v).lower() or "recon" in str(k).lower()):
        print("==",k); print(json.dumps(v, indent=2)[:2500])
print("n", len(built.cohort))
