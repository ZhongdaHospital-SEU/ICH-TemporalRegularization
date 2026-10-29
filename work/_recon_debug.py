import sys, json
sys.path.insert(0, r"D:\TT paper\1024 TP Intracerebral Hemorrhage\work")
import build_landmark_inputs as B
from src.ich_dynamic.common import load_config
cfg = load_config()
built = B.build_landmark_dataset("mimic4", 24, channels=cfg["channels"], strict_reference=False)
rec = built.audit.get("reconciliation", {})
print(json.dumps(rec, indent=2)[:3000])
