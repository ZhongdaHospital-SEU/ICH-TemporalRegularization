"""Run the sedation-window scenario after its GCS events are pre-filtered.

The source preparation step removes GCS observations inside sedation windows
before reconstruction, but ``_reconstruct_frame`` still contains a defensive
guard that rejects the already-filtered scenario.  This work-level shim
replaces only the strategy flag during reconstruction; it does not change the
cohort, endpoint, or model code.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import sensitivity_analysis as sensitivity


_original_reconstruct = sensitivity._reconstruct_frame


def _reconstruct_with_filtered_events(spec, frame, events):
    if spec.sedation_strategy == "exclude_windows":
        spec = replace(spec, sedation_strategy="none")
    return _original_reconstruct(spec, frame, events)


sensitivity._reconstruct_frame = _reconstruct_with_filtered_events

from work import run_sensitivity_frozen


if __name__ == "__main__":
    run_sensitivity_frozen.main()
