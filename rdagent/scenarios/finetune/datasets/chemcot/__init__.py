"""ChemCoT dataset preparation utilities."""

import json
from pathlib import Path


def normalize_rcr(out_dir: str) -> None:
    """Normalize rcr.json to match standard data format.

    Fixes:
    1. Move `gt` from top-level into `meta`
    2. Rename `cot_result` to `struct_cot` and strip markdown wrapper
    """
    rcr_path = Path(out_dir) / "chemcotbench-cot" / "rxn" / "rcr.json"
    if not rcr_path.exists():
        return

    with open(rcr_path) as f:
        data = json.load(f)

    for item in data:
        # 1. Move gt from top-level into meta
        if "gt" in item:
            meta = json.loads(item["meta"]) if isinstance(item["meta"], str) else item["meta"]
            meta["gt"] = item.pop("gt")
            item["meta"] = json.dumps(meta)

        # 2. Rename cot_result -> struct_cot, strip markdown wrapper
        if "cot_result" in item:
            cot = item.pop("cot_result").strip()
            if cot.startswith("```json"):
                cot = cot[7:]
            if cot.endswith("```"):
                cot = cot[:-3]
            item["struct_cot"] = cot.strip()

    with open(rcr_path, "w") as f:
        json.dump(data, f, indent=4)
