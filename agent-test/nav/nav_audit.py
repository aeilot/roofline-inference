"""Static audit of the blinded candidate engines before timing (PILOT.md, Revision 3).

The evaluation workers are not sandboxed, and TASK.md forbids state kept between requests or
processes. This lists every line of a candidate's engine/ that touches files, processes, the network,
the environment or serialization, for manual review; it decides nothing by itself. Results go to
runs/audit.json and the "可信度复核" section of nav_report.md.

    python agent-test/nav/nav_audit.py        # reads runs/candidates.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r"\bopen\(|\.write_|\bwrite\(|\bsave|\bload\(|pickle|shelve|sqlite|np\.save|mx\.save|"
                     r"\bos\.environ|\bgetenv|subprocess|socket|urllib|requests|Path\.home|expanduser|"
                     r"/tmp|tempfile|lru_cache|functools\.cache|\bglobal\b")


def main():
    cands = json.loads((HERE / "runs" / "candidates.json").read_text())
    audit = {}
    for name, d in sorted(cands.items()):
        hits = []
        for f in sorted(Path(d).rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            try:
                text = f.read_text()
            except UnicodeDecodeError:
                hits.append(dict(file=str(f.relative_to(d)), line=0, match="(binary file)"))
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if PATTERN.search(line) and not line.lstrip().startswith("#"):
                    hits.append(dict(file=str(f.relative_to(d)), line=i, match=line.strip()[:120]))
        audit[name] = hits
        print(f"{name}: {len(hits)} lines to review")
        for h in hits:
            print(f"  {h['file']}:{h['line']}  {h['match']}")
    (HERE / "runs" / "audit.json").write_text(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
