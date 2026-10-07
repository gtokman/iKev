"""Reference probabilities from kev's own MLX backend for the records in records.json -> reference.json.
Run from the kev checkout: uv run --extra mlx python <this file>"""
import json, sys
from pathlib import Path
from kev.checkpoint import Checkpoint, LoadOptions

here = Path(__file__).parent
run = sys.argv[1] if len(sys.argv) > 1 else "jaredpalmer/kev-0.8b@v1.0"
tok, m = Checkpoint(run).load("mlx", LoadOptions(backend="mlx"))
out = []
for rec in json.loads((here / "records.json").read_text()):
    enc = m.encode(tok, rec)
    logits = [z.detach().float().tolist() for z in m.forward(enc)]
    probs = [p.detach().float().tolist() for p in m.probs(enc)]
    out.append({"ids": enc["ids"], "decide_idx": enc["decide_idx"], "opt_idx": enc["opt_idx"], "logits": logits, "probs": probs})
(here / "reference.json").write_text(json.dumps({"run": run, "temperature": float(m.head.temperature), "records": out}, indent=1))
print("wrote", here / "reference.json", "temperature", float(m.head.temperature))
for r in out: print([[round(x, 3) for x in p] for p in r["probs"]])
