"""Build the on-device Kev checkpoint: the Kev LoRA folded into its Qwen3.5 base with mlx-lm, quantized, plus the pointer
head as safetensors and a kev.json with the calibration. The output directory is what KevModel loads (locally or from the Hub).

    uv run --python 3.12 --with mlx-lm --with torch --with huggingface_hub --with safetensors \
        scripts/convert.py --run jaredpalmer/kev-0.8b --revision v1.0 --bits 4 --out build/kev-0.8b-mlx-4bit
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model, quantize_model, save_config, save_model

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_lora(lm, adapter_dir):
    """kev.mlx_model.merge_lora: W + (B @ A) * alpha / r in fp32 on the CPU, rounded once to the backbone dtype."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("trainable_token_indices"):
        raise SystemExit("checkpoints with trained token embeddings are not supported")
    alpha = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    weights = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(tree_flatten(lm.parameters()))
    merged = {}
    with mx.stream(mx.cpu):
        for name, a in weights.items():
            if not name.endswith(".lora_A.weight"):
                continue
            stem = name[: -len(".lora_A.weight")]
            target = stem.replace("base_model.model.", "language_model.model.", 1) + ".weight"
            if target not in params:
                raise SystemExit(f"adapter tensor {stem} has no weight in the mlx-lm model (looked for {target})")
            base = params[target]
            delta = (weights[stem + ".lora_B.weight"].astype(mx.float32) @ a.astype(mx.float32)) * alpha
            merged[target] = (base.astype(mx.float32) + delta).astype(base.dtype)
            mx.eval(merged[target])
    lm.load_weights(list(merged.items()), strict=False)
    mx.eval(lm.parameters())
    return len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--revision", default="v1.0")
    ap.add_argument("--bits", type=int, default=4, help="0 keeps the backbone dtype (bf16)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adapter_dir = Path(snapshot_download(args.run, revision=args.revision))
    head = torch.load(adapter_dir / "head.pt", map_location="cpu", weights_only=False)
    base_id, base_rev = head["base"], head.get("base_revision")
    base_dir = Path(snapshot_download(base_id, revision=base_rev, allow_patterns=["*.json", "*.safetensors", "merges.txt"]))

    lm, config = load_model(base_dir, lazy=False)
    merged = merge_lora(lm, adapter_dir)
    print(f"merged {merged} tensors from {args.run}@{args.revision} into {base_id}@{base_rev}")

    if "lm_head" in lm.language_model:
        del lm.language_model["lm_head"]
    config.pop("vision_config", None)

    if args.bits:
        _, config = quantize_model(lm, config, args.group_size, args.bits)
        print(f"quantized to {args.bits} bits (group size {args.group_size})")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_model(out, lm, donate_model=True)
    save_config(config, config_path=out / "config.json")
    for name in TOKENIZER_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy(src, out / name)

    pointer = {k: mx.array(v.float().numpy()) for k, v in head["head"].items()}
    mx.save_safetensors(str(out / "head.safetensors"), pointer)
    meta = {
        "format": 1,
        "run": args.run,
        "revision": args.revision,
        "base": base_id,
        "base_revision": base_rev,
        "head_dim": int(head.get("head_dim", 256)),
        "temperature": float(head.get("temperature", 1.0)),
        "option_isolation": bool(head.get("option_isolation", False)),
        "bits": args.bits,
        "group_size": args.group_size if args.bits else None,
        "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
        "head_sha256": sha256(adapter_dir / "head.pt"),
    }
    (out / "kev.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
