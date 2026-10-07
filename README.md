# kev-swift

[Kev](https://github.com/jaredpalmer/kev) decision models on Apple devices, running on [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

Kev is not a chat model. It answers fixed-choice questions about a piece of text ("state") in one prefill pass: no
generation, calibrated probabilities, a confidence you can threshold on. This package is a faithful port of
`kev.mlx_model` so the same checkpoints give the same answers on iPhone, iPad and Mac.

```swift
import Kev

let model = try await KevModel.load(hubID: "HumanInterfaceDesign/kev-0.8b-mlx-4bit")   // downloads once, then cached
let answers = try await model.decide(
    state: "User message: set a timer for 10 minutes",
    questions: [
        .choice("Which route should handle this message?", options: ["casual", "deviceAction", "assistant"]),
        .yesNo("Does the message ask for an action on this device?"),
    ])
answers[0].choice          // "deviceAction"
answers[0].probabilities   // [0.03, 0.80, 0.17], sums to 1, temperature-calibrated
answers[0].confidence      // 0 = uniform, 1 = certain
answers[1].yes             // 0.93
```

`KevRouter` wraps the one question OS1 asks before every chat turn (casual / deviceAction / assistant) and returns
`nil` when Kev is not confident enough, so the caller falls back to its default path.

## What is ported

| kev (Python) | here |
|---|---|
| `kev.model.encode` — `<state>` prefix, `<q> instr <opt> … </opt> <decide>` branches, `<\|x\|>` → `<¦x¦>` sanitising | `KevEncoder` |
| `kev.mlx_model.MLXDecisionModel` — merged Qwen3.5 backbone, hidden state at `<decide>` / `</opt>` | `KevModel` on `MLXLLM.Qwen35Model` |
| `kev.model.PointerHead` — `(k(h_opts) @ q(h_decide)) / sqrt(dp) / temperature` | `PointerHead` |
| `confidence` for choice / score questions | `KevAnswer.confidence` |

Each question is run as its own row (`state + branch`), which is exactly what kev's block-causal mask lets a branch
attend to; batching rows is how kev's MLX backend runs too. `option_isolation` checkpoints are refused (they need the
packed mask, which kev's MLX backend does not implement either).

Hidden states come out of mlx-swift-lm's Qwen3.5 model through its `mtp.lastHiddenStates` output key (the model emits
the final-norm hidden states for every token when `mtp.emitDrafterState` is set); nothing in mlx-swift-lm is forked.

## Checkpoints

A Kev release on the Hub is a LoRA adapter + `head.pt` on top of `Qwen/Qwen3.5-0.8B-Base`. `scripts/convert.py` merges
the adapter (fp32, one rounding, like `kev.mlx_model.merge_lora`), drops the LM head, optionally quantizes, and writes
an mlx-swift-lm-loadable directory with `head.safetensors` and a `kev.json` (head dim, fitted temperature, hashes):

```bash
uv run --python 3.12 --with mlx-lm --with torch --with huggingface_hub --with safetensors --with numpy \
  scripts/convert.py --bits 4 --out build/kev-0.8b-mlx-4bit      # 428 MiB; --bits 8 → 787 MiB; --bits 0 → bf16, 1.4 GiB
```

Upload the directory to a Hub model repo and pass its id to `KevModel.load(hubID:)`, or ship it in the app bundle and
use `KevModel.load(directory:)`.

## Parity

`scripts/parity/records.json` holds a few records; `reference.py` scores them with kev's own MLX backend
(`uv run --extra mlx python scripts/parity/reference.py` inside a kev checkout) into `reference.json`, and
`KevParityTests` compares token ids, readout positions and probabilities against it. Token ids and positions match
exactly on every record. Probabilities against kev's reference (8 questions, M-series Mac):

| checkpoint | size | max \|Δp\| | clear decisions kept (ref margin > 0.05) | near-ties flipped |
|---|---|---|---|---|
| bf16 | 1.4 GiB | 0.009 | 6 / 6 | 0 |
| 8-bit | 787 MiB | 0.020 | 6 / 6 | 1 (0.497 / 0.503) |
| 4-bit | 428 MiB | 0.149 | 6 / 6 | 2 (margins 0.017 and 0.006) |

So the port is exact to the precision of the weights. 4-bit squashes the margins (a 0.56 / 0.25 call became 0.41 / 0.36),
which matters because `confidence` is what the router thresholds on; 8-bit is the sensible default for a phone.

Run tests with `xcodebuild`, not `swift test` — MLX's Metal library is only built that way:

```bash
xcodebuild test -scheme Kev -destination 'platform=macOS' -skipPackagePluginValidation -skipMacroValidation
```

The parity suite is skipped unless `build/kev-0.8b-mlx-4bit` (or `KEV_CHECKPOINT`) and `reference.json` exist.

## Status

Verified on an Apple Silicon Mac. Not yet measured on iPhone: load time, memory (the 4-bit backbone is ~0.5 GB of
weights), decision latency, battery. Do not treat it as production-ready for iOS until those numbers exist.
