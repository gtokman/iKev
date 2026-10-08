# iKev

[Kev](https://github.com/jaredpalmer/kev) decision models on Apple devices, running on [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

Kev is not a chat model. It answers fixed-choice questions about a piece of text ("state") in one prefill pass: no
generation, calibrated probabilities, a confidence you can threshold on. This package is a faithful port of
`kev.mlx_model` so the same checkpoints give the same answers on iPhone, iPad and Mac.

```swift
import Kev

let model = try await KevModel.load(hubID: "gtokman/iKev")   // downloads once, then cached
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

Published checkpoint: [`gtokman/iKev`](https://huggingface.co/gtokman/iKev) (8-bit, 787 MB, public). Smoke-test the
download + one decision from the Hub (needs network, ~0.8 GB cache):

```sh
TEST_RUNNER_KEV_HUB=gtokman/iKev xcodebuild test -scheme Kev -destination 'platform=macOS' \
  -skipPackagePluginValidation -skipMacroValidation -only-testing:KevTests/KevHubTests
```

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

## Benchmark

`scripts/bench/messages.json` holds 50 labelled messages (16 casual, 14 deviceAction, 20 assistant) and
`scripts/bench/questions.json` the router wordings tried. `KevBenchmark` loads a checkpoint cold, routes each
message once and writes `build/bench-<checkpoint>-<platform>.md` (per-message probabilities, confidence
sweep, question-wording comparison):

```sh
TEST_RUNNER_KEV_BENCH=1 TEST_RUNNER_KEV_CHECKPOINT=$PWD/build/kev-0.8b-mlx-8bit \
  xcodebuild test -scheme Kev -destination 'platform=macOS' \
  -skipPackagePluginValidation -skipMacroValidation -only-testing:KevTests/KevBenchmark
```

Apple M4 Pro (virtualised, 16 GB), macOS 26.5, `Memory.cacheLimit = 256 MB`:

| checkpoint | cold load | decide p50 / p95 | footprint loaded / peak | accuracy @0 | handled / accuracy @0.3 |
|---|---:|---:|---:|---:|---:|
| 8-bit (787 MB) | 1.4 s | 29 / 32 ms | 1071 / 1190 MB | 43/50 | 43/50 → 91% |
| 4-bit (428 MB) | 1.5 s | 30 / 32 ms | 637 / 835 MB | 46/50 | 40/50 → 92% |

Notes:

- Decide latency is prefill-bound and identical for both precisions; the difference is memory.
- Without a cache limit MLX's buffer cache let the 8-bit footprint peak at 1.7 GB; 32 MB halved
  throughput (63 ms). 256 MB is the compromise, set in `KevModel.load`.
- The 4-bit/8-bit accuracy gap (3 messages) is within noise for 50 samples; the parity run shows 4-bit
  squashes probability margins, which is what the confidence threshold acts on.
- Remaining misses are all "chatty acknowledgement vs assistant" (`sounds good`, `you're the best`) and
  "read my data vs act on my device" (`what's on my calendar tomorrow?`, `remember that …`). The second
  group is a product decision about where calendar/memory *reads* should go; the router question can be
  reworded in `scripts/bench/questions.json` and re-measured.
- **iOS Simulator is not supported** — MLX needs a real Metal GPU family (see mlx-swift's
  troubleshooting doc); the simulator process aborts inside Metal device setup. `KevModel.load` throws
  `KevModelError.simulatorUnsupported` there so an app can fall back instead of crashing. The package
  itself builds for the simulator, so the rest of an app still runs. Device numbers need a real iPhone.

## Example: kev-dungeon

`Examples/Dungeon` is a game loop with Kev as the player. A tiny roguelike (reach the exit alive, gold is a bonus,
goblins bite) runs as a `while` loop; every turn the game writes the situation in plain words plus an ASCII map as the
Kev *state*, offers the legal moves as the *options* of one choice question, and takes whatever the pointer head picks.
No generation, no parsing, no illegal moves. A second score question rates how dangerous the spot is.

```sh
scripts/kev-dungeon.sh --release                                   # downloads gtokman/iKev, random seed
scripts/kev-dungeon.sh --release --checkpoint build/kev-0.8b-mlx-8bit --seed 7 --delay 0.5
scripts/kev-dungeon.sh --random --seed 7                           # uniform legal moves, no model: the baseline
```

```
── turn 6 ──────────────────────────────────
  north        ███░░░░░░░ 0.30  attack the goblin standing there (it has 2 HP left)
  south        ██░░░░░░░░ 0.20  pick up the potion lying there
→ east         ████░░░░░░ 0.38  attack the goblin standing there (it has 2 HP left)
  west         █░░░░░░░░░ 0.13  step away from the exit, back where you just came from
  danger: risky (safe 0.39, risky 0.47, deadly 0.14), confidence 0.17
  You strike the goblin to the east (1 HP left). The goblin bites you (HP 3/5).
```

The script wraps `xcodebuild` because `swift run` cannot find MLX's metallib (same reason as the tests). Kev-0.8B
escapes 6 of seeds 1–11 with the 8-bit checkpoint and fights goblins more than it should; the random baseline escapes 1.
Each decision is one prefill over ~400 tokens, ≈140 ms on an M4 Pro in Release. The option texts are the game being
honest about what a move does (`step closer to the exit` uses the walking distance around walls, `back where you just
came from` marks a reversal, and every option ends with its consequence: `kill it`, `leaving you at 2/5 HP`, `and you
die` — a 0.8B model will not combine the hero's HP in the state with the goblin's HP in the option by itself); change them in `Dungeon.describe` and the narrative in `Dungeon.narrative` to see how the
play changes. The engine tests never touch the model, so they run with plain `swift test --filter DungeonTests`.

### On an iPhone

`Examples/DungeonApp/KevDungeon.xcodeproj` is the same game as a SwiftUI app (iOS 17+, iPhone only): open it, pick
your team, run on a device. It downloads `gtokman/iKev@4bit` on first launch (0.43 GB, progress shown), then plays
with Kev's probabilities, the danger meter and a speed slider; Step/Play/Pause, New (random seed) and Replay (same
seed). The project is generated from `project.yml` with [xcodegen](https://github.com/yonaskolb/XcodeGen) and depends
on the `KevDungeonGame` product of this package, so the engine and the Kev questions are shared with the CLI. In the
Simulator MLX cannot run, so the app falls back to the random player and says so.

## Status

Verified on an Apple Silicon Mac. Not yet measured on iPhone: load time, memory (the 4-bit backbone is ~0.5 GB of
weights), decision latency, battery. Do not treat it as production-ready for iOS until those numbers exist.
