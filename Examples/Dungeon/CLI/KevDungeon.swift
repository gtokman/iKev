import Foundation
import Kev
import KevDungeonGame

/// `kev-dungeon`: a game loop that asks Kev for the hero's next move every turn.
///
///     kev-dungeon [--hub gtokman/iKev | --checkpoint DIR] [--seed N] [--turns N] [--delay SECONDS] [--random] [--quiet]
@main
struct KevDungeon {
    struct Options {
        var hubID = "gtokman/iKev"
        var checkpoint: URL?
        var seed = UInt64.random(in: 0 ..< 1_000_000)
        var turns = 60
        var delay: Double = 0
        var random = false
        var quiet = false

        init(arguments: [String]) throws {
            var it = arguments.dropFirst().makeIterator()
            func value(_ flag: String) throws -> String {
                guard let v = it.next() else { throw Usage.missingValue(flag) }
                return v
            }
            func number<T: LosslessStringConvertible>(_ flag: String) throws -> T {
                guard let n = T(try value(flag)) else { throw Usage.badValue(flag) }
                return n
            }
            while let arg = it.next() {
                switch arg {
                case "--hub": hubID = try value(arg)
                case "--checkpoint": checkpoint = URL(filePath: try value(arg))
                case "--seed": seed = try number(arg)
                case "--turns": turns = try number(arg)
                case "--delay": delay = try number(arg)
                case "--random": random = true
                case "--quiet": quiet = true
                case "-h", "--help": throw Usage.help
                default: throw Usage.unknown(arg)
                }
            }
        }
    }

    enum Usage: Error, CustomStringConvertible {
        case help, missingValue(String), badValue(String), unknown(String)

        var description: String {
            let usage = """
                usage: kev-dungeon [--hub ID | --checkpoint DIR] [--seed N] [--turns N] [--delay SECONDS] [--random] [--quiet]
                  --hub ID          Hub repo to download the checkpoint from (default gtokman/iKev)
                  --checkpoint DIR  local converted checkpoint instead of the Hub
                  --seed N          dungeon layout and goblin dice (printed each game, so a game can be replayed)
                  --turns N         turns before the torch burns out (default 60)
                  --delay SECONDS   pause between turns so you can watch
                  --random          play uniformly random legal moves instead of asking Kev (no model needed)
                  --quiet           only print the final result
                """
            switch self {
            case .help: return usage
            case .missingValue(let f): return "\(f) needs a value\n\(usage)"
            case .badValue(let f): return "bad value for \(f)\n\(usage)"
            case .unknown(let a): return "unknown argument \(a)\n\(usage)"
            }
        }
    }

    static func main() async {
        do {
            let options = try Options(arguments: CommandLine.arguments)
            var player: any Player
            if options.random {
                player = RandomPlayer(rng: SeededGenerator(seed: options.seed &+ 1))
            } else {
                let started = Date()
                let model: KevModel
                if let checkpoint = options.checkpoint {
                    model = try await KevModel.load(directory: checkpoint)
                } else {
                    model = try await KevModel.load(hubID: options.hubID)
                }
                print("Loaded \(await model.metadata.run) in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
                player = KevPlayer(model: model)
            }
            let result = try await play(seed: options.seed, turns: options.turns, player: &player, delay: options.delay, quiet: options.quiet)
            print(result)
        } catch let usage as Usage {
            print(usage.description)
            exit(usage.description.hasPrefix("usage") ? 0 : 2)
        } catch {
            print("error: \(error)")
            exit(1)
        }
    }

    static func play(seed: UInt64, turns: Int, player: inout any Player, delay: Double, quiet: Bool) async throws -> String {
        var dungeon = Dungeon(seed: seed, maxTurns: turns)
        print("Seed \(seed). Hero @ must reach the exit >. g goblin, ! potion, $ gold.")
        if !quiet { print(dungeon.map) }
        var decideTimes: [Double] = []
        while dungeon.outcome == .playing {
            let started = Date()
            let turn = try await player.choose(in: dungeon)
            decideTimes.append(Date().timeIntervalSince(started))
            let events = dungeon.apply(turn.move)
            if !quiet {
                print(render(turn: turn, dungeon: dungeon, events: events))
                if delay > 0 { try await Task.sleep(for: .seconds(delay)) }
            }
        }
        let verdict: String
        switch dungeon.outcome {
        case .escaped: verdict = "ESCAPED"
        case .died: verdict = "DIED"
        case .outOfTurns: verdict = "LOST IN THE DARK"
        case .playing: verdict = "?"
        }
        let p50 = decideTimes.sorted()[decideTimes.count / 2]
        return
            "\(verdict) on turn \(dungeon.turn) with \(dungeon.gold) gold and \(max(dungeon.hp, 0)) HP (seed \(seed), decide p50 \(Int(p50 * 1000)) ms)"
    }

    static func bar(_ p: Double, width: Int = 10) -> String {
        let filled = Int((p * Double(width)).rounded())
        return String(repeating: "█", count: filled) + String(repeating: "░", count: width - filled)
    }

    static func render(turn: Turn, dungeon: Dungeon, events: [String]) -> String {
        var lines = ["", "── turn \(dungeon.turn) ──────────────────────────────────"]
        for (i, move) in turn.moves.enumerated() {
            let marker = move == turn.move ? "→" : " "
            lines.append(
                "\(marker) \(move.name.padding(toLength: 12, withPad: " ", startingAt: 0)) \(bar(turn.probabilities[i])) \(String(format: "%.2f", turn.probabilities[i]))  \(turn.descriptions[i])"
            )
        }
        if !turn.danger.isEmpty {
            let level = KevPlayer.dangerLevels[turn.danger.indices.max { turn.danger[$0] < turn.danger[$1] }!]
            lines.append(
                "  danger: \(level) (" + zip(KevPlayer.dangerLevels, turn.danger).map { "\($0) \(String(format: "%.2f", $1))" }.joined(separator: ", ")
                    + "), confidence \(String(format: "%.2f", turn.confidence))")
        }
        lines.append("  " + events.joined(separator: " "))
        lines.append(dungeon.map)
        lines.append("  HP \(max(dungeon.hp, 0))/\(Dungeon.heroMaxHP)  potions \(dungeon.potions)  gold \(dungeon.gold)")
        return lines.joined(separator: "\n")
    }
}
