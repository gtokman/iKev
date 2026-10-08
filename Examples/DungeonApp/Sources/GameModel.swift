import Foundation
import Kev
import KevDungeonGame
import Observation

/// The game loop as UI state: load Kev once, then `step()` asks the player for a move and applies it. `play()` keeps
/// stepping with a pause in between until the game ends or the user stops it.
@MainActor
@Observable
final class GameModel {
    enum Brain {
        case loading(fraction: Double?)
        case kev
        /// MLX cannot run in the Simulator (and loading can fail for other reasons); the random player keeps the UI alive.
        case random(reason: String)
    }

    static let hubID = "gtokman/iKev"
    /// 4-bit is the phone-sized checkpoint (0.43 GB download, ~0.65 GB resident); `main` is 8-bit.
    static let revision = "4bit"

    private(set) var brain = Brain.loading(fraction: nil)
    private(set) var dungeon: Dungeon
    private(set) var lastTurn: Turn?
    private(set) var events: [String] = []
    private(set) var isPlaying = false
    private(set) var isThinking = false
    private(set) var lastDecideMilliseconds: Int?
    var secondsPerTurn: Double = 0.8

    private var player: (any Player)?
    private var playTask: Task<Void, Never>?

    init(seed: UInt64 = .random(in: 0 ..< 1_000_000)) {
        dungeon = Dungeon(seed: seed)
    }

    var seed: UInt64 { dungeon.seed }

    func loadModel() async {
        if player != nil { return }
        do {
            let model = try await KevModel.load(hubID: Self.hubID, revision: Self.revision) { progress in
                let fraction = progress.fractionCompleted
                Task { @MainActor [weak self] in
                    guard let self, case .loading = self.brain else { return }
                    self.brain = .loading(fraction: fraction)
                }
            }
            player = KevPlayer(model: model)
            brain = .kev
        } catch KevModelError.simulatorUnsupported {
            player = RandomPlayer(rng: SeededGenerator(seed: seed))
            brain = .random(reason: "MLX needs a real GPU; the Simulator plays random moves. Run on an iPhone to see Kev.")
        } catch {
            player = RandomPlayer(rng: SeededGenerator(seed: seed))
            brain = .random(reason: "Kev failed to load (\(error)); playing random moves.")
        }
    }

    func newGame(seed: UInt64 = .random(in: 0 ..< 1_000_000)) {
        pause()
        dungeon = Dungeon(seed: seed)
        lastTurn = nil
        events = []
        lastDecideMilliseconds = nil
    }

    func replay() { newGame(seed: seed) }

    /// One turn: ask the player, apply the move. No-op while loading, thinking or after the game ended.
    func step() async {
        guard var player, !isThinking, dungeon.outcome == .playing else { return }
        isThinking = true
        defer { isThinking = false }
        let started = ContinuousClock.now
        do {
            let turn = try await player.choose(in: dungeon)
            self.player = player
            lastDecideMilliseconds = Int((ContinuousClock.now - started) / .milliseconds(1))
            lastTurn = turn
            events = dungeon.apply(turn.move)
        } catch {
            events = ["Kev could not decide: \(error)"]
            pause()
        }
    }

    func play() {
        guard !isPlaying, dungeon.outcome == .playing else { return }
        isPlaying = true
        playTask = Task { [weak self] in
            while let self, !Task.isCancelled, self.isPlaying, self.dungeon.outcome == .playing {
                await self.step()
                try? await Task.sleep(for: .seconds(self.secondsPerTurn))
            }
            self?.isPlaying = false
        }
    }

    func pause() {
        isPlaying = false
        playTask?.cancel()
        playTask = nil
    }

    func togglePlay() { isPlaying ? pause() : play() }
}
