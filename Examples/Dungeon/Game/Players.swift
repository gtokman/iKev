import Foundation
import Kev

/// One decision per turn: which legal move, plus how dangerous the situation looks.
public struct Turn: Sendable {
    public let moves: [Move]
    /// What each entry of `moves` would do, as shown to the player.
    public let descriptions: [String]
    public let move: Move
    /// Probability per entry of `moves`, summing to 1.
    public let probabilities: [Double]
    public let confidence: Double
    /// Probability per `KevPlayer.dangerLevels`; empty for players that do not estimate it.
    public let danger: [Double]
}

public protocol Player: Sendable {
    mutating func choose(in dungeon: Dungeon) async throws -> Turn
}

/// Asks Kev. The state is the dungeon's narrative, the options are the legal moves with what each one would do, and
/// the pointer head picks one in a single prefill pass: no generation, no parsing, no illegal moves.
public struct KevPlayer: Player {
    public static let dangerLevels = ["safe", "risky", "deadly"]

    public let model: KevModel

    public init(model: KevModel) { self.model = model }

    public func choose(in dungeon: Dungeon) async throws -> Turn {
        let moves = dungeon.availableMoves
        let descriptions = moves.map(dungeon.describe)
        let options = zip(moves, descriptions).map { "\($0.name): \($1)" }
        let answers = try await model.decide(
            state: dungeon.narrative,
            questions: [
                .choice("What should the hero do next to reach the exit alive?", options: options),
                .score("How dangerous is the hero's situation right now?", levels: Self.dangerLevels),
            ])
        let decision = answers[0]
        return Turn(
            moves: moves, descriptions: descriptions, move: moves[decision.argmax], probabilities: decision.probabilities,
            confidence: decision.confidence, danger: answers[1].probabilities)
    }
}

/// Picks uniformly among the legal moves: the baseline Kev has to beat.
public struct RandomPlayer: Player {
    public var rng: SeededGenerator

    public init(rng: SeededGenerator) { self.rng = rng }

    public mutating func choose(in dungeon: Dungeon) async throws -> Turn {
        let moves = dungeon.availableMoves
        let index = Int.random(in: 0 ..< moves.count, using: &rng)
        return Turn(
            moves: moves, descriptions: moves.map(dungeon.describe), move: moves[index], probabilities: Array(repeating: 1 / Double(moves.count), count: moves.count),
            confidence: 0, danger: [])
    }
}
