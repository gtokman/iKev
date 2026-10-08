import Testing

@testable import KevDungeonGame

/// The game engine alone: no model, no Metal, so these run anywhere.
struct DungeonTests {
    @Test func seededGamesReplayIdentically() {
        let a = Dungeon(seed: 42)
        let b = Dungeon(seed: 42)
        #expect(a.map == b.map)
        #expect(a.narrative == b.narrative)
    }

    @Test(arguments: [UInt64](1 ... 50))
    func exitIsReachableAndMovesAreLegal(seed: UInt64) {
        let dungeon = Dungeon(seed: seed)
        #expect(!dungeon.availableMoves.isEmpty)
        for move in dungeon.availableMoves {
            if case .step(let d) = move {
                #expect(!dungeon.walls.contains(dungeon.hero + d.delta))
            }
        }
        #expect(dungeon.goblins.count == 3)
        #expect(dungeon.goblins.allSatisfy { $0.position.distance(to: dungeon.hero) >= 3 })
        #expect(!dungeon.availableMoves.contains(.drinkPotion), "no potion to drink at the start")
    }

    @Test(arguments: [UInt64](1 ... 50))
    func randomPlayerGameEnds(seed: UInt64) async throws {
        var dungeon = Dungeon(seed: seed, maxTurns: 40)
        var player = RandomPlayer(rng: SeededGenerator(seed: seed))
        var turns = 0
        while dungeon.outcome == .playing {
            let turn = try await player.choose(in: dungeon)
            #expect(turn.moves == dungeon.availableMoves)
            #expect(turn.descriptions.count == turn.moves.count)
            dungeon.apply(turn.move)
            turns += 1
        }
        #expect(turns == dungeon.turn)
        #expect(turns <= 40)
        if dungeon.outcome == .outOfTurns { #expect(dungeon.turn == 40) }
        if dungeon.outcome == .died { #expect(dungeon.hp <= 0) }
        if dungeon.outcome == .escaped { #expect(dungeon.hero == dungeon.exit) }
    }

    @Test func narrativeMentionsWhatMatters() {
        let dungeon = Dungeon(seed: 7)
        let text = dungeon.narrative
        #expect(text.contains("HP 5/5 (unhurt)"))
        #expect(text.contains("The exit is"))
        #expect(text.contains("A goblin (2 HP) is"))
        #expect(text.contains(dungeon.map))
        for move in dungeon.availableMoves {
            #expect(!dungeon.describe(move).isEmpty)
        }
    }

    @Test func steppingOntoLootPicksItUp() {
        for seed in UInt64(1) ... 200 {
            var dungeon = Dungeon(seed: seed)
            guard
                let move = dungeon.availableMoves.first(where: {
                    if case .step(let d) = $0 { return dungeon.goldOnFloor.contains(dungeon.hero + d.delta) }
                    return false
                })
            else { continue }
            #expect(dungeon.describe(move).contains("gold"))
            let events = dungeon.apply(move)
            #expect(dungeon.gold == 1)
            #expect(events.contains { $0.contains("pick up gold") })
            return
        }
        Issue.record("no seed in 1...200 starts next to gold")
    }
}
