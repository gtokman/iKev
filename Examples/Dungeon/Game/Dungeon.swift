import Foundation

/// A tiny roguelike: reach the exit alive, grab gold on the way, don't get eaten. Deterministic for a seed so a run
/// can be replayed; no model code in here.
public struct Point: Hashable, Sendable {
    public var x: Int
    public var y: Int

    public init(x: Int, y: Int) {
        self.x = x
        self.y = y
    }

    public static func + (lhs: Point, rhs: Point) -> Point { Point(x: lhs.x + rhs.x, y: lhs.y + rhs.y) }

    public func distance(to other: Point) -> Int { abs(x - other.x) + abs(y - other.y) }

    public func isAdjacent(to other: Point) -> Bool { distance(to: other) == 1 }
}

public enum Direction: String, CaseIterable, Sendable {
    case north, south, east, west

    public var delta: Point {
        switch self {
        case .north: Point(x: 0, y: -1)
        case .south: Point(x: 0, y: 1)
        case .east: Point(x: 1, y: 0)
        case .west: Point(x: -1, y: 0)
        }
    }

    public var opposite: Direction {
        switch self {
        case .north: .south
        case .south: .north
        case .east: .west
        case .west: .east
        }
    }
}

public enum Move: Hashable, Sendable {
    case step(Direction)
    case drinkPotion

    public var name: String {
        switch self {
        case .step(let direction): direction.rawValue
        case .drinkPotion: "drink potion"
        }
    }
}

public enum Outcome: Equatable, Sendable {
    case playing
    case escaped
    case died
    case outOfTurns
}

public struct Goblin: Hashable, Sendable {
    public var position: Point
    public var hp: Int
}

/// SplitMix64, so games replay identically across platforms.
public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public struct Dungeon: Sendable {
    public static let width = 11
    public static let height = 7
    public static let heroMaxHP = 5
    public static let goblinHP = 2
    public static let potionHeal = 3

    public let seed: UInt64
    public let maxTurns: Int
    private(set) var rng: SeededGenerator
    public private(set) var walls: Set<Point> = []
    public private(set) var exit: Point
    public private(set) var hero: Point
    public private(set) var hp = Dungeon.heroMaxHP
    public private(set) var potions = 0
    public private(set) var gold = 0
    public private(set) var turn = 0
    public private(set) var goblins: [Goblin] = []
    public private(set) var potionsOnFloor: Set<Point> = []
    public private(set) var goldOnFloor: Set<Point> = []
    public private(set) var outcome = Outcome.playing
    /// Direction of the hero's last step, so retracing it can be called out.
    public private(set) var lastStep: Direction?

    public init(seed: UInt64, maxTurns: Int = 60) {
        self.seed = seed
        self.maxTurns = maxTurns
        rng = SeededGenerator(seed: seed)
        hero = Point(x: 1, y: Int.random(in: 1 ..< Dungeon.height - 1, using: &rng))
        exit = Point(x: Dungeon.width - 2, y: Int.random(in: 1 ..< Dungeon.height - 1, using: &rng))
        repeat {
            walls = Set(Self.border)
            for _ in 0 ..< 8 {
                let p = randomFreeCell()
                if p != hero && p != exit { walls.insert(p) }
            }
        } while !exitReachable()
        goblins = (0 ..< 3).map { _ in
            Goblin(position: randomFreeCell(minDistanceFromHero: 3), hp: Dungeon.goblinHP)
        }
        for _ in 0 ..< 2 { potionsOnFloor.insert(randomFreeCell(minDistanceFromHero: 1)) }
        for _ in 0 ..< 4 { goldOnFloor.insert(randomFreeCell(minDistanceFromHero: 1)) }
    }

    private static var border: [Point] {
        var cells: [Point] = []
        for x in 0 ..< width { cells += [Point(x: x, y: 0), Point(x: x, y: height - 1)] }
        for y in 0 ..< height { cells += [Point(x: 0, y: y), Point(x: width - 1, y: y)] }
        return cells
    }

    private mutating func randomFreeCell(minDistanceFromHero: Int = 0) -> Point {
        while true {
            let p = Point(
                x: Int.random(in: 1 ..< Dungeon.width - 1, using: &rng),
                y: Int.random(in: 1 ..< Dungeon.height - 1, using: &rng))
            if isFree(p), p != exit, p.distance(to: hero) >= minDistanceFromHero { return p }
        }
    }

    private func isFree(_ p: Point) -> Bool {
        !walls.contains(p) && p != hero && goblin(at: p) == nil && !potionsOnFloor.contains(p)
            && !goldOnFloor.contains(p)
    }

    private func exitReachable() -> Bool {
        var seen: Set<Point> = [hero]
        var queue = [hero]
        while let p = queue.popLast() {
            if p == exit { return true }
            for d in Direction.allCases {
                let n = p + d.delta
                if !walls.contains(n), !seen.contains(n) {
                    seen.insert(n)
                    queue.append(n)
                }
            }
        }
        return false
    }

    public func goblin(at p: Point) -> Goblin? { goblins.first { $0.position == p } }

    /// Walking distance to the exit around the walls (goblins and loot do not block), or nil when cut off.
    public func stepsToExit(from start: Point) -> Int? {
        var distance: [Point: Int] = [exit: 0]
        var queue = [exit]
        var index = 0
        while index < queue.count {
            let p = queue[index]
            index += 1
            if p == start { return distance[p] }
            for d in Direction.allCases {
                let n = p + d.delta
                if !walls.contains(n), distance[n] == nil {
                    distance[n] = distance[p]! + 1
                    queue.append(n)
                }
            }
        }
        return nil
    }

    // MARK: Moves

    /// Only legal moves are offered, so the player never has to reason about walls.
    public var availableMoves: [Move] {
        var moves: [Move] = Direction.allCases.compactMap { d in
            walls.contains(hero + d.delta) ? nil : .step(d)
        }
        if potions > 0 && hp < Dungeon.heroMaxHP { moves.append(.drinkPotion) }
        return moves
    }

    /// What the hero would find by taking `move`: the option text the player chooses among.
    public func describe(_ move: Move) -> String {
        switch move {
        case .drinkPotion:
            return "drink a potion to heal \(Dungeon.potionHeal) HP (you have \(potions))"
        case .step(let d):
            let target = hero + d.delta
            if let g = goblin(at: target) {
                return "attack the goblin standing there (it has \(g.hp) HP left)"
            }
            if target == exit { return "walk through the exit and win" }
            var parts: [String] = []
            if potionsOnFloor.contains(target) { parts.append("pick up the potion lying there") }
            if goldOnFloor.contains(target) { parts.append("pick up the gold lying there") }
            if parts.isEmpty {
                let before = stepsToExit(from: hero) ?? .max
                let after = stepsToExit(from: target) ?? .max
                parts.append(after < before ? "step closer to the exit" : "step away from the exit")
                if let lastStep, d == lastStep.opposite { parts.append("back where you just came from") }
            }
            if goblins.contains(where: { $0.position.isAdjacent(to: target) }) {
                parts.append("a goblin could bite you there")
            }
            return parts.joined(separator: ", ")
        }
    }

    /// Resolve one turn: the hero acts, then every goblin acts. Returns what happened, in order.
    @discardableResult
    public mutating func apply(_ move: Move) -> [String] {
        precondition(outcome == .playing, "game is over")
        precondition(availableMoves.contains(move), "illegal move \(move)")
        var events: [String] = []
        turn += 1
        switch move {
        case .drinkPotion:
            potions -= 1
            hp = min(Dungeon.heroMaxHP, hp + Dungeon.potionHeal)
            events.append("You drink a potion (HP \(hp)/\(Dungeon.heroMaxHP)).")
        case .step(let d):
            let target = hero + d.delta
            if let i = goblins.firstIndex(where: { $0.position == target }) {
                goblins[i].hp -= 1
                if goblins[i].hp <= 0 {
                    goblins.remove(at: i)
                    gold += 1
                    events.append("You slay the goblin to the \(d.rawValue)! It drops a gold coin.")
                } else {
                    events.append("You strike the goblin to the \(d.rawValue) (\(goblins[i].hp) HP left).")
                }
            } else {
                hero = target
                lastStep = d
                events.append("You step \(d.rawValue).")
                if potionsOnFloor.remove(target) != nil {
                    potions += 1
                    events.append("You pick up a potion (\(potions) carried).")
                }
                if goldOnFloor.remove(target) != nil {
                    gold += 1
                    events.append("You pick up gold (\(gold) total).")
                }
                if hero == exit {
                    outcome = .escaped
                    events.append("You escape the dungeon with \(gold) gold!")
                    return events
                }
            }
        }
        for i in goblins.indices {
            let g = goblins[i]
            if g.position.isAdjacent(to: hero) {
                hp -= 1
                events.append("The goblin bites you (HP \(max(hp, 0))/\(Dungeon.heroMaxHP)).")
                if hp <= 0 {
                    outcome = .died
                    events.append("You die on turn \(turn) with \(gold) gold.")
                    return events
                }
            } else if g.position.distance(to: hero) <= 4, Int.random(in: 0 ..< 10, using: &rng) < 7 {
                let dx = hero.x - g.position.x
                let dy = hero.y - g.position.y
                let preferred: [Point] =
                    abs(dx) >= abs(dy)
                    ? [Point(x: dx.signum(), y: 0), Point(x: 0, y: dy.signum())]
                    : [Point(x: 0, y: dy.signum()), Point(x: dx.signum(), y: 0)]
                for step in preferred where step != Point(x: 0, y: 0) {
                    let next = g.position + step
                    if next != hero, next != exit, isFree(next) {
                        goblins[i].position = next
                        break
                    }
                }
            }
        }
        if turn >= maxTurns {
            outcome = .outOfTurns
            events.append("The torch burns out after \(maxTurns) turns. You are lost in the dark with \(gold) gold.")
        }
        return events
    }

    // MARK: Text

    public var map: String {
        (0 ..< Dungeon.height).map { y in
            String(
                (0 ..< Dungeon.width).map { x -> Character in
                    let p = Point(x: x, y: y)
                    if p == hero { return "@" }
                    if p == exit { return ">" }
                    if goblin(at: p) != nil { return "g" }
                    if walls.contains(p) { return "#" }
                    if potionsOnFloor.contains(p) { return "!" }
                    if goldOnFloor.contains(p) { return "$" }
                    return "."
                })
        }.joined(separator: "\n")
    }

    private var healthWord: String {
        switch hp {
        case Dungeon.heroMaxHP: "unhurt"
        case 3...: "lightly hurt"
        case 2: "badly hurt"
        default: "nearly dead"
        }
    }

    public func relative(_ p: Point) -> String {
        let dx = p.x - hero.x
        let dy = p.y - hero.y
        var parts: [String] = []
        if dx != 0 { parts.append("\(abs(dx)) step\(abs(dx) == 1 ? "" : "s") \(dx > 0 ? "east" : "west")") }
        if dy != 0 { parts.append("\(abs(dy)) step\(abs(dy) == 1 ? "" : "s") \(dy > 0 ? "south" : "north")") }
        return parts.isEmpty ? "right here" : parts.joined(separator: " and ")
    }

    /// The state Kev reads each turn: the situation in plain words, then the map.
    public var narrative: String {
        var lines: [String] = []
        lines.append(
            "You are a hero in a dungeon. Reach the exit alive; gold is a bonus. Goblins bite for 1 HP each turn they stand next to you; hitting a goblin twice kills it."
        )
        lines.append(
            "Turn \(turn + 1) of \(maxTurns). HP \(hp)/\(Dungeon.heroMaxHP) (\(healthWord)). Potions carried: \(potions). Gold: \(gold)."
        )
        lines.append("The exit is \(relative(exit)), \(stepsToExit(from: hero) ?? 0) steps away by the shortest path.")
        if goblins.isEmpty {
            lines.append("No goblins are left.")
        } else {
            for g in goblins.sorted(by: { $0.position.distance(to: hero) < $1.position.distance(to: hero) }) {
                if g.position.isAdjacent(to: hero) {
                    lines.append("A goblin (\(g.hp) HP) is right next to you, \(relative(g.position)), about to bite.")
                } else {
                    lines.append("A goblin (\(g.hp) HP) is \(relative(g.position)).")
                }
            }
        }
        if let p = potionsOnFloor.min(by: { $0.distance(to: hero) < $1.distance(to: hero) }) {
            lines.append("The nearest potion lies \(relative(p)).")
        }
        if let p = goldOnFloor.min(by: { $0.distance(to: hero) < $1.distance(to: hero) }) {
            lines.append("The nearest gold lies \(relative(p)).")
        }
        lines.append("Map (@ you, > exit, g goblin, ! potion, $ gold, # wall):")
        lines.append(map)
        return lines.joined(separator: "\n")
    }
}
