import KevDungeonGame
import SwiftUI

struct ContentView: View {
    @State private var model = GameModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    StatusBar(dungeon: model.dungeon)
                    DungeonGrid(dungeon: model.dungeon)
                    OutcomeBanner(dungeon: model.dungeon)
                    DecisionPanel(turn: model.lastTurn, thinking: model.isThinking, milliseconds: model.lastDecideMilliseconds)
                    if !model.events.isEmpty {
                        Text(model.events.joined(separator: " "))
                            .font(.callout)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                    }
                    BrainStatus(brain: model.brain)
                }
                .padding(.vertical)
            }
            .navigationTitle("Kev's Dungeon")
            .navigationBarTitleDisplayMode(.inline)
            .safeAreaInset(edge: .bottom) {
                Controls(model: model)
            }
        }
        .task { await model.loadModel() }
    }
}

private struct StatusBar: View {
    let dungeon: Dungeon

    var body: some View {
        HStack(spacing: 16) {
            Label {
                Text("\(max(dungeon.hp, 0))/\(Dungeon.heroMaxHP)")
            } icon: {
                Image(systemName: "heart.fill").foregroundStyle(.red)
            }
            Label("\(dungeon.potions)", systemImage: "flask.fill").foregroundStyle(.purple)
            Label("\(dungeon.gold)", systemImage: "dollarsign.circle.fill").foregroundStyle(.yellow)
            Spacer()
            Text("turn \(dungeon.turn)/\(dungeon.maxTurns)")
                .foregroundStyle(.secondary)
            Text("seed \(dungeon.seed)")
                .foregroundStyle(.secondary)
        }
        .font(.subheadline.monospacedDigit())
        .padding(.horizontal)
    }
}

private struct DungeonGrid: View {
    let dungeon: Dungeon

    var body: some View {
        GeometryReader { geometry in
            let cell = geometry.size.width / CGFloat(Dungeon.width)
            VStack(spacing: 0) {
                ForEach(0 ..< Dungeon.height, id: \.self) { y in
                    HStack(spacing: 0) {
                        ForEach(0 ..< Dungeon.width, id: \.self) { x in
                            Cell(dungeon: dungeon, point: Point(x: x, y: y))
                                .frame(width: cell, height: cell)
                        }
                    }
                }
            }
        }
        .aspectRatio(CGFloat(Dungeon.width) / CGFloat(Dungeon.height), contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
        .animation(.easeInOut(duration: 0.2), value: dungeon.hero)
    }
}

private struct Cell: View {
    let dungeon: Dungeon
    let point: Point

    var body: some View {
        ZStack {
            Rectangle().fill(background)
            if let symbol {
                Text(symbol)
                    .font(.system(size: 20))
                    .minimumScaleFactor(0.5)
            }
        }
    }

    private var background: Color {
        if dungeon.walls.contains(point) { return Color(.sRGB, white: 0.18) }
        if point == dungeon.exit { return .green.opacity(0.35) }
        return (point.x + point.y).isMultiple(of: 2) ? Color(.sRGB, white: 0.92) : Color(.sRGB, white: 0.86)
    }

    private var symbol: String? {
        if point == dungeon.hero { return dungeon.hp > 0 ? "🧙" : "💀" }
        if dungeon.goblin(at: point) != nil { return "👺" }
        if point == dungeon.exit { return "🚪" }
        if dungeon.potionsOnFloor.contains(point) { return "🧪" }
        if dungeon.goldOnFloor.contains(point) { return "💰" }
        return nil
    }
}

private struct OutcomeBanner: View {
    let dungeon: Dungeon

    var body: some View {
        switch dungeon.outcome {
        case .playing:
            EmptyView()
        case .escaped:
            banner("Escaped with \(dungeon.gold) gold on turn \(dungeon.turn)!", color: .green)
        case .died:
            banner("Died on turn \(dungeon.turn) with \(dungeon.gold) gold.", color: .red)
        case .outOfTurns:
            banner("The torch burned out. Lost in the dark with \(dungeon.gold) gold.", color: .orange)
        }
    }

    private func banner(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(color.opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
    }
}

/// Kev's probabilities over the legal moves for the last turn, and the danger score.
private struct DecisionPanel: View {
    let turn: Turn?
    let thinking: Bool
    let milliseconds: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("What should the hero do next?")
                    .font(.headline)
                Spacer()
                if thinking {
                    ProgressView().controlSize(.small)
                } else if let milliseconds {
                    Text("\(milliseconds) ms").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            if let turn {
                ForEach(turn.moves.indices, id: \.self) { i in
                    OptionRow(
                        name: turn.moves[i].name, description: turn.descriptions[i],
                        probability: turn.probabilities[i], chosen: turn.moves[i] == turn.move)
                }
                if !turn.danger.isEmpty {
                    DangerMeter(probabilities: turn.danger, confidence: turn.confidence)
                        .padding(.top, 4)
                }
            } else {
                Text("Press play. Every turn the dungeon is written out as text, the legal moves become the options of one Kev question, and the pointer head picks one.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }
}

private struct OptionRow: View {
    let name: String
    let description: String
    let probability: Double
    let chosen: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(name)
                    .font(.subheadline.weight(chosen ? .bold : .regular))
                Spacer()
                Text(probability, format: .percent.precision(.fractionLength(0)))
                    .font(.subheadline.monospacedDigit())
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color(.tertiarySystemFill))
                    Capsule()
                        .fill(chosen ? Color.accentColor : Color.secondary.opacity(0.5))
                        .frame(width: geometry.size.width * probability)
                }
            }
            .frame(height: 8)
            Text(description)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .animation(.easeOut(duration: 0.25), value: probability)
    }
}

private struct DangerMeter: View {
    let probabilities: [Double]
    let confidence: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("How dangerous is this?").font(.subheadline)
                Spacer()
                Text("confidence \(confidence, format: .number.precision(.fractionLength(2)))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            GeometryReader { geometry in
                HStack(spacing: 2) {
                    ForEach(probabilities.indices, id: \.self) { i in
                        Rectangle()
                            .fill(colors[i])
                            .frame(width: max(0, geometry.size.width * probabilities[i] - 2))
                    }
                }
            }
            .frame(height: 10)
            .clipShape(Capsule())
            HStack {
                ForEach(probabilities.indices, id: \.self) { i in
                    Text("\(KevPlayer.dangerLevels[i]) \(probabilities[i], format: .percent.precision(.fractionLength(0)))")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(colors[i])
                    if i < probabilities.count - 1 { Spacer() }
                }
            }
        }
        .animation(.easeOut(duration: 0.25), value: probabilities)
    }

    private let colors: [Color] = [.green, .orange, .red]
}

private struct BrainStatus: View {
    let brain: GameModel.Brain

    var body: some View {
        Group {
            switch brain {
            case .loading(let fraction):
                VStack(spacing: 6) {
                    ProgressView(value: fraction)
                    Text(fraction.map { "Downloading \(GameModel.hubID)@\(GameModel.revision)… \($0, format: .percent.precision(.fractionLength(0)))" }
                        ?? "Loading Kev…")
                }
            case .kev:
                Label("Kev-0.8B (\(GameModel.hubID)@\(GameModel.revision)) is playing", systemImage: "brain")
            case .random(let reason):
                Label(reason, systemImage: "dice")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal)
    }
}

private struct Controls: View {
    @Bindable var model: GameModel

    private var ready: Bool {
        if case .loading = model.brain { return false }
        return true
    }

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Button {
                    model.newGame()
                } label: {
                    Label("New", systemImage: "shuffle")
                }
                Button {
                    model.replay()
                } label: {
                    Label("Replay", systemImage: "arrow.counterclockwise")
                }
                Spacer()
                Button {
                    Task { await model.step() }
                } label: {
                    Label("Step", systemImage: "forward.frame.fill")
                }
                .disabled(!ready || model.isPlaying || model.dungeon.outcome != .playing)
                Button {
                    model.togglePlay()
                } label: {
                    Label(model.isPlaying ? "Pause" : "Play", systemImage: model.isPlaying ? "pause.fill" : "play.fill")
                        .frame(minWidth: 80)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!ready || model.dungeon.outcome != .playing)
            }
            HStack {
                Image(systemName: "tortoise")
                Slider(value: $model.secondsPerTurn, in: 0.1 ... 2.0)
                Image(systemName: "hare")
            }
            .foregroundStyle(.secondary)
        }
        .padding()
        .background(.bar)
    }
}

#Preview {
    ContentView()
}
