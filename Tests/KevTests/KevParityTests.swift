import Foundation
import Testing

@testable import Kev

/// Compares against kev's own MLX backend on scripts/parity/records.json. Needs a converted checkpoint at
/// `KEV_CHECKPOINT` (default build/kev-0.8b-mlx-4bit) and scripts/parity/reference.json from reference.py.
@Suite(.serialized) struct KevParityTests {
    struct Reference: Decodable {
        struct Record: Decodable {
            let ids: [Int]
            let decide_idx: [Int]
            let opt_idx: [[Int]]
            let logits: [[Double]]
            let probs: [[Double]]
        }
        let temperature: Double
        let records: [Record]
    }
    struct Record: Decodable {
        struct Question: Decodable {
            let instr: String
            let options: [String]
            let label: Int
        }
        let state: String
        let questions: [Question]
    }

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    static let checkpoint = URL(
        fileURLWithPath: ProcessInfo.processInfo.environment["KEV_CHECKPOINT"]
            ?? root.appending(path: "build/kev-0.8b-mlx-4bit").path)
    static let parity = root.appending(path: "scripts/parity")

    static func fixtures() throws -> ([Record], Reference)? {
        let referenceURL = parity.appending(path: "reference.json")
        guard FileManager.default.fileExists(atPath: referenceURL.path),
            FileManager.default.fileExists(atPath: checkpoint.appending(path: "kev.json").path)
        else { return nil }
        let records = try JSONDecoder().decode(
            [Record].self, from: Data(contentsOf: parity.appending(path: "records.json")))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL))
        return (records, reference)
    }

    static func questions(_ record: Record) -> [KevQuestion] {
        record.questions.map { KevQuestion.choice($0.instr, options: $0.options) }
    }

    @Test func encodingMatchesKev() async throws {
        guard let (records, reference) = try Self.fixtures() else { return }
        let model = try await KevModel.load(directory: Self.checkpoint)
        let encoder = await model.encoderForTesting
        for (record, expected) in zip(records, reference.records) {
            let enc = try encoder.encode(state: record.state, questions: Self.questions(record))
            var packed = enc.state
            var start = enc.state.count
            for (k, branch) in enc.branches.enumerated() {
                #expect(start + branch.decide == expected.decide_idx[k])
                #expect(branch.options.map { start + $0 } == expected.opt_idx[k])
                packed += branch.ids
                start += branch.ids.count
            }
            #expect(packed == expected.ids, "token ids differ for state \(record.state.prefix(40))")
        }
    }

    @Test func probabilitiesMatchKev() async throws {
        guard let (records, reference) = try Self.fixtures() else { return }
        let model = try await KevModel.load(directory: Self.checkpoint)
        #expect(abs(Double(await model.metadata.temperature) - reference.temperature) < 1e-6)
        var maxDelta = 0.0
        var agree = 0
        var total = 0
        var decided = 0
        var lines: [String] = []
        for (record, expected) in zip(records, reference.records) {
            let answers = try await model.decide(state: record.state, questions: Self.questions(record))
            for (answer, probs) in zip(answers, expected.probs) {
                total += 1
                let delta = zip(answer.probabilities, probs).map { abs($0 - $1) }.max() ?? 1
                maxDelta = max(maxDelta, delta)
                let sorted = probs.sorted(by: >)
                if sorted[0] - sorted[1] > 0.05 {
                    decided += 1
                    let referenceArgmax = probs.indices.max { probs[$0] < probs[$1] } ?? 0
                    if referenceArgmax == answer.argmax { agree += 1 }
                }
                lines.append(
                    "kev-swift \(answer.probabilities.map { String(format: "%.3f", $0) })  ref \(probs.map { String(format: "%.3f", $0) })"
                )
            }
        }
        let clock = ContinuousClock()
        let warm = try await clock.measure {
            _ = try await model.decide(state: records[0].state, questions: Self.questions(records[0]))
        }
        let summary =
            "\(Self.checkpoint.lastPathComponent): warm 1-question decide \(warm), \(total) questions, max |Δp| = \(maxDelta), argmax agreement \(agree)/\(decided) where the reference margin > 0.05\n"
        print(summary)
        let report = Self.root.appending(path: "build/parity-\(Self.checkpoint.lastPathComponent).txt")
        try (lines.joined(separator: "\n") + "\n" + summary).write(to: report, atomically: true, encoding: .utf8)
        // kev scores in fp32 from bf16 weights; bf16 / 8-bit should track it closely. 4-bit drifts visibly.
        let tolerance = (await model.metadata.bits ?? 16) <= 4 ? 0.2 : 0.03
        #expect(agree == decided)
        #expect(maxDelta < tolerance)
    }
}
