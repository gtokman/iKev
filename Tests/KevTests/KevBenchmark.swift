import Foundation
import Testing

@testable import Kev

/// Opt-in benchmark (`TEST_RUNNER_KEV_BENCH=1`): cold load, warm decide latency and routing accuracy on
/// scripts/bench/messages.json. Writes build/bench-<checkpoint>-<platform>.md.
@Suite(.serialized) struct KevBenchmark {
    struct Message: Decodable {
        let text: String
        let route: String
    }

    struct Variant: Decodable {
        let name: String
        let instructions: String
        let criteria: [String: String]

        var question: KevQuestion {
            KevQuestion.choice(
                instructions, options: KevRouter.Route.allCases.map { KevQuestion.optionText($0.rawValue, criteria[$0.rawValue]) })
        }
    }

    static var enabled: Bool { ProcessInfo.processInfo.environment["KEV_BENCH"] != nil }

    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : .nan
    }

    static var platform: String {
        #if targetEnvironment(simulator)
            return "ios-simulator"
        #elseif os(iOS)
            return "ios"
        #else
            return "macos"
        #endif
    }

    @Test func benchmark() async throws {
        guard Self.enabled else { return }
        let checkpoint = KevParityTests.checkpoint
        let messages = try JSONDecoder().decode(
            [Message].self, from: Data(contentsOf: KevParityTests.root.appending(path: "scripts/bench/messages.json")))
        let clock = ContinuousClock()
        let baseline = Self.footprintMB()
        var model: KevModel?
        let cold = try await clock.measure { model = try await KevModel.load(directory: checkpoint) }
        let loaded = Self.footprintMB()
        let router = KevRouter(model: model!, threshold: 0)
        _ = try await router.route(latest: "warm up")
        let afterFirst = Self.footprintMB()

        var latencies: [Duration] = []
        var rows: [String] = []
        var correct = 0
        var confidences: [(Double, Bool)] = []
        var peak = afterFirst
        for message in messages {
            var decision: KevRouter.Decision?
            let elapsed = try await clock.measure { decision = try await router.route(latest: message.text) }
            latencies.append(elapsed)
            peak = max(peak, Self.footprintMB())
            guard let decision else { continue }
            let ok = decision.route.rawValue == message.route
            if ok { correct += 1 }
            confidences.append((decision.confidence, ok))
            let probs = KevRouter.Route.allCases.map { String(format: "%.2f", decision.probabilities[$0] ?? 0) }
            rows.append(
                "| \(message.text.replacingOccurrences(of: "|", with: "\\|")) | \(message.route) | \(decision.route.rawValue) \(ok ? "" : "✗") | \(probs.joined(separator: " / ")) | \(String(format: "%.2f", decision.confidence)) | \(Self.ms(elapsed)) |"
            )
        }
        let sorted = latencies.sorted()
        func pct(_ p: Double) -> Duration { sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))] }
        var thresholdLines: [String] = []
        for t in stride(from: 0.0, through: 0.6, by: 0.1) {
            let kept = confidences.filter { $0.0 >= t }
            let acc = kept.isEmpty ? 0 : Double(kept.filter(\.1).count) / Double(kept.count)
            thresholdLines.append(
                "| \(String(format: "%.1f", t)) | \(kept.count)/\(confidences.count) | \(String(format: "%.0f%%", acc * 100)) |")
        }
        let report = """
            # kev-swift benchmark — \(checkpoint.lastPathComponent) on \(Self.platform)

            \(Self.device())

            | metric | value |
            |---|---|
            | cold load (weights → ready) | \(Self.ms(cold)) |
            | first decide (JIT/Metal warm-up) | included above as warm-up |
            | decide latency p50 / p95 / max (\(latencies.count) msgs) | \(Self.ms(pct(0.5))) / \(Self.ms(pct(0.95))) / \(Self.ms(sorted.last!)) |
            | memory footprint: before / loaded / peak | \(Int(baseline)) / \(Int(loaded)) / \(Int(peak)) MB |
            | routing accuracy (threshold 0) | \(correct)/\(messages.count) |

            ## Accuracy vs confidence threshold

            | threshold | handled locally | accuracy of handled |
            |---|---|---|
            \(thresholdLines.joined(separator: "\n"))

            ## Per message (casual / deviceAction / assistant)

            | message | expected | kev | p | confidence | ms |
            |---|---|---|---|---|---|
            \(rows.joined(separator: "\n"))

            """
        let variants = try JSONDecoder().decode(
            [Variant].self, from: Data(contentsOf: KevParityTests.root.appending(path: "scripts/bench/questions.json")))
        var variantLines: [String] = []
        for variant in variants {
            let r = KevRouter(model: model!, threshold: 0, question: variant.question)
            var hits = 0
            var confident = 0
            var confidentHits = 0
            for message in messages {
                guard let d = try await r.route(latest: message.text) else { continue }
                let ok = d.route.rawValue == message.route
                if ok { hits += 1 }
                if d.confidence >= 0.3 {
                    confident += 1
                    if ok { confidentHits += 1 }
                }
            }
            variantLines.append("| \(variant.name) | \(hits)/\(messages.count) | \(confidentHits)/\(confident) |")
        }
        let fullReport = report + """

            ## Question wording (scripts/bench/questions.json)

            | variant | accuracy (threshold 0) | accuracy / handled at threshold 0.3 |
            |---|---|---|
            \(variantLines.joined(separator: "\n"))

            """
        let out = KevParityTests.root.appending(path: "build/bench-\(checkpoint.lastPathComponent)-\(Self.platform).md")
        try fullReport.write(to: out, atomically: true, encoding: .utf8)
        print(fullReport)
    }

    static func ms(_ d: Duration) -> String {
        let ms = Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
        return String(format: "%.0f ms", ms)
    }

    static func device() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        var model = ProcessInfo.processInfo.hostName
        #if os(iOS)
            model = String(decoding: machine.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        #else
            var msize = 0
            sysctlbyname("machdep.cpu.brand_string", nil, &msize, nil, 0)
            var brand = [CChar](repeating: 0, count: msize)
            sysctlbyname("machdep.cpu.brand_string", &brand, &msize, nil, 0)
            model = String(decoding: brand.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        #endif
        let mem = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return "Device: \(model), \(String(format: "%.0f", mem)) GB RAM, \(ProcessInfo.processInfo.operatingSystemVersionString)"
    }
}
