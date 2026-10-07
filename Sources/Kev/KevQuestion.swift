import Foundation

/// One question Kev answers about a state. Mirrors kev's System One request types: a yes/no (`noul`), a fixed choice
/// among named options, or a score along ordered levels.
public struct KevQuestion: Sendable, Hashable {
    public enum Kind: Sendable, Hashable {
        case yesNo
        case choice
        case score
    }

    public let kind: Kind
    public let instructions: String
    /// The option texts the model sees, in order (option boundaries are read out in this order).
    public let options: [String]
    /// The keys probabilities are reported under, in option order: option names (choice), `false`/`true` (yesNo),
    /// level indices as strings (score).
    public let keys: [String]

    public static let maxOptions = 16

    public static func yesNo(_ instructions: String, no: String? = nil, yes: String? = nil) -> KevQuestion {
        KevQuestion(
            kind: .yesNo, instructions: instructions,
            options: [optionText("no", no), optionText("yes", yes)], keys: ["false", "true"])
    }

    /// `criteria` keeps declaration order; a description may be empty.
    public static func choice(_ instructions: String, criteria: KeyValuePairs<String, String>) -> KevQuestion {
        precondition((1 ... maxOptions).contains(criteria.count), "criteria must have 1...\(maxOptions) options")
        return KevQuestion(
            kind: .choice, instructions: instructions,
            options: criteria.map { optionText($0.key, $0.value) }, keys: criteria.map(\.key))
    }

    public static func choice(_ instructions: String, options: [String]) -> KevQuestion {
        precondition((1 ... maxOptions).contains(options.count), "options must have 1...\(maxOptions) entries")
        return KevQuestion(kind: .choice, instructions: instructions, options: options, keys: options)
    }

    public static func score(_ instructions: String, levels: [String]) -> KevQuestion {
        precondition((1 ... maxOptions).contains(levels.count), "levels must have 1...\(maxOptions) entries")
        return KevQuestion(
            kind: .score, instructions: instructions, options: levels,
            keys: levels.indices.map(String.init))
    }

    static func optionText(_ name: String, _ description: String?) -> String {
        guard let description, !description.isEmpty else { return name }
        return "\(name): \(description)"
    }
}

/// Calibrated probabilities for one question, in option order, plus the derived decision.
public struct KevAnswer: Sendable, Hashable {
    public let question: KevQuestion
    /// Probability per option, summing to 1.
    public let probabilities: [Double]
    /// Raw pointer logits before softmax (already divided by the checkpoint temperature).
    public let logits: [Double]

    public init(question: KevQuestion, logits: [Double]) {
        self.question = question
        self.logits = logits
        let m = logits.max() ?? 0
        let e = logits.map { exp($0 - m) }
        let z = e.reduce(0, +)
        self.probabilities = e.map { $0 / z }
    }

    /// Probability by key.
    public var distribution: [String: Double] {
        Dictionary(uniqueKeysWithValues: zip(question.keys, probabilities))
    }

    /// Index of the most likely option (first on ties).
    public var argmax: Int {
        var best = 0
        for i in probabilities.indices where probabilities[i] > probabilities[best] { best = i }
        return best
    }

    /// Key of the most likely option.
    public var choice: String { question.keys[argmax] }

    /// P(yes) for a yes/no question.
    public var yes: Double { probabilities.count == 2 ? probabilities[1] : 0 }

    /// Expected level for a score question.
    public var score: Double {
        zip(probabilities.indices, probabilities).reduce(0) { $0 + Double($1.0) * $1.1 }
    }

    /// 0 at uniform, 1 at certainty. Choice/yesNo: (p_max - 1/K) / (1 - 1/K). Score: 1 - E|level - mode| / D with D the
    /// mean absolute deviation of a uniform distribution over the levels. Same formulas as kev.api.
    public var confidence: Double {
        let p = probabilities
        let k = p.count
        guard k > 1 else { return 1 }
        switch question.kind {
        case .yesNo, .choice:
            return ((p.max() ?? 0) - 1 / Double(k)) / (1 - 1 / Double(k))
        case .score:
            let mode = argmax
            let mean = Double(k - 1) / 2
            let d = (0 ..< k).reduce(0.0) { $0 + abs(Double($1) - mean) } / Double(k)
            let spread = zip(p.indices, p).reduce(0.0) { $0 + $1.1 * abs(Double($1.0 - mode)) }
            return max(0, 1 - spread / d)
        }
    }
}
