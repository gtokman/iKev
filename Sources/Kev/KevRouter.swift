import Foundation

/// Decides where a chat turn should go. One Kev choice question over the latest message (plus a little history),
/// answered locally only when the calibrated confidence clears `threshold`; otherwise the caller falls back.
public struct KevRouter: Sendable {
    public enum Route: String, Sendable, CaseIterable {
        case casual
        case deviceAction
        case assistant
    }

    public struct Decision: Sendable {
        public let route: Route
        public let probabilities: [Route: Double]
        public let confidence: Double
    }

    public let model: KevModel
    /// Minimum `KevAnswer.confidence` (0 = uniform, 1 = certain) to act on the decision.
    public let threshold: Double
    public let maxHistory: Int

    public static let question = KevQuestion.choice(
        "Which route should handle the latest user message?",
        criteria: [
            "casual": "a standalone greeting, thanks, goodbye, or light social chat that needs no facts, memory, or action",
            "deviceAction": "a request to do one thing on this device (timer, alarm, reminder, calendar, message, call)",
            "assistant":
                "anything else: questions, advice, requests that need facts, tools, or earlier messages, or a greeting combined with a request",
        ])

    public init(model: KevModel, threshold: Double = 0.5, maxHistory: Int = 6) {
        self.model = model
        self.threshold = threshold
        self.maxHistory = maxHistory
    }

    public static func state(latest: String, history: [(role: String, text: String)], maxHistory: Int) -> String {
        guard !history.isEmpty else { return "User message: \(latest)" }
        let turns = history.suffix(maxHistory).map { "\($0.role): \($0.text)" }
        return "Conversation:\n" + turns.joined(separator: "\n") + "\nuser: \(latest)"
    }

    /// `nil` when Kev is not confident enough; the caller should take its default path.
    public func route(latest: String, history: [(role: String, text: String)] = []) async throws -> Decision? {
        let state = Self.state(latest: latest, history: history, maxHistory: maxHistory)
        guard let answer = try await model.decide(state: state, questions: [Self.question]).first else { return nil }
        let routes = Route.allCases
        let probabilities = Dictionary(uniqueKeysWithValues: zip(routes, answer.probabilities))
        let decision = Decision(
            route: routes[answer.argmax], probabilities: probabilities, confidence: answer.confidence)
        return decision.confidence >= threshold ? decision : nil
    }
}
