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

    /// Best of the wordings in scripts/bench/questions.json on scripts/bench/messages.json (43/50 at threshold 0).
    public static let question = KevQuestion.choice(
        "What does the user want from the assistant?",
        criteria: [
            "casual": "just a friendly reply; they are chatting, greeting, thanking or signing off",
            "deviceAction": "perform one phone action now (timer, alarm, reminder, calendar event, text, call)",
            "assistant": "an answer, information, content, or help that needs knowledge, tools, data or earlier context",
        ])

    public let question: KevQuestion

    public init(
        model: KevModel, threshold: Double = 0.5, maxHistory: Int = 6, question: KevQuestion = KevRouter.question
    ) {
        precondition(question.options.count == Route.allCases.count, "router question needs one option per Route")
        self.model = model
        self.threshold = threshold
        self.maxHistory = maxHistory
        self.question = question
    }

    public static func state(latest: String, history: [(role: String, text: String)], maxHistory: Int) -> String {
        guard !history.isEmpty else { return "User message: \(latest)" }
        let turns = history.suffix(maxHistory).map { "\($0.role): \($0.text)" }
        return "Conversation:\n" + turns.joined(separator: "\n") + "\nuser: \(latest)"
    }

    /// `nil` when Kev is not confident enough; the caller should take its default path.
    public func route(latest: String, history: [(role: String, text: String)] = []) async throws -> Decision? {
        let state = Self.state(latest: latest, history: history, maxHistory: maxHistory)
        guard let answer = try await model.decide(state: state, questions: [question]).first else { return nil }
        let routes = Route.allCases
        let probabilities = Dictionary(uniqueKeysWithValues: zip(routes, answer.probabilities))
        let decision = Decision(
            route: routes[answer.argmax], probabilities: probabilities, confidence: answer.confidence)
        return decision.confidence >= threshold ? decision : nil
    }
}
