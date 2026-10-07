import Foundation

/// Token access the encoder needs. `MLXLMCommon.Tokenizer` satisfies it; tests use a fake.
public protocol KevTokenizing: Sendable {
    func tokenIDs(for text: String) -> [Int]
    func tokenID(forSpecial token: String) -> Int?
}

public struct KevContextOverflow: Error, Sendable, Equatable {
    public let message: String
    public let stateTokens: Int?
    public let maxState: Int?
}

/// One question's branch tokens and where to read out within the branch.
public struct KevBranch: Sendable, Equatable {
    public let ids: [Int]
    /// Offset of `<decide>` within `ids` (always the last token).
    public let decide: Int
    /// Offset of each option's `</opt>` within `ids`.
    public let options: [Int]
}

/// A record packed the way kev.model.encode packs it, split into the shared state prefix and one branch per question.
/// Feeding `state + branch` as one causal row is kev's row form: exactly the tokens question k may attend to.
public struct KevEncoding: Sendable, Equatable {
    public let state: [Int]
    public let branches: [KevBranch]
    /// Whole state length, the `<state>` token included (before any truncation).
    public let stateTokens: Int
    public let stateTruncated: Bool

    /// Token ids of the row for question `index`: state followed by its branch.
    public func row(_ index: Int) -> [Int] { state + branches[index].ids }
}

/// Kev's token layout over Qwen's spare special tokens: `<state> ...` then per question
/// `<q> instructions <opt> option </opt> ... <decide>`. Caller text can never forge a delimiter: `<|name|>` is rewritten
/// to `<¦name¦>` before tokenizing, like kev.model.user_tokens.
public struct KevEncoder: Sendable {
    public static let specialTokens = (
        state: "<|fim_prefix|>", question: "<|fim_middle|>", optionStart: "<|box_start|>",
        optionEnd: "<|box_end|>", decide: "<|fim_suffix|>"
    )

    /// kev's training context (MAX_STATE / MAX_BRANCH). Serving may raise these; rows past `maxBranch` tokens are refused.
    public static let trainingMaxState = 384
    public static let trainingMaxBranch = 1024

    let tokenizer: any KevTokenizing
    let stateID: Int
    let questionID: Int
    let optionStartID: Int
    let optionEndID: Int
    let decideID: Int

    public init(tokenizer: any KevTokenizing) throws {
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.tokenID(forSpecial: token) else {
                throw KevContextOverflow(message: "tokenizer has no special token \(token)", stateTokens: nil, maxState: nil)
            }
            return id
        }
        self.tokenizer = tokenizer
        stateID = try id(Self.specialTokens.state)
        questionID = try id(Self.specialTokens.question)
        optionStartID = try id(Self.specialTokens.optionStart)
        optionEndID = try id(Self.specialTokens.optionEnd)
        decideID = try id(Self.specialTokens.decide)
    }

    private static let specialPattern = try! NSRegularExpression(pattern: "<\\|([A-Za-z0-9_]+)\\|>")

    static func sanitize(_ text: String) -> String {
        let range = NSRange(text.startIndex..., in: text)
        return specialPattern.stringByReplacingMatches(in: text, range: range, withTemplate: "<¦$1¦>")
    }

    func userTokens(_ text: String) -> [Int] {
        tokenizer.tokenIDs(for: Self.sanitize(text))
    }

    /// - Parameters:
    ///   - maxState: a longer state is cut to its first `maxState` tokens (the `<state>` token counted), or refused when
    ///     `strict`.
    ///   - maxBranch: row limit (state + branch); a question over it is always refused.
    public func encode(
        state: String, questions: [KevQuestion],
        maxState: Int = KevEncoder.trainingMaxState, maxBranch: Int = KevEncoder.trainingMaxBranch,
        strict: Bool = false
    ) throws -> KevEncoding {
        let stateTokens = userTokens(state)
        let stateCount = stateTokens.count + 1
        if strict, stateCount > maxState {
            throw KevContextOverflow(
                message: "state exceeds \(maxState) tokens: \(stateCount)", stateTokens: stateCount, maxState: maxState)
        }
        let s = [stateID] + stateTokens.prefix(max(0, maxState - 1))
        var branches: [KevBranch] = []
        for question in questions {
            var ids = [questionID] + userTokens(question.instructions)
            var ends: [Int] = []
            for option in question.options {
                ids += [optionStartID] + userTokens(option) + [optionEndID]
                ends.append(ids.count - 1)
            }
            ids.append(decideID)
            if ids.count > maxBranch - s.count {
                throw KevContextOverflow(
                    message: "branch too long: \(ids.count) tokens with a \(s.count)-token state (row limit \(maxBranch))",
                    stateTokens: nil, maxState: nil)
            }
            branches.append(KevBranch(ids: ids, decide: ids.count - 1, options: ends))
        }
        return KevEncoding(
            state: s, branches: branches, stateTokens: stateCount, stateTruncated: stateCount > maxState)
    }
}
