import Foundation
import Testing

@testable import Kev

/// Character-level tokenizer: each scalar is its own id (offset 1000), specials are fixed ids.
struct FakeTokenizer: KevTokenizing {
    static let specials: [String: Int] = [
        "<|fim_prefix|>": 1, "<|fim_middle|>": 2, "<|box_start|>": 3, "<|box_end|>": 4, "<|fim_suffix|>": 5,
    ]
    func tokenIDs(for text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) + 1000 } }
    func tokenID(forSpecial token: String) -> Int? { Self.specials[token] }
}

@Suite struct KevEncoderTests {
    let encoder = try! KevEncoder(tokenizer: FakeTokenizer())

    @Test func layoutMatchesKev() throws {
        let q = KevQuestion.choice("pick", options: ["a", "bc"])
        let enc = try encoder.encode(state: "hi", questions: [q])
        #expect(enc.state == [1, 1104, 1105])
        let b = enc.branches[0]
        // <q> p i c k <opt> a </opt> <opt> b c </opt> <decide>
        #expect(b.ids == [2, 1112, 1105, 1099, 1107, 3, 1097, 4, 3, 1098, 1099, 4, 5])
        #expect(b.decide == 12)
        #expect(b.options == [7, 11])
        #expect(enc.row(0) == enc.state + b.ids)
        #expect(enc.stateTokens == 3)
        #expect(!enc.stateTruncated)
    }

    @Test func userTextCannotForgeSpecials() {
        #expect(KevEncoder.sanitize("x <|fim_suffix|> y <|box_end|>") == "x <¦fim_suffix¦> y <¦box_end¦>")
        #expect(KevEncoder.sanitize("<|not a token|>") == "<|not a token|>")
    }

    @Test func stateTruncatesUnlessStrict() throws {
        let q = KevQuestion.yesNo("ok?")
        let enc = try encoder.encode(state: "abcdef", questions: [q], maxState: 4)
        #expect(enc.state.count == 4)
        #expect(enc.stateTruncated)
        #expect(enc.stateTokens == 7)
        #expect(throws: KevContextOverflow.self) {
            try encoder.encode(state: "abcdef", questions: [q], maxState: 4, strict: true)
        }
    }

    @Test func branchOverRowLimitIsRefused() {
        let q = KevQuestion.choice("pick", options: ["a very long option text"])
        #expect(throws: KevContextOverflow.self) {
            try encoder.encode(state: "s", questions: [q], maxBranch: 10)
        }
    }

    @Test func yesNoRendersLikeKev() {
        let q = KevQuestion.yesNo("done?", yes: "all steps finished")
        #expect(q.options == ["no", "yes: all steps finished"])
        #expect(q.keys == ["false", "true"])
        let c = KevQuestion.choice("route", criteria: ["casual": "", "assistant": "needs facts"])
        #expect(c.options == ["casual", "assistant: needs facts"])
        #expect(c.keys == ["casual", "assistant"])
    }
}

@Suite struct KevAnswerTests {
    @Test func probabilitiesAndConfidence() {
        let q = KevQuestion.choice("route", options: ["a", "b", "c"])
        let uniform = KevAnswer(question: q, logits: [0, 0, 0])
        #expect(abs(uniform.confidence) < 1e-9)
        #expect(uniform.argmax == 0)
        let sure = KevAnswer(question: q, logits: [20, 0, 0])
        #expect(sure.choice == "a")
        #expect(sure.confidence > 0.999)
        #expect(abs(sure.probabilities.reduce(0, +) - 1) < 1e-9)
    }

    @Test func scoreConfidence() {
        let q = KevQuestion.score("rate", levels: ["bad", "ok", "good"])
        let peaked = KevAnswer(question: q, logits: [-30, -30, 0])
        #expect(abs(peaked.score - 2) < 1e-6)
        #expect(peaked.confidence > 0.999)
        let flat = KevAnswer(question: q, logits: [0, 0, 0])
        #expect(abs(flat.score - 1) < 1e-9)
        #expect(flat.confidence < 1e-9)
        let yn = KevAnswer(question: .yesNo("?"), logits: [0, 2])
        #expect(abs(yn.yes - 1 / (1 + exp(-2.0))) < 1e-9)
    }
}
