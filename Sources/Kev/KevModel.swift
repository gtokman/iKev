import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// `kev.json` written next to the weights by scripts/convert.py.
public struct KevCheckpointMetadata: Codable, Sendable {
    public let format: Int
    public let run: String
    public let revision: String?
    public let base: String
    public let headDim: Int
    public let temperature: Float
    public let optionIsolation: Bool
    public let bits: Int?

    enum CodingKeys: String, CodingKey {
        case format, run, revision, base, bits
        case headDim = "head_dim"
        case temperature
        case optionIsolation = "option_isolation"
    }
}

public enum KevModelError: Error, Sendable {
    case unsupportedBackbone(String)
    case unsupportedCheckpoint(String)
    case missingHiddenStates
    /// MLX needs a real Metal GPU family; the iOS Simulator aborts inside Metal device setup.
    case simulatorUnsupported
}

extension MLXLMCommon.Tokenizer {
    var kev: any KevTokenizing { TokenizerBridge(tokenizer: self) }
}

private struct TokenizerBridge: KevTokenizing, @unchecked Sendable {
    let tokenizer: MLXLMCommon.Tokenizer
    func tokenIDs(for text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    func tokenID(forSpecial token: String) -> Int? { tokenizer.convertTokenToId(token) }
}

/// A loaded Kev checkpoint: the merged Qwen3.5 backbone plus the pointer head. Prefill only, no text generation.
public actor KevModel {
    public let metadata: KevCheckpointMetadata
    private let container: ModelContainer
    private let head: PointerHead
    private let encoder: KevEncoder
    private let padID: Int

    /// Load a checkpoint directory produced by scripts/convert.py (or downloaded from the Hub).
    public static func load(directory: URL, tokenizerLoader: any TokenizerLoader) async throws -> KevModel {
        // MLX reports Metal failures (e.g. GPU work refused while an iOS app is in the background)
        // through fatalError unless a handler is installed; surface them as thrown MLXError instead.
        try await withError { try await loadUnguarded(directory: directory, tokenizerLoader: tokenizerLoader) }
    }

    private static func loadUnguarded(directory: URL, tokenizerLoader: any TokenizerLoader) async throws -> KevModel {
        #if targetEnvironment(simulator)
            throw KevModelError.simulatorUnsupported
        #endif
        let metaURL = directory.appending(path: "kev.json")
        let metadata = try JSONDecoder().decode(KevCheckpointMetadata.self, from: Data(contentsOf: metaURL))
        guard metadata.format == 1 else {
            throw KevModelError.unsupportedCheckpoint("kev.json format \(metadata.format)")
        }
        guard !metadata.optionIsolation else {
            throw KevModelError.unsupportedCheckpoint("option_isolation checkpoints are not supported")
        }
        Memory.cacheLimit = 256 * 1024 * 1024
        let container = try await LLMModelFactory.shared.loadContainer(
            from: directory, using: tokenizerLoader)
        try await container.perform { context in
            guard context.model is Qwen35Model else {
                throw KevModelError.unsupportedBackbone(String(describing: type(of: context.model)))
            }
        }
        let head = try PointerHead.load(
            from: directory.appending(path: "head.safetensors"),
            pointerDim: metadata.headDim, temperature: metadata.temperature)
        let tokenizer = await container.tokenizer
        let encoder = try KevEncoder(tokenizer: tokenizer.kev)
        let padID = tokenizer.convertTokenToId("<|endoftext|>") ?? 0
        return KevModel(metadata: metadata, container: container, head: head, encoder: encoder, padID: padID)
    }

    private init(
        metadata: KevCheckpointMetadata, container: ModelContainer, head: PointerHead, encoder: KevEncoder, padID: Int
    ) {
        self.metadata = metadata
        self.container = container
        self.head = head
        self.encoder = encoder
        self.padID = padID
    }

    var encoderForTesting: KevEncoder { encoder }

    /// Answer `questions` about `state` in one batched prefill (one row per question: state + branch).
    public func decide(
        state: String, questions: [KevQuestion],
        maxState: Int = 4096, maxBranch: Int = 4096 + 1024
    ) async throws -> [KevAnswer] {
        guard !questions.isEmpty else { return [] }
        let encoding = try encoder.encode(
            state: state, questions: questions, maxState: maxState, maxBranch: maxBranch)
        let rows = encoding.branches.indices.map(encoding.row)
        let length = rows.map(\.count).max() ?? 0
        let padded = rows.map { $0 + Array(repeating: padID, count: length - $0.count) }
        let stateCount = encoding.state.count
        let head = self.head
        let logits: [[Float]] = try await container.perform { context in
          try withError {
            guard let model = context.model as? Qwen35Model else {
                throw KevModelError.unsupportedBackbone(String(describing: type(of: context.model)))
            }
            let tokens = MLXArray(padded.flatMap { $0.map(Int32.init) }).reshaped(padded.count, length)
            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
            guard let hidden = output.state?[mtpLastHiddenStatesKey] else {
                throw KevModelError.missingHiddenStates
            }
            var out: [MLXArray] = []
            for (i, branch) in encoding.branches.enumerated() {
                let decide = hidden[i, stateCount + branch.decide]
                let options = hidden[i, MLXArray(branch.options.map { Int32(stateCount + $0) })]
                out.append(head(decide: decide, options: options))
            }
            eval(out)
            return out.map { $0.asArray(Float.self) }
          }
        }
        return zip(questions, logits).map { KevAnswer(question: $0, logits: $1.map(Double.init)) }
    }
}

extension KevModel {
    /// Drops MLX's cached buffers (inference scratch memory). Call after a decision or under memory pressure;
    /// the next decision reallocates what it needs.
    public nonisolated static func releaseCache() {
        Memory.clearCache()
    }
}
