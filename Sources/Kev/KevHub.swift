import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

extension KevModel {
    /// Load a checkpoint directory with the Hugging Face tokenizer loader.
    public static func load(directory: URL) async throws -> KevModel {
        try await load(directory: directory, tokenizerLoader: #huggingFaceTokenizerLoader())
    }

    /// Download (or reuse from the Hub cache) a converted checkpoint repo and load it. Safe to call again: a complete
    /// download is reused without network access.
    public static func load(
        hubID: String, revision: String = "main", hub: HubClient = HubClient(),
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> KevModel {
        let directory = try await download(hubID: hubID, revision: revision, hub: hub, progressHandler: progressHandler)
        return try await load(directory: directory)
    }

    /// Fetch the checkpoint files into the Hub cache and return the local directory.
    public static func download(
        hubID: String, revision: String = "main", hub: HubClient = HubClient(),
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> URL {
        let downloader = #hubDownloader(hub)
        return try await downloader.download(
            id: hubID, revision: revision, matching: ["*.json", "*.safetensors", "*.txt"], useLatest: false,
            progressHandler: progressHandler)
    }
}
