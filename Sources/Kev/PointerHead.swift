import Foundation
import MLX
import MLXNN

/// Kev's readout: a bilinear score between the hidden state at `<decide>` and each option's `</opt>` hidden state,
/// scaled by 1/sqrt(dp) and divided by the checkpoint's fitted temperature.
final class PointerHead: Module, @unchecked Sendable {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    let scale: Float
    let temperature: Float

    init(hiddenSize: Int, pointerDim: Int, temperature: Float) {
        _q.wrappedValue = Linear(hiddenSize, pointerDim, bias: true)
        _k.wrappedValue = Linear(hiddenSize, pointerDim, bias: true)
        self.scale = 1 / Float(pointerDim).squareRoot()
        self.temperature = temperature
    }

    /// - Parameters:
    ///   - decide: `[d]` hidden state at the question's `<decide>` token
    ///   - options: `[n, d]` hidden states at each option's `</opt>` token
    /// - Returns: `[n]` logits in float32
    func callAsFunction(decide: MLXArray, options: MLXArray) -> MLXArray {
        let z = matmul(k(options.asType(.float32)), q(decide.asType(.float32))) * scale
        return temperature == 1 ? z : z / temperature
    }

    /// Reads `q.weight`/`q.bias`/`k.weight`/`k.bias`; the hidden size comes from the weight shape.
    static func load(from url: URL, pointerDim: Int, temperature: Float) throws -> PointerHead {
        let weights = try loadArrays(url: url)
        guard let q = weights["q.weight"], q.ndim == 2, q.dim(0) == pointerDim else {
            throw KevModelError.unsupportedCheckpoint("head.safetensors has no [\(pointerDim), d] q.weight")
        }
        let hiddenSize = q.dim(1)
        let head = PointerHead(hiddenSize: hiddenSize, pointerDim: pointerDim, temperature: temperature)
        try head.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(head)
        return head
    }
}
