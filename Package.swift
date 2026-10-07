// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Kev",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "Kev", targets: ["Kev"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3"),
            traits: []),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.13.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4"),
    ],
    targets: [
        .target(
            name: "Kev",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "KevTests",
            dependencies: ["Kev"]
        ),
    ]
)
