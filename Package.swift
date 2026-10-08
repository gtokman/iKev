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
        .library(name: "Kev", targets: ["Kev"]),
        .executable(name: "kev-dungeon", targets: ["KevDungeon"]),
        .library(name: "KevDungeonGame", targets: ["KevDungeonGame"]),
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
        // Example: a roguelike that asks Kev for every move. The engine is a library so it can be tested without Metal.
        .target(
            name: "KevDungeonGame",
            dependencies: ["Kev"],
            path: "Examples/Dungeon/Game",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "KevDungeon",
            dependencies: ["KevDungeonGame"],
            path: "Examples/Dungeon/CLI",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "KevDungeonTests",
            dependencies: ["KevDungeonGame"]
        ),
    ]
)
