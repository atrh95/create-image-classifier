// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CreateImageClassifier",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "CreateImageClassifier", targets: [
            "CICInterface",
            "CICConfusionMatrix",
            "CICFileManager",
            "CICTrainingResult",
            "BinaryClassifier",
            "MultiClassClassifier",
            "OvOClassifier",
            "OvRClassifier",
        ]),
        .library(name: "CICInterface", targets: ["CICInterface"]),
        .library(name: "CICConfusionMatrix", targets: ["CICConfusionMatrix"]),
        .library(name: "CICFileManager", targets: ["CICFileManager"]),
        .library(name: "CICTrainingResult", targets: ["CICTrainingResult"]),
        .library(name: "BinaryClassifier", targets: ["BinaryClassifier"]),
        .library(name: "MultiClassClassifier", targets: ["MultiClassClassifier"]),
        .library(name: "OvOClassifier", targets: ["OvOClassifier"]),
        .library(name: "OvRClassifier", targets: ["OvRClassifier"]),
    ],
    targets: [
        .target(
            name: "CICInterface",
            path: "CICInterface"
        ),
        .target(
            name: "CICConfusionMatrix",
            path: "CICConfusionMatrix",
            sources: [
                "CICBinaryConfusionMatrix.swift",
                "CICMultiClassConfusionMatrix.swift",
            ]
        ),
        .testTarget(
            name: "CICConfusionMatrixTests",
            dependencies: ["CICConfusionMatrix"],
            path: "CICConfusionMatrix/Tests",
            sources: [
                "CICBinaryConfusionMatrixTests.swift",
                "CICMultiClassConfusionMatrixTests.swift",
            ]
        ),
        .target(
            name: "CICFileManager",
            path: "CICFileManager",
            sources: [
                "CICFileManager.swift",
            ]
        ),
        .testTarget(
            name: "CICFileManagerTests",
            dependencies: ["CICFileManager"],
            path: "CICFileManager/Tests",
            sources: ["CICFileManagerTests.swift"]
        ),
        .target(
            name: "CICTrainingResult",
            dependencies: ["CICInterface", "CICConfusionMatrix"],
            path: "CICTrainingResult"
        ),
        .target(
            name: "BinaryClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/BinaryClassifier/Sources"
        ),
        .testTarget(
            name: "BinaryClassifierTests",
            dependencies: ["BinaryClassifier"],
            path: "Classifiers/BinaryClassifier/Tests",
            resources: [.process("TestResources")]
        ),
        .target(
            name: "MultiClassClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/MultiClassClassifier/Sources"
        ),
        .testTarget(
            name: "MultiClassClassifierTests",
            dependencies: ["MultiClassClassifier"],
            path: "Classifiers/MultiClassClassifier/Tests",
            resources: [.process("TestResources")]
        ),
        .target(
            name: "OvOClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/OvOClassifier/Sources"
        ),
        .testTarget(
            name: "OvOClassifierTests",
            dependencies: ["OvOClassifier"],
            path: "Classifiers/OvOClassifier/Tests",
            resources: [.process("TestResources")]
        ),
        .target(
            name: "OvRClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/OvRClassifier/Sources"
        ),
        .testTarget(
            name: "OvRClassifierTests",
            dependencies: ["OvRClassifier"],
            path: "Classifiers/OvRClassifier/Tests",
            resources: [.process("TestResources")]
        ),
    ]
)
