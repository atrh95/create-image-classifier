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
            "MultiClassifier",
            "MultiLabelClassifier",
            "OvOClassifier",
            "OvRClassifier",
        ]),
        .library(name: "CICInterface", targets: ["CICInterface"]),
        .library(name: "CICConfusionMatrix", targets: ["CICConfusionMatrix"]),
        .library(name: "CICFileManager", targets: ["CICFileManager"]),
        .library(name: "CICTrainingResult", targets: ["CICTrainingResult"]),
        .library(name: "BinaryClassifier", targets: ["BinaryClassifier"]),
        .library(name: "MultiClassifier", targets: ["MultiClassifier"]),
        .library(name: "MultiLabelClassifier", targets: ["MultiLabelClassifier"]),
        .library(name: "OvOClassifier", targets: ["OvOClassifier"]),
        .library(name: "OvRClassifier", targets: ["OvRClassifier"]),
    ],
    dependencies: [
        // Dependencies will be added here
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
                "CICMultiLabelConfusionMatrix.swift",
            ]
        ),
        .testTarget(
            name: "CICConfusionMatrixTests",
            dependencies: ["CICConfusionMatrix"],
            path: "CICConfusionMatrix/Tests/CICConfusionMatrixTests",
            sources: [
                "CICBinaryConfusionMatrixTests.swift",
                "CICMultiClassConfusionMatrixTests.swift",
                "CICMultiLabelConfusionMatrixTests.swift",
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
            path: "CICFileManager/Tests/CICFileManagerTests",
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
            path: "Classifiers/BinaryClassifier",
            sources: [
                "BinaryClassifier.swift",
                "BinaryTrainingResult.swift",
            ],
            resources: [.copy("Resources")]
        ),
        .target(
            name: "MultiClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/MultiClassifier",
            sources: [
                "MultiClassClassifier.swift",
                "MultiClassTrainingResult.swift",
            ],
            resources: [.copy("Resources")]
        ),
        .target(
            name: "MultiLabelClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/MultiLabelClassifier",
            sources: [
                "MultiLabelClassifier.swift",
                "MultiLabelTrainingResult.swift",
            ],
            resources: [.copy("Resources")]
        ),
        .target(
            name: "OvOClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/OvOClassifier",
            sources: [
                "OvOClassifier.swift",
                "OvOTrainingResult.swift",
            ],
            resources: [.copy("Resources")]
        ),
        .target(
            name: "OvRClassifier",
            dependencies: [
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/OvRClassifier",
            sources: [
                "OvRClassifier.swift",
                "OvRTrainingResult.swift",
            ],
            resources: [.copy("Resources")]
        ),
        .testTarget(
            name: "ClassifierTests",
            dependencies: [
                "BinaryClassifier",
                "MultiClassifier",
                "MultiLabelClassifier",
                "OvOClassifier",
                "OvRClassifier",
                "CICInterface",
                "CICConfusionMatrix",
                "CICFileManager",
                "CICTrainingResult",
            ],
            path: "Classifiers/Tests",
            sources: [
                "Tests/BinaryClassifierTests.swift",
                "Tests/MultiClassClassifierTests.swift",
                "Tests/MultiLabelClassifierTests.swift",
                "Tests/OvOClassifierTests.swift",
                "Tests/OvRClassifierTests.swift",
                "TestUtils.swift",
                "ClassifierTestsError.swift",
            ],
            resources: [
                .process("TestResources"),
            ]
        ),
    ]
)
