import BinaryClassification
import CICInterface
import CreateML
import CreateMLComponents
import Foundation
import MultiClassClassification
import MultiLabelClassification
import OvOClassification
import OvRClassification

// 分類器の種類
enum ClassifierType: String {
    case binary, multiClass, multiLabel, ovr, ovo

    func makeClassifier() -> any ClassifierProtocol {
        switch self {
            case .binary: BinaryClassificationClassifier()
            case .multiClass: MultiClassClassificationClassifier()
            case .multiLabel: MultiLabelClassificationClassifier()
            case .ovr: OvRClassificationClassifier()
            case .ovo: OvOClassificationClassifier()
        }
    }
}

// モデルの種類
enum MLModelType: String {
    case scaryCatScreeningML

    struct ModelConfig {
        let name: String
        let supportedClassifierVersions: [ClassifierType: String]
        let author: String
        let modelParameters: CreateML.MLImageClassifier.ModelParameters
        let scenePrintRevision: Int?
    }

    private static let configs: [MLModelType: ModelConfig] = [
        .scaryCatScreeningML: ModelConfig(
            name: "ScaryCatScreeningML",
            supportedClassifierVersions: [
                .binary: "v6",
                .multiClass: "v3",
                .multiLabel: "v1",
                .ovr: "v20",
                .ovo: "v1",
            ],
            author: "akitora",
            modelParameters: MLImageClassifier.ModelParameters(
                validation: .split(strategy: .automatic),
                maxIterations: 8,
                augmentation: [],
                algorithm: .transferLearning(
                    featureExtractor: .scenePrint(revision: 1),
                    classifier: .logisticRegressor
                )
            ),
            scenePrintRevision: 1
        ),
    ]

    private var config: ModelConfig {
        guard let config = Self.configs[self] else {
            fatalError("Configが存在しないモデルタイプ: \(self)")
        }
        return config
    }

    var name: String { config.name }
    var supportedClassifierTypes: [ClassifierType] { Array(config.supportedClassifierVersions.keys) }
    func version(for classifier: ClassifierType) -> String? { config.supportedClassifierVersions[classifier] }
    var author: String { config.author }
    var modelParameters: CreateML.MLImageClassifier.ModelParameters { config.modelParameters }
    var scenePrintRevision: Int? { config.scenePrintRevision }
}

let semaphore = DispatchSemaphore(value: 0)

Task {
    let selectedModel: MLModelType = .scaryCatScreeningML
    let selectedClassifier: ClassifierType = .binary
    let trainingCount = 1

    guard trainingCount > 0 else {
        print("トレーニングの回数は1以上を指定してください")
        semaphore.signal()
        return
    }

    guard selectedModel.supportedClassifierTypes.contains(selectedClassifier),
          let version = selectedModel.version(for: selectedClassifier)
    else {
        print("❌ エラー: 選択されたモデルは指定された分類器タイプをサポートしていません")
        exit(1)
    }

    let classifier = selectedClassifier.makeClassifier()

    // 指定された回数分トレーニングを実行
    for i in 1 ... trainingCount {
        print("トレーニング開始: \(i)/\(trainingCount)")

        // モデルの作成
        print("\n🚀 モデル作成開始...")
        guard let result = await classifier.create(
            author: selectedModel.author,
            modelName: selectedModel.name,
            version: version,
            modelParameters: selectedModel.modelParameters,
            scenePrintRevision: selectedModel.scenePrintRevision
        ) else {
            print("❌ モデル作成失敗")
            continue
        }

        result.saveLog(
            modelAuthor: selectedModel.author,
            modelName: selectedModel.name,
            modelVersion: version
        )

        print("トレーニング完了: \(selectedModel.name) [\(selectedClassifier.rawValue)] - \(i)/\(trainingCount)")
    }

    semaphore.signal()
}

semaphore.wait()
