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
            case .binary: BinaryClassifier()
            case .multiClass: MultiClassClassifier()
            case .multiLabel: MultiLabelClassifier()
            case .ovr: OvRClassifier()
            case .ovo: OvOClassifier()
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
    }

    static let configs: [MLModelType: ModelConfig] = [
        .scaryCatScreeningML: ModelConfig(
            name: "ScaryCatScreeningML",
            supportedClassifierVersions: [
                .binary: "v6",
                .multiClass: "v3",
                .multiLabel: "v1",
                .ovr: "v27",
                .ovo: "v1",
            ],
            author: "akitora",
            modelParameters: MLImageClassifier.ModelParameters(
                validation: .split(strategy: .automatic),
                maxIterations: 20,
                augmentation: [],
                algorithm: .transferLearning(
                    featureExtractor: .scenePrint(revision: 2),
                    classifier: .logisticRegressor
                )
            )
        ),
    ]

    var config: ModelConfig {
        guard let config = Self.configs[self] else {
            fatalError("Configが存在しないモデルタイプ: \(self)")
        }
        return config
    }
}

let semaphore = DispatchSemaphore(value: 0)

Task {
    let selectedModel: MLModelType = .scaryCatScreeningML
    let selectedClassifier: ClassifierType = .ovr
    let trainingCount = 5

    guard selectedModel.config.supportedClassifierVersions.keys.contains(selectedClassifier),
          let version = selectedModel.config.supportedClassifierVersions[selectedClassifier]
    else {
        print("❌ エラー: 選択されたモデルは指定された分類器タイプをサポートしていません")
        exit(1)
    }

    let classifier = selectedClassifier.makeClassifier()

    // 指定された回数分トレーニングを実行
    for i in 1 ... trainingCount {
        print("トレーニング開始: \(i)/\(trainingCount)")

        // モデルの作成
        guard let result = await classifier.create(
            author: selectedModel.config.author,
            modelName: selectedModel.config.name,
            version: version,
            modelParameters: selectedModel.config.modelParameters
        ) else {
            print("❌ モデル作成失敗")
            continue
        }

        result.saveLog(
            modelAuthor: selectedModel.config.author,
            modelName: selectedModel.config.name,
            modelVersion: version
        )

        print("トレーニング完了: \(selectedModel.config.name) [\(selectedClassifier.rawValue)] \(version) - \(i)/\(trainingCount)")
    }

    semaphore.signal()
}

semaphore.wait()
