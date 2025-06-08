import BinaryClassifier
import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import CreateML
import CreateMLComponents
import Foundation
import MultiClassClassifier
import OvOClassifier
import OvRClassifier

// 分類器の種類
enum ClassifierType: String {
    case binary, multiClass, ovr, ovo

    func makeClassifier() -> any ClassifierProtocol {
        switch self {
            case .binary: BinaryClassifier()
            case .multiClass: MultiClassClassifier()
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
        let shouldEqualizeFileCount: Bool
    }

    static let configs: [MLModelType: ModelConfig] = [
        .scaryCatScreeningML: ModelConfig(
            name: "ScaryCatScreeningML",
            supportedClassifierVersions: [
                .binary: "v6",
                .multiClass: "v3",
                .ovr: "v33",
                .ovo: "v1",
            ],
            author: "akitora",
            modelParameters: MLImageClassifier.ModelParameters(
                validation: .split(strategy: .automatic),
                maxIterations: 15,
                augmentation: [.flip, .rotation],
                algorithm: .transferLearning(
                    featureExtractor: .scenePrint(revision: 2),
                    classifier: .logisticRegressor
                )
            ),
            shouldEqualizeFileCount: true
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
    let trainingCount = 10

    guard selectedModel.config.supportedClassifierVersions.keys.contains(selectedClassifier),
          let version = selectedModel.config.supportedClassifierVersions[selectedClassifier]
    else {
        print("❌ エラー: 選択されたモデルは指定された分類器タイプをサポートしていません")
        exit(1)
    }

    let classifier = selectedClassifier.makeClassifier()

    // 指定された回数分トレーニングを実行
    for i in 1 ... trainingCount {
        print("\n📚 トレーニング開始: \(i)/\(trainingCount)")

        do {
            // モデルの作成
            try classifier.createAndSaveModel(
                author: selectedModel.config.author,
                modelName: selectedModel.config.name,
                version: version,
                modelParameters: selectedModel.config.modelParameters,
                shouldEqualizeFileCount: selectedModel.config.shouldEqualizeFileCount
            )

            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("🎉 MLModelの作成が完了しました！")
            print("  モデル: \(selectedModel.config.name)")
            print("  分類器: \(selectedClassifier.rawValue)")
            print("  バージョン: \(version)")
            print("  進捗: \(i)/\(trainingCount)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        } catch {
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("❌ エラーが発生しました: \(error)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            continue
        }
    }

    semaphore.signal()
}

semaphore.wait()
