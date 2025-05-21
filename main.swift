import BinaryClassification
import CreateML
import CreateMLComponents
import CSInterface
import Foundation
import MultiClassClassification
import MultiLabelClassification
import OvOClassification
import OvRClassification

// 分類器の種類
enum TrainerType: String {
    case binary, multiClass, multiLabel, ovr, ovo

    func makeTrainer() -> any ScreeningTrainerProtocol {
        switch self {
            case .binary: BinaryClassificationTrainer()
            case .multiClass: MultiClassClassificationTrainer()
            case .multiLabel: MultiLabelClassificationTrainer()
            case .ovr: OvRClassificationTrainer()
            case .ovo: OvOClassificationTrainer()
        }
    }
}

// モデルの種類
enum MLModelType: String {
    case scaryCatScreeningML

    struct ModelConfig {
        let name: String
        let supportedTrainerVersions: [TrainerType: String]
    }

    private static let configs: [MLModelType: ModelConfig] = [
        .scaryCatScreeningML: ModelConfig(
            name: "ScaryCatScreeningML",
            supportedTrainerVersions: [
                .binary: "v5",
                .multiClass: "v3",
                .multiLabel: "v1",
                .ovr: "v20",
                .ovo: "v1",
            ]
        ),
    ]

    private var config: ModelConfig {
        guard let config = Self.configs[self] else {
            fatalError("Configが存在しないモデルタイプ: \(self)")
        }
        return config
    }

    var name: String { config.name }
    var supportedTrainerTypes: [TrainerType] { Array(config.supportedTrainerVersions.keys) }
    func version(for trainer: TrainerType) -> String? { config.supportedTrainerVersions[trainer] }
}

let semaphore = DispatchSemaphore(value: 0)

Task {
    let selectedModel: MLModelType = .scaryCatScreeningML
    let selectedTrainer: TrainerType = .ovo
    let author = "akitora"

    // ModelParametersの設定
    // 特徴抽出器の設定：ScenePrint の場合のみリビジョンを指定する
    let scenePrintRevision: Int? = 1
    let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
        featureExtractor: .scenePrint(revision: scenePrintRevision),
        classifier: .logisticRegressor
    )

    let modelParameters = MLImageClassifier.ModelParameters(
        validation: .split(strategy: .automatic),
        maxIterations: 10,
        augmentation: [],
        algorithm: algorithm
    )

    guard selectedModel.supportedTrainerTypes.contains(selectedTrainer),
          let version = selectedModel.version(for: selectedTrainer)
    else {
        print("無効な組み合わせです")
        semaphore.signal()
        return
    }

    let trainer = selectedTrainer.makeTrainer()

    guard let result = await trainer.train(
        author: author,
        modelName: selectedModel.name,
        version: version,
        modelParameters: modelParameters,
        scenePrintRevision: scenePrintRevision
    ) as? TrainingResultProtocol else {
        print("トレーニングに失敗しました")
        semaphore.signal()
        return
    }

    result.saveLog(
        modelAuthor: author,
        modelName: selectedModel.name,
        modelVersion: version
    )

    print("トレーニング完了: \(selectedModel.name) [\(selectedTrainer.rawValue)]")
    semaphore.signal()
}

semaphore.wait()
