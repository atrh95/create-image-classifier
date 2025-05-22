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
                .binary: "v6",
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
    let selectedTrainer: TrainerType = .ovr
    let author = "akitora"
    let trainingCount = 1

    guard trainingCount > 0 else {
        print("トレーニングの回数は1以上を指定してください")
        semaphore.signal()
        return
    }

    // ModelParametersの設定
    // 特徴抽出器の設定：ScenePrint の場合のみリビジョンを指定する
    let scenePrintRevision: Int? = 1
    let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
        featureExtractor: .scenePrint(revision: scenePrintRevision),
        classifier: .logisticRegressor
    )

    let modelParameters = MLImageClassifier.ModelParameters(
        validation: .split(strategy: .automatic),
        maxIterations: 8,
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

    // 指定された回数分トレーニングを実行
    for i in 1 ... trainingCount {
        print("トレーニング開始: \(i)/\(trainingCount)")

        guard let result = await trainer.train(
            author: author,
            modelName: selectedModel.name,
            version: version,
            modelParameters: modelParameters,
            scenePrintRevision: scenePrintRevision
        ) as? TrainingResultProtocol else {
            print("トレーニングに失敗しました: \(i)/\(trainingCount)")
            continue
        }

        result.saveLog(
            modelAuthor: author,
            modelName: selectedModel.name,
            modelVersion: version
        )

        print("トレーニング完了: \(selectedModel.name) [\(selectedTrainer.rawValue)] - \(i)/\(trainingCount)")
    }

    semaphore.signal()
}

semaphore.wait()
