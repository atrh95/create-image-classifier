import BinaryClassification
import CSInterface
import Foundation
import CreateMLComponents
import CreateML
import MultiClassClassification
import MultiLabelClassification
import OvRClassification

// --- トレーナータイプ ---
enum TrainerType {
    case binary
    case multiClass
    case multiLabel
    case ovr

    var definedVersion: String {
        switch self {
            case .binary: "v5"
            case .multiClass: "v3"
            case .multiLabel: "v1"
            case .ovr: "v16"
        }
    }
}

// --- トレーニング設定 ---
let currentTrainerType: TrainerType = .ovr
let maxTrainingIterations = 5

// --- メタデータ定義 ---
let modelAuthor = "akitora"
let modelShortDescription = "ScaryCatScreener Training"
let modelVersion = currentTrainerType.definedVersion

print("🚀 トレーニングを開始します... 設定タイプ: \(currentTrainerType), バージョン: \(modelVersion)")

// トレーナーの選択と実行
let trainer: any ScreeningTrainerProtocol
var trainingResult: Any?
let shortDescription: String

switch currentTrainerType {
    case .binary:
        let binaryTrainer = BinaryClassificationTrainer()
        trainer = binaryTrainer
        shortDescription = "Binary Classification: \(modelShortDescription)"
    case .multiClass:
        let multiClassTrainer = MultiClassClassificationTrainer()
        trainer = multiClassTrainer
        shortDescription = "Multi-Class Classification: \(modelShortDescription)"
    case .multiLabel:
        let multiLabelTrainer = MultiLabelClassificationTrainer()
        trainer = multiLabelTrainer
        shortDescription = "Multi-Label Classification: \(modelShortDescription)"
    case .ovr:
        let ovrTrainer = OvRClassificationTrainer()
        trainer = ovrTrainer
        shortDescription = "One-vs-Rest (OvR) Batch: \(modelShortDescription)"
}

trainingResult = await trainer.train(
    author: modelAuthor,
    shortDescription: shortDescription,
    version: modelVersion,
    maxIterations: maxTrainingIterations
)

// 結果の処理
if let result = trainingResult {
    print("🎉 トレーニングが正常に完了しました。")

    // 結果をログに保存 (TrainingResultDataプロトコルのsaveLogメソッドを利用)
    if let resultData = result as? any TrainingResultProtocol {
        resultData.saveLog(
            modelAuthor: modelAuthor,
            modelDescription: modelShortDescription,
            modelVersion: modelVersion
        )
        print("💾 トレーニング結果をログに保存しました。")
    } else {
        print("⚠️ 結果の型がTrainingResultDataに準拠していません。ログは保存されませんでした。")
    }
} else {
    print("🛑 トレーニングまたはモデルの保存中にエラーが発生しました。")
}

print("✅ すべての処理が完了しました。")
