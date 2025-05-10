import BinaryClassification
import Foundation
import MultiClassClassification
import MultiLabelClassification
import OvRClassification
import SCSInterface

// --- トレーナータイプ ---
enum TrainerType {
    case binary
    case multiClass
    case multiLabel
    case ovr

    var definedVersion: String {
        switch self {
        case .binary: return "v2"
        case .multiClass: return "v2"
        case .multiLabel: return "v1"
        case .ovr: return "v4"
        }
    }
}

// --- トレーニング設定 ---
let currentTrainerType: TrainerType = .ovr

// --- メタデータ定義 ---
let modelAuthor = "akitora"
let modelShortDescription = "ScaryCatScreener Training"
let modelVersion = currentTrainerType.definedVersion

print("🚀 トレーニングを開始します... 設定タイプ: \(currentTrainerType), バージョン: \(modelVersion)")

// トレーナーの選択と実行
let trainer: any ScreeningTrainerProtocol
var trainingResult: Any?

switch currentTrainerType {
    case .binary:
        let binaryTrainer = BinaryClassificationTrainer()
        trainer = binaryTrainer
        trainingResult = await binaryTrainer.train(
            author: modelAuthor,
            shortDescription: "Binary Classification: \(modelShortDescription)",
            version: modelVersion
        )
    case .multiClass:
        let multiClassTrainer = MultiClassClassificationTrainer()
        trainer = multiClassTrainer
        trainingResult = await multiClassTrainer.train(
            author: modelAuthor,
            shortDescription: "Multi-Class Classification: \(modelShortDescription)",
            version: modelVersion
        )
    case .multiLabel:
        let multiLabelTrainer = MultiLabelClassificationTrainer()
        trainer = multiLabelTrainer
        trainingResult = await multiLabelTrainer.train(
            author: modelAuthor,
            shortDescription: "Multi-Label Classification: \(modelShortDescription)",
            version: modelVersion
        )
    case .ovr:
        let ovrTrainer = OvRClassificationTrainer()
        trainer = ovrTrainer
        trainingResult = await ovrTrainer.train(
            author: modelAuthor,
            shortDescription: "One-vs-Rest (OvR) Batch: \(modelShortDescription)",
            version: modelVersion
        )
}

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
