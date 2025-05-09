import BinaryClassification
import MultiClassClassification
import MultiLabelClassification
import SCSInterface
import Foundation

// --- トレーナータイプ ---
enum TrainerType {
    case binary
    case multiClass
    case multiLabel
}

// --- トレーニング設定 ---
let currentTrainerType: TrainerType = .multiLabel

// --- メタデータ定義 ---
let modelAuthor = "akitora"
let modelShortDescription = "ScaryCatScreener - \(currentTrainerType)"
let modelVersion = "v1"
// ---------------------

print("🚀 トレーニングを開始します... 設定タイプ: \(currentTrainerType)")

// トレーナーの選択と実行
let trainer: any ScreeningTrainerProtocol
var trainingResult: Any? // Any? because the result type varies

switch currentTrainerType {
case .binary:
    let binaryTrainer = BinaryClassificationTrainer()
    trainer = binaryTrainer
    trainingResult = await binaryTrainer.train(
        author: modelAuthor,
        shortDescription: modelShortDescription,
        version: modelVersion
    )
case .multiClass:
    let multiClassTrainer = MultiClassClassificationTrainer()
    trainer = multiClassTrainer
    trainingResult = await multiClassTrainer.train(
        author: modelAuthor,
        shortDescription: modelShortDescription,
        version: modelVersion
    )
case .multiLabel:
    let multiLabelTrainer = MultiLabelClassificationTrainer()
    trainer = multiLabelTrainer
    trainingResult = await multiLabelTrainer.train(
        author: modelAuthor,
        shortDescription: modelShortDescription,
        version: modelVersion
    )
}

// 結果の処理
if let result = trainingResult {
    print("🎉 トレーニングが正常に完了しました。")

    // 結果をログに保存 (TrainingResultDataプロトコルのsaveLogメソッドを利用)
    if let resultData = result as? any TrainingResultData {
        resultData.saveLog(
            trainer: trainer,
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
