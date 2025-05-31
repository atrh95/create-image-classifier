import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

public struct MultiClassTrainingResult: TrainingResultProtocol {
    // メタデータ
    public let metadata: CICTrainingMetadata

    // パフォーマンス指標
    public let trainingMetrics: (accuracy: Double, errorRate: Double)
    public let validationMetrics: (accuracy: Double, errorRate: Double)

    // 詳細な性能指標
    public let confusionMatrix: CICMultiClassConfusionMatrix?
    public let classMetrics: [ClassMetrics]

    public var modelOutputPath: String {
        URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent().path
    }

    public init(
        metadata: CICTrainingMetadata,
        trainingMetrics: (accuracy: Double, errorRate: Double),
        validationMetrics: (accuracy: Double, errorRate: Double),
        confusionMatrix: CICMultiClassConfusionMatrix?
    ) {
        self.metadata = metadata
        self.trainingMetrics = trainingMetrics
        self.validationMetrics = validationMetrics
        self.confusionMatrix = confusionMatrix
        classMetrics = confusionMatrix?.calculateMetrics() ?? []
    }

    public func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String
    ) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        let trainingAccStr = String(format: "%.2f", trainingMetrics.accuracy)
        let validationAccStr = String(format: "%.2f", validationMetrics.accuracy)
        let trainingErrStr = String(format: "%.2f", trainingMetrics.errorRate * 100)
        let validationErrStr = String(format: "%.2f", validationMetrics.errorRate * 100)
        let durationStr = String(format: "%.2f", metadata.trainingDurationInSeconds)

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(metadata.maxIterations)
        データ拡張       : \(metadata.dataAugmentationDescription)
        特徴抽出器       : \(metadata.featureExtractorDescription)

        ## トレーニング設定
        使用されたクラスラベル : \(metadata.detectedClassLabelsList.joined(separator: ", "))

        ## パフォーマンス指標 (全体)
        トレーニング所要時間: \(durationStr) 秒
        トレーニング誤分類率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)%
        検証誤分類率 (学習時自動検証) : \(validationErrStr)%
        """

        if let confusionMatrix {
            markdownText += """
            ## クラス別性能指標
            | クラス | 再現率 | 適合率 | F1スコア |
            |--------|--------|--------|----------|
            \(classMetrics.map { metric in
                "| \(metric.label) | \(String(format: "%.1f%%", metric.recall * 100.0)) | \(String(format: "%.1f%%", metric.precision * 100.0)) | \(String(format: "%.3f", metric.f1Score)) |"
            }.joined(separator: "\n"))
            """
        }

        markdownText += """

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        let outputDir = URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent()
        let textFileName = "MultiClass_Run_Report_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdownText.write(toFile: textFilePath, atomically: true, encoding: String.Encoding.utf8)
            print("✅ [\(modelName)] モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ [\(modelName)] Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            print("--- [\(modelName)] モデル情報 (Markdown) ---:")
            print(markdownText)
            print("--- ここまで --- ")
        }
    }
}
