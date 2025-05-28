import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

public struct IndividualModelReport {
    public let modelName: String
    public let positiveClassName: String
    public let trainingAccuracyRate: Double
    public let validationAccuracyPercentage: Double
    public let confusionMatrix: CSBinaryConfusionMatrix?

    func generateMarkdownReport() -> String {
        var report = """
        ## \(positiveClassName)
        - 訓練正解率: \(String(format: "%.1f%%", trainingAccuracyRate))
        - 検証正解率: \(String(format: "%.1f%%", validationAccuracyPercentage))
        """
        if let confusionMatrix = confusionMatrix {
            report += """

            - 再現率 (Recall)    : \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))
            - 適合率 (Precision) : \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))
            - F1スコア          : \(String(format: "%.1f%%", confusionMatrix.f1Score * 100.0))

            \(confusionMatrix.getMatrixGraph())
            """
        } else {
            report += "\n⚠️ 検証データが不十分なため、混同行列の計算をスキップしました\n"
        }
        report += "\n"
        return report
    }
}

public struct OvOTrainingResult: TrainingResultProtocol {
    public let metadata: CICTrainingMetadata
    public let individualReports: [IndividualModelReport]

    public var modelOutputPath: String {
        URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent().path
    }

    public init(
        modelName: String,
        trainingDurationInSeconds: TimeInterval,
        trainedModelFilePath: String,
        sourceTrainingDataDirectoryPath: String,
        detectedClassLabelsList: [String],
        maxIterations: Int,
        dataAugmentationDescription: String,
        featureExtractorDescription: String,
        individualReports: [IndividualModelReport]
    ) {
        self.metadata = CICTrainingMetadata(
            modelName: modelName,
            trainingDurationInSeconds: trainingDurationInSeconds,
            trainedModelFilePath: trainedModelFilePath,
            sourceTrainingDataDirectoryPath: sourceTrainingDataDirectoryPath,
            detectedClassLabelsList: detectedClassLabelsList,
            maxIterations: maxIterations,
            dataAugmentationDescription: dataAugmentationDescription,
            featureExtractorDescription: featureExtractorDescription
        )
        self.individualReports = individualReports
    }

    public func saveLog(modelAuthor _: String, modelName: String, modelVersion: String) {
        // ファイル生成日時フォーマッタ
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        // 各ペアの個別指標を表示
        var individualPairSections = ""
        for report in individualReports {
            individualPairSections += report.generateMarkdownReport() + "\n"
        }

        let markdown = """
        # OvO (One-vs-One) トレーニング実行レポート

        ## 実行概要
        モデル群         : OvOモデル群 (One-vs-One)
        モデルベース名   : \(modelName)
        レポート生成日時   : \(generatedDateString)
        最大反復回数     : \(metadata.maxIterations) (各ペアモデル共通)
        データ拡張       : \(metadata.dataAugmentationDescription)
        特徴抽出器       : \(metadata.featureExtractorDescription)
        検出されたクラス: \(metadata.detectedClassLabelsList.joined(separator: ", "))

        ## 個別ペアのトレーニング結果
        \(individualPairSections)
        """

        // モデルファイルと同じディレクトリに保存
        let outputDir = URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent()
        let textFileName = "OvO_Run_Report_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdown.write(toFile: textFilePath, atomically: true, encoding: .utf8)
            print("✅ ログファイル保存完了: \(textFilePath)")
        } catch {
            print("❌ エラー: ログファイル保存失敗: \(error.localizedDescription)")
        }
    }
}
