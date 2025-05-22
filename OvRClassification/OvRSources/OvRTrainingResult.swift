import CSInterface
import Foundation
import CSConfusionMatrix

public struct IndividualModelReport {
    public let modelName: String
    public let positiveClassName: String
    public let trainingAccuracyRate: Double
    public let validationAccuracyPercentage: Double
    public let confusionMatrix: CSBinaryConfusionMatrix?
}

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelName: String
    public let trainingDurationInSeconds: TimeInterval
    public let trainedModelFilePath: String
    public let sourceTrainingDataDirectoryPath: String
    public let detectedClassLabelsList: [String]
    public let maxIterations: Int
    public let dataAugmentationDescription: String
    public let baseFeatureExtractorDescription: String
    public let scenePrintRevision: Int?
    public let individualReports: [IndividualModelReport]

    public var modelOutputPath: String {
        URL(fileURLWithPath: trainedModelFilePath).deletingLastPathComponent().path
    }

    public init(
        modelName: String,
        trainingDurationInSeconds: TimeInterval,
        trainedModelFilePath: String,
        sourceTrainingDataDirectoryPath: String,
        detectedClassLabelsList: [String],
        maxIterations: Int,
        dataAugmentationDescription: String,
        baseFeatureExtractorDescription: String,
        scenePrintRevision: Int?,
        individualReports: [IndividualModelReport]
    ) {
        self.modelName = modelName
        self.trainingDurationInSeconds = trainingDurationInSeconds
        self.trainedModelFilePath = trainedModelFilePath
        self.sourceTrainingDataDirectoryPath = sourceTrainingDataDirectoryPath
        self.detectedClassLabelsList = detectedClassLabelsList
        self.maxIterations = maxIterations
        self.dataAugmentationDescription = dataAugmentationDescription
        self.baseFeatureExtractorDescription = baseFeatureExtractorDescription
        self.scenePrintRevision = scenePrintRevision
        self.individualReports = individualReports
    }

    public func saveLog(modelAuthor: String, modelName: String, modelVersion: String) {
        // ファイル生成日時フォーマッタ
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        // 各ペアの個別指標を表示
        var individualPairSections = ""
        for report in individualReports {
            individualPairSections += """
            ## \(report.positiveClassName)
            - 訓練正解率: \(String(format: "%.1f%%", report.trainingAccuracyRate))
            - 検証正解率: \(String(format: "%.1f%%", report.validationAccuracyPercentage))
            """
            if let confusionMatrix = report.confusionMatrix {
                individualPairSections += """
                
                - 再現率 (Recall)    : \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))
                - 適合率 (Precision) : \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))
                - F1スコア          : \(String(format: "%.1f%%", confusionMatrix.f1Score * 100.0))
                
                \(confusionMatrix.getMatrixGraph())
                """
            } else {
                individualPairSections += "\n⚠️ 検証データが不十分なため、混同行列の計算をスキップしました\n"
            }
            individualPairSections += "\n"
        }

        let markdown = """
        # OvR (One-vs-Rest) トレーニング実行レポート

        ## 実行概要
        モデル群         : OvRモデル群 (One-vs-Rest)
        モデルベース名   : \(modelName)
        レポート生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations) (各ペアモデル共通)
        データ拡張       : \(dataAugmentationDescription)
        特徴抽出器       : \(baseFeatureExtractorDescription)
        検出されたクラス: \(detectedClassLabelsList.joined(separator: ", "))

        ## 個別ペアのトレーニング結果
        \(individualPairSections)
        """

        // モデルファイルと同じディレクトリに保存
        let outputDir = URL(fileURLWithPath: trainedModelFilePath)
        let textFileName = "OvR_Run_Report_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdown.write(toFile: textFilePath, atomically: true, encoding: .utf8)
            print("✅ ログファイル保存完了: \(textFilePath)")
        } catch {
            print("❌ エラー: ログファイル保存失敗: \(error.localizedDescription)")
        }
    }
}
