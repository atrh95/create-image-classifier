import CSInterface
import Foundation

public struct MultiClassTrainingResult: TrainingResultProtocol {
    public let modelName: String
    public let trainingDataAccuracy: Double
    public let validationDataAccuracy: Double
    public let trainingDataErrorRate: Double
    public let validationDataErrorRate: Double
    public let trainingTimeInSeconds: TimeInterval
    public let modelOutputPath: String
    public let trainingDataPath: String
    public let classLabels: [String]
    public let maxIterations: Int
    public let macroAverageRecall: Double
    public let macroAveragePrecision: Double
    public let detectedClassLabelsList: [String]
    public let dataAugmentationDescription: String
    public let featureExtractorDescription: String

    public init(
        modelName: String,
        trainingDataAccuracy: Double,
        validationDataAccuracy: Double,
        trainingDataErrorRate: Double,
        validationDataErrorRate: Double,
        trainingTimeInSeconds: TimeInterval,
        modelOutputPath: String,
        trainingDataPath: String,
        classLabels: [String],
        maxIterations: Int,
        macroAverageRecall: Double,
        macroAveragePrecision: Double,
        detectedClassLabelsList: [String],
        dataAugmentationDescription: String,
        baseFeatureExtractorDescription: String,
        scenePrintRevision: Int?
    ) {
        self.modelName = modelName
        self.trainingDataAccuracy = trainingDataAccuracy
        self.validationDataAccuracy = validationDataAccuracy
        self.trainingDataErrorRate = trainingDataErrorRate
        self.validationDataErrorRate = validationDataErrorRate
        self.trainingTimeInSeconds = trainingTimeInSeconds
        self.modelOutputPath = modelOutputPath
        self.trainingDataPath = trainingDataPath
        self.classLabels = classLabels
        self.maxIterations = maxIterations
        self.macroAverageRecall = macroAverageRecall
        self.macroAveragePrecision = macroAveragePrecision
        self.detectedClassLabelsList = detectedClassLabelsList
        self.dataAugmentationDescription = dataAugmentationDescription
        if let revision = scenePrintRevision {
            featureExtractorDescription = "\(baseFeatureExtractorDescription)(revision: \(revision))"
        } else {
            featureExtractorDescription = baseFeatureExtractorDescription
        }
    }

    public func saveLog(
        modelAuthor: String,
        modelName _: String,
        modelVersion: String
    ) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let classLabelsString = classLabels.isEmpty ? "不明" : classLabels.joined(separator: ", ")

        let trainingAccStr = String(format: "%.2f", trainingDataAccuracy)
        let validationAccStr = String(format: "%.2f", validationDataAccuracy)
        let trainingErrStr = String(format: "%.2f", trainingDataErrorRate * 100)
        let validationErrStr = String(format: "%.2f", validationDataErrorRate * 100)
        let durationStr = String(format: "%.2f", trainingTimeInSeconds)

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations)
        データ拡張       : \(dataAugmentationDescription)
        特徴抽出器       : \(featureExtractorDescription)

        ## トレーニング設定
        使用されたクラスラベル : \(classLabelsString)

        ## パフォーマンス指標 (全体)
        トレーニング所要時間: \(durationStr) 秒
        トレーニング誤分類率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)%
        検証誤分類率 (学習時自動検証) : \(validationErrStr)%
        """

        markdownText += """

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        let outputDir = URL(fileURLWithPath: modelOutputPath).deletingLastPathComponent()
        let textFileName = "\(modelName)_\(modelVersion).md"
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
