import CSInterface
import CSConfusionMatrix
import Foundation

public struct MultiLabelTrainingResult: TrainingResultProtocol {
    // 型定義
    public typealias LabelMetric = LabelMetrics
    
    // 基本情報
    public let modelName: String
    public let modelOutputPath: String
    public let trainingDataPath: String
    
    // トレーニング設定
    public let classLabels: [String]
    public let maxIterations: Int
    public let dataAugmentationDescription: String
    public let featureExtractorDescription: String
    
    // パフォーマンス指標
    public let trainingMetrics: (accuracy: Double, errorRate: Double)
    public let validationMetrics: (accuracy: Double, errorRate: Double)
    public let trainingDurationInSeconds: TimeInterval
    
    // 詳細な性能指標
    public let confusionMatrix: CSMultiLabelConfusionMatrix?
    public let labelMetrics: [LabelMetric]?

    public init(
        modelName: String,
        trainingDurationInSeconds: TimeInterval,
        modelOutputPath: String,
        trainingDataPath: String,
        classLabels: [String],
        maxIterations: Int,
        trainingMetrics: (accuracy: Double, errorRate: Double),
        validationMetrics: (accuracy: Double, errorRate: Double),
        dataAugmentationDescription: String,
        featureExtractorDescription: String,
        scenePrintRevision: Int?,
        confusionMatrix: CSMultiLabelConfusionMatrix? = nil
    ) {
        self.modelName = modelName
        self.modelOutputPath = modelOutputPath
        self.trainingDataPath = trainingDataPath
        self.classLabels = classLabels
        self.maxIterations = maxIterations
        self.trainingMetrics = trainingMetrics
        self.validationMetrics = validationMetrics
        self.trainingDurationInSeconds = trainingDurationInSeconds
        self.dataAugmentationDescription = dataAugmentationDescription
        self.featureExtractorDescription = featureExtractorDescription
        self.confusionMatrix = confusionMatrix
        self.labelMetrics = confusionMatrix?.calculateMetrics()
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

        let classLabelsString = classLabels.isEmpty ? "不明" : classLabels.joined(separator: ", ")
        let durationStr = String(format: "%.2f", trainingDurationInSeconds)
        let trainingAccStr = String(format: "%.2f", trainingMetrics.accuracy)
        let validationAccStr = String(format: "%.2f", validationMetrics.accuracy)
        let trainingErrStr = String(format: "%.2f", trainingMetrics.errorRate * 100)
        let validationErrStr = String(format: "%.2f", validationMetrics.errorRate * 100)

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations) (注: CreateMLComponentsでは直接使用されません)
        データ拡張       : \(dataAugmentationDescription)
        特徴抽出器       : \(featureExtractorDescription)

        ## トレーニング設定
        アノテーションファイル: \(URL(fileURLWithPath: trainingDataPath).lastPathComponent)
        検出された全ラベル : \(classLabelsString)

        ## 全体のパフォーマンス指標
        トレーニング所要時間: \(durationStr) 秒
        トレーニング誤分類率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)%
        検証誤分類率 (学習時自動検証) : \(validationErrStr)%

        ## ラベル別性能指標
        """

        if let confusionMatrix {
            if let labelMetrics {
                for metric in labelMetrics {
                    markdownText += """
                    
                    ### \(metric.label)
                    再現率: \(String(format: "%.1f%%", metric.recall * 100.0)), \
                    適合率: \(String(format: "%.1f%%", metric.precision * 100.0)), \
                    F1スコア: \(String(format: "%.1f%%", metric.f1Score * 100.0))
                    """
                }
            }

            markdownText += "\n\n## 混同行列\n"
            markdownText += confusionMatrix.getMatrixGraph()
        }

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
