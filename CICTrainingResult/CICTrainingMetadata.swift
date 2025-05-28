import CICConfusionMatrix
import CICInterface
import Foundation

/// 画像分類モデルのトレーニング結果に関するメタデータを格納する構造体
public struct CICTrainingMetadata: TrainingResultProtocol {
    public let modelName: String
    public let trainingDurationInSeconds: TimeInterval
    public let trainedModelFilePath: String
    public let sourceTrainingDataDirectoryPath: String
    public let detectedClassLabelsList: [String]
    public let maxIterations: Int
    public let dataAugmentationDescription: String
    public let featureExtractorDescription: String

    public init(
        modelName: String,
        trainingDurationInSeconds: TimeInterval,
        trainedModelFilePath: String,
        sourceTrainingDataDirectoryPath: String,
        detectedClassLabelsList: [String],
        maxIterations: Int,
        dataAugmentationDescription: String,
        featureExtractorDescription: String
    ) {
        self.modelName = modelName
        self.trainingDurationInSeconds = trainingDurationInSeconds
        self.trainedModelFilePath = trainedModelFilePath
        self.sourceTrainingDataDirectoryPath = sourceTrainingDataDirectoryPath
        self.detectedClassLabelsList = detectedClassLabelsList
        self.maxIterations = maxIterations
        self.dataAugmentationDescription = dataAugmentationDescription
        self.featureExtractorDescription = featureExtractorDescription
    }

    public func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String
    ) {
        // ファイル生成日時フォーマッタ
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        // 文字列フォーマット
        let durationStr = String(format: "%.2f", trainingDurationInSeconds)

        // Markdownの基本内容
        let infoText = """
        # モデルトレーニング情報

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations)
        データ拡張       : \(dataAugmentationDescription)
        特徴抽出器       : \(featureExtractorDescription)

        ## トレーニング設定
        使用されたクラスラベル : \(detectedClassLabelsList.joined(separator: ", "))

        ## パフォーマンス指標
        トレーニング所要時間: \(durationStr) 秒

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        // Markdownファイルのパスを作成
        let outputDir = URL(fileURLWithPath: trainedModelFilePath).deletingLastPathComponent()
        let textFileName = "\(modelName)_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        // Markdownファイルに書き込み
        do {
            try infoText.write(toFile: textFilePath, atomically: true, encoding: .utf8)
            print("✅ モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            // 書き込み失敗時はコンソールに出力
            print("--- モデル情報 (Markdown) ---:")
            print(infoText)
            print("--- ここまで --- ")
        }
    }
} 
