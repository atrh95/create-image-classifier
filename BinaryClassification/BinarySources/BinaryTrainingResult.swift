import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

/// 画像分類モデルのトレーニング結果を格納する構造体
public struct BinaryTrainingResult: TrainingResultProtocol {
    public let metadata: CICTrainingMetadata
    public let trainingDataAccuracyPercentage: Double
    public let validationDataAccuracyPercentage: Double?
    public let trainingDataMisclassificationRate: Double
    public let validationDataMisclassificationRate: Double?
    public let confusionMatrix: CSBinaryConfusionMatrix?

    public init(
        modelName: String,
        trainingDataAccuracyPercentage: Double,
        validationDataAccuracyPercentage: Double?,
        trainingDataMisclassificationRate: Double,
        validationDataMisclassificationRate: Double?,
        trainingDurationInSeconds: TimeInterval,
        trainedModelFilePath: String,
        sourceTrainingDataDirectoryPath: String,
        detectedClassLabelsList: [String],
        maxIterations: Int,
        dataAugmentationDescription: String,
        featureExtractorDescription: String,
        confusionMatrix: CSBinaryConfusionMatrix?
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
        self.trainingDataAccuracyPercentage = trainingDataAccuracyPercentage
        self.validationDataAccuracyPercentage = validationDataAccuracyPercentage
        self.trainingDataMisclassificationRate = trainingDataMisclassificationRate
        self.validationDataMisclassificationRate = validationDataMisclassificationRate
        self.confusionMatrix = confusionMatrix
    }

    public func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String
    ) {
        // ファイル生成日時フォーマッタ
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        // 文字列フォーマット
        let trainingAccStr = String(format: "%.2f", trainingDataAccuracyPercentage)
        let validationAccStr = validationDataAccuracyPercentage.map { String(format: "%.2f", $0) } ?? "N/A"
        let trainingErrStr = String(format: "%.2f", trainingDataMisclassificationRate * 100)
        let validationErrStr = validationDataMisclassificationRate
            .map { String(format: "%.2f", $0 * 100) } ?? "N/A"

        // 混同行列から計算した指標
        let confusionMatrixSection = if let confusionMatrix {
            """
            - 再現率 (Recall)    : \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))
              - 正例を正しく識別する能力を示す指標
            - 適合率 (Precision) : \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))
              - 予測が正例と判断した中で、実際に正例だった割合
            - F1スコア          : \(String(format: "%.1f%%", confusionMatrix.f1Score * 100.0))
              - 再現率と適合率の調和平均。両方の指標のバランスを表し、1に近いほどモデルの性能が高いことを示す

            \(confusionMatrix.getMatrixGraph())
            """
        } else {
            "⚠️ 検証データが不十分なため、混同行列の計算をスキップしました"
        }

        // Markdownの内容
        let infoText = """
        # モデルトレーニング情報

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(metadata.maxIterations)
        データ拡張       : \(metadata.dataAugmentationDescription)
        特徴抽出器       : \(metadata.featureExtractorDescription)

        ## トレーニング設定
        使用されたクラスラベル : \(metadata.detectedClassLabelsList.joined(separator: ", "))

        ## パフォーマンス指標
        トレーニング所要時間: \(String(format: "%.2f", metadata.trainingDurationInSeconds)) 秒
        トレーニング誤分類率 : \(trainingErrStr)%
        訓練データ正解率 : \(trainingAccStr)% ※モデルが学習に使用したデータでの正解率
        検証データ正解率 : \(validationAccStr)% ※モデルが学習に使用していない未知のデータでの正解率
        検証誤分類率       : \(validationErrStr)%

        \(confusionMatrixSection)

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        // Markdownファイルのパスを作成
        let outputDir = URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent()
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
