import CoreML
import CreateML
import Foundation

/// 画像分類モデルトレーナー
public protocol ClassifierProtocol {
    associatedtype TrainingResultType: TrainingResultProtocol

    var outputDirPath: String { get }
    var classificationMethod: String { get }
    var resourcesDirectoryPath: String { get }
    var testResourcesDirectoryPath: String? { get set }
    var outputDirectoryPathOverride: String? { get set }

    /// トレーニングの実行
    func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> TrainingResultType?

    /// モデル出力用のディレクトリを設定
    func setupOutputDirectory(modelName: String, version: String) throws -> URL

    /// クラスラベルディレクトリの一覧を取得
    func getClassLabelDirectories() throws -> [URL]

    /// トレーニングのデータソースを準備
    func prepareTrainingData(from classLabelDirURLs: [URL]) throws -> MLImageClassifier.DataSource

    /// モデルのトレーニングを実行
    func trainModel(
        trainingDataSource: MLImageClassifier.DataSource,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) throws -> (MLImageClassifier, TimeInterval)

    /// モデルのメタデータを作成
    func createModelMetadata(
        author: String,
        version: String,
        classLabelDirURLs: [URL],
        trainingMetrics: MLClassifierMetrics,
        validationMetrics: MLClassifierMetrics,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) -> MLModelMetadata

    /// トレーニング済みモデルを保存
    func saveModel(
        imageClassifier: MLImageClassifier,
        modelName: String,
        modelFileName: String,
        version: String,
        outputDirectoryURL: URL,
        metadata: MLModelMetadata
    ) throws -> String

    /// トレーニングの結果を作成
    func createTrainingResult(
        modelName: String,
        classLabelDirURLs: [URL],
        trainingMetrics: MLClassifierMetrics,
        validationMetrics: MLClassifierMetrics,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?,
        trainingDurationSeconds: TimeInterval,
        modelFilePath: String
    ) -> TrainingResultType
}
