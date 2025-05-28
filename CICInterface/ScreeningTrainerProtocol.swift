import CreateML
import Foundation

/// 画像分類モデルトレーナー
public protocol ScreeningTrainerProtocol {
    associatedtype TrainingResultType

    var outputDirPath: String { get }
    var resourcesDirectoryPath: String { get }
    var classificationMethod: String { get }

    /// トレーニング実行
    func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async
        -> TrainingResultType?
}
