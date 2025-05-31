import CoreML
import CreateML
import Foundation

/// 画像分類モデルトレーナー
public protocol ClassifierProtocol {
    associatedtype TrainingResultType: TrainingResultProtocol

    var outputParentDirPath: String { get }
    var classificationMethod: String { get }
    var resourcesDirectoryPath: String { get }
    var outputDirectoryPathOverride: String? { get set }
    var resourceDirPathOverride: String? { get set }

    /// モデルの作成
    func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> TrainingResultType?
}
