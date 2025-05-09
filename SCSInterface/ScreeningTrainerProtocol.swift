import Foundation

/// 画像分類モデルトレーナー
public protocol ScreeningTrainerProtocol {
    associatedtype TrainingResultType

    var modelName: String { get }

    /// 出力先ディレクトリパス
    var customOutputDirPath: String { get }

    /// リソースディレクトリ絶対パス
    var resourcesDirectoryPath: String { get }

    /// トレーニング実行 (読み込み、学習、評価、保存)
    /// - Parameters:
    ///   - author: モデルの作成者
    ///   - shortDescription: モデルの簡単な説明
    ///   - version: モデルのバージョン
    /// - Returns: トレーニング結果 (成功時) または nil (失敗時)
    func train(author: String, shortDescription: String, version: String) async -> TrainingResultType?
} 