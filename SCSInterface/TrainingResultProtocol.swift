import Foundation
import SCSInterface // To access ScreeningTrainerProtocol

/// 画像分類モデルの共通トレーニング結果データプロトコル
public protocol TrainingResultProtocol {
    /// トレーニング結果をMarkdownファイルとして保存する
    /// - Parameters:
    ///   - trainer: 使用されたトレーナーのインスタンス
    ///   - modelAuthor: モデルの作成者
    ///   - modelDescription: モデルの簡単な説明
    ///   - modelVersion: モデルのバージョン
    func saveLog(
        modelAuthor: String,
        modelDescription: String,
        modelVersion: String
    )
}
