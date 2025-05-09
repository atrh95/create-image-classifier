import Foundation
import SCSInterface // To access ScreeningTrainerProtocol

/// 画像分類モデルの共通トレーニング結果データプロトコル
public protocol TrainingResultData {
    /// トレーニングデータでの正解率 (0.0 ~ 100.0)
    var trainingAccuracy: Double { get }
    /// 検証データでの正解率 (0.0 ~ 100.0)
    var validationAccuracy: Double { get }
    /// トレーニングデータでのエラー率 (0.0 ~ 1.0)
    var trainingError: Double { get }
    /// 検証データでのエラー率 (0.0 ~ 1.0)
    var validationError: Double { get }
    /// トレーニングにかかった時間（秒）
    var trainingDuration: TimeInterval { get }
    /// 生成されたモデルファイルの出力パス
    var modelOutputPath: String { get }
    /// トレーニングに使用されたデータのパス
    var trainingDataPath: String { get }
    /// 検出されたクラスラベルのリスト
    var classLabels: [String] { get }

    /// トレーニング結果をMarkdownファイルとして保存する
    /// - Parameters:
    ///   - trainer: 使用されたトレーナーのインスタンス
    ///   - modelAuthor: モデルの作成者
    ///   - modelDescription: モデルの簡単な説明
    ///   - modelVersion: モデルのバージョン
    func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String)
} 