import Foundation

/// 特定のラベルに対する評価メトリクス
public struct LabelMetrics {
    public let precision: Double
    public let recall: Double
    public let f1Score: Double
    public let support: Int // そのラベルの真のインスタンス数 (TP + FN)
}

/// 画像分類モデルのトレーニング結果（ログ記録用）を格納する構造体
public struct TrainingResultLogModel {
    /// トレーニングデータでの正解率 (0.0 ~ 100.0)
    public let trainingAccuracy: Double
    /// 検証データでの正解率 (0.0 ~ 100.0)
    public let validationAccuracy: Double
    /// トレーニングデータでのエラー率 (0.0 ~ 1.0)
    public let trainingError: Double
    /// 検証データでのエラー率 (0.0 ~ 1.0)
    public let validationError: Double
    /// トレーニングにかかった時間（秒）
    public let trainingDuration: TimeInterval
    /// 生成されたモデルファイルの出力パス
    public let modelOutputPath: String
    /// トレーニングに使用されたデータのパス
    public let trainingDataPath: String
    /// 検出されたクラスラベルのリスト
    public let classLabels: [String]
    /// 各ラベルごとの詳細なメトリクス (オプション)
    public let perLabelMetrics: [String: LabelMetrics]?
}
