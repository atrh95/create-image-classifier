import Foundation

public protocol CICModelReportProtocol {
    /// モデルファイル名
    var modelFileName: String { get }

    /// トレーニングと検証のメトリクス
    var metrics: (
        training: (accuracy: Double, errorRate: Double),
        validation: (accuracy: Double, errorRate: Double)
    ) { get }

    /// クラスごとの画像枚数
    var classCounts: [String: Int] { get }

    /// メトリクスの説明文を生成
    func generateMetricsDescription() -> String
}
