import Foundation

public protocol CSBinaryConfusionMatrixProtocol {
    /// 例：猫vs犬の判定の場合、「実際が猫で予測も猫」の数
    var truePositive: Int { get }
    
    /// 例：猫vs犬の判定の場合、「実際は犬だが予測が猫」の数
    var falsePositive: Int { get }
    
    /// 例：猫vs犬の判定の場合、「実際は猫だが予測が犬」の数
    var falseNegative: Int { get }
    
    /// 例：猫vs犬の判定の場合、「実際が犬で予測も犬」の数
    var trueNegative: Int { get }
    
    /// 例：猫vs犬の判定の場合、「実際の猫のうち、正しく猫と判定できた割合」
    var recall: Double { get }
    
    /// 例：猫vs犬の判定の場合、「猫と判定したうち、実際に猫だった割合」
    var precision: Double { get }
    
    /// 例：猫vs犬の判定の場合、「(実際が猫で予測も猫 + 実際が犬で予測も犬) / 全サンプル数」
    var accuracy: Double { get }
    
    /// 例：猫vs犬の判定の場合、「猫の判定の正確さ（適合率）と網羅性（再現率）のバランスを示す指標。
    /// 例：猫の判定で「見逃しが少ない（再現率が高い）」かつ「誤判定が少ない（適合率が高い）」場合に高くなる。
    /// 値の範囲は0.0（最悪）から1.0（最良）」
    var f1Score: Double { get }
}

public protocol CSMultiClassConfusionMatrixProtocol {
    /// 例：labels = ["猫", "犬", "鳥"] の場合、
    /// matrix[0] は「実際が猫」の場合の予測結果の数
    /// matrix[0][0] は「実際が猫で予測も猫」の数
    /// matrix[0][1] は「実際が猫だが予測が犬」の数
    var labels: [String] { get }
    
    /// 例：labels = ["猫", "犬", "鳥"] の場合、
    /// matrix = [
    ///     [10, 2, 1],  // 猫の行：実際が猫で、予測が[猫,犬,鳥]の数
    ///     [1, 15, 0],  // 犬の行：実際が犬で、予測が[猫,犬,鳥]の数
    ///     [0, 1, 8]    // 鳥の行：実際が鳥で、予測が[猫,犬,鳥]の数
    /// ]
    var matrix: [[Int]] { get }
    
    /// 各クラスの性能指標を計算する
    /// 例：labels = ["猫", "犬", "鳥"] の場合、
    /// 戻り値: [
    ///     (label: "猫", recall: 0.77, precision: 0.91, f1Score: 0.83),  // 猫の再現率、適合率、F1スコア
    ///     (label: "犬", recall: 0.94, precision: 0.83, f1Score: 0.88),  // 犬の再現率、適合率、F1スコア
    ///     (label: "鳥", recall: 0.89, precision: 0.89, f1Score: 0.89)   // 鳥の再現率、適合率、F1スコア
    /// ]
    func calculateMetrics() -> [(label: String, recall: Double, precision: Double, f1Score: Double)]
} 