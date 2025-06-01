import Foundation

/// 分類器のテストで使用する共通のエラー定義
enum ClassifierTestsError: Error {
    /// モデルの訓練に失敗した場合
    case trainingFailed

    /// モデルファイルが見つからない場合
    case modelFileMissing

    /// 予測に失敗した場合
    case predictionFailed

    /// テストのセットアップに失敗した場合
    case setupFailed

    /// リソースパスに関するエラー
    case resourcePathError

    /// テストリソースが見つからない場合
    case testResourceMissing
}
