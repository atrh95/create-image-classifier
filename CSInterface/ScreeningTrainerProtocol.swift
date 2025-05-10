import Foundation

/// 画像分類モデルトレーナー
public protocol ScreeningTrainerProtocol {
    associatedtype TrainingResultType

    var modelName: String { get }

    /// 出力先ディレクトリパス
    var customOutputDirPath: String { get }

    /// リソースディレクトリ絶対パス
    var resourcesDirectoryPath: String { get }

    /// 各トレーナー固有の接頭辞 (例: "OvR", "MultiLabel")
    var outputRunNamePrefix: String { get }

    /// トレーニング実行 (読み込み、学習、評価、保存)
    /// - Parameters:
    ///   - author: モデルの作成者
    ///   - shortDescription: モデルの簡単な説明
    ///   - version: モデルのバージョン
    ///   - maxIterations: トレーニングの反復数
    /// - Returns: トレーニング結果 (成功時) または nil (失敗時)
    func train(author: String, shortDescription: String, version: String, maxIterations: Int) async -> TrainingResultType?
}

public extension ScreeningTrainerProtocol {
    /// バージョン管理された実行出力ディレクトリを設定し、URLを返す
    /// 形式: OutputModels/[version]/[Prefix]_[version]_Result_[index]
    ///
    /// - Parameters:
    ///   - version: トレーニングバージョン
    ///   - fileManager: ファイル操作に使用する FileManager
    ///   - trainerFilePath: #filePath プロジェクトルート特定用
    /// - Throws: ディレクトリ作成失敗時
    /// - Returns: 実行出力ディレクトリURL
    func setupVersionedRunOutputDirectory(
        version: String,
        fileManager: FileManager = .default,
        trainerFilePath: String
    ) throws -> URL {
        var projectRootURL = URL(fileURLWithPath: trainerFilePath)
        // #filePath は通常、呼び出し元のファイルパスを指すため、
        // プロジェクトルートに到達するために適切な回数だけ lastPathComponent を削除する
        // この回数はプロジェクト構造に依存する可能性があるため、通常は2回または3回
        projectRootURL.deleteLastPathComponent() // Trainer.swift -> Sources dir
        projectRootURL.deleteLastPathComponent() // Sources dir -> Module dir
        // OvRClassificationTrainer の場合、さらに1つ上
        if trainerFilePath.contains("OvRClassification") || trainerFilePath
            .contains("MultiLabelClassification") || trainerFilePath
            .contains("MultiClassClassification") || trainerFilePath.contains("BinaryClassification")
        {
            projectRootURL.deleteLastPathComponent() // Module dir -> Project Root
        }

        // ベース出力ディレクトリ (例: OvRClassification/OutputModels)
        let baseOutputDirURL = projectRootURL.appendingPathComponent(customOutputDirPath)

        // バージョン別ディレクトリ (例: OvRClassification/OutputModels/v1)
        let versionedOutputDirURL = baseOutputDirURL.appendingPathComponent(version)
        try fileManager.createDirectory(at: versionedOutputDirURL, withIntermediateDirectories: true, attributes: nil)
        print("📂 バージョン別出力ディレクトリ: \(versionedOutputDirURL.path)")

        // バージョン別ディレクトリ内の既存の実行をリスト
        let existingRuns = (try? fileManager.contentsOfDirectory(
            at: versionedOutputDirURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        )) ?? []

        // 実行名のプレフィックス (例: "OvR_v1_Result_")
        let runNamePrefixWithVersion = "\(outputRunNamePrefix)_\(version)_Result_"

        // 次の実行インデックスを計算
        let nextIndex = (existingRuns.compactMap { url -> Int? in
            let runName = url.lastPathComponent
            if runName.hasPrefix(runNamePrefixWithVersion) {
                return Int(runName.replacingOccurrences(of: runNamePrefixWithVersion, with: ""))
            }
            return nil
        }.max() ?? 0) + 1

        // 最終的な実行出力ディレクトリURLを構築 (例: OvRClassification/OutputModels/v1/OvR_v1_Result_1)
        let finalOutputRunURL = versionedOutputDirURL.appendingPathComponent("\(runNamePrefixWithVersion)\(nextIndex)")

        try fileManager
            .createDirectory(at: finalOutputRunURL, withIntermediateDirectories: true, attributes: nil) // ここを true に変更
        print("💾 結果保存ディレクトリ: \(finalOutputRunURL.path)")

        return finalOutputRunURL
    }
}
