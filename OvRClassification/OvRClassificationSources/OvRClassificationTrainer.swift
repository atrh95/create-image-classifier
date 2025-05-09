import Foundation
import CreateML
import TabularData
import SCSInterface
import Combine

public class OvRClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvRBatchResult

    public var modelName: String {
        return "OvR_BatchCoordinator"
    }

    public var customOutputDirPath: String {
        return "OvRClassification/OutputModels"
    }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init() {
    }

    static let fileManager = FileManager.default
    static let tempBaseDirName = "TempOvRTrainingData"

    // Helper function to convert snake_case to UpperCamelCase
    private func toUpperCamelCase(fromSnakeCase string: String) -> String {
        return string.split(separator: "_")
                     .map { $0.capitalized }
                     .joined()
    }

    private enum TrainerError: Error {
        case directoryCreationFailed(path: String, underlyingError: Error)
        case resourceListingFailed(path: String, underlyingError: Error)
        case noPrimaryLabelsFound(path: String)
    }

    private func setupOutputDirectories(version: String, baseProjectURL: URL) throws -> (mainRunURL: URL, tempOvRBaseURL: URL) {
        let mainOutputRunURL = baseProjectURL.appendingPathComponent(customOutputDirPath).appendingPathComponent("OvR_Run_\(version)")
        let tempOvRBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName)

        do {
            try Self.fileManager.createDirectory(at: mainOutputRunURL, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print("❌ メイン出力実行ディレクトリの作成エラー \(mainOutputRunURL.path): \(error.localizedDescription)")
            throw TrainerError.directoryCreationFailed(path: mainOutputRunURL.path, underlyingError: error)
        }
        
        if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
            do {
                try Self.fileManager.removeItem(at: tempOvRBaseURL)
            } catch {
                print("⚠️ 既存の一時ベースディレクトリの削除失敗 \(tempOvRBaseURL.path): \(error.localizedDescription)")
                // 続行。createDirectoryが成功するか、より明確に失敗する可能性があるため
            }
        }
        do {
            try Self.fileManager.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print("❌ 一時ベースディレクトリの作成エラー \(tempOvRBaseURL.path): \(error.localizedDescription)")
            throw TrainerError.directoryCreationFailed(path: tempOvRBaseURL.path, underlyingError: error)
        }
        return (mainOutputRunURL, tempOvRBaseURL)
    }

    public func train(author: String, shortDescription: String, version: String) async -> OvRBatchResult? {
        let baseProjectURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        
        let mainOutputRunURL: URL
        let tempOvRBaseURL: URL

        do {
            (mainOutputRunURL, tempOvRBaseURL) = try setupOutputDirectories(version: version, baseProjectURL: baseProjectURL)
        } catch {
            // エラーはsetupOutputDirectories内で出力済み
            return nil
        }
        
        let ovrResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)

        print("🚀 OvR分類トレーニングバッチを開始します...")
        print("  バッチコーディネーター: \(self.modelName)")
        print("  今回の実行バージョン: \(version)")
        print("  作成者: \(author)")
        print("  バッチ説明: \(shortDescription)")
        print("  リソースパス: \(ovrResourcesURL.path)")
        print("  今回の実行のメイン出力パス: \(mainOutputRunURL.path)")
        print("  一時データパス: \(tempOvRBaseURL.path)")

        let allLabelSourceDirectories: [URL]
        do {
            allLabelSourceDirectories = try Self.fileManager.contentsOfDirectory(
                at: ovrResourcesURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ).filter { url in
                var isDirectory: ObjCBool = false
                Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
                return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
            }
        } catch {
             print("❌ ラベルディレクトリの一覧取得エラー \(ovrResourcesURL.path): \(error.localizedDescription)")
            // ここではまだcleanupTemporaryDataを呼び出す必要はない。
            // この関数が失敗した場合、またはsetupOutputDirectoriesが失敗した場合に最後にクリーンアップされる。
            return nil
        }
        

        let primaryLabelSourceDirs = allLabelSourceDirectories.filter { $0.lastPathComponent.lowercased() != "rest" }

        if primaryLabelSourceDirs.isEmpty {
            print("❌ エラー: プライマリラベルディレクトリが見つかりません \(ovrResourcesURL.path)。各クラスのサブディレクトリが必要です。")
            print("         ('Rest'という名前のディレクトリ（大文字・小文字を区別しない）は、'rest'グループを形成するために他のラベルと同様に扱われます。)")
            OvRClassificationTrainer.cleanupTemporaryData(at: tempOvRBaseURL)
            return nil
        }

        print("\(primaryLabelSourceDirs.count)件のプライマリラベルを処理します: \(primaryLabelSourceDirs.map {$0.lastPathComponent}.joined(separator: ", "))")

        var individualTrainingResults: [OvRTrainingResult] = []

        for oneLabelSourceDirURL in primaryLabelSourceDirs {
            let result = await trainSingleOvRPair(
                oneLabelSourceDirURL: oneLabelSourceDirURL,
                ovrResourcesURL: ovrResourcesURL,
                mainRunURL: mainOutputRunURL,
                tempOvRBaseURL: tempOvRBaseURL,
                author: author,
                shortDescription: shortDescription,
                version: version
            )
            if let validResult = result {
                individualTrainingResults.append(validResult)
            }
        }
        
        OvRClassificationTrainer.cleanupTemporaryData(at: tempOvRBaseURL)
        print("\n🏁 OvR分類トレーニングバッチが完了しました。")
        
        if individualTrainingResults.isEmpty {
            print("  このバッチで正常にトレーニングされたOvRモデルはありませんでした。")
            return nil
        }
        
        print("  正常にトレーニングされた個別OvRモデルの総数: \(individualTrainingResults.count)")
        print("  このバッチのすべての出力は次の場所にあります: \(mainOutputRunURL.path)")
        
        let batchResult = OvRBatchResult(
            batchVersion: version,
            individualResults: individualTrainingResults,
            mainOutputDirectoryPath: mainOutputRunURL.path
        )
        
        batchResult.saveLog(trainer: self, modelAuthor: author, modelDescription: shortDescription, modelVersion: version)

        return batchResult
    }

    private func trainSingleOvRPair(
        oneLabelSourceDirURL: URL,
        ovrResourcesURL: URL,
        mainRunURL: URL,
        tempOvRBaseURL: URL,
        author: String,
        shortDescription: String,
        version: String
    ) async -> OvRTrainingResult? {
        let originalOneLabelName = oneLabelSourceDirURL.lastPathComponent
        let upperCamelCaseOneLabelName = toUpperCamelCase(fromSnakeCase: originalOneLabelName)
        
        print("\n--- ラベルのOvR処理を開始: \(originalOneLabelName) (\(upperCamelCaseOneLabelName)として) ---")

        let ovrPairOutputDir = mainRunURL.appendingPathComponent("\(upperCamelCaseOneLabelName)_vs_Rest")
        do {
            try Self.fileManager.createDirectory(at: ovrPairOutputDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print("  ❌ OvRペアの出力ディレクトリ作成エラー \(upperCamelCaseOneLabelName): \(ovrPairOutputDir.path) - \(error.localizedDescription)")
            return nil
        }
        
        // 1. OvRペアごとの一時的な訓練ルートディレクトリを作成
        let tempOvRPairRootName = "\(upperCamelCaseOneLabelName)_vs_Rest_TrainingData"
        let tempOvRPairRootURL = tempOvRBaseURL.appendingPathComponent(tempOvRPairRootName)
        
        // 2. その下に "PositiveLabel" (実際のラベル名) と "Rest" のサブディレクトリを作成
        let tempPositiveDataDirForML = tempOvRPairRootURL.appendingPathComponent(upperCamelCaseOneLabelName)
        let tempRestDataDirForML = tempOvRPairRootURL.appendingPathComponent("Rest")

        do {
            // Ensure the root for this pair is clean or created
            if Self.fileManager.fileExists(atPath: tempOvRPairRootURL.path) {
                try Self.fileManager.removeItem(at: tempOvRPairRootURL)
            }
            try Self.fileManager.createDirectory(at: tempPositiveDataDirForML, withIntermediateDirectories: true, attributes: nil)
            try Self.fileManager.createDirectory(at: tempRestDataDirForML, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print("  ❌ \(originalOneLabelName)の一時データディレクトリ作成エラー: \(error.localizedDescription)")
            return nil
        }

        var positiveSamplesCount = 0
        var negativeSamplesCount = 0
        var restLabelNamesForThisPair: [String] = []

        // 3. ポジティブサンプルの準備 (tempPositiveDataDirForML へコピー)
        do {
            let positiveSourceFiles = try getFilesInDirectory(oneLabelSourceDirURL)
            for fileURL in positiveSourceFiles {
                try Self.fileManager.copyItem(at: fileURL, to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent))
            }
            positiveSamplesCount = positiveSourceFiles.count
            print("  ポジティブサンプル('\(upperCamelCaseOneLabelName)\')を準備中: \(tempPositiveDataDirForML.path) - 数: \(positiveSamplesCount)")
        } catch {
            print("  ❌ \(originalOneLabelName)のポジティブデータ準備エラー: \(error.localizedDescription)")
            // Attempt to clean up the pair-specific temp directory on error
            try? Self.fileManager.removeItem(at: tempOvRPairRootURL)
            return nil
        }

        // 4. ネガティブサンプルの準備 (tempRestDataDirForML へコピー)
        do {
            let globalRestDirURL = ovrResourcesURL.appendingPathComponent("rest")
            
            if Self.fileManager.fileExists(atPath: globalRestDirURL.path) {
                let negativeSourceFiles = try getFilesInDirectory(globalRestDirURL)
                for fileURL in negativeSourceFiles {
                     try Self.fileManager.copyItem(at: fileURL, to: tempRestDataDirForML.appendingPathComponent(fileURL.lastPathComponent))
                }
                negativeSamplesCount = negativeSourceFiles.count
                if negativeSamplesCount > 0 {
                    restLabelNamesForThisPair.append("Rest") // Or globalRestDirURL.lastPathComponent
                }
            } else {
                print("  ⚠️ グローバルRestディレクトリが見つかりません: \(globalRestDirURL.path)")
                // Consider if this is an error or acceptable. For now, count remains 0.
            }
            
            print("  ネガティブサンプル(\'Rest\')を準備中: \(tempRestDataDirForML.path) - 数: \(negativeSamplesCount)")
            if !restLabelNamesForThisPair.isEmpty {
                 print("    (ネガティブデータのソース: \(restLabelNamesForThisPair.sorted().joined(separator: ", ")))")
            }
        } catch {
             print("  ❌ \(originalOneLabelName)のネガティブデータ準備エラー: \(error.localizedDescription)")
            try? Self.fileManager.removeItem(at: tempOvRPairRootURL)
            return nil
        }

        if positiveSamplesCount == 0 || negativeSamplesCount == 0 {
            print("  ⚠️ データ準備後、ポジティブ(\(positiveSamplesCount))またはネガティブ(\(negativeSamplesCount))サンプルがないため、'\(originalOneLabelName)\'をスキップします。")
            try? Self.fileManager.removeItem(at: tempOvRPairRootURL)
            return nil
        }
        
        var singleOvRTrainingResult: OvRTrainingResult?

        // 5. MLトレーニングの実行
        do {
            print("  ⏳ CreateMLイメージ分類器ジョブを開始中 (\(upperCamelCaseOneLabelName) vs Rest)...")
            let trainingStartTime = Date()
            
            // データソースの指定を簡略化
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempOvRPairRootURL) // ★ 変更点
            
            var parameters = MLImageClassifier.ModelParameters()
            parameters.featureExtractor = .scenePrint(revision: 1)
            parameters.validation = .split(strategy: .automatic)
            parameters.maxIterations = 25
            parameters.augmentationOptions = [.crop, .rotation, .blur]

            let job = try MLImageClassifier.train(
                trainingData: trainingDataSource,
                parameters: parameters
            )
            
            let trainingTimeInSeconds = Date().timeIntervalSince(trainingStartTime)
            print("  ⏱️ CreateMLジョブ完了 (\(upperCamelCaseOneLabelName))。時間: \(String(format: "%.2f", trainingTimeInSeconds))秒")

            // Swift Concurrency compatible way to get the first (and expected only) value from the publisher
            var iterator = job.result.values.makeAsyncIterator()
            guard let classifier = try await iterator.next() else {
                // This case means the publisher completed without emitting a value,
                // which is unexpected if no error was thrown by iterator.next().
                struct TrainingJobDidNotYieldClassifierError: Error, LocalizedError {
                    let modelName: String
                    var errorDescription: String? {
                        "CreateML training job for '\(modelName)' completed without producing a classifier model or an explicit error."
                    }
                }
                print("  ⚠️ (\(upperCamelCaseOneLabelName)) Training job completed without a classifier result.")
                throw TrainingJobDidNotYieldClassifierError(modelName: upperCamelCaseOneLabelName)
            }

            let modelFileName = "\(upperCamelCaseOneLabelName)_OvR_\(version).mlmodel"
            let modelOutputPath = ovrPairOutputDir.appendingPathComponent(modelFileName).path
            let reportFileName = "\(upperCamelCaseOneLabelName)_OvR_\(version)_Report.md"
            let reportPath = ovrPairOutputDir.appendingPathComponent(reportFileName).path
            
            let metadata = MLModelMetadata(
                author: author,
                shortDescription: "\(shortDescription) — Binary classification for '\(upperCamelCaseOneLabelName)\' vs Rest.",
                version: version
            )
            
            try classifier.write(to: URL(fileURLWithPath: modelOutputPath), metadata: metadata)
            print("  ✅ (\(upperCamelCaseOneLabelName)) モデルを保存しました: \(modelOutputPath)")

            singleOvRTrainingResult = OvRTrainingResult(
                modelName: modelFileName,
                modelOutputPath: modelOutputPath,
                reportPath: reportPath,
                oneLabelName: upperCamelCaseOneLabelName,
                restLabelNames: restLabelNamesForThisPair.sorted(),
                positiveSamplesCount: positiveSamplesCount,
                negativeSamplesCount: negativeSamplesCount,
                trainingAccuracy: 0,
                validationAccuracy: 0,
                trainingError: 0,
                validationError: 0,
                trainingDuration: trainingTimeInSeconds,
                trainingDataPath: tempOvRPairRootURL.path
            )
            
            singleOvRTrainingResult?.saveLog(trainer: self, modelAuthor: author, modelDescription: shortDescription, modelVersion: version)

        } catch {
            print("  ❌ (\(upperCamelCaseOneLabelName)) 不明なトレーニングエラー: \(error.localizedDescription)")
            return nil
        }
        
        print("  --- ラベルのOvR処理完了: \(originalOneLabelName) ---")
        return singleOvRTrainingResult
    }

    private func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        return try Self.fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            // Corrected the line below: isDirectory: &isDirectory instead of nil
            Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return !isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }

    static func cleanupTemporaryData(at tempBaseDir: URL) {
        do {
            print("\n🧹 一時トレーニングデータをクリーンアップ中: \(tempBaseDir.path)")
            try fileManager.removeItem(at: tempBaseDir)
            print("  ✅ 一時トレーニングデータを削除しました。")
        } catch {
            print("  ⚠️ 一時トレーニングデータのクリーンアップ失敗: \(error.localizedDescription)")
        }
    }
}
