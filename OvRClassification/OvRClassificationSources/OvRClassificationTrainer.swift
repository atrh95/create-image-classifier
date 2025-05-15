import Combine
import CreateML
import CSInterface
import Foundation
import TabularData
import CoreML

private struct OvRPairTrainingResult {
    let modelPath: String
    let modelName: String
    let positiveClassName: String
    let trainingAccuracyRate: Double
    let validationAccuracyRate: Double
    let trainingErrorRate: Double
    let validationErrorRate: Double
    let trainingTime: TimeInterval
    let trainingDataPath: String
    let recallRate: Double
    let precisionRate: Double
    let individualModelDescription: String
}

public class OvRClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvRTrainingResult

    public var customOutputDirPath: String {
        "OvRClassification/OutputModels"
    }

    public var outputRunNamePrefix: String { "OvR" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    static let fileManager = FileManager.default
    static let tempBaseDirName = "TempOvRTrainingData"

    public func train(
        author: String,
        version: String,
        maxIterations: Int
    ) async -> OvRTrainingResult? {
        let mainOutputRunURL: URL
        do {
            mainOutputRunURL = try setupVersionedRunOutputDirectory(
                version: version,
                trainerFilePath: #filePath
            )
        } catch {
            print("🛑 出力ディレクトリの設定に失敗しました: \(error.localizedDescription)")
            return nil
        }

        let baseProjectURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let tempOvRBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName)
        defer {
            if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
                do {
                    try Self.fileManager.removeItem(at: tempOvRBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvRBaseURL.path) をクリーンアップしました。")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvRBaseURL.path) のクリーンアップに失敗しました: \(error.localizedDescription)")
                }
            }
        }

        if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 一時ディレクトリ \(tempOvRBaseURL.path) の作成に失敗しました。処理を中止します。")
            return nil
        }

        let ovrResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)

        print("🚀 OvRトレーニングを開始します: バージョン \(version)")

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
            print("🛑 リソースディレクトリ内のラベルディレクトリの取得に失敗しました: \(error.localizedDescription)")
            return nil
        }

        let primaryLabelSourceDirs = allLabelSourceDirectories.filter { $0.lastPathComponent.lowercased() != "safe" }

        if primaryLabelSourceDirs.isEmpty {
            print("🛑 プライマリトレーニングターゲットとなるディレクトリが見つかりません ('safe' ディレクトリを除く)。処理を中止します。")
            return nil
        }

        print("  処理対象となる主要ラベル数 (safeを除く): \(primaryLabelSourceDirs.count)")

        var allPairTrainingResults: [OvRPairTrainingResult] = []

        for (index, dir) in primaryLabelSourceDirs.enumerated() {
            if let result = await trainSingleOvRPair(
                oneLabelSourceDirURL: dir,
                allLabelSourceDirs: allLabelSourceDirectories,
                mainRunURL: mainOutputRunURL,
                tempOvRBaseURL: tempOvRBaseURL,
                author: author,
                version: version,
                pairIndex: index,
                maxIterations: maxIterations
            ) {
                allPairTrainingResults.append(result)
            }
        }

        guard !allPairTrainingResults.isEmpty else {
            print("🛑 有効なOvRペアトレーニングが一つも完了しませんでした。処理を中止します。")
            return nil
        }

        let individualReports: [IndividualModelReport] = allPairTrainingResults.map { result in
            IndividualModelReport(
                modelName: result.modelName,
                positiveClassName: result.positiveClassName,
                trainingAccuracyRate: result.trainingAccuracyRate,
                validationAccuracyRate: result.validationAccuracyRate,
                recallRate: result.recallRate,
                precisionRate: result.precisionRate,
                modelDescription: result.individualModelDescription
            )
        }
        
        let trainingDataPaths = allPairTrainingResults.map(\.trainingDataPath).joined(separator: "; ")

        let finalRunOutputPath = mainOutputRunURL.path

        let trainingResult = OvRTrainingResult(
            modelName: outputRunNamePrefix,
            modelOutputPath: finalRunOutputPath,
            trainingDataPaths: trainingDataPaths,
            maxIterations: maxIterations,
            individualReports: individualReports
        )

        return trainingResult
    }

    private func trainSingleOvRPair(
        oneLabelSourceDirURL: URL,
        allLabelSourceDirs: [URL],
        mainRunURL: URL,
        tempOvRBaseURL: URL,
        author: String,
        version: String,
        pairIndex: Int,
        maxIterations: Int
    ) async -> OvRPairTrainingResult? {
        let originalOneLabelName = oneLabelSourceDirURL.lastPathComponent
        let positiveClassNameForModel = originalOneLabelName.components(separatedBy: CharacterSet(charactersIn: "_-"))
                                             .map { $0.capitalized }
                                             .joined()

        let tempOvRPairRootName = "OvR_\(positiveClassNameForModel)_vs_Rest_TempData_v\(version)_idx\(pairIndex)"
        let tempOvRPairRootURL = tempOvRBaseURL.appendingPathComponent(tempOvRPairRootName)

        let tempPositiveDataDirForML = tempOvRPairRootURL.appendingPathComponent(positiveClassNameForModel)
        let tempRestDataDirForML = tempOvRPairRootURL.appendingPathComponent("Rest")

        if Self.fileManager.fileExists(atPath: tempOvRPairRootURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRPairRootURL)
        }
        do {
            try Self.fileManager.createDirectory(at: tempPositiveDataDirForML, withIntermediateDirectories: true)
            try Self.fileManager.createDirectory(at: tempRestDataDirForML, withIntermediateDirectories: true)
        } catch {
            print("🛑 OvRペア [\(positiveClassNameForModel)] の一時学習ディレクトリ作成に失敗: \(error.localizedDescription)")
            return nil
        }
        
        // Copy positive samples
        if let positiveSourceFiles = try? getFilesInDirectory(oneLabelSourceDirURL) {
            for fileURL in positiveSourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
        }
        // Count positive samples from the destination directory
        guard let positiveSamplesCount = try? getFilesInDirectory(tempPositiveDataDirForML).count, positiveSamplesCount > 0 else {
            print("⚠️ OvRペア [\(positiveClassNameForModel)]: ポジティブサンプルが見つからないか空です。学習をスキップ。 Path: \(tempPositiveDataDirForML.path)")
            return nil
        }

        // Start: Logic for collecting balanced "Rest" samples (from user-provided older code)
        let otherDirsForNegativeSampling = allLabelSourceDirs.filter { dirURL in
            // Ensure we are not comparing standardizedFileURLs if one of them might not be standardized yet
            // Direct path comparison should be fine if oneLabelSourceDirURL is from allLabelSourceDirs
            return dirURL.path != oneLabelSourceDirURL.path
        }

        if otherDirsForNegativeSampling.isEmpty {
            print("ℹ️ OvRペア [\(positiveClassNameForModel)]: ネガティブサンプリング対象の他のディレクトリがありません。このペアの学習はスキップされます。")
            return nil
        }

        let numFilesToCollectPerOtherDir =
            Int(ceil(Double(positiveSamplesCount) / Double(otherDirsForNegativeSampling.count)))

        var collectedNegativeFilesCount = 0
        for otherDirURL in otherDirsForNegativeSampling {
            guard let filesInOtherDir = try? getFilesInDirectory(otherDirURL), !filesInOtherDir.isEmpty else {
                print("ℹ️ OvRペア [\(positiveClassNameForModel)]: ディレクトリ \(otherDirURL.lastPathComponent) は空かアクセス不能なため、ネガティブサンプル収集からスキップします。")
                continue
            }

            let filesToCopy = filesInOtherDir.shuffled().prefix(numFilesToCollectPerOtherDir)
            for fileURL in filesToCopy {
                let sourceDirNamePrefix = otherDirURL.lastPathComponent
                // Sanitize names as in the provided older code
                let sanitizedSourceDirNamePrefix = sourceDirNamePrefix.replacingOccurrences(
                    of: "[^a-zA-Z0-9_.-]",
                    with: "_",
                    options: .regularExpression
                )
                let sanitizedOriginalFileName = fileURL.lastPathComponent.replacingOccurrences(
                    of: "[^a-zA-Z0-9_.-]",
                    with: "_",
                    options: .regularExpression
                )
                let newFileName = "\(sanitizedSourceDirNamePrefix)_\(sanitizedOriginalFileName)"

                do {
                    try Self.fileManager.copyItem(
                        at: fileURL,
                        to: tempRestDataDirForML.appendingPathComponent(newFileName)
                    )
                    collectedNegativeFilesCount += 1
                } catch {
                    print("⚠️ OvRペア [\(positiveClassNameForModel)]: ファイルコピーに失敗: \(fileURL.path) から \(tempRestDataDirForML.appendingPathComponent(newFileName).path) へ。エラー: \(error.localizedDescription)")
                }
            }
        }
        // End: Logic for collecting balanced "Rest" samples

        // Ensure collectedNegativeFilesCount is the actual count from the directory, not just the sum of successful copies
        let actualRestSamplesCount = (try? getFilesInDirectory(tempRestDataDirForML).count) ?? 0

        if actualRestSamplesCount == 0 {
             print("🛑 OvRペア [\(positiveClassNameForModel)]: ネガティブサンプルを1つも収集できませんでした。ポジティブサンプル数: \(positiveSamplesCount), 他カテゴリ数: \(otherDirsForNegativeSampling.count), 各カテゴリからの目標収集数: \(numFilesToCollectPerOtherDir)。学習をスキップします。")
            return nil
        }
        
        print("  🔄 OvRペア [\(positiveClassNameForModel) vs Rest] のトレーニングを開始 (サンプル数: Pos \(positiveSamplesCount), Rest \(actualRestSamplesCount))...")

        let trainingDataSource: MLImageClassifier.DataSource
        do {
            trainingDataSource = .labeledDirectories(at: tempOvRPairRootURL)
        } catch {
             print("    ❌ OvRペア [\(positiveClassNameForModel) vs Rest] のデータソース作成エラー: \(error.localizedDescription)")
            return nil
        }
        
        var parameters = MLImageClassifier.ModelParameters()
        parameters.featureExtractor = .scenePrint(revision: 1)
        parameters.maxIterations = maxIterations
        parameters.validation = .split(strategy: .automatic)

        let startTime = Date()
        do {
            let model = try MLImageClassifier(trainingData: trainingDataSource, parameters: parameters)
            let endTime = Date()
            let trainingDurationSeconds = endTime.timeIntervalSince(startTime)

            let trainingMetrics = model.trainingMetrics
            let validationMetrics = model.validationMetrics
            
            let pairTrainingAccuracyRate = (1.0 - trainingMetrics.classificationError)
            let pairValidationAccuracyRate = (1.0 - validationMetrics.classificationError)

            var pairRecallRate: Double = 0.0
            var pairPrecisionRate: Double = 0.0
            
            // Mirroring BinaryClassificationTrainer.swift logic for confusion matrix
            let confusionValue = validationMetrics.confusion

            if let confusionTable = confusionValue as? MLDataTable {
                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0
                
                let ovrPositiveLabel = positiveClassNameForModel
                let ovrNegativeLabel = "Rest"

                // Exact loop and parsing from BinaryClassificationTrainer
                for row in confusionTable.rows {
                    guard
                        let actualLabel = row["True Label"]?.stringValue, // Corrected key
                        let predictedLabel = row["Predicted"]?.stringValue, // Corrected key
                        let count = row["Count"]?.intValue // Corrected key
                    else {
                        print("    ⚠️ OvRペア [\(ovrPositiveLabel)]: 混同行列(MLDataTable)の行の解析に失敗。Row: \(row)")
                        continue
                    }

                    if actualLabel == ovrPositiveLabel, predictedLabel == ovrPositiveLabel {
                        truePositives += count
                    } else if actualLabel == ovrNegativeLabel, predictedLabel == ovrPositiveLabel {
                        falsePositives += count
                    } else if actualLabel == ovrPositiveLabel, predictedLabel == ovrNegativeLabel {
                        falseNegatives += count
                    }
                }

                if (truePositives + falseNegatives) > 0 {
                    pairRecallRate = Double(truePositives) / Double(truePositives + falseNegatives)
                }
                if (truePositives + falsePositives) > 0 {
                    pairPrecisionRate = Double(truePositives) / Double(truePositives + falsePositives)
                }

                if confusionTable.rows.isEmpty {
                    print("    ℹ️ OvRペア [\(ovrPositiveLabel)]: 混同行列(MLDataTable)が空でした。再現率/適合率は0です。")
                } else if truePositives == 0 && falsePositives == 0 && falseNegatives == 0 {
                    // Log if all TP, FP, FN are zero but table was not empty
                    print("    ℹ️ OvRペア [\(ovrPositiveLabel)]: 混同行列(MLDataTable)からTP,FP,FNが全て0。ラベル名('\(ovrPositiveLabel)','\(ovrNegativeLabel)')やデータを確認。再現率/適合率0。 Table: \(confusionTable.description)")
                }
                // Print calculated rates like in Binary Trainer (optional, but good for debug)
                print("    🔍 OvRペア [\(ovrPositiveLabel)] 検証データ 再現率: \(String(format: "%.2f", pairRecallRate * 100))%")
                print("    🎯 OvRペア [\(ovrPositiveLabel)] 検証データ 適合率: \(String(format: "%.2f", pairPrecisionRate * 100))%")

            } else {
                print("    ⚠️ OvRペア [\(positiveClassNameForModel)]: 混同行列が期待される MLDataTable 型ではありませんでした (型: \(type(of: confusionValue)))。再現率/適合率は0として扱います。")
            }

            let pairModelFileName = "OvR_\(positiveClassNameForModel)_vs_Rest_v\(version).mlmodel"
            let pairModelOutputURL = mainRunURL.appendingPathComponent(pairModelFileName)

            var individualModelDesc = String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                pairTrainingAccuracyRate * 100,
                pairValidationAccuracyRate * 100
            )
            // Now that we are calculating them (hopefully correctly), include them
            individualModelDesc += String(
                format: ", 再現率(%@): %.1f%%, 適合率(%@): %.1f%%",
                positiveClassNameForModel, pairRecallRate * 100,
                positiveClassNameForModel, pairPrecisionRate * 100
            )
            individualModelDesc += String(format: ". サンプル (陽性/Rest): %d/%d (自動分割)", positiveSamplesCount, actualRestSamplesCount)

            let metadata = MLModelMetadata(
                author: author,
                shortDescription: individualModelDesc,
                version: version
            )
            
            try model.write(to: pairModelOutputURL, metadata: metadata)
            print("    ✅ OvRペア [\(positiveClassNameForModel) vs Rest] トレーニング成功。モデル保存先: \(pairModelOutputURL.path) (時間: \(String(format: "%.2f", trainingDurationSeconds))秒)")
            print("      📈 検証正解率: \(String(format: "%.2f", pairValidationAccuracyRate * 100))%, 再現率: \(String(format: "%.2f", pairRecallRate*100))%, 適合率: \(String(format: "%.2f", pairPrecisionRate*100))%")

            return OvRPairTrainingResult(
                modelPath: pairModelOutputURL.path,
                modelName: pairModelFileName,
                positiveClassName: positiveClassNameForModel,
                trainingAccuracyRate: pairTrainingAccuracyRate,
                validationAccuracyRate: pairValidationAccuracyRate,
                trainingErrorRate: trainingMetrics.classificationError,
                validationErrorRate: validationMetrics.classificationError,
                trainingTime: trainingDurationSeconds,
                trainingDataPath: tempOvRPairRootURL.path,
                recallRate: pairRecallRate,
                precisionRate: pairPrecisionRate,
                individualModelDescription: individualModelDesc
            )
        } catch {
            print("    ❌ OvRペア [\(positiveClassNameForModel) vs Rest] のトレーニングまたは保存中にエラー: \(error.localizedDescription)")
            // Removed specific CreateMLError catch, now using generic error.
            // For more details, you might need to inspect the `error` object further, e.g., `error as NSError`
            return nil
        }
    }

    // Simplified getFilesInDirectory closer to original working version
    private func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try Self.fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey, .nameKey], // .nameKey can be useful for debugging
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            // Check if it's a directory first
            if Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return false // Exclude directories
            }
            // Ensure it's not a hidden file (redundant with .skipsHiddenFiles but safe)
            if url.lastPathComponent.hasPrefix(".") {
                return false
            }
            // Optionally, be more explicit about wanting regular files if symbolic links etc. are an issue
            // var isRegular: ObjCBool = false
            // if Self.fileManager.fileExists(atPath: url.path, isDirectory: &isRegular) { // This checks if it IS a directory
            //    // To check if it's a regular file, more specific attribute check might be needed if problems persist
            // }
            return true // If not a directory and not hidden, include it
        }
    }
}
