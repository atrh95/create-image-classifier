import Combine
import CoreML
import CreateML
import CSInterface
import Foundation
import TabularData

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

    // DI 用のプロパティ
    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "OvR" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init(
        resourcesDirectoryPathOverride: String? = nil,
        outputDirectoryPathOverride: String? = nil
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    static let fileManager = FileManager.default
    static let tempBaseDirName = "TempOvRTrainingData"

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvRTrainingResult? {
        let mainOutputRunURL: URL
        do {
            mainOutputRunURL = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("🛑 エラー: 出力ディレクトリ設定失敗: \(error.localizedDescription)")
            return nil
        }

        let baseProjectURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let tempOvRBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName)
        defer {
            if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
                do {
                    try Self.fileManager.removeItem(at: tempOvRBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvRBaseURL.path) クリーンアップ完了")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvRBaseURL.path) クリーンアップ失敗: \(error.localizedDescription)")
                }
            }
        }

        if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 エラー: 一時ディレクトリ \(tempOvRBaseURL.path) 作成失敗。処理中止。")
            return nil
        }

        let ovrResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)

        print("🚀 OvRトレーニング開始 (バージョン: \(version))...")

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
            print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
            return nil
        }

        let primaryLabelSourceDirs = allLabelSourceDirectories.filter { $0.lastPathComponent.lowercased() != "safe" }

        if primaryLabelSourceDirs.isEmpty {
            print("🛑 エラー: プライマリトレーニング対象ディレクトリが見つかりません ('safe'除く)。処理中止。")
            return nil
        }

        print("  処理対象主要ラベル数 ('safe'除く): \(primaryLabelSourceDirs.count)")

        // データ拡張と特徴抽出器の説明を生成 (モデル全体で共通、TrainingResult用)
        let commonDataAugmentationDesc: String
        if !modelParameters.augmentationOptions.isEmpty {
            commonDataAugmentationDesc = String(describing: modelParameters.augmentationOptions)
        } else {
            commonDataAugmentationDesc = "なし"
        }
        
        let featureExtractorString = String(describing: modelParameters.featureExtractor)
        var commonFeatureExtractorDesc: String
        if let revision = scenePrintRevision {
            commonFeatureExtractorDesc = "\(featureExtractorString)(revision: \(revision))"
        } else {
            commonFeatureExtractorDesc = featureExtractorString
        }

        var allPairTrainingResults: [OvRPairTrainingResult] = []

        for (index, dir) in primaryLabelSourceDirs.enumerated() {
            print(
                "🔄 OvRペア \(index + 1)/\(primaryLabelSourceDirs.count): [\(dir.lastPathComponent)] vs Rest トレーニング開始..."
            )
            if let result = await trainSingleOvRPair(
                oneLabelSourceDirURL: dir,
                allLabelSourceDirs: allLabelSourceDirectories,
                mainRunURL: mainOutputRunURL,
                tempOvRBaseURL: tempOvRBaseURL,
                modelName: modelName,
                author: author,
                version: version,
                pairIndex: index,
                modelParameters: modelParameters,
                scenePrintRevision: scenePrintRevision
            ) {
                allPairTrainingResults.append(result)
                print("  ✅ OvRペア [\(dir.lastPathComponent)] vs Rest トレーニング成功")
            } else {
                print("  ⚠️ OvRペア [\(dir.lastPathComponent)] vs Rest トレーニング失敗またはスキップ")
            }
        }

        guard !allPairTrainingResults.isEmpty else {
            print("🛑 エラー: 有効なOvRペアトレーニングが一つも完了しませんでした。処理中止。")
            return nil
        }

        // トレーニング完了後のパフォーマンス指標を表示
        print("\n📊 トレーニング結果サマリー:")
        for result in allPairTrainingResults {
            print(String(format: "  %@: 訓練正解率 %.1f%%, 検証正解率 %.1f%%, 再現率 %.1f%%, 適合率 %.1f%%",
                result.positiveClassName,
                result.trainingAccuracyRate,
                result.validationAccuracyRate,
                result.recallRate * 100,
                result.precisionRate * 100))
        }

        let individualReports: [IndividualModelReport] = allPairTrainingResults.map { result in
            let confusionMatrix = ConfusionMatrix(
                truePositive: 0,
                falsePositive: 0,
                falseNegative: 0,
                trueNegative: 0
            )
            let individualModelReport = IndividualModelReport(
                modelName: result.modelName,
                positiveClassName: result.positiveClassName,
                trainingAccuracyRate: result.trainingAccuracyRate,
                validationAccuracyPercentage: result.validationAccuracyRate,
                recallRate: result.recallRate,
                precisionRate: result.precisionRate,
                modelDescription: result.individualModelDescription,
                confusionMatrix: confusionMatrix
            )
            return individualModelReport
        }

        let trainingDataPaths = allPairTrainingResults.map(\.trainingDataPath).joined(separator: "; ")

        let finalRunOutputPath = mainOutputRunURL.path

        print("🎉 OvRトレーニング全体完了")
        print("結果出力先: \(finalRunOutputPath)")

        let trainingResult = OvRTrainingResult(
            modelOutputPath: finalRunOutputPath,
            trainingDataPaths: trainingDataPaths,
            maxIterations: modelParameters.maxIterations,
            individualReports: individualReports,
            dataAugmentationDescription: commonDataAugmentationDesc,
            baseFeatureExtractorDescription: featureExtractorString,
            scenePrintRevision: scenePrintRevision
        )

        return trainingResult
    }

    private func trainSingleOvRPair(
        oneLabelSourceDirURL: URL,
        allLabelSourceDirs: [URL],
        mainRunURL: URL,
        tempOvRBaseURL: URL,
        modelName: String,
        author: String,
        version: String,
        pairIndex: Int,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvRPairTrainingResult? {
        let originalOneLabelName = oneLabelSourceDirURL.lastPathComponent
        let positiveClassNameForModel = originalOneLabelName.components(separatedBy: CharacterSet(charactersIn: "_-"))
            .map(\.capitalized)
            .joined()
            .replacingOccurrences(of: "[^a-zA-Z0-9]", with: "", options: .regularExpression)

        let modelFileNameBase =
            "\(modelName)_\(classificationMethod)_\(positiveClassNameForModel)_\(version)"
        let tempOvRPairRootName = "\(modelFileNameBase)_TempData"
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
            print("🛑 エラー: OvRペア [\(positiveClassNameForModel)] 一時学習ディレクトリ作成失敗: \(error.localizedDescription)")
            return nil
        }

        var positiveSamplesCount = 0
        if let positiveSourceFiles = try? getFilesInDirectory(oneLabelSourceDirURL) {
            for fileURL in positiveSourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            positiveSamplesCount = (try? getFilesInDirectory(tempPositiveDataDirForML).count) ?? 0
        }

        guard positiveSamplesCount > 0 else {
            return nil
        }

        let otherDirsForNegativeSampling = allLabelSourceDirs.filter { $0.path != oneLabelSourceDirURL.path }

        if otherDirsForNegativeSampling.isEmpty {
            return nil
        }

        let numFilesToCollectPerOtherDir =
            Int(ceil(Double(positiveSamplesCount) / Double(otherDirsForNegativeSampling.count)))
        var totalNegativeSamplesCollected = 0

        for otherDirURL in otherDirsForNegativeSampling {
            guard let filesInOtherDir = try? getFilesInDirectory(otherDirURL), !filesInOtherDir.isEmpty else {
                continue
            }

            let filesToCopy = filesInOtherDir.shuffled().prefix(numFilesToCollectPerOtherDir)
            for fileURL in filesToCopy {
                let sourceDirNamePrefix = otherDirURL.lastPathComponent
                    .replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "_", options: .regularExpression)
                let originalFileName = fileURL.lastPathComponent
                    .replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "_", options: .regularExpression)
                let newFileName = "\(sourceDirNamePrefix)_\(originalFileName)"

                do {
                    try Self.fileManager.copyItem(
                        at: fileURL,
                        to: tempRestDataDirForML.appendingPathComponent(newFileName)
                    )
                    totalNegativeSamplesCollected += 1
                } catch {
                    continue
                }
            }
        }

        guard totalNegativeSamplesCollected > 0 else {
            return nil
        }

        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempOvRPairRootURL)
        let modelForPairName = "\(modelName)_\(classificationMethod)_\(positiveClassNameForModel)"

        do {
            let trainingStartTime = Date()
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracy = (1.0 - trainingMetrics.classificationError) * 100.0
            let validationAccuracy = (1.0 - validationMetrics.classificationError) * 100.0

            var recall = 0.0
            var precision = 0.0

            let confusionMatrix = validationMetrics.confusion
            var labelSet = Set<String>()
            for row in confusionMatrix.rows {
                if let actual = row["True Label"]?.stringValue {
                    labelSet.insert(actual)
                }
                if let predicted = row["Predicted"]?.stringValue {
                    labelSet.insert(predicted)
                }
            }
            let classLabelsFromConfusion = Array(labelSet).sorted()

            if classLabelsFromConfusion.count == 2 {
                let negativeLabel = classLabelsFromConfusion[0]
                let positiveLabel = classLabelsFromConfusion[1]

                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0

                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let cnt = row["Count"]?.intValue
                    else { continue }

                    if actual == positiveLabel, predicted == positiveLabel {
                        truePositives += cnt
                    } else if actual == negativeLabel, predicted == positiveLabel {
                        falsePositives += cnt
                    } else if actual == positiveLabel, predicted == negativeLabel {
                        falseNegatives += cnt
                    }
                }

                if (truePositives + falseNegatives) > 0 {
                    recall = Double(truePositives) / Double(truePositives + falseNegatives)
                }
                if (truePositives + falsePositives) > 0 {
                    precision = Double(truePositives) / Double(truePositives + falsePositives)
                }
            }

            let positiveCountForDesc = (try? getFilesInDirectory(tempPositiveDataDirForML).count) ?? 0
            let restCountForDesc = (try? getFilesInDirectory(tempRestDataDirForML).count) ?? 0

            var descriptionParts: [String] = []

            descriptionParts.append(String(
                format: "クラス構成 (陽性/他): %@ (%d枚) / Rest (%d枚)",
                positiveClassNameForModel, positiveCountForDesc, restCountForDesc
            ))

            descriptionParts.append("最大反復回数: \(modelParameters.maxIterations)回")

            descriptionParts.append(String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracy,
                validationAccuracy
            ))

            if classLabelsFromConfusion.count == 2 {
                let positiveLabelForDesc = classLabelsFromConfusion.first { $0 == positiveClassNameForModel } ?? classLabelsFromConfusion[1]
                descriptionParts.append(String(
                    format: "陽性クラス (%@): 再現率 %.1f%%, 適合率 %.1f%%",
                    positiveLabelForDesc,
                    max(0.0, recall * 100),
                    max(0.0, precision * 100)
                ))
            }

            let augmentationFinalDescription: String
            if !modelParameters.augmentationOptions.isEmpty {
                augmentationFinalDescription = String(describing: modelParameters.augmentationOptions)
                descriptionParts.append("データ拡張: \(augmentationFinalDescription)")
            } else {
                augmentationFinalDescription = "なし"
                descriptionParts.append("データ拡張: なし")
            }

            let featureExtractorStringForPair = String(describing: modelParameters.featureExtractor)
            var featureExtractorDescForPairMetadata: String
            if let revision = scenePrintRevision {
                featureExtractorDescForPairMetadata = "\(featureExtractorStringForPair)(revision: \(revision))"
                descriptionParts.append("特徴抽出器: \(featureExtractorDescForPairMetadata)")
            } else {
                featureExtractorDescForPairMetadata = featureExtractorStringForPair
                descriptionParts.append("特徴抽出器: \(featureExtractorDescForPairMetadata)")
            }

            let individualDesc = descriptionParts.joined(separator: "\n")

            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: individualDesc,
                version: version
            )

            let modelFileName = "\(modelFileNameBase).mlmodel"
            let modelFilePath = mainRunURL.appendingPathComponent(modelFileName).path

            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)

            return OvRPairTrainingResult(
                modelPath: modelFilePath,
                modelName: modelFileNameBase,
                positiveClassName: positiveClassNameForModel,
                trainingAccuracyRate: trainingAccuracy,
                validationAccuracyRate: validationAccuracy,
                trainingErrorRate: trainingMetrics.classificationError,
                validationErrorRate: validationMetrics.classificationError,
                trainingTime: trainingDurationSeconds,
                trainingDataPath: tempOvRPairRootURL.path,
                recallRate: recall,
                precisionRate: precision,
                individualModelDescription: individualDesc
            )

        } catch let createMLError as CreateML.MLCreateError {
            print("🛑 エラー: OvRペア [\(positiveClassNameForModel)] トレーニング/保存失敗 (CreateML): \(createMLError.localizedDescription)")
            return nil
        } catch {
            print("🛑 エラー: OvRペア [\(positiveClassNameForModel)] トレーニング/保存中に予期しないエラー: \(error.localizedDescription)")
            return nil
        }
    }

    // 指定されたディレクトリ内のファイル一覧を取得する
    private func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try Self.fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return !isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }
}
