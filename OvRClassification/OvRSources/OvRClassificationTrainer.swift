import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import Combine
import CoreML
import CreateML
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
    let confusionMatrix: CICBinaryConfusionMatrix?
}

public class OvRClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvRTrainingResult

    // DI 用のプロパティ
    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?
    private let fileManager: CICFileManager

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
        outputDirectoryPathOverride: String? = nil,
        fileManager: CICFileManager = CICFileManager()
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.fileManager = fileManager
    }

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
            mainOutputRunURL = try fileManager.createOutputDirectory(
                modelName: modelName,
                version: version,
                classificationMethod: classificationMethod,
                moduleOutputPath: outputDirPath
            )
        } catch {
            print("🛑 エラー: 出力ディレクトリ設定失敗: \(error.localizedDescription)")
            return nil
        }

        let baseProjectURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let tempOvRBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName)
        defer {
            if FileManager.default.fileExists(atPath: tempOvRBaseURL.path) {
                do {
                    try FileManager.default.removeItem(at: tempOvRBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvRBaseURL.path) クリーンアップ完了")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvRBaseURL.path) クリーンアップ失敗: \(error.localizedDescription)")
                }
            }
        }

        if FileManager.default.fileExists(atPath: tempOvRBaseURL.path) {
            try? FileManager.default.removeItem(at: tempOvRBaseURL)
        }
        guard (try? FileManager.default.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 エラー: 一時ディレクトリ \(tempOvRBaseURL.path) 作成失敗。処理中止。")
            return nil
        }

        let ovrResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)

        print("🚀 OvRトレーニング開始 (バージョン: \(version))...")

        let allLabelSourceDirectories: [URL]
        do {
            allLabelSourceDirectories = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
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
        let commonDataAugmentationDesc = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)
        var featureExtractorDesc: String = if let revision = scenePrintRevision {
            "\(featureExtractorDescription)(revision: \(revision))"
        } else {
            featureExtractorDescription
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
                modelParameters: modelParameters
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
        print("\n📊 トレーニング結果サマリー")
        for result in allPairTrainingResults {
            print(String(
                format: "  %@: 訓練正解率 %.1f%%, 検証正解率 %.1f%%, 再現率 %.1f%%, 適合率 %.1f%%",
                result.positiveClassName,
                result.trainingAccuracyRate,
                result.validationAccuracyRate,
                result.confusionMatrix?.recall ?? 0.0 * 100,
                result.confusionMatrix?.precision ?? 0.0 * 100
            ))
        }

        let trainingDataPaths = allPairTrainingResults.map(\.trainingDataPath).joined(separator: "; ")

        let finalRunOutputPath = mainOutputRunURL.path

        print("🎉 OvRトレーニング全体完了")
        print("結果出力先: \(finalRunOutputPath)")

        let metadata = CICTrainingMetadata(
            modelName: modelName,
            trainingDurationInSeconds: allPairTrainingResults.map(\.trainingTime).reduce(0.0, +),
            trainedModelFilePath: finalRunOutputPath,
            sourceTrainingDataDirectoryPath: trainingDataPaths,
            detectedClassLabelsList: allLabelSourceDirectories.map(\.lastPathComponent),
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: commonDataAugmentationDesc,
            featureExtractorDescription: featureExtractorDesc
        )

        let individualModelReports = allPairTrainingResults.map { result in
            CICIndividualModelReport(
                modelName: result.modelName,
                positiveClassName: result.positiveClassName,
                trainingAccuracyRate: result.trainingAccuracyRate,
                validationAccuracyPercentage: result.validationAccuracyRate,
                confusionMatrix: result.confusionMatrix
            )
        }

        let trainingResult = OvRTrainingResult(
            metadata: metadata,
            trainingMetrics: (
                accuracy: 1.0 - allPairTrainingResults.map(\.trainingErrorRate)
                    .reduce(0.0, +) / Double(allPairTrainingResults.count),
                errorRate: allPairTrainingResults.map(\.trainingErrorRate)
                    .reduce(0.0, +) / Double(allPairTrainingResults.count)
            ),
            validationMetrics: (
                accuracy: 1.0 - allPairTrainingResults.map(\.validationErrorRate)
                    .reduce(0.0, +) / Double(allPairTrainingResults.count),
                errorRate: allPairTrainingResults.map(\.validationErrorRate)
                    .reduce(0.0, +) / Double(allPairTrainingResults.count)
            ),
            confusionMatrix: nil,
            individualModelReports: individualModelReports
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
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> OvRPairTrainingResult? {
        let positiveClassNameForModel = oneLabelSourceDirURL.lastPathComponent
        let modelFileNameBase = "\(modelName)_\(classificationMethod)_\(positiveClassNameForModel)_\(version)"
        let tempOvRPairRootURL = tempOvRBaseURL.appendingPathComponent(modelFileNameBase)

        let tempPositiveDataDirForML = tempOvRPairRootURL.appendingPathComponent(positiveClassNameForModel)
        let tempRestDataDirForML = tempOvRPairRootURL.appendingPathComponent("Rest")

        if FileManager.default.fileExists(atPath: tempOvRPairRootURL.path) {
            try? FileManager.default.removeItem(at: tempOvRPairRootURL)
        }
        do {
            try FileManager.default.createDirectory(at: tempPositiveDataDirForML, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: tempRestDataDirForML, withIntermediateDirectories: true)
        } catch {
            print("🛑 エラー: OvRペア [\(positiveClassNameForModel)] vs Rest 一時学習ディレクトリ作成失敗: \(error.localizedDescription)")
            return nil
        }

        var positiveSamplesCount = 0
        if let positiveSourceFiles = try? fileManager.getFilesInDirectory(oneLabelSourceDirURL) {
            for fileURL in positiveSourceFiles {
                try? FileManager.default.copyItem(
                    at: fileURL,
                    to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            positiveSamplesCount = (try? fileManager.getFilesInDirectory(tempPositiveDataDirForML).count) ?? 0
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
            guard let filesInOtherDir = try? fileManager.getFilesInDirectory(otherDirURL),
                  !filesInOtherDir.isEmpty
            else {
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
                    try FileManager.default.copyItem(
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

        do {
            let trainingStartTime = Date()
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracyPercent = (1.0 - trainingMetrics.classificationError) * 100.0
            let validationAccuracyPercent = (1.0 - validationMetrics.classificationError) * 100.0

            // 混同行列の計算をCICBinaryConfusionMatrixに委任
            let confusionMatrix = CICBinaryConfusionMatrix(
                dataTable: validationMetrics.confusion,
                predictedColumn: "Predicted",
                actualColumn: "True Label",
                positiveClass: positiveClassNameForModel
            )

            if let confusionMatrix {
                // 混同行列の表示
                print(confusionMatrix.getMatrixGraph())
            } else {
                print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
            }

            let trainingDataPath = tempOvRPairRootURL.path

            // データ拡張の説明
            let dataAugmentationDesc = if !modelParameters.augmentationOptions.isEmpty {
                String(describing: modelParameters.augmentationOptions)
            } else {
                "なし"
            }

            // 特徴抽出器の説明
            let featureExtractorDesc = String(describing: modelParameters.featureExtractor)

            // モデルのメタデータを作成
            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: """
                クラス: \(positiveClassNameForModel), Rest
                訓練正解率: \(String(format: "%.1f%%", trainingAccuracyPercent))
                検証正解率: \(String(format: "%.1f%%", validationAccuracyPercent))
                \(confusionMatrix.map { matrix in
                    """
                    再現率: \(String(format: "%.1f%%", matrix.recall * 100.0)), \
                    適合率: \(String(format: "%.1f%%", matrix.precision * 100.0)), \
                    F1スコア: \(String(format: "%.1f%%", matrix.f1Score * 100.0))
                    """
                } ?? "")
                データ拡張: \(dataAugmentationDesc)
                特徴抽出器: \(featureExtractorDesc)
                """,
                version: version
            )

            let modelFileName = "\(modelFileNameBase).mlmodel"
            let modelFilePath = mainRunURL.appendingPathComponent(modelFileName).path

            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)

            return OvRPairTrainingResult(
                modelPath: modelFilePath,
                modelName: modelFileNameBase,
                positiveClassName: positiveClassNameForModel,
                trainingAccuracyRate: trainingAccuracyPercent,
                validationAccuracyRate: validationAccuracyPercent,
                trainingErrorRate: trainingMetrics.classificationError,
                validationErrorRate: validationMetrics.classificationError,
                trainingTime: trainingDurationSeconds,
                trainingDataPath: trainingDataPath,
                confusionMatrix: confusionMatrix
            )

        } catch let createMLError as CreateML.MLCreateError {
            print(
                "🛑 エラー: OvRペア [\(positiveClassNameForModel)] vs Rest トレーニング/保存失敗 (CreateML): \(createMLError.localizedDescription)"
            )
            return nil
        } catch {
            print(
                "🛑 エラー: OvRペア [\(positiveClassNameForModel)] vs Rest トレーニング/保存中に予期しないエラー: \(error.localizedDescription)"
            )
            return nil
        }
    }
}
