import Combine
import CoreML
import CreateML
import CSInterface
import CSConfusionMatrix
import Foundation
import TabularData

// OvOペアのトレーニング結果を格納する
private struct OvOPairTrainingResult {
    let modelPath: String
    let modelName: String
    let class1Name: String // OvOペアのクラス1
    let class2Name: String // OvOペアのクラス2
    let trainingAccuracyRate: Double
    let validationAccuracyRate: Double
    let trainingErrorRate: Double
    let validationErrorRate: Double
    let trainingTime: TimeInterval
    let trainingDataPath: String // このペアのトレーニングに使用されたデータのパス
    let confusionMatrix: CSBinaryConfusionMatrix?
}

public class OvOClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvOTrainingResult

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

    public var classificationMethod: String { "OvO" }

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
    static let tempBaseDirName = "TempOvOTrainingData"

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvOTrainingResult? {
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

        let baseProjectURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let tempOvOBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName) // OvO用一時ディレクトリベースパス
        defer {
            if Self.fileManager.fileExists(atPath: tempOvOBaseURL.path) {
                do {
                    try Self.fileManager.removeItem(at: tempOvOBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvOBaseURL.path) クリーンアップ完了")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvOBaseURL.path) クリーンアップ失敗: \(error.localizedDescription)")
                }
            }
        }

        if Self.fileManager.fileExists(atPath: tempOvOBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvOBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvOBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 エラー: 一時ディレクトリ \(tempOvOBaseURL.path) 作成失敗。処理中止。")
            return nil
        }

        let ovoResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath) // Use the (potentially overridden) property

        print("🚀 OvOトレーニング開始 (バージョン: \(version))...")

        let allLabelSourceDirectories: [URL]
        do {
            allLabelSourceDirectories = try Self.fileManager.contentsOfDirectory(
                at: ovoResourcesURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ).filter { url in
                var isDirectory: ObjCBool = false
                Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
                return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".") // 隠しファイルを除外
            }
        } catch {
            print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
            return nil
        }

        // OvOでは最低2つのクラスが必要
        guard allLabelSourceDirectories.count >= 2 else {
            print("🛑 エラー: OvOトレーニングには最低2つのクラスラベルディレクトリが必要です。現在 \(allLabelSourceDirectories.count)個。処理中止。")
            return nil
        }

        print("  検出された総ラベル数: \(allLabelSourceDirectories.count)")

        // クラスペアを生成 (例: [A,B], [A,C], [B,C])
        var classPairs: [(URL, URL)] = []
        for i in 0 ..< allLabelSourceDirectories.count {
            for j in (i + 1) ..< allLabelSourceDirectories.count {
                classPairs.append((allLabelSourceDirectories[i], allLabelSourceDirectories[j]))
            }
        }

        if classPairs.isEmpty {
            print("🛑 エラー: 有効なクラスペアが生成できませんでした。処理中止。")
            return nil
        }

        print("  生成されたOvOペア数: \(classPairs.count)")

        // 各ペアモデル共通設定の記述を生成 (TrainingResult用)
        let commonDataAugmentationDesc = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorString = String(describing: modelParameters.featureExtractor)
        var commonFeatureExtractorDesc: String = if let revision = scenePrintRevision {
            "\(featureExtractorString)(revision: \(revision))"
        } else {
            featureExtractorString
        }

        var allPairTrainingResults: [OvOPairTrainingResult] = []
        var pairIndex = 0

        for pair in classPairs {
            let dir1 = pair.0
            let dir2 = pair.1
            print(
                "🔄 OvOペア \(pairIndex + 1)/\(classPairs.count): [\(dir1.lastPathComponent)] vs [\(dir2.lastPathComponent)] トレーニング開始..."
            )
            if let result = await trainSingleOvOPair(
                class1DirURL: dir1,
                class2DirURL: dir2,
                mainRunURL: mainOutputRunURL,
                tempOvOBaseURL: tempOvOBaseURL, // OvO用一時ベースURL
                modelName: modelName, // ベースモデル名
                author: author,
                version: version,
                pairIndex: pairIndex,
                modelParameters: modelParameters,
                scenePrintRevision: scenePrintRevision
            ) {
                allPairTrainingResults.append(result)
                print("  ✅ OvOペア [\(dir1.lastPathComponent)] vs [\(dir2.lastPathComponent)] トレーニング成功")
            } else {
                print("  ⚠️ OvOペア [\(dir1.lastPathComponent)] vs [\(dir2.lastPathComponent)] トレーニング失敗またはスキップ")
            }
            pairIndex += 1
        }

        guard !allPairTrainingResults.isEmpty else {
            print("🛑 エラー: 有効なOvOペアトレーニングが一つも完了しませんでした。処理中止。")
            return nil
        }

        // IndividualModelReportの作成
        let individualReports: [IndividualModelReport] = allPairTrainingResults.map { result in
            return IndividualModelReport(
                modelName: result.modelName,
                positiveClassName: "\(result.class1Name)_vs_\(result.class2Name)",
                trainingAccuracyRate: result.trainingAccuracyRate,
                validationAccuracyPercentage: result.validationAccuracyRate,
                confusionMatrix: result.confusionMatrix
            )
        }

        let trainingDataPaths = allPairTrainingResults.map(\.trainingDataPath).joined(separator: "; ")
        let finalRunOutputPath = mainOutputRunURL.path

        print("🎉 OvOトレーニング全体完了")
        print("結果出力先: \(finalRunOutputPath)")
        print("📁 デバッグ: 出力ディレクトリの内容:")
        do {
            let contents = try FileManager.default.contentsOfDirectory(at: mainOutputRunURL, includingPropertiesForKeys: nil)
            for url in contents {
                print("  - \(url.lastPathComponent)")
            }
        } catch {
            print("❌ デバッグ: ディレクトリ内容の取得に失敗: \(error.localizedDescription)")
        }

        let trainingResult = OvOTrainingResult(
            modelName: modelName,
            trainingDurationInSeconds: allPairTrainingResults.map(\.trainingTime).reduce(0.0, +),
            trainedModelFilePath: finalRunOutputPath,
            sourceTrainingDataDirectoryPath: trainingDataPaths,
            detectedClassLabelsList: allLabelSourceDirectories.map(\.lastPathComponent),
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: commonDataAugmentationDesc,
            baseFeatureExtractorDescription: featureExtractorString,
            scenePrintRevision: scenePrintRevision,
            individualReports: individualReports
        )

        return trainingResult
    }

    // 1つのOvOペアのモデルをトレーニングする関数
    private func trainSingleOvOPair(
        class1DirURL: URL,
        class2DirURL: URL,
        mainRunURL: URL,
        tempOvOBaseURL: URL,
        modelName: String,
        author: String,
        version: String,
        pairIndex: Int,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvOPairTrainingResult? {
        let class1NameOriginal = class1DirURL.lastPathComponent
        let class2NameOriginal = class2DirURL.lastPathComponent

        // モデル名やディレクトリ名に使用するクラス名 (英数字のみに整形)
        let modelClass1Name = class1NameOriginal.components(separatedBy: CharacterSet(charactersIn: "_-"))
            .map(\.capitalized)
            .joined()
            .replacingOccurrences(of: "[^a-zA-Z0-9]", with: "", options: .regularExpression)

        let modelClass2Name = class2NameOriginal.components(separatedBy: CharacterSet(charactersIn: "_-"))
            .map(\.capitalized)
            .joined()
            .replacingOccurrences(of: "[^a-zA-Z0-9]", with: "", options: .regularExpression)

        // モデルファイル名と一時ディレクトリ名を作成
        let modelFileNameBase =
            "\(modelName)_\(classificationMethod)_\(modelClass1Name)_vs_\(modelClass2Name)_\(version)"
        // Ensure unique temp dir per pair using pairIndex
        let tempOvOPairRootName = "\(modelFileNameBase)_TempData_idx\(pairIndex)"
        let tempOvOPairRootURL = tempOvOBaseURL.appendingPathComponent(tempOvOPairRootName)

        let tempClass1DataDirForML = tempOvOPairRootURL.appendingPathComponent(modelClass1Name)
        let tempClass2DataDirForML = tempOvOPairRootURL.appendingPathComponent(modelClass2Name)

        if Self.fileManager.fileExists(atPath: tempOvOPairRootURL.path) {
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
        }
        do {
            try Self.fileManager.createDirectory(at: tempClass1DataDirForML, withIntermediateDirectories: true)
            try Self.fileManager.createDirectory(at: tempClass2DataDirForML, withIntermediateDirectories: true)
        } catch {
            print(
                "🛑 エラー: OvOペア [\(modelClass1Name) vs \(modelClass2Name)] 一時学習ディレクトリ作成失敗: \(error.localizedDescription)"
            )
            return nil
        }

        var class1SamplesCount = 0
        if let class1SourceFiles = try? getFilesInDirectory(class1DirURL) {
            for fileURL in class1SourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempClass1DataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            class1SamplesCount = (try? getFilesInDirectory(tempClass1DataDirForML).count) ?? 0
        }
        guard class1SamplesCount > 0 else {
            print(
                "⚠️ OvOペア [\(modelClass1Name) vs \(modelClass2Name)]: クラス1 [\(modelClass1Name)] のサンプルなし。学習スキップ。Path: \(tempClass1DataDirForML.path)"
            )
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
            return nil
        }

        var class2SamplesCount = 0
        if let class2SourceFiles = try? getFilesInDirectory(class2DirURL) {
            for fileURL in class2SourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempClass2DataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
            class2SamplesCount = (try? getFilesInDirectory(tempClass2DataDirForML).count) ?? 0
        }
        guard class2SamplesCount > 0 else {
            print(
                "⚠️ OvOペア [\(modelClass1Name) vs \(modelClass2Name)]: クラス2 [\(modelClass2Name)] のサンプルなし。学習スキップ。Path: \(tempClass2DataDirForML.path)"
            )
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
            return nil
        }

        print(
            "  準備完了: [\(modelClass1Name)] (\(class1SamplesCount)枚) vs [\(modelClass2Name)] (\(class2SamplesCount)枚)"
        )

        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempOvOPairRootURL)
        let modelFilePath = mainRunURL.appendingPathComponent("\(modelFileNameBase).mlmodel").path

        do {
            let trainingStartTime = Date()

            print(
                "  ⏳ OvOペア [\(modelClass1Name) vs \(modelClass2Name)] モデルトレーニング実行中 (最大反復: \(modelParameters.maxIterations)回)... "
            )
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            print("  ✅ OvOペア [\(modelClass1Name) vs \(modelClass2Name)] モデルトレーニング完了")

            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracyPercent = (1.0 - trainingMetrics.classificationError) * 100.0
            let validationAccuracyPercent = (1.0 - validationMetrics.classificationError) * 100.0
            let trainingErrorRate = trainingMetrics.classificationError
            let validationErrorRate = validationMetrics.classificationError

            // トレーニング完了後のパフォーマンス指標を表示
            print("\n📊 トレーニング結果サマリー")
            print(String(
                format: "  訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercent,
                validationAccuracyPercent
            ))

            // 混同行列の計算をCSBinaryConfusionMatrixに委任
            let confusionMatrix = CSBinaryConfusionMatrix(
                dataTable: validationMetrics.confusion,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            )

            if let confusionMatrix {
                // 混同行列の表示
                print(confusionMatrix.getMatrixGraph())
            } else {
                print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
            }

            // モデルの説明文を構築
            var descriptionParts: [String] = []
            descriptionParts.append(String(
                format: "クラス構成 (%@/%@): %@ (%d枚) / %@ (%d枚)",
                modelClass1Name, modelClass2Name, modelClass1Name, class1SamplesCount, modelClass2Name,
                class2SamplesCount
            ))
            descriptionParts.append("最大反復回数: \(modelParameters.maxIterations)回")
            descriptionParts.append(String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercent,
                validationAccuracyPercent
            ))

            if let confusionMatrix {
                descriptionParts.append(String(
                    format: "クラス '%@': 再現率 %.1f%%, 適合率 %.1f%%",
                    modelClass1Name,
                    confusionMatrix.recall * 100.0,
                    confusionMatrix.precision * 100.0
                ))
                descriptionParts.append(String(
                    format: "クラス '%@': 再現率 %.1f%%, 適合率 %.1f%%",
                    modelClass2Name,
                    confusionMatrix.recall * 100.0,
                    confusionMatrix.precision * 100.0
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

            // モデルのメタデータを作成
            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: """
                クラス: \(modelClass1Name), \(modelClass2Name)
                訓練正解率: \(String(format: "%.1f%%", trainingAccuracyPercent))
                検証正解率: \(String(format: "%.1f%%", validationAccuracyPercent))
                \(
                    confusionMatrix != nil ?
                        "性能指標: [再現率: \(String(format: "%.1f%%", confusionMatrix!.recall * 100.0)), " +
                        "適合率: \(String(format: "%.1f%%", confusionMatrix!.precision * 100.0)), " +
                        "F1スコア: \(String(format: "%.1f%%", confusionMatrix!.f1Score * 100.0))]" :
                        ""
                )
                データ拡張: \(augmentationFinalDescription)
                特徴抽出器: \(featureExtractorDescForPairMetadata)
                """,
                version: version
            )

            print("💾 OvOペア [\(modelClass1Name) vs \(modelClass2Name)] モデル保存中: \(modelFilePath)")
            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
            print("✅ OvOペア [\(modelClass1Name) vs \(modelClass2Name)] モデル保存完了")

            print(String(
                format: "  ⏱️ OvOペア [\(modelClass1Name) vs \(modelClass2Name)] トレーニング所要時間: %.2f 秒",
                trainingDurationSeconds
            ))
            print(String(
                format: "  📊 OvOペア [\(modelClass1Name) vs \(modelClass2Name)] 訓練正解率: %.2f%%",
                trainingAccuracyPercent
            ))
            print(String(
                format: "  📈 OvOペア [\(modelClass1Name) vs \(modelClass2Name)] 検証正解率: %.2f%%",
                validationAccuracyPercent
            ))

            return OvOPairTrainingResult(
                modelPath: modelFilePath,
                modelName: modelFileNameBase,
                class1Name: modelClass1Name,
                class2Name: modelClass2Name,
                trainingAccuracyRate: trainingAccuracyPercent,
                validationAccuracyRate: validationAccuracyPercent,
                trainingErrorRate: trainingErrorRate,
                validationErrorRate: validationErrorRate,
                trainingTime: trainingDurationSeconds,
                trainingDataPath: tempOvOPairRootURL.path,
                confusionMatrix: confusionMatrix
            )

        } catch let createMLError as CreateML.MLCreateError {
            print(
                "🛑 エラー: OvOペア [\(modelClass1Name) vs \(modelClass2Name)] トレーニング/保存失敗 (CreateML): \(createMLError.localizedDescription)"
            )
            print("  詳細情報: \(createMLError)")
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
            return nil
        } catch {
            print(
                "🛑 エラー: OvOペア [\(modelClass1Name) vs \(modelClass2Name)] トレーニング/保存中に予期しないエラー: \(error.localizedDescription)"
            )
            try? Self.fileManager.removeItem(at: tempOvOPairRootURL)
            return nil
        }
    }

    // 指定されたディレクトリ内のファイル一覧を取得する
    private func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try Self.fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ).filter { url in
            !url.lastPathComponent
                .hasPrefix(".") && (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
    }
}
