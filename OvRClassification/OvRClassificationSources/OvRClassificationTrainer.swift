import Combine
import CreateML
import CSInterface
import Foundation
import TabularData

private struct OvRPairTrainingResult {
    let modelPath: String
    let trainingAccuracy: Double
    let validationAccuracy: Double
    let trainingErrorRate: Double
    let validationErrorRate: Double
    let trainingTime: TimeInterval
    let trainingDataPath: String
}

public class OvRClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = OvRTrainingResult

    public var modelName: String {
        "OvR_BatchCoordinator"
    }

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

    public func train(author: String, shortDescription: String, version: String) async -> OvRTrainingResult? {
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
        defer { // この行からコメントアウトを解除
            if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
                do {
                    try Self.fileManager.removeItem(at: tempOvRBaseURL)
                    print("🗑️ 一時ディレクトリ \(tempOvRBaseURL.path) をクリーンアップしました。")
                } catch {
                    print("⚠️ 一時ディレクトリ \(tempOvRBaseURL.path) のクリーンアップに失敗しました: \(error.localizedDescription)")
                }
            }
        } // ここまでコメントアウトを解除

        if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true)) != nil
        else {
            print("🛑 一時ディレクトリ \(tempOvRBaseURL.path) の作成に失敗しました。処理を中止します。")
            return nil
        }

        let ovrResourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)

        print("🚀 OvRトレーニングを開始します: \(version)")

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
            return nil
        }

        let primaryLabelSourceDirs = allLabelSourceDirectories.filter { $0.lastPathComponent.lowercased() != "safe" }

        if primaryLabelSourceDirs.isEmpty {
            print("🛑 プライマリトレーニングターゲットとなるディレクトリが見つかりません ('safe' ディレクトリを除く)。処理を中止します。")
            return nil
        }

        print("  処理対象ラベル数: \(primaryLabelSourceDirs.count)")

        var allTrainingResults: [OvRPairTrainingResult] = []

        for (index, dir) in primaryLabelSourceDirs.enumerated() {
            if let result = await trainSingleOvRPair(
                oneLabelSourceDirURL: dir,
                allLabelSourceDirs: allLabelSourceDirectories,
                ovrResourcesURL: ovrResourcesURL,
                mainRunURL: mainOutputRunURL,
                tempOvRBaseURL: tempOvRBaseURL,
                author: author,
                shortDescription: shortDescription,
                version: version,
                pairIndex: index
            ) {
                allTrainingResults.append(result)
            }
        }

        guard !allTrainingResults.isEmpty else {
            return nil
        }

        let avgTrainingAccuracy = allTrainingResults.map(\.trainingAccuracy)
            .reduce(0, +) / Double(allTrainingResults.count)
        let avgValidationAccuracy = allTrainingResults.map(\.validationAccuracy)
            .reduce(0, +) / Double(allTrainingResults.count)
        let avgTrainingErrorRate = allTrainingResults.map(\.trainingErrorRate)
            .reduce(0, +) / Double(allTrainingResults.count)
        let avgValidationErrorRate = allTrainingResults.map(\.validationErrorRate)
            .reduce(0, +) / Double(allTrainingResults.count)
        let avgTrainingTime = allTrainingResults.map(\.trainingTime).reduce(0, +) / Double(allTrainingResults.count)
        let trainingDataPaths = allTrainingResults.map(\.trainingDataPath).joined(separator: ", ")

        // OvRPairTrainingResultからIndividualModelReportに変換
        let individualReports: [IndividualModelReport] = allTrainingResults.map { result in
            IndividualModelReport(
                modelName: URL(fileURLWithPath: result.modelPath).lastPathComponent,
                trainingAccuracy: result.trainingAccuracy,
                validationAccuracy: result.validationAccuracy
            )
        }

        let representativeModelPath = allTrainingResults.first?.modelPath ?? mainOutputRunURL.path

        let trainingResult = OvRTrainingResult(
            modelOutputPath: representativeModelPath,
            trainingDataAccuracy: avgTrainingAccuracy,
            validationDataAccuracy: avgValidationAccuracy,
            trainingDataErrorRate: avgTrainingErrorRate,
            validationDataErrorRate: avgValidationErrorRate,
            trainingTimeInSeconds: avgTrainingTime,
            trainingDataPath: trainingDataPaths,
            individualReports: individualReports
        )

        return trainingResult
    }

    private func trainSingleOvRPair(
        oneLabelSourceDirURL: URL,
        allLabelSourceDirs: [URL],
        ovrResourcesURL _: URL,
        mainRunURL: URL,
        tempOvRBaseURL: URL,
        author: String,
        shortDescription _: String,
        version: String,
        pairIndex _: Int
    ) async -> OvRPairTrainingResult? {
        let originalOneLabelName = oneLabelSourceDirURL.lastPathComponent
        let upperCamelCaseOneLabelName = originalOneLabelName.split(separator: "_").map(\.capitalized).joined()

        let tempOvRPairRootName = "\(upperCamelCaseOneLabelName)_vs_Rest_TrainingData_\(version)"
        let tempOvRPairRootURL = tempOvRBaseURL.appendingPathComponent(tempOvRPairRootName)

        let tempPositiveDataDirForML = tempOvRPairRootURL.appendingPathComponent(upperCamelCaseOneLabelName)
        let tempRestDataDirForML = tempOvRPairRootURL.appendingPathComponent("Rest")

        if Self.fileManager.fileExists(atPath: tempOvRPairRootURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRPairRootURL)
        }
        try? Self.fileManager.createDirectory(at: tempPositiveDataDirForML, withIntermediateDirectories: true)
        try? Self.fileManager.createDirectory(at: tempRestDataDirForML, withIntermediateDirectories: true)

        if let positiveSourceFiles = try? getFilesInDirectory(oneLabelSourceDirURL) {
            for fileURL in positiveSourceFiles {
                try? Self.fileManager.copyItem(
                    at: fileURL,
                    to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent)
                )
            }
        }

        guard let positiveSourceFilesForCount = try? getFilesInDirectory(oneLabelSourceDirURL),
              !positiveSourceFilesForCount.isEmpty
        else {
            print(
                "⚠️ ポジティブサンプルが見つからないか空です: \(oneLabelSourceDirURL.lastPathComponent)。ペア \(originalOneLabelName) vs Rest の学習をスキップします。"
            )
            return nil
        }
        let positiveSamplesCount = positiveSourceFilesForCount.count

        let safeDirName = "safe"
        let otherDirsForNegativeSampling = allLabelSourceDirs.filter { dirURL in
            let isCurrentPositiveDir = dirURL.resolvingSymlinksInPath().standardizedFileURL == oneLabelSourceDirURL
                .resolvingSymlinksInPath().standardizedFileURL
            return !isCurrentPositiveDir
        }

        if otherDirsForNegativeSampling.isEmpty {
            print(
                "ℹ️ ネガティブサンプリング対象の他のディレクトリがありません (safeディレクトリ以外に、現在のラベル \(originalOneLabelName) と比較できるものがありません)。このペアの学習はスキップされます。"
            )
            return nil
        }

        let numFilesToCollectPerOtherDir =
            Int(ceil(Double(positiveSamplesCount) / Double(otherDirsForNegativeSampling.count)))

        var collectedNegativeFilesCount = 0
        for otherDirURL in otherDirsForNegativeSampling {
            guard let filesInOtherDir = try? getFilesInDirectory(otherDirURL), !filesInOtherDir.isEmpty else {
                print("ℹ️ ディレクトリ \(otherDirURL.lastPathComponent) は空かアクセス不能なため、ネガティブサンプル収集からスキップします。")
                continue
            }

            let filesToCopy = filesInOtherDir.shuffled().prefix(numFilesToCollectPerOtherDir)
            for fileURL in filesToCopy {
                let sourceDirNamePrefix = otherDirURL.lastPathComponent
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
                    print(
                        "⚠️ ファイルコピーに失敗: \(fileURL.path) から \(tempRestDataDirForML.appendingPathComponent(newFileName).path) へ。エラー: \(error.localizedDescription)"
                    )
                }
            }
        }

        if collectedNegativeFilesCount == 0 {
            print(
                "🛑 ネガティブサンプルを1つも収集できませんでした。ポジティブサンプル数: \(positiveSamplesCount), 他カテゴリ数: \(otherDirsForNegativeSampling.count), 各カテゴリからの目標収集数: \(numFilesToCollectPerOtherDir)。ペア \(originalOneLabelName) vs Rest の学習をスキップします。"
            )
            return nil
        }
        print(
            "ℹ️ \(originalOneLabelName) vs Rest: \(collectedNegativeFilesCount) 枚のネガティブサンプルを \(otherDirsForNegativeSampling.count) カテゴリから収集しました (目標 各\(numFilesToCollectPerOtherDir)枚)。"
        )

        do {
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempOvRPairRootURL)

            var parameters = MLImageClassifier.ModelParameters()
            parameters.featureExtractor = .scenePrint(revision: 1)
            parameters.validation = .split(strategy: .automatic)
            parameters.maxIterations = 25
            parameters.augmentationOptions = [.crop, .rotation, .blur]

            let startTime = Date()
            let job = try MLImageClassifier.train(
                trainingData: trainingDataSource,
                parameters: parameters
            )
            let trainingTime = Date().timeIntervalSince(startTime)

            var iterator = job.result.values.makeAsyncIterator()
            guard let classifier = try await iterator.next() else {
                return nil
            }

            let modelFileName = "\(upperCamelCaseOneLabelName)_OvR_\(version).mlmodel"
            let modelOutputPath = mainRunURL.appendingPathComponent(modelFileName).path

            let metadata = MLModelMetadata(
                author: author,
                shortDescription: "\(upperCamelCaseOneLabelName) 対 Rest の2値分類モデル。",
                version: version
            )

            try classifier.write(to: URL(fileURLWithPath: modelOutputPath), metadata: metadata)

            // Extract training and validation metrics
            let trainingErrorRate = classifier.trainingMetrics.classificationError
            let validationErrorRate = classifier.validationMetrics.classificationError
            let trainingAccuracy = 1.0 - trainingErrorRate
            let validationAccuracy = 1.0 - validationErrorRate

            print("✅ トレーニング完了: \(modelFileName)")
            print("  トレーニングデータ正解率: \(String(format: "%.4f", trainingAccuracy))")
            print("  検証データ正解率: \(String(format: "%.4f", validationAccuracy))")

            return OvRPairTrainingResult(
                modelPath: modelOutputPath,
                trainingAccuracy: trainingAccuracy,
                validationAccuracy: validationAccuracy,
                trainingErrorRate: trainingErrorRate,
                validationErrorRate: validationErrorRate,
                trainingTime: trainingTime,
                trainingDataPath: tempOvRPairRootURL.path
            )

        } catch {
            return nil
        }
    }

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
