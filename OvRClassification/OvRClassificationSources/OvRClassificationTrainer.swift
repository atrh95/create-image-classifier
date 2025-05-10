import Combine
import CreateML
import Foundation
import SCSInterface
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
        let baseProjectURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()

        let batchRootURL = baseProjectURL.appendingPathComponent(customOutputDirPath)
        guard (try? Self.fileManager.createDirectory(at: batchRootURL, withIntermediateDirectories: true)) != nil else {
            return nil
        }

        let existingRuns = (try? Self.fileManager.contentsOfDirectory(at: batchRootURL, includingPropertiesForKeys: nil)) ?? []
        let nextIndex = (existingRuns.compactMap { Int($0.lastPathComponent.replacingOccurrences(of: "OvR_Result_", with: "")) }.max() ?? 0) + 1
        let mainOutputRunURL = batchRootURL.appendingPathComponent("OvR_Result_\(nextIndex)")

        guard (try? Self.fileManager.createDirectory(at: mainOutputRunURL, withIntermediateDirectories: true)) != nil else {
            return nil
        }

        let tempOvRBaseURL = baseProjectURL.appendingPathComponent(Self.tempBaseDirName)
        if Self.fileManager.fileExists(atPath: tempOvRBaseURL.path) {
            try? Self.fileManager.removeItem(at: tempOvRBaseURL)
        }
        guard (try? Self.fileManager.createDirectory(at: tempOvRBaseURL, withIntermediateDirectories: true)) != nil else {
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

        let avgTrainingAccuracy = allTrainingResults.map { $0.trainingAccuracy }.reduce(0, +) / Double(allTrainingResults.count)
        let avgValidationAccuracy = allTrainingResults.map { $0.validationAccuracy }.reduce(0, +) / Double(allTrainingResults.count)
        let avgTrainingErrorRate = allTrainingResults.map { $0.trainingErrorRate }.reduce(0, +) / Double(allTrainingResults.count)
        let avgValidationErrorRate = allTrainingResults.map { $0.validationErrorRate }.reduce(0, +) / Double(allTrainingResults.count)
        let avgTrainingTime = allTrainingResults.map { $0.trainingTime }.reduce(0, +) / Double(allTrainingResults.count)
        let trainingDataPaths = allTrainingResults.map { $0.trainingDataPath }.joined(separator: ", ")

        let representativeModelPath = allTrainingResults.first?.modelPath ?? mainOutputRunURL.path

        let trainingResult = OvRTrainingResult(
            modelOutputPath: representativeModelPath,
            trainingDataAccuracy: avgTrainingAccuracy,
            validationDataAccuracy: avgValidationAccuracy,
            trainingDataErrorRate: avgTrainingErrorRate,
            validationDataErrorRate: avgValidationErrorRate,
            trainingTimeInSeconds: avgTrainingTime,
            trainingDataPath: trainingDataPaths
        )

        return trainingResult
    }

    private func trainSingleOvRPair(
        oneLabelSourceDirURL: URL,
        allLabelSourceDirs: [URL],
        ovrResourcesURL: URL,
        mainRunURL: URL,
        tempOvRBaseURL: URL,
        author: String,
        shortDescription: String,
        version: String,
        pairIndex: Int
    ) async -> OvRPairTrainingResult? {
        let originalOneLabelName = oneLabelSourceDirURL.lastPathComponent
        let upperCamelCaseOneLabelName = originalOneLabelName.split(separator: "_").map { $0.capitalized }.joined()

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
                try? Self.fileManager.copyItem(at: fileURL, to: tempPositiveDataDirForML.appendingPathComponent(fileURL.lastPathComponent))
            }
        }

        guard let positiveSourceFilesForCount = try? getFilesInDirectory(oneLabelSourceDirURL), !positiveSourceFilesForCount.isEmpty else {
            print("⚠️ ポジティブサンプルが見つからないか空です: \(oneLabelSourceDirURL.lastPathComponent)。ペア \(originalOneLabelName) vs Rest の学習をスキップします。")
            return nil
        }
        let positiveSamplesCount = positiveSourceFilesForCount.count

        let safeDirName = "safe"
        let otherDirsForNegativeSampling = allLabelSourceDirs.filter { dirURL in
            let dirNameLowercased = dirURL.lastPathComponent.lowercased()
            let isCurrentPositiveDir = dirURL.resolvingSymlinksInPath().standardizedFileURL == oneLabelSourceDirURL.resolvingSymlinksInPath().standardizedFileURL
            return !isCurrentPositiveDir && dirNameLowercased != safeDirName
        }

        if otherDirsForNegativeSampling.isEmpty {
            print("ℹ️ ネガティブサンプリング対象の他のディレクトリがありません (safeディレクトリ以外に、現在のラベル \(originalOneLabelName) と比較できるものがありません)。このペアの学習はスキップされます。")
            return nil
        }
        
        let numFilesToCollectPerOtherDir = Int(ceil(Double(positiveSamplesCount) / Double(otherDirsForNegativeSampling.count)))

        var collectedNegativeFilesCount = 0
        for otherDirURL in otherDirsForNegativeSampling {
            guard let filesInOtherDir = try? getFilesInDirectory(otherDirURL), !filesInOtherDir.isEmpty else {
                print("ℹ️ ディレクトリ \(otherDirURL.lastPathComponent) は空かアクセス不能なため、ネガティブサンプル収集からスキップします。")
                continue
            }
            
            let filesToCopy = filesInOtherDir.shuffled().prefix(numFilesToCollectPerOtherDir)
            for fileURL in filesToCopy {
                let sourceDirNamePrefix = otherDirURL.lastPathComponent
                let sanitizedSourceDirNamePrefix = sourceDirNamePrefix.replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "_", options: .regularExpression)
                let sanitizedOriginalFileName = fileURL.lastPathComponent.replacingOccurrences(of: "[^a-zA-Z0-9_.-]", with: "_", options: .regularExpression)
                let newFileName = "\(sanitizedSourceDirNamePrefix)_\(sanitizedOriginalFileName)"
                
                do {
                    try Self.fileManager.copyItem(at: fileURL, to: tempRestDataDirForML.appendingPathComponent(newFileName))
                    collectedNegativeFilesCount += 1
                } catch {
                    print("⚠️ ファイルコピーに失敗: \(fileURL.path) から \(tempRestDataDirForML.appendingPathComponent(newFileName).path) へ。エラー: \(error.localizedDescription)")
                }
            }
        }

        if collectedNegativeFilesCount == 0 {
            print("🛑 ネガティブサンプルを1つも収集できませんでした。ポジティブサンプル数: \(positiveSamplesCount), 他カテゴリ数: \(otherDirsForNegativeSampling.count), 各カテゴリからの目標収集数: \(numFilesToCollectPerOtherDir)。ペア \(originalOneLabelName) vs Rest の学習をスキップします。")
            return nil
        }
        print("ℹ️ \(originalOneLabelName) vs Rest: \(collectedNegativeFilesCount) 枚のネガティブサンプルを \(otherDirsForNegativeSampling.count) カテゴリから収集しました (目標 各\(numFilesToCollectPerOtherDir)枚)。")

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
