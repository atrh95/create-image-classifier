import CICFileManager
import CoreML
import CreateML
import Foundation
import OvOClassifier
import Vision
import XCTest

final class OvOClassifierTests: XCTestCase {
    var classifier: OvOClassifier!
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestModel"
    let testModelVersion: String = "v1"

    let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
        featureExtractor: .scenePrint(revision: 1),
        classifier: .logisticRegressor
    )

    var modelParameters: MLImageClassifier.ModelParameters {
        MLImageClassifier.ModelParameters(
            validation: .split(strategy: .automatic),
            maxIterations: 1,
            augmentation: [],
            algorithm: algorithm
        )
    }

    var temporaryOutputDirectoryURL: URL!

    override func setUpWithError() throws {
        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_OvO")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // OvOClassifierTests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("OvOResources")
            .path

        classifier = OvOClassifier(
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path,
            resourceDirPathOverride: resourceDirectoryPath
        )
    }

    override func tearDownWithError() throws {
        if let tempDir = temporaryOutputDirectoryURL, fileManager.fileExists(atPath: tempDir.path) {
            try? fileManager.removeItem(at: tempDir)
        }
        temporaryOutputDirectoryURL = nil

        classifier.resourceDirPathOverride = nil
        classifier = nil
        try super.tearDownWithError()
    }

    func testClassifierDIConfiguration() async throws {
        // モデルの作成
        try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        XCTAssertNotNil(classifier, "OvOClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    // モデルの訓練と成果物の生成をテスト
    func testModelTrainingAndArtifactGeneration() async throws {
        // モデルの作成
        try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let resultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let latestResultDir = resultDirs.first else {
            XCTFail("結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // 各クラスラベルペアに対応するモデルファイルの存在確認
        for i in 0 ..< classLabelDirs.count {
            for j in (i + 1) ..< classLabelDirs.count {
                let classLabel1 = classLabelDirs[i]
                let classLabel2 = classLabelDirs[j]
                let expectedModelFileName =
                    "\(testModelName)_\(classifier.classificationMethod)_\(classLabel1)_vs_\(classLabel2)_\(testModelVersion).mlmodel"
                let modelFilePath = latestResultDir.appendingPathComponent(expectedModelFileName).path

                XCTAssertTrue(
                    fileManager.fileExists(atPath: modelFilePath),
                    "クラスペア '\(classLabel1)_vs_\(classLabel2)' の訓練モデルファイルが期待されるパス「\(modelFilePath)」に見つかりません"
                )
            }
        }

        // クラスラベルの存在確認
        XCTAssertFalse(
            classLabelDirs.isEmpty,
            "リソースディレクトリにクラスラベルディレクトリが見つかりません: \(classifier.resourcesDirectoryPath)"
        )

        // 各クラスラベルディレクトリにファイルが存在することを確認
        for classLabel in classLabelDirs {
            let classDirURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
                .appendingPathComponent(classLabel)
            let files = try fileManager.contentsOfDirectory(
                at: classDirURL,
                includingPropertiesForKeys: nil
            )

            XCTAssertFalse(
                files.isEmpty,
                "クラスラベル「\(classLabel)」のディレクトリにファイルが見つかりません: \(classDirURL.path)"
            )
        }

        // ログファイルの存在確認
        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = latestResultDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(
            fileManager.fileExists(atPath: expectedLogFilePath),
            "ログファイルが期待パス「\(expectedLogFilePath)」に未生成"
        )

        // モデルファイル名の検証
        for i in 0 ..< classLabelDirs.count {
            for j in (i + 1) ..< classLabelDirs.count {
                let classLabel1 = classLabelDirs[i]
                let classLabel2 = classLabelDirs[j]
                let modelFileName =
                    "\(testModelName)_\(classifier.classificationMethod)_\(classLabel1)_vs_\(classLabel2)_\(testModelVersion).mlmodel"
                let regex = #"^TestModel_OvO_\w+_vs_\w+_v\d+\.mlmodel$"#
                XCTAssertTrue(
                    modelFileName.range(of: regex, options: .regularExpression) != nil,
                    """
                    モデルファイル名が期待パターンに一致しません。
                    期待パターン: \(regex)
                    実際: \(modelFileName)
                    """
                )

                let modelFilePath = latestResultDir.appendingPathComponent(modelFileName).path
                XCTAssertTrue(
                    modelFilePath.contains(testModelVersion),
                    "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
                )
                XCTAssertTrue(
                    modelFilePath.contains("OvO"),
                    "モデルファイルパスに分類法「OvO」が含まれていません"
                )
                XCTAssertTrue(
                    modelFilePath.contains("\(classLabel1)_vs_\(classLabel2)"),
                    "モデルファイルパスにクラスラベルペア「\(classLabel1)_vs_\(classLabel2)」が含まれていません"
                )
            }
        }
    }

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() async throws {
        // モデルの作成
        try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let resultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let latestResultDir = resultDirs.first else {
            XCTFail("結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        print("結果ディレクトリ: \(latestResultDir.path)")

        // モデルファイルの存在確認
        let modelFiles = try fileManager.contentsOfDirectory(
            at: latestResultDir,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "mlmodel" }

        print("検出されたモデルファイル: \(modelFiles.map(\.lastPathComponent).joined(separator: ", "))")

        guard let modelFile = modelFiles.first else {
            XCTFail("モデルファイルが見つかりません: \(latestResultDir.path)")
            return
        }

        // モデルのコンパイル
        let compiledModelURL = try await MLModel.compileModel(at: modelFile)
        print("コンパイルされたモデル: \(compiledModelURL.path)")

        // コンパイルされたモデルファイルの存在確認
        guard fileManager.fileExists(atPath: compiledModelURL.path) else {
            XCTFail("コンパイルされたモデルファイルが存在しません: \(compiledModelURL.path)")
            throw ClassifierTestsError.modelFileMissing
        }

        // モデルの読み込みを試みる
        let mlModel: MLModel
        do {
            mlModel = try MLModel(contentsOf: compiledModelURL)
        } catch {
            XCTFail("モデルの読み込みに失敗しました: \(error.localizedDescription)")
            throw error
        }

        let vnCoreMLModel = try VNCoreMLModel(for: mlModel)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        let classLabels = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        guard classLabels.count >= 2 else {
            XCTFail("テストには少なくとも2つのクラスラベルが必要です。検出されたラベル: \(classLabels)")
            throw ClassifierTestsError.setupFailed
        }

        let classLabel1 = classLabels[0]
        let classLabel2 = classLabels[1]

        print("動的に識別されたテスト用クラスラベル: '\(classLabel1)' および '\(classLabel2)'")

        // classLabel1 の画像取得処理
        let imageURL1: URL
        do {
            imageURL1 = try TestUtils.getRandomImageURL(
                forClassLabel: classLabel1,
                resourcesDirectoryPath: classifier.resourcesDirectoryPath,
                fileManager: fileManager
            )
        } catch {
            XCTFail("'\(classLabel1)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        // classLabel2 の画像取得処理
        let imageURL2: URL
        do {
            imageURL2 = try TestUtils.getRandomImageURL(
                forClassLabel: classLabel2,
                resourcesDirectoryPath: classifier.resourcesDirectoryPath,
                fileManager: fileManager
            )
        } catch {
            XCTFail("'\(classLabel2)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        let imageHandler1 = VNImageRequestHandler(url: imageURL1, options: [:])
        try imageHandler1.perform([predictionRequest])
        guard let observations1 = predictionRequest.results as? [VNClassificationObservation],
              let topResult1 = observations1.first
        else {
            XCTFail("クラス '\(classLabel1)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw ClassifierTestsError.predictionFailed
        }
        XCTAssertNotNil(topResult1.identifier, "クラス '\(classLabel1)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel1)' 画像の予測 (正解ラベル): \(topResult1.identifier) (確信度: \(topResult1.confidence))")

        let imageHandler2 = VNImageRequestHandler(url: imageURL2, options: [:])
        try imageHandler2.perform([predictionRequest])
        guard let observations2 = predictionRequest.results as? [VNClassificationObservation],
              let topResult2 = observations2.first
        else {
            XCTFail("クラス '\(classLabel2)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw ClassifierTestsError.predictionFailed
        }
        XCTAssertNotNil(topResult2.identifier, "クラス '\(classLabel2)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel2)' 画像の予測 (正解ラベル): \(topResult2.identifier) (確信度: \(topResult2.confidence))")
    }

    // 出力ディレクトリの連番を検証
    func testSequentialOutputDirectoryNumbering() async throws {
        // 1回目のモデル作成
        try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let firstResultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let firstResultDir = firstResultDirs.first else {
            XCTFail("1回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 2回目のモデル作成を実行
        try await classifier.create(
            author: "TestAuthor",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters
        )

        // 出力ディレクトリから最新の結果を取得
        let secondResultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let secondResultDir = secondResultDirs.first else {
            XCTFail("2回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 連番の検証
        let firstResultNumber = Int(firstResultDir.lastPathComponent.replacingOccurrences(
            of: "OvO_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondResultDir.lastPathComponent.replacingOccurrences(
            of: "OvO_Result_",
            with: ""
        )) ?? 0

        print("1回目の結果ディレクトリ: \(firstResultDir.lastPathComponent)")
        print("2回目の結果ディレクトリ: \(secondResultDir.lastPathComponent)")
        print("1回目の番号: \(firstResultNumber)")
        print("2回目の番号: \(secondResultNumber)")

        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }

    func testClassFileCountBalance() async throws {
        // モデルの作成
        try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let resultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let latestResultDir = resultDirs.first else {
            XCTFail("結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // 各クラスの画像ファイル数を取得
        var classFileCounts: [String: Int] = [:]
        for classLabel in classLabelDirs {
            let classDir = URL(fileURLWithPath: classifier.resourcesDirectoryPath).appendingPathComponent(classLabel)
            let files = try fileManager.contentsOfDirectory(
                at: classDir,
                includingPropertiesForKeys: nil
            )
            classFileCounts[classLabel] = files.count
        }

        // 各クラスペアのモデルファイルを検証
        for i in 0 ..< classLabelDirs.count {
            for j in (i + 1) ..< classLabelDirs.count {
                let class1 = classLabelDirs[i]
                let class2 = classLabelDirs[j]
                let expectedModelFileName =
                    "\(testModelName)_\(classifier.classificationMethod)_\(class1)_vs_\(class2)_\(testModelVersion).mlmodel"
                let modelFilePath = latestResultDir.appendingPathComponent(expectedModelFileName).path

                // モデルファイルの存在確認
                XCTAssertTrue(
                    fileManager.fileExists(atPath: modelFilePath),
                    "クラスペア '\(class1) vs \(class2)' の訓練モデルファイルが期待されるパス「\(modelFilePath)」に見つかりません"
                )

                // モデルのメタデータを読み込み
                let modelURL = URL(fileURLWithPath: modelFilePath)
                let compiledModelURL = try await MLModel.compileModel(at: modelURL)
                let model = try MLModel(contentsOf: compiledModelURL)

                // メタデータから説明文を取得
                guard let description = model.modelDescription.metadata[MLModelMetadataKey.description] as? String
                else {
                    XCTFail("クラスペア '\(class1) vs \(class2)' のモデルメタデータから説明文を取得できません")
                    continue
                }

                // 説明文からファイル数を抽出
                let lines = description.components(separatedBy: CharacterSet.newlines)
                var class1Count = 0
                var class2Count = 0

                for line in lines {
                    if line.contains("\(class1):") {
                        if let count = extractFileCount(from: line) {
                            class1Count = count
                        }
                    } else if line.contains("\(class2):") {
                        if let count = extractFileCount(from: line) {
                            class2Count = count
                        }
                    }
                }

                // 両クラスのファイル数が一致することを確認
                XCTAssertEqual(
                    class1Count,
                    class2Count,
                    """
                    クラスペア '\(class1) vs \(class2)' のモデルで、両クラスのファイル数が一致しません。
                    \(class1): \(class1Count)枚
                    \(class2): \(class2Count)枚
                    """
                )

                // ファイル数が期待値（両クラスの最小枚数）と一致することを確認
                let originalClass1Count = classFileCounts[class1] ?? 0
                let originalClass2Count = classFileCounts[class2] ?? 0
                let expectedCount = min(originalClass1Count, originalClass2Count)

                XCTAssertEqual(
                    class1Count,
                    expectedCount,
                    """
                    クラスペア '\(class1) vs \(class2)' のモデルで、ファイル数が期待値と一致しません。
                    期待値: \(expectedCount)枚（\(class1): \(originalClass1Count)枚, \(class2): \(originalClass2Count)枚 の最小値）
                    実際: \(class1Count)枚
                    """
                )
            }
        }
    }

    private func extractFileCount(from line: String) -> Int? {
        let components = line.components(separatedBy: ":")
        guard components.count >= 2 else { return nil }
        let countString = components[1].trimmingCharacters(in: .whitespaces)
        return Int(countString.replacingOccurrences(of: "枚", with: ""))
    }
}
