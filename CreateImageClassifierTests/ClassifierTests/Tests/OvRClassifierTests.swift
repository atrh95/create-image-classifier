import CICFileManager
import CoreML
import CreateML
import Foundation
@testable import OvRClassification
import Vision
import XCTest

final class OvRClassifierTests: XCTestCase {
    var classifier: OvRClassifier!
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
            .appendingPathComponent("TestOutput_OvR")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // OvRClassifierTests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("OvR")
            .path

        classifier = OvRClassifier(
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
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )
        
        XCTAssertNotNil(classifier, "OvRClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    func testModelTrainingAndArtifactGeneration() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.modelOutputPath),
            "訓練モデルファイルが期待されるパス「\(result.modelOutputPath)」に見つかりません"
        )

        let resourceURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)

        var expectedClassLabels: [String] = []
        do {
            let subdirectories = try fileManager.contentsOfDirectory(
                at: resourceURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
            expectedClassLabels = subdirectories.filter { url in
                var isDirectory: ObjCBool = false
                return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            }.map(\.lastPathComponent).sorted()
        } catch {
            XCTFail("テストリソースのサブディレクトリからのクラスラベルの取得に失敗しました: \(error.localizedDescription)")
            throw ClassifierTestsError.setupFailed
        }

        // モデルファイル名の検証
        let modelFilePath = result.metadata.trainedModelFilePath
        let modelFileName = URL(fileURLWithPath: modelFilePath).lastPathComponent

        // OvR分類器では各クラスに対して1つの分類器を作成するため、各クラス名を含むパターンを期待
        let regex = #"^TestModel_OvR_[a-z_]+_v\d+\.mlmodel$"#
        XCTAssertTrue(
            modelFileName.range(of: regex, options: .regularExpression) != nil,
            """
            モデルファイル名が期待パターンに一致しません。
            期待パターン: \(regex)
            実際: \(modelFileName)
            """
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.metadata.trainedModelFilePath).deletingLastPathComponent()
        let expectedLogFileName = "OvR_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        XCTAssertEqual(result.metadata.modelName, testModelName)
    }

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // モデルのコンパイル
        let trainedModelURL = URL(fileURLWithPath: result.metadata.trainedModelFilePath)
        let compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)

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

        let imageURL: URL
        do {
            // 存在するディレクトリからランダムに選択
            let availableClassLabels = ["sphynx", "mouth_open"]
            let randomClassLabel = availableClassLabels.randomElement()!
            imageURL = try TestUtils.getRandomImageURL(
                forClassLabel: randomClassLabel,
                resourcesDirectoryPath: classifier.resourcesDirectoryPath,
                fileManager: fileManager
            )
        } catch {
            XCTFail("テストリソースからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        print("Test image for prediction: \(imageURL.path)")

        guard fileManager.fileExists(atPath: imageURL.path) else {
            XCTFail("OvRテスト用画像ファイルが見つかりません: \(imageURL.path)")
            throw ClassifierTestsError.resourcePathError
        }

        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error {
                XCTFail("VNCoreMLRequest failed: \(error.localizedDescription)")
                return
            }
            guard let observations = request.results as? [VNClassificationObservation] else {
                XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
                return
            }

            XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

            if let topResult = observations.first {
                print(
                    "OvR Top prediction for \(imageURL.lastPathComponent) using \(URL(fileURLWithPath: result.metadata.trainedModelFilePath).lastPathComponent): \(topResult.identifier) with confidence \(topResult.confidence)"
                )
            } else {
                print(
                    "OvR prediction for \(imageURL.lastPathComponent) using \(URL(fileURLWithPath: result.metadata.trainedModelFilePath).lastPathComponent): No observations found, though this should have been caught by XCTAssertFalse."
                )
            }
        }

        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])
    }

    /// 生成されるmlmodelファイルの数がクラスの数と一致することを確認する
    func testModelFileCountMatchesClassCount() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // リソースディレクトリからクラスラベルの一覧を取得
        let resourceURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
        let classLabels = try FileManager.default.contentsOfDirectory(
            at: resourceURL,
            includingPropertiesForKeys: [.isDirectoryKey]
        )
        .filter(\.hasDirectoryPath)
        .map(\.lastPathComponent)
        .sorted()

        // モデルファイルのディレクトリを取得
        let modelFileDir = URL(fileURLWithPath: result.metadata.trainedModelFilePath).deletingLastPathComponent()

        // 生成されたmlmodelファイルの数を取得
        let modelFiles = try FileManager.default.contentsOfDirectory(at: modelFileDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "mlmodel" }

        // クラスの数とmlmodelファイルの数が一致することを確認
        XCTAssertEqual(
            modelFiles.count,
            classLabels.count,
            """
            生成されたmlmodelファイルの数がクラスの数と一致しません。
            クラス数: \(classLabels.count)
            生成されたmlmodelファイル数: \(modelFiles.count)
            クラスラベル: \(classLabels.joined(separator: ", "))
            """
        )

        // 各モデルファイル名が正しいクラス名を含むことを確認
        for modelFile in modelFiles {
            let fileName = modelFile.lastPathComponent
            // ファイル名からクラス名を抽出（例: "TestModel_OvR_class1_v1.mlmodel"）
            let components = fileName.components(separatedBy: "_")
            guard components.count >= 4 else {
                XCTFail("モデルファイル名の形式が不正です: \(fileName)")
                continue
            }

            // クラス名を抽出（OvRの後の部分、バージョン番号の前まで）
            let classComponents = components[2 ..< (components.count - 1)]
            let className = classComponents.joined(separator: "_")

            // 抽出したクラス名が実際のクラスラベルのリストに含まれていることを確認
            XCTAssertTrue(
                classLabels.contains(className),
                """
                モデルファイル名に含まれるクラス名が実際のクラスラベルのリストに存在しません。
                ファイル名: \(fileName)
                抽出されたクラス名: \(className)
                実際のクラスラベル: \(classLabels.joined(separator: ", "))
                """
            )
        }
    }

    // 出力ディレクトリの連番を検証
    func testSequentialOutputDirectoryNumbering() async throws {
        // 1回目のモデル作成
        let firstResult = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        let firstModelFileDir = URL(fileURLWithPath: firstResult.metadata.trainedModelFilePath)
            .deletingLastPathComponent()

        // 2回目のモデル作成を実行
        let secondResult = try await classifier.create(
            author: "TestAuthor",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters
        )

        let secondModelFileDir = URL(fileURLWithPath: secondResult.metadata.trainedModelFilePath)
            .deletingLastPathComponent()

        // 連番の検証
        let firstResultNumber = Int(firstModelFileDir.lastPathComponent.replacingOccurrences(
            of: "OvR_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondModelFileDir.lastPathComponent.replacingOccurrences(
            of: "OvR_Result_",
            with: ""
        )) ?? 0
        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }
}
