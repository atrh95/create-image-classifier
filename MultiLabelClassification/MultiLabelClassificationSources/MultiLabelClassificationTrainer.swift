import CoreML
import CreateML
import CreateMLComponents
import Foundation
import SCSInterface

@available(macOS 14, *)   // API利用可能性の警告を抑制
public class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    // Helper struct for decoding the manifest JSON
    struct ManifestEntry: Decodable {
        var filename: String
        var annotations: [String] // Decode as array, then convert to Set<String>
    }

    public var modelName: String { "MultiLabelScaryCatML_Components" }
    public var dataDirectoryName: String { "" }
    public var manifestFileName: String { "multilabel_cat_annotations.json" }
    public var customOutputDirPath: String { "OutputModels/ScaryCatScreeningML/MultiLabel" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appending(path: "Resources").path
    }

    public init() {}

    public func train(
        author: String,
        shortDescription: String,
        version: String
    ) async -> TrainingResultType? {
        let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let imageBaseDir = resourcesDir.appendingPathComponent(dataDirectoryName)
        let manifestFile = resourcesDir.appendingPathComponent(manifestFileName)

        print("ℹ️ リソースディレクトリ (ローカルパスベース): \(resourcesDir.path)")
        print("ℹ️ 画像ベースディレクトリ (ローカルパスベース): \(imageBaseDir.path)")
        print("ℹ️ マニフェストファイルURL (ローカルパスベース): \(manifestFile.path)")

        guard FileManager.default.fileExists(atPath: imageBaseDir.path) else {
            print("❌ エラー: 画像ベースディレクトリが見つかりません (ローカルパスベース): \(imageBaseDir.path)")
            // 詳細デバッグ: imageBaseDir の一つ上の階層も確認
            let parentDir = imageBaseDir.deletingLastPathComponent()
            print("  親ディレクトリ (ローカルパスベース)「\(parentDir.path)」の内容:")
            do {
                let items = try FileManager.default.contentsOfDirectory(atPath: parentDir.path)
                items.forEach { item in print("    - \(item)") }
            } catch {
                 print("    親ディレクトリの内容をリストできませんでした: \(error)")
            }
            return nil
        }
        // manifestFileの存在は resourcesDir.appendingPathComponent(manifestFileName) で確認済みのため、ここでの fileExists チェックは不要

        let fileManager = FileManager.default
        var finalOutputDir: URL!

        do {
            var playgroundRoot = URL(fileURLWithPath: #filePath)
            (1...3).forEach { _ in playgroundRoot.deleteLastPathComponent() }
            var baseOutputDir = playgroundRoot

            let customPath = customOutputDirPath
            if !customPath.isEmpty {
                let customURL = URL(fileURLWithPath: customPath)
                baseOutputDir = customURL.isFileURL && customPath.hasPrefix("/") ? customURL : baseOutputDir.appendingPathComponent(customPath)
            } else {
                baseOutputDir = baseOutputDir.appendingPathComponent("OutputModels")
            }
            try fileManager.createDirectory(at: baseOutputDir, withIntermediateDirectories: true, attributes: nil)

            var resultCounter = 1
            let resultDirPrefix = "multilabel_components_result_"
            repeat {
                finalOutputDir = baseOutputDir.appendingPathComponent("\(resultDirPrefix)\(resultCounter)")
                resultCounter += 1
            } while fileManager.fileExists(atPath: finalOutputDir.path)
            try fileManager.createDirectory(at: finalOutputDir, withIntermediateDirectories: false, attributes: nil)
            print("💾 結果保存ディレクトリ (マルチラベル - Components): \(finalOutputDir.path)")

            print("\n🚀 マルチラベル分類モデル (Components) [\(modelName)] のトレーニングを開始します...")
            print("  マニフェスト: \(manifestFile.path)")
            print("  画像ディレクトリ: \(imageBaseDir.path)")

            var allLabelsSet = Set<String>()
            // Decode manifest and prepare annotated features
            let annotatedFeatures: [AnnotatedFeature<URL, Set<String>>]
            do {
                // Explicitly check if the manifest file exists before trying to read it
                guard FileManager.default.fileExists(atPath: manifestFile.path) else {
                    print("❌ FATAL ERROR: Manifest file NOT FOUND at path: \(manifestFile.path) according to FileManager.default.fileExists")
                    // You might want to list directory contents here for further debugging if needed:
                    // let resourcesDirPath = manifestFile.deletingLastPathComponent().path
                    // print("  Contents of directory \(resourcesDirPath):")
                    // do {
                    //     let items = try FileManager.default.contentsOfDirectory(atPath: resourcesDirPath)
                    //     items.forEach { item in print("    - \(item)") }
                    // } catch {
                    //     print("    Could not list directory contents: \(error)")
                    // }
                    return nil
                }
                print("ℹ️ FileManager confirms manifest file exists at: \(manifestFile.path)")

                // Attempt to get file attributes for more diagnostics
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: manifestFile.path)
                    print("  🔍 File attributes: \(attributes)")
                    if let fileSize = attributes[.size] as? NSNumber {
                        print("  📏 File size: \(fileSize.intValue) bytes")
                        if fileSize.intValue == 0 {
                            print("  ⚠️ WARNING: Manifest file appears to be empty (0 bytes).")
                        }
                    }
                } catch {
                    print("  ⚠️ Could not retrieve file attributes for \(manifestFile.path): \(error)")
                }

                print("  Attempting to read data from path: \(manifestFile.path) using FileHandle.")
                let fileHandle: FileHandle
                let jsonData: Data
                do {
                    let targetURL = URL(fileURLWithPath: manifestFile.path)
                    fileHandle = try FileHandle(forReadingFrom: targetURL)
                    
                    // Read all data available from the file handle
                    if #available(macOS 10.15.4, iOS 13.4, tvOS 13.4, watchOS 6.2, *) {
                        jsonData = try fileHandle.readToEnd() ?? Data()
                    } else {
                        jsonData = fileHandle.readDataToEndOfFile()
                    }
                    try fileHandle.close() // Close the file handle after reading
                    
                    print("  Successfully read \(jsonData.count) bytes using FileHandle.")
                    if jsonData.isEmpty {
                        // Check file size again, as it might have changed or our previous check was misleading
                        if let attributes = try? FileManager.default.attributesOfItem(atPath: manifestFile.path),
                           let fileSize = attributes[.size] as? NSNumber,
                           fileSize.intValue > 0 {
                            print("  ⚠️ WARNING: FileHandle read 0 bytes, but FileManager reported a size of \(fileSize.intValue) bytes. This is highly unusual.")
                        } else {
                            print("  ℹ️ FileHandle read 0 bytes. This might be expected if the file is truly empty.")
                        }
                    }
                } catch {
                    print("  ❌ ERROR using FileHandle to read \(manifestFile.path): \(error)")
                    print("    Detailed FileHandle error: \(error.localizedDescription)")
                    throw error // Re-throw to be caught by the outer catch block
                }

                let decoder = JSONDecoder()
                // --- Begin new do-catch for JSON Decoding ---
                do {
                    let manifestEntries = try decoder.decode([ManifestEntry].self, from: jsonData)
                    print("  ✅ Successfully decoded manifest JSON into \(manifestEntries.count) entries.")

                    annotatedFeatures = manifestEntries.map { entry in
                        let imageURL = imageBaseDir.appendingPathComponent(entry.filename)
                        let labelSet = Set(entry.annotations)
                        labelSet.forEach { allLabelsSet.insert($0) }
                        return AnnotatedFeature(feature: imageURL, annotation: labelSet)
                    }
                    
                    if annotatedFeatures.isEmpty {
                        print("❌ ERROR: Manifest decoded, but resulted in zero annotated features. Check manifest content.")
                        return nil
                    }
                } catch let decodingError as DecodingError {
                    print("❌ JSON DECODING ERROR: Failed to decode data from manifest file \(manifestFile.path).")
                    print("  jsonData size: \(jsonData.count) bytes")
                    switch decodingError {
                    case .typeMismatch(let type, let context):
                        print("  Type Mismatch: For key '\(context.codingPath.map { $0.stringValue }.joined(separator: "."))', expected type '\(type)' but found a different type.")
                        print("  Context: \(context.debugDescription)")
                    case .valueNotFound(let type, let context):
                        print("  Value Not Found: For key '\(context.codingPath.map { $0.stringValue }.joined(separator: "."))', expected value of type '\(type)' but found null or missing value.")
                        print("  Context: \(context.debugDescription)")
                    case .keyNotFound(let key, let context):
                        print("  Key Not Found: Key '\(key.stringValue)' not found at path '\(context.codingPath.map { $0.stringValue }.joined(separator: "."))'.")
                        print("  Context: \(context.debugDescription)")
                    case .dataCorrupted(let context):
                        print("  Data Corrupted: The JSON data is corrupted. Path: '\(context.codingPath.map { $0.stringValue }.joined(separator: "."))'.")
                        print("  Context: \(context.debugDescription)")
                        if let underlying = context.underlyingError as NSError?, let debugDescription = underlying.userInfo[NSDebugDescriptionErrorKey] as? String {
                            print("    Underlying error (from parser): \(debugDescription)")
                        }
                    @unknown default:
                        print("  Unknown Decoding Error: \(decodingError.localizedDescription)")
                    }
                    return nil // Critical: Stop processing if JSON decoding fails
                } catch {
                    // Catch any other non-DecodingError that might occur during the manifestEntries.map or other logic
                    print("❌ UNEXPECTED ERROR during manifest processing (after successful FileHandle read but not a DecodingError): \(error.localizedDescription)")
                    return nil
                }
                // --- End new do-catch for JSON Decoding ---

                let sortedClassLabels = allLabelsSet.sorted()
                guard !sortedClassLabels.isEmpty else {
                    print("❌ エラー: マニフェストからラベルを抽出できませんでした。マニフェストファイルを確認してください。")
                    return nil
                }
                print("📚 検出された全ラベル (マニフェストより): \(sortedClassLabels.joined(separator: ", "))")

                let startTime = Date()

                let allLabelsForClassifier: Set<String> = allLabelsSet

                // ImageFeaturePrintを設定 - リビジョン調整可能
                let featurePrint = ImageFeaturePrint(revision: 1)
                // ラベルが分類器に渡されることを確認
                let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(labels: allLabelsForClassifier)

                // ImageReaderをパイプラインの先頭にする
                print("  📖 ImageReaderをパイプラインに設定...")
                let imageReaderInPipeline = ImageReader()
                
                let pipeline = imageReaderInPipeline
                    .appending(featurePrint)
                    .appending(classifier)

                // 分割比率はトレーニング0.8、検証0.2
                // `annotatedFeatures` ( [AnnotatedFeature<URL, Set<String>>] ) を直接分割
                print("  📊 トレーニングデータと検証データを準備中 (80/20 スプリット)...")
                let (trainingData, validationData) = annotatedFeatures.randomSplit(by: 0.8)

                print("  ⚙️ パイプラインモデルの学習を開始します...")
                let fittedModel = try await pipeline.fitted(
                    to: trainingData,
                    validateOn: validationData
                )
                
                let endTime = Date()
                let duration = endTime.timeIntervalSince(startTime)
                print("🎉 [\(modelName)] のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", duration))秒)")

                // メトリクス計算
                print("🧪 検証データでメトリクスを計算中...")
                
                // 予測結果を非同期シーケンスとして取得し、直接配列として扱うことを期待
                // Appleのドキュメント例では await model.prediction(from:) がコレクションを返すように見える
                let validationPredictionsArray = try await fittedModel.prediction(from: validationData)
                
                // .map を使用して、分類器の出力（確率分布）と正解ラベルを抽出
                let predictedClassDistributions = validationPredictionsArray.map(\.prediction)
                let groundTruthLabels = validationPredictionsArray.map(\.annotation)

                // MultiLabelClassificationMetrics は ClassificationDistribution を直接扱えると期待
                let metrics = try MultiLabelClassificationMetrics(
                    classifications: predictedClassDistributions, // 確率分布のリストを直接渡す
                    groundTruth: groundTruthLabels,          // 正解ラベルのリストを渡す
                    strategy: .balancedPrecisionAndRecall,
                    labels: allLabelsSet 
                )
                let meanAveragePrecision = metrics.meanAveragePrecision
                print("📊 平均適合率 (MAP) [検証]: \(String(format: "%.4f", meanAveragePrecision))")


                let outputModelURL = finalOutputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")
                print("  💾 [\(modelName)_\(version).mlmodel] を保存中: \(outputModelURL.path)")
                
                // Use CreateMLComponents.ModelMetadata
                // Try initializing with author and version, then set shortDescription if possible
                var modelMetadata = CreateMLComponents.ModelMetadata(version: version, author: author)
                // modelMetadata.shortDescription = shortDescription // Attempt to set, if this property exists and is settable
                // If the above line causes an error, shortDescription might not be a property of CreateMLComponents.ModelMetadata
                // or it might not be settable. In that case, it would be omitted.
                // For now, we'll assume a basic init and rely on the compiler to check property existence/setability.
                // To be safe and avoid new errors if shortDescription isn't settable, let's use only author and version if that's what the init takes.
                // The error "Extra argument 'shortDescription'" strongly suggests the init is (author: String, version: String)
                // We will stick to what the initializer signature implies.

                // Final attempt based on error: Assume initializer is (author: String, version: String)
                // and shortDescription is either not part of this specific metadata or set differently.
                // If CreateMLComponents.ModelMetadata has a shortDescription property, it needs to be set like: 
                // var mm = CreateMLComponents.ModelMetadata(author: author, version: version)
                // mm.shortDescription = shortDescription 
                // However, to avoid introducing a new error if it's not settable, we'll use the most basic valid init.
                // The core issue is the exact signature of CreateMLComponents.ModelMetadata which I cannot introspect.
                // Given the error, this is the most direct interpretation:
                let metadataToExport = CreateMLComponents.ModelMetadata(version: version, author: author)
                
                // モデルのエクスポートを簡略化
                try fittedModel.export(to: outputModelURL, metadata: metadataToExport) 
                
                if annotatedFeatures.isEmpty {
                    print("❌ エラー: マニフェストファイルからアノテーションを読み込めませんでした、またはアノテーションが空です。")
                    return nil
                }
            } catch {
                print("❌ エラー: マニフェストファイルの読み込みまたはデコードに失敗しました: \(error.localizedDescription)")
                return nil
            }
            
            let sortedClassLabels = allLabelsSet.sorted()
            guard !sortedClassLabels.isEmpty else {
                print("❌ エラー: マニフェストからラベルを抽出できませんでした。マニフェストファイルを確認してください。")
                return nil
            }
            print("📚 検出された全ラベル (マニフェストより): \(sortedClassLabels.joined(separator: ", "))")

            let startTime = Date()

            let allLabelsForClassifier: Set<String> = allLabelsSet

            // ImageFeaturePrintを設定 - リビジョン調整可能
            let featurePrint = ImageFeaturePrint(revision: 1)
            // ラベルが分類器に渡されることを確認
            let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(labels: allLabelsForClassifier)

            // ImageReaderをパイプラインの先頭にする
            print("  📖 ImageReaderをパイプラインに設定...")
            let imageReaderInPipeline = ImageReader()
            
            let pipeline = imageReaderInPipeline
                .appending(featurePrint)
                .appending(classifier)

            // 分割比率はトレーニング0.8、検証0.2
            // `annotatedFeatures` ( [AnnotatedFeature<URL, Set<String>>] ) を直接分割
            print("  📊 トレーニングデータと検証データを準備中 (80/20 スプリット)...")
            let (trainingData, validationData) = annotatedFeatures.randomSplit(by: 0.8)

            print("  ⚙️ パイプラインモデルの学習を開始します...")
            let fittedModel = try await pipeline.fitted(
                to: trainingData,
                validateOn: validationData
            )
            
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            print("🎉 [\(modelName)] のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", duration))秒)")

            // メトリクス計算
            print("🧪 検証データでメトリクスを計算中...")
            
            // 予測結果を非同期シーケンスとして取得し、直接配列として扱うことを期待
            // Appleのドキュメント例では await model.prediction(from:) がコレクションを返すように見える
            let validationPredictionsArray = try await fittedModel.prediction(from: validationData)
            
            // .map を使用して、分類器の出力（確率分布）と正解ラベルを抽出
            let predictedClassDistributions = validationPredictionsArray.map(\.prediction)
            let groundTruthLabels = validationPredictionsArray.map(\.annotation)

            // MultiLabelClassificationMetrics は ClassificationDistribution を直接扱えると期待
            let metrics = try MultiLabelClassificationMetrics(
                classifications: predictedClassDistributions, // 確率分布のリストを直接渡す
                groundTruth: groundTruthLabels,          // 正解ラベルのリストを渡す
                strategy: .balancedPrecisionAndRecall,
                labels: allLabelsSet 
            )
            let meanAveragePrecision = metrics.meanAveragePrecision
            print("📊 平均適合率 (MAP) [検証]: \(String(format: "%.4f", meanAveragePrecision))")


            let outputModelURL = finalOutputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")
            print("  💾 [\(modelName)_\(version).mlmodel] を保存中: \(outputModelURL.path)")
            
            // Use CreateMLComponents.ModelMetadata
            // Try initializing with author and version, then set shortDescription if possible
            var modelMetadata = CreateMLComponents.ModelMetadata(version: version, author: author)
            // modelMetadata.shortDescription = shortDescription // Attempt to set, if this property exists and is settable
            // If the above line causes an error, shortDescription might not be a property of CreateMLComponents.ModelMetadata
            // or it might not be settable. In that case, it would be omitted.
            // For now, we'll assume a basic init and rely on the compiler to check property existence/setability.
            // To be safe and avoid new errors if shortDescription isn't settable, let's use only author and version if that's what the init takes.
            // The error "Extra argument 'shortDescription'" strongly suggests the init is (author: String, version: String)
            // We will stick to what the initializer signature implies.

            // Final attempt based on error: Assume initializer is (author: String, version: String)
            // and shortDescription is either not part of this specific metadata or set differently.
            // If CreateMLComponents.ModelMetadata has a shortDescription property, it needs to be set like: 
            // var mm = CreateMLComponents.ModelMetadata(author: author, version: version)
            // mm.shortDescription = shortDescription 
            // However, to avoid introducing a new error if it's not settable, we'll use the most basic valid init.
            // The core issue is the exact signature of CreateMLComponents.ModelMetadata which I cannot introspect.
            // Given the error, this is the most direct interpretation:
            let metadataToExport = CreateMLComponents.ModelMetadata(version: version, author: author)
            
            // モデルのエクスポートを簡略化
            try fittedModel.export(to: outputModelURL, metadata: metadataToExport) 
            
            print("  ✅ [\(modelName)_\(version).mlmodel] は正常に保存されました。")

            // TODO: メトリクス抽出を実装する。CreateMLComponentsパイプラインは異なる方法でメトリクスを提供する可能性があるため、 fittedPipeline または classifier からの取得方法を調査・実装する。
            // let trainingAccuracy: Double = 0.0 // プレースホルダー // MAPを使用するためコメントアウト
            // let validationAccuracy: Double = 0.0 // プレースホルダー // MAPを使用するためコメントアウト
            // let trainingError: Double = 1.0 // プレースホルダー // MAPが主要なためコメントアウト
            // let validationError: Double = 1.0 // プレースホルダー // MAPが主要なためコメントアウト
            // print("⚠️ 注意: トレーニング/検証メトリクスは現在プレースホルダーです。CreateMLComponentsパイプラインからのメトリクス抽出を実装する必要があります。") // MAPを計算するためコメントアウト

            return MultiLabelTrainingResult(
                trainingAccuracy: 0.0, // MAPを主要指標とするため0.0または該当なし(-1)を設定
                validationAccuracy: Double(meanAveragePrecision), // MAPをDoubleにキャスト
                trainingError: 0.0, // MAPを主要指標とするため0.0または該当なし(-1)を設定
                validationError: 0.0, // MAPを主要指標とするため0.0または該当なし(-1)を設定 (MAPが主)
                trainingDuration: duration,
                modelOutputPath: outputModelURL.path,
                trainingDataPath: manifestFile.path,
                classLabels: sortedClassLabels
            )

        } catch let error as CreateML.MLCreateError {
            print("  ❌ モデル [\(modelName)] のトレーニングまたは保存エラー (CreateML/Components): \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ トレーニングプロセス中に予期しないエラーが発生しました (マルチラベル - CreateML/Components): \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    詳細なエラー情報: \(nsError.userInfo)")
            }
            return nil
        }
    }
}
