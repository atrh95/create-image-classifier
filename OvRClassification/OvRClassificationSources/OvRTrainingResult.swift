import Foundation
import SCSInterface

public struct OvRTrainingResult: TrainingResultData {
    public let modelName: String
    public let modelOutputPath: String
    public let reportPath: String
    public let oneLabelName: String
    public let restLabelNames: [String]
    public let positiveSamplesCount: Int
    public let negativeSamplesCount: Int
    public let trainingAccuracy: Double
    public let validationAccuracy: Double
    public let trainingError: Double
    public let validationError: Double
    public let trainingDuration: TimeInterval
    public let trainingDataPath: String

    public var classLabels: [String] {
        return ([oneLabelName] + restLabelNames).sorted()
    }

    public func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let classLabelsString = self.classLabels.joined(separator: ", ")
        let restLabelsSummary = restLabelNames.isEmpty ? "N/A" : restLabelNames.sorted().joined(separator: ", ")

        let trainingAccStr = String(format: "%.2f", trainingAccuracy * 100)
        let validationAccStr = String(format: "%.2f", validationAccuracy * 100)
        let trainingErrStr = String(format: "%.2f", trainingError * 100)
        let validationErrStr = String(format: "%.2f", validationError * 100)
        let durationStr = String(format: "%.2f", trainingDuration)

        let reportTitleOneLabelName = oneLabelName
        let specificModelDescription = "\(modelDescription) — Binary classification of '\(reportTitleOneLabelName)' versus all other classes."

        let reportContent = """
        # OvR Classification Report: \(reportTitleOneLabelName) vs Rest
        
        ## Model Details
        - **Individual Model Name**: \(self.modelName) 
        - **Saved Model Path**: \(self.modelOutputPath)
        - **Report Generated**: \(generatedDateString)
        
        ## Training Configuration
        - **Target Label (One)**: \(reportTitleOneLabelName) (\(positiveSamplesCount) samples)
        - **Rest Labels**: \(restLabelsSummary) (\(negativeSamplesCount) samples total)
        - **All Labels Considered for this OvR pair**: \(classLabelsString)
        
        ## Performance Metrics
        - **Training Duration**: \(durationStr) seconds
        - **Training Accuracy**: \(trainingAccStr)%
        - **Training Error**: \(trainingErrStr)%
        - **Validation Accuracy**: \(validationAccStr)%
        - **Validation Error**: \(validationErrStr)%
        
        ## Model Metadata (as written to .mlmodel)
        - **Author**: \(modelAuthor)
        - **Description**: \(specificModelDescription) 
        - **Version**: \(modelVersion)
        
        *This report describes the training of a binary classifier to distinguish '\(reportTitleOneLabelName)' from all other labels.*
        """
        do {
            try reportContent.write(to: URL(fileURLWithPath: self.reportPath), atomically: true, encoding: .utf8)
            print("  ✅ 個別OvRレポートを保存しました: \\(self.reportPath)")
        } catch {
            print("  ❌ 個別OvRレポートの保存エラー (\\(self.modelName)): \\(error.localizedDescription)")
        }
    }
}

public struct OvRBatchResult {
    public let batchVersion: String
    public let individualResults: [OvRTrainingResult]
    public let mainOutputDirectoryPath: String

    public func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        var summaryContent = """
        # OvR Batch Classification Report
        
        ## Batch Details
        - **Batch Coordinator Model Name**: \\(trainer.modelName) 
        - **Batch Version**: \\(batchVersion)
        - **Report Generated**: \\(generatedDateString)
        - **Total OvR Models Trained**: \\(individualResults.count)
        - **Main Output Directory**: \\(mainOutputDirectoryPath)

        ## Model Metadata (for the batch)
        - **Author**: \\(modelAuthor)
        - **Description**: \\(modelDescription)
        - **Version**: \\(modelVersion)
        
        ## Individual OvR Model Summaries
        
        """

        if individualResults.isEmpty {
            summaryContent += "このバッチで正常にトレーニングされた個別のOvRモデルはありませんでした。\\n"
        } else {
            for result in individualResults {
                let relativeModelPath = result.modelOutputPath.replacingOccurrences(of: mainOutputDirectoryPath, with: ".")
                let relativeReportPath = result.reportPath.replacingOccurrences(of: mainOutputDirectoryPath, with: ".")
                summaryContent += """
                ---
                - **Target Label (One)**: \\(result.oneLabelName)
                  - **Model Name**: \\(result.modelName)
                  - **Model Path**: \\(relativeModelPath)
                  - **Report Path**: \\(relativeReportPath)
                  - **Training Accuracy**: \\(result.trainingAccuracy != nil ? String(format: "%.2f", result.trainingAccuracy! * 100) : "N/A")%
                  - **Validation Accuracy**: \\(result.validationAccuracy != nil ? String(format: "%.2f", result.validationAccuracy! * 100) : "N/A")%
                
                """
            }
        }
        
        let summaryReportFileName = "OvR_Batch_Summary_\\(batchVersion).md"
        let summaryReportURL = URL(fileURLWithPath: mainOutputDirectoryPath).appendingPathComponent(summaryReportFileName)

        do {
            try summaryContent.write(to: summaryReportURL, atomically: true, encoding: .utf8)
            print("✅ OvRバッチ概要レポートを保存しました: \\(summaryReportURL.path)")
        } catch {
            print("❌ OvRバッチ概要レポートの保存エラー: \\(error.localizedDescription)")
        }
    }
}
