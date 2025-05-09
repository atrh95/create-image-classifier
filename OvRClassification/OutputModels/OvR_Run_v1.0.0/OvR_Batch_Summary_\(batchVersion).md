# OvR Batch Classification Report

## Batch Details
- **Batch Coordinator Model Name**: \(trainer.modelName) 
- **Batch Version**: \(batchVersion)
- **Report Generated**: \(generatedDateString)
- **Total OvR Models Trained**: \(individualResults.count)
- **Main Output Directory**: \(mainOutputDirectoryPath)

## Model Metadata (for the batch)
- **Author**: \(modelAuthor)
- **Description**: \(modelDescription)
- **Version**: \(modelVersion)

## Individual OvR Model Summaries
---
- **Target Label (One)**: \(result.oneLabelName)
  - **Model Name**: \(result.modelName)
  - **Model Path**: \(relativeModelPath)
  - **Report Path**: \(relativeReportPath)
  - **Training Accuracy**: \(result.trainingAccuracy != nil ? String(format: "%.2f", result.trainingAccuracy! * 100) : "N/A")%
  - **Validation Accuracy**: \(result.validationAccuracy != nil ? String(format: "%.2f", result.validationAccuracy! * 100) : "N/A")%
---
- **Target Label (One)**: \(result.oneLabelName)
  - **Model Name**: \(result.modelName)
  - **Model Path**: \(relativeModelPath)
  - **Report Path**: \(relativeReportPath)
  - **Training Accuracy**: \(result.trainingAccuracy != nil ? String(format: "%.2f", result.trainingAccuracy! * 100) : "N/A")%
  - **Validation Accuracy**: \(result.validationAccuracy != nil ? String(format: "%.2f", result.validationAccuracy! * 100) : "N/A")%
---
- **Target Label (One)**: \(result.oneLabelName)
  - **Model Name**: \(result.modelName)
  - **Model Path**: \(relativeModelPath)
  - **Report Path**: \(relativeReportPath)
  - **Training Accuracy**: \(result.trainingAccuracy != nil ? String(format: "%.2f", result.trainingAccuracy! * 100) : "N/A")%
  - **Validation Accuracy**: \(result.validationAccuracy != nil ? String(format: "%.2f", result.validationAccuracy! * 100) : "N/A")%
---
- **Target Label (One)**: \(result.oneLabelName)
  - **Model Name**: \(result.modelName)
  - **Model Path**: \(relativeModelPath)
  - **Report Path**: \(relativeReportPath)
  - **Training Accuracy**: \(result.trainingAccuracy != nil ? String(format: "%.2f", result.trainingAccuracy! * 100) : "N/A")%
  - **Validation Accuracy**: \(result.validationAccuracy != nil ? String(format: "%.2f", result.validationAccuracy! * 100) : "N/A")%
