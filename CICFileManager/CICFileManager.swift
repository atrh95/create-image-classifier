import Foundation

public class CICFileManager {
    private let fileManager: Foundation.FileManager
    
    public init(fileManager: Foundation.FileManager = .default) {
        self.fileManager = fileManager
    }
    
    public func createOutputDirectory(
        modelName: String,
        version: String,
        classificationMethod: String,
        moduleOutputPath: String
    ) throws -> URL {
        let baseDirURL = URL(fileURLWithPath: moduleOutputPath)
            .appendingPathComponent(modelName)
            .appendingPathComponent(version)

        var resultNumber = 1

        // 既存のディレクトリを確認して次の番号を決定
        do {
            let contents = try fileManager.contentsOfDirectory(at: baseDirURL, includingPropertiesForKeys: nil)
            let existingNumbers = contents.compactMap { url -> Int? in
                let dirName = url.lastPathComponent
                guard dirName.hasPrefix("\(classificationMethod)_Result_") else { return nil }
                let numberStr = dirName.replacingOccurrences(of: "\(classificationMethod)_Result_", with: "")
                return Int(numberStr)
            }

            if let maxNumber = existingNumbers.max() {
                resultNumber = maxNumber + 1
            }
        } catch {
            // ディレクトリが存在しない場合は1から開始
            resultNumber = 1
        }

        let outputDirURL = baseDirURL.appendingPathComponent("\(classificationMethod)_Result_\(resultNumber)")

        try fileManager.createDirectory(
            at: outputDirURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        return outputDirURL
    }
    
    public func getClassLabelDirectories(resourcesPath: String) throws -> [URL] {
        let resourcesDirURL = URL(fileURLWithPath: resourcesPath)
        return try fileManager.contentsOfDirectory(
            at: resourcesDirURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }
} 