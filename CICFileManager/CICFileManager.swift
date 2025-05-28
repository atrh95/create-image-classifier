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
        let baseDir = URL(fileURLWithPath: moduleOutputPath)
            .appendingPathComponent(modelName)
            .appendingPathComponent(version)

        // 親ディレクトリが存在しない場合は作成
        if !fileManager.fileExists(atPath: baseDir.path) {
            try fileManager.createDirectory(at: baseDir, withIntermediateDirectories: true)
        }

        // 既存の実行を確認
        var runIndex = 1
        var finalOutputDir = baseDir.appendingPathComponent("\(classificationMethod)_Result_\(runIndex)")

        // 既存のディレクトリを確認
        while fileManager.fileExists(atPath: finalOutputDir.path) {
            runIndex += 1
            finalOutputDir = baseDir.appendingPathComponent("\(classificationMethod)_Result_\(runIndex)")
        }

        // 最終的なディレクトリを作成
        try fileManager.createDirectory(at: finalOutputDir, withIntermediateDirectories: true)
        return finalOutputDir
    }

    public func getClassLabelDirectories(resourcesPath: String) throws -> [URL] {
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        return try fileManager.contentsOfDirectory(
            at: resourcesDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }

    public func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return !isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }
}
