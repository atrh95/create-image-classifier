import Foundation

public final class CICFileManager: FileManager {
    override public init() {
        super.init()
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
        if !fileExists(atPath: baseDir.path) {
            try createDirectory(at: baseDir, withIntermediateDirectories: true)
        }

        // 既存の実行を確認
        var runIndex = 1
        var finalOutputDir = baseDir.appendingPathComponent("\(classificationMethod)_Result_\(runIndex)")

        // 既存のディレクトリを確認
        while fileExists(atPath: finalOutputDir.path) {
            runIndex += 1
            finalOutputDir = baseDir.appendingPathComponent("\(classificationMethod)_Result_\(runIndex)")
        }

        // 最終的なディレクトリを作成
        try createDirectory(at: finalOutputDir, withIntermediateDirectories: true)
        return finalOutputDir
    }

    public func getClassLabelDirectories(resourcesPath: String) throws -> [URL] {
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        return try contentsOfDirectory(
            at: resourcesDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }

    public func getFilesInDirectory(_ directoryURL: URL) throws -> [URL] {
        try contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileExists(atPath: url.path, isDirectory: &isDirectory)
            return !isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }
    }

    // 最小枚数に揃えた画像セットを準備する
    public func prepareEqualizedMinimumImageSet(
        classDirs: [URL],
        shouldEqualize: Bool
    ) throws -> [String: URL] {
        // 1. 各クラスの画像ファイルを取得
        var classFiles: [String: [URL]] = [:]
        for classDir in classDirs {
            let files = try getFilesInDirectory(classDir)
            classFiles[classDir.lastPathComponent] = files
        }

        // 2. 最小枚数を計算
        let minCount = shouldEqualize ? (classFiles.values.map(\.count).min() ?? 0) : (classFiles.values.map(\.count).max() ?? 0)

        // 3. 一時ディレクトリを作成
        let tempBaseDir = temporaryDirectory
            .appendingPathComponent("TempBalancedImages")
        if fileExists(atPath: tempBaseDir.path) {
            try removeItem(at: tempBaseDir)
        }
        try createDirectory(at: tempBaseDir, withIntermediateDirectories: true)

        // 4. 各クラスのサブディレクトリを作成し、画像をコピー
        var result: [String: URL] = [:]
        for (className, files) in classFiles {
            let tempClassDir = tempBaseDir.appendingPathComponent(className)
            try createDirectory(at: tempClassDir, withIntermediateDirectories: true)

            // ランダムに選択した画像をコピー
            let selectedFiles = shouldEqualize ? Array(files.shuffled().prefix(minCount)) : files
            for (index, file) in selectedFiles.enumerated() {
                let destination = tempClassDir.appendingPathComponent("\(index).\(file.pathExtension)")
                try copyItem(at: file, to: destination)
            }

            result[className] = tempClassDir
        }

        return result
    }
}
