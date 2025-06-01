import Foundation
import XCTest

enum TestUtils {
    static func getRandomImageURL(
        forClassLabel classLabel: String,
        resourcesDirectoryPath: String,
        fileManager: FileManager
    ) throws -> URL {
        let resourceURL = URL(fileURLWithPath: resourcesDirectoryPath)
        let classLabelURL = resourceURL.appendingPathComponent(classLabel)

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: classLabelURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            let message = "サブディレクトリ '\(classLabel)' が見つからないか、ディレクトリではありません: \(classLabelURL.path)"
            XCTFail(message)
            throw ClassifierTestsError.testResourceMissing
        }

        let validExtensions = ["jpg", "jpeg", "png"]
        let allFiles = try fileManager.contentsOfDirectory(
            at: classLabelURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ).filter { url in
            validExtensions.contains(url.pathExtension.lowercased())
        }

        guard !allFiles.isEmpty else {
            throw ClassifierTestsError.testResourceMissing
        }

        guard let randomFile = allFiles.randomElement() else {
            XCTFail("利用可能な画像ファイルが見つかりません")
            throw ClassifierTestsError.testResourceMissing
        }

        return randomFile
    }
} 