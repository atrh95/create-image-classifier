import XCTest
@testable import CICFileManager

final class CICFileManagerTests: XCTestCase {
    private var sut: CICFileManager!
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        sut = CICFileManager()
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        sut = nil
        tempDir = nil
        super.tearDown()
    }

    // 新規モデルの出力ディレクトリが正しく作成されることを確認
    func testCreateOutputDirectory() throws {
        let outputDir = try sut.createOutputDirectory(
            modelName: "TestModel",
            version: "1.0",
            classificationMethod: "Binary",
            moduleOutputPath: tempDir.path
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputDir.path))
        XCTAssertEqual(outputDir.lastPathComponent, "Binary_Result_1")
    }

    // 既存の実行がある場合、次の番号のディレクトリが作成されることを確認
    func testCreateOutputDirectoryWithExistingRuns() throws {
        // 既存の実行ディレクトリを作成
        let existingDir = tempDir
            .appendingPathComponent("TestModel")
            .appendingPathComponent("1.0")
            .appendingPathComponent("Binary_Result_1")
        try FileManager.default.createDirectory(at: existingDir, withIntermediateDirectories: true)

        let outputDir = try sut.createOutputDirectory(
            modelName: "TestModel",
            version: "1.0",
            classificationMethod: "Binary",
            moduleOutputPath: tempDir.path
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputDir.path))
        XCTAssertEqual(outputDir.lastPathComponent, "Binary_Result_2")
    }

    // クラスラベルディレクトリのみが取得され、隠しディレクトリが除外されることを確認
    func testGetClassLabelDirectories() throws {
        // テスト用のディレクトリ構造を作成
        let resourcesDir = tempDir.appendingPathComponent("Resources")
        try FileManager.default.createDirectory(at: resourcesDir, withIntermediateDirectories: true)

        // クラスラベルディレクトリを作成
        let class1Dir = resourcesDir.appendingPathComponent("Class1")
        let class2Dir = resourcesDir.appendingPathComponent("Class2")
        let hiddenDir = resourcesDir.appendingPathComponent(".hidden")
        try FileManager.default.createDirectory(at: class1Dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: class2Dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: hiddenDir, withIntermediateDirectories: true)

        let classDirs = try sut.getClassLabelDirectories(resourcesPath: resourcesDir.path)

        XCTAssertEqual(classDirs.count, 2)
        XCTAssertTrue(classDirs.contains { $0.lastPathComponent == "Class1" })
        XCTAssertTrue(classDirs.contains { $0.lastPathComponent == "Class2" })
        XCTAssertFalse(classDirs.contains { $0.lastPathComponent == ".hidden" })
    }
} 