public protocol TrainingResultProtocol {
    func saveLog(
        modelAuthor: String,
        modelName: String,
        modelDescription: String,
        modelVersion: String
    )
}
