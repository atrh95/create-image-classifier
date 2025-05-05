# train-cat-screening-ml

Swift Playground と画像データセットを使用して、猫スクリーニング用の機械学習モデルをトレーニングします。

## 概要

このリポジトリには、Core ML Create を使って画像分類モデル (`.mlmodel`) をトレーニングするための Swift Playground が含まれています。

## 内容物

*   `CatScreeningML.playground`: モデルトレーニング用の Playground。
    *   `Contents.swift`: メインのトレーニングスクリプト。
    *   `Sources/`: トレーナー、ロガーなどの補助コード。
    *   `Resources/`: トレーニング用画像データセット (`ScaryCatScreenerData` など)。
*   `OutputModels/`: トレーニング済み `.mlmodel` ファイルの出力先ディレクトリ。

## セットアップ (依存関係)

この Playground が `CatScreeningKit` のコード (例: `CSKShared`) を参照する場合、以下の手順で依存関係を設定します (推奨)。

1.  リポジトリのルートに、以下の内容で `Package.swift` ファイルを作成します。
    ```swift
    // swift-tools-version:6.0
    import PackageDescription

    let package = Package(
        name: "TrainingEnvironment",
        platforms: [.macOS(.v14)],
        dependencies: [
            // 必要に応じてバージョンやブランチを調整
            .package(url: "https://github.com/terrio32/cat-screening-kit.git", from: "1.0.0")
        ],
        targets: [
            // 依存関係解決のためのダミーターゲット
            .target(name: "DummyTarget", dependencies: [])
        ]
    )
    ```
2.  ターミナルでリポジトリのルートディレクトリに移動し、`xed .` を実行して Xcode でパッケージとして開きます。
3.  Xcode のファイルナビゲータから `CatScreeningML.playground` を開きます。これで `import CSKShared` などが可能になります。

*注: Playground の設定でローカルパスを直接指定する方法もありますが、管理が複雑になるため推奨されません。*

## ディレクトリ構成

```bash
.
├── CatScreeningML.playground
│   ├── Contents.swift             # メインのトレーニングスクリプト
│   ├── contents.xcplayground
│   ├── playground.xcworkspace
│   ├── Resources
│   │   └── ScaryCatScreenerData   # 画像データセット
│   │       ├── Not Scary
│   │       └── Scary
│   └── Sources
│       ├── AccuracyImprovementTips.md # 精度改善Tips
│       ├── ScaryCatScreenerTrainer.swift # トレーナ実装
│       ├── ScreeningTrainerProtocol.swift # トレーナプロトコル
│       ├── TrainingResult.swift         # 結果モデル
│       └── TrainingResultLogger.swift   # ロガー
├── OutputModels                       # トレーニング済みモデルの出力先（git管理外推奨）
├── Package.swift                      # 依存関係定義
└── README.md
```

## Playground の実行

1.  Xcode で `CatScreeningML.playground` を開きます (上記セットアップ後)。
2.  Playground の実行ボタン (▶︎) をクリックします。
3.  `Contents.swift` が `Sources` 内のトレーナー (例: `ScaryCatScreenerTrainer`) を使用してトレーニングを開始します。
4.  進捗、結果、エラー、モデルの保存場所は Xcode のコンソールに出力されます。

## トレーニング設定の変更 (`Contents.swift`)

*   **パスと名前:**
    *   `modelName`: 出力する `.mlmodel` ファイル名。
    *   `dataDirectoryName`: `Resources` 内のトレーニングデータディレクトリ名 (例: `ScaryCatScreenerData`)。
    *   `customOutputDirPath`: モデル出力先 (デフォルト: `./OutputModels`)。
*   **トレーニングパラメータ:**
    *   `executeTrainingCore` 内の `MLImageClassifier.ModelParameters` でデータ拡張 (`augmentation`) や最大反復回数 (`maxIterations`) などを設定します。
*   **モデルメタデータ:**
    *   `author`, `shortDescription`, `version` などのメタデータを設定します。これらは `.mlmodel` に埋め込まれます。

## トレーニングデータの管理

1.  `CatScreeningML.playground/Resources/` 内に、モデル用のデータディレクトリ (例: `ScaryCatScreenerData`) を作成または編集します。
2.  データディレクトリ内に、分類クラス名のサブディレクトリ (例: `Scary`, `Not Scary`) を作成します。
3.  各クラスディレクトリに、対応するトレーニング画像 (.jpg, .png など) を追加します。

## 出力モデルの使用

1.  トレーニングが成功すると、`customOutputDirPath` 内の `result_N` ディレクトリに `.mlmodel` ファイルが生成されます。
2.  生成された `.mlmodel` を `CatScreeningKit` リポジトリの `Sources/Screeners/該当スクリーナ/Resources/` ディレクトリにコピーします。
3.  `CatScreeningKit` リポジトリで変更をコミット＆プッシュします。

## 精度改善

モデルの精度を改善するためのヒントは、[AccuracyImprovementTips.md](CatScreeningML.playground/Sources/AccuracyImprovementTips.md) を参照してください。