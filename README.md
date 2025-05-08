# TrainCatScreeningML

## 概要

Core ML Create を使用して画像分類モデル (`.mlmodel`) を作成するSwiftコマンドラインプロジェクトです。

## プロジェクト構成

*   `TrainCatScreeningML/main.swift`: メインのトレーニング実行スクリプト。
*   `BinaryClassification/`: 2項分類タスク関連のモジュール。
    *   `Sources/`: トレーニングロジック。
    *   `Resources/`: 学習用画像データセット。
    *   `OutputModels/`: 学習済み `.mlmodel` ファイルの出力先。
*   `Package.swift`: プロジェクト定義と依存関係。

## セットアップ

プロジェクトの依存関係は `Package.swift` で管理されています。
外部ライブラリを追加・変更した場合は、ターミナルで `swift package resolve` を実行してください。
詳細は `Package.swift` をご確認ください。

## ディレクトリ構造 (主要部分)

```bash
.
├── BinaryClassification
│   ├── OutputModels
│   ├── Resources
│   │   └── ScaryCatScreenerData # データセット例
│   └── Sources
└── TrainCatScreeningML
```
*(注: `CatScreeningML.playground` ディレクトリは古い構成のものです)*

## トレーニングの実行

1.  ターミナルでプロジェクトのルートディレクトリに移動します。
2.  コマンド `swift run TrainCatScreeningML` を実行します。
    (またはXcodeで `Package.swift` を開き、`TrainCatScreeningML` スキームを実行)
3.  トレーニングの進捗や結果はコンソールに出力されます。

## トレーニング設定

*   モデルのメタデータ (作成者、バージョン等) は `TrainCatScreeningML/main.swift` で設定します。
*   トレーニング固有のパラメータ (モデル名、データパス、学習パラメータ等) は、主に `BinaryClassification/Sources/` 内のトレーナークラス (例: `ScaryCatScreenerTrainer.swift`) で定義されています。適宜編集してください。

## トレーニングデータ

学習データは `BinaryClassification/Resources/{データセット名}/{クラス名}/` の形式で配置してください。
(例: `BinaryClassification/Resources/ScaryCatScreenerData/Scary/`)
データがプロジェクトに正しくバンドルされるよう、必要に応じて `Package.swift` の `resources` 設定もご確認ください。

## 出力モデルの使用

トレーニングが成功すると、`BinaryClassification/OutputModels/result_N/` ディレクトリに `.mlmodel` ファイルが生成されます (Nは実行ごとの番号)。
このモデルファイルを、推論を行うアプリケーションやライブラリで使用してください。

## 精度改善

モデルの精度を改善するための一般的なヒント (データの量や質、ハイパーパラメータ調整など) は、機械学習のベストプラクティスを参照してください。