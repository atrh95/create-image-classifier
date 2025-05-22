# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 14:05:45 +0900
最大反復回数     : 8 (注: CreateMLComponentsでは直接使用されません)
データ拡張       : なし
特徴抽出器       : ImageFeaturePrint

## トレーニング設定
元データ(マニフェスト): /Users/akitora.hayashi/iOS Projects/cat-screening-ml/MultiLabelClassification/Resources/multilabel_cat_annotations.json
検出された全ラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 2.14 秒
トレーニング誤分類率 (学習時) : 100.00%
訓練データ正解率 (学習時) : 0.00%
検証データ正解率 (学習時自動検証) : 0.59%
検証誤分類率 (学習時自動検証) : 40.83%
## ラベル別性能指標


## 混同行列
  ┌──────────────────────┬────────┬────────┐
  │                      │ 予測値    │ 実際値    │
  ├──────────────────────┼────────┼────────┤
  │ black_and_white      │     11 │     18 │
  │ human_hands_detected │      7 │     12 │
  │ mouth_open           │      1 │     17 │
  │ sphynx               │     10 │     13 │
  └──────────────────────┴────────┴────────┘

## モデルメタデータ
作成者            : akitora
バージョン          : v1