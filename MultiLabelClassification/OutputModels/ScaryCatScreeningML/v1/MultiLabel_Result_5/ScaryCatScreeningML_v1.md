# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 14:12:47 +0900
最大反復回数     : 8 (注: CreateMLComponentsでは直接使用されません)
データ拡張       : なし
特徴抽出器       : ImageFeaturePrint

## トレーニング設定
元データ(マニフェスト): /Users/akitora.hayashi/iOS Projects/cat-screening-ml/MultiLabelClassification/Resources/multilabel_cat_annotations.json
検出された全ラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 2.11 秒
トレーニング誤分類率 (学習時) : 100.00%
訓練データ正解率 (学習時) : 0.00%
検証データ正解率 (学習時自動検証) : 0.48%
検証誤分類率 (学習時自動検証) : 52.43%
## ラベル別性能指標

### black_and_white
再現率: 33.3%, 適合率: 85.7%, F1スコア: 48.0%
### human_hands_detected
再現率: 66.7%, 適合率: 61.5%, F1スコア: 64.0%
### mouth_open
再現率: 0.0%, 適合率: 0.0%, F1スコア: 0.0%
### sphynx
再現率: 69.2%, 適合率: 90.0%, F1スコア: 78.3%

## 混同行列
Label	True Positives	Total Actual	Precision	Recall	F1 Score
black_and_white	6	18	0.86	0.33	0.48
human_hands_detected	8	12	0.62	0.67	0.64
mouth_open	0	17	0.00	0.00	0.00
sphynx	9	13	0.90	0.69	0.78

## モデルメタデータ
作成者            : akitora
バージョン          : v1