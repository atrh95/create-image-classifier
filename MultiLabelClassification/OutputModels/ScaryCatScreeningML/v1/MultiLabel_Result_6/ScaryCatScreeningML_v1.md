# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 14:14:15 +0900
最大反復回数     : 8 (注: CreateMLComponentsでは直接使用されません)
データ拡張       : なし
特徴抽出器       : ImageFeaturePrint

## トレーニング設定
元データ(マニフェスト): /Users/akitora.hayashi/iOS Projects/cat-screening-ml/MultiLabelClassification/Resources/multilabel_cat_annotations.json
検出された全ラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 2.21 秒
トレーニング誤分類率 (学習時) : 100.00%
訓練データ正解率 (学習時) : 0.00%
検証データ正解率 (学習時自動検証) : 0.60%
検証誤分類率 (学習時自動検証) : 39.52%
## ラベル別性能指標
### black_and_white
再現率: 38.9%, 適合率: 100.0%, F1スコア: 56.0%
### human_hands_detected
再現率: 66.7%, 適合率: 66.7%, F1スコア: 66.7%
### mouth_open
再現率: 29.4%, 適合率: 33.3%, F1スコア: 31.2%
### sphynx
再現率: 84.6%, 適合率: 91.7%, F1スコア: 88.0%

## 混同行列
Label	True Positives	Total Actual	Precision	Recall	F1 Score
black_and_white	7	18	1.00	0.39	0.56
human_hands_detected	8	12	0.67	0.67	0.67
mouth_open	5	17	0.33	0.29	0.31
sphynx	11	13	0.92	0.85	0.88


## モデルメタデータ
作成者            : akitora
バージョン          : v1