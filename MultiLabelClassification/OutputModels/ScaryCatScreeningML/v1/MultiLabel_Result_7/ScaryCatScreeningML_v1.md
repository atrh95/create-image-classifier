# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 14:18:35 +0900
最大反復回数     : 8 (注: CreateMLComponentsでは直接使用されません)
データ拡張       : なし
特徴抽出器       : ImageFeaturePrint

## トレーニング設定
アノテーションファイル: multilabel_cat_annotations.json
検出された全ラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 2.25 秒
トレーニング誤分類率 (学習時) : 100.00%
訓練データ正解率 (学習時) : 0.00%
検証データ正解率 (学習時自動検証) : 0.54%
検証誤分類率 (学習時自動検証) : 45.90%
## ラベル別性能指標
### black_and_white
再現率: 66.7%, 適合率: 92.3%, F1スコア: 77.4%

### human_hands_detected
再現率: 50.0%, 適合率: 66.7%, F1スコア: 57.1%

### mouth_open
再現率: 0.0%, 適合率: 0.0%, F1スコア: 0.0%

### sphynx
再現率: 69.2%, 適合率: 100.0%, F1スコア: 81.8%

## 混同行列
Label	True Positives	Total Actual	Precision	Recall	F1 Score
black_and_white	12	18	0.92	0.67	0.77
human_hands_detected	6	12	0.67	0.50	0.57
mouth_open	0	17	0.00	0.00	0.00
sphynx	9	13	1.00	0.69	0.82


## モデルメタデータ
作成者            : akitora
バージョン          : v1