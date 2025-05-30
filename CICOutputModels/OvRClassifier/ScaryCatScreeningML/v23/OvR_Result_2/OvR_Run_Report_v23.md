# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-30 21:54:39 +0900
最大反復回数     : 11
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 2)

## トレーニング設定
使用されたクラスラベル : sphynx, safe, human_hands_detected, black_and_white, mouth_open

## パフォーマンス指標 (全体)
トレーニング所要時間: 15.27 秒
トレーニング誤分類率 (学習時) : 0.00%
訓練データ正解率 (学習時) : 1.00%
検証データ正解率 (学習時自動検証) : 0.90%
検証誤分類率 (学習時自動検証) : 10.00%## クラス別性能指標

### black_and_white
再現率: 100.0%, 適合率: 92.9%, F1スコア: 96.3%

### human_hands_detected
再現率: 90.0%, 適合率: 90.0%, F1スコア: 90.0%

### mouth_open
再現率: 57.1%, 適合率: 100.0%, F1スコア: 72.7%

### safe
再現率: 100.0%, 適合率: 75.0%, F1スコア: 85.7%

### sphynx
再現率: 100.0%, 適合率: 100.0%, F1スコア: 100.0%

## 混同行列
Actual\Predicted	black_and_white	human_hands_detected	mouth_open	safe	sphynx
black_and_white	13	0	0	0	0
human_hands_detected	0	9	0	1	0
mouth_open	1	1	4	1	0
safe	0	0	0	6	0
sphynx	0	0	0	0	4

## 個別モデルの性能指標


## モデルメタデータ
作成者            : akitora
バージョン          : v23