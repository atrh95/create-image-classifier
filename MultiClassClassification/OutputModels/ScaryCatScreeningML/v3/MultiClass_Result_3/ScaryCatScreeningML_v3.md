# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 13:36:30 +0900
最大反復回数     : 8
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## トレーニング設定
使用されたクラスラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 2.98 秒
トレーニング誤分類率 (学習時) : 19.47%
訓練データ正解率 (学習時) : 0.81%
検証データ正解率 (学習時自動検証) : 0.75%
検証誤分類率 (学習時自動検証) : 25.00%## クラス別性能指標
【black_and_white】
再現率: 60.0%, 適合率: 60.0%, F1スコア: 60.0%
【human_hands_detected】
再現率: 100.0%, 適合率: 100.0%, F1スコア: 100.0%
【mouth_open】
再現率: 50.0%, 適合率: 66.7%, F1スコア: 57.1%
【sphynx】
再現率: 100.0%, 適合率: 80.0%, F1スコア: 88.9%

## 混同行列
Actual\Predicted	black_and_white	human_hands_detected	mouth_open	sphynx
black_and_white	3	0	1	1
human_hands_detected	0	3	0	0
mouth_open	2	0	2	0
sphynx	0	0	0	4

## モデルメタデータ
作成者            : akitora
バージョン          : v3