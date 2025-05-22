# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 13:25:47 +0900
最大反復回数     : 8
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## トレーニング設定
使用されたクラスラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 2.98 秒
トレーニング誤分類率 (学習時) : 19.47%
訓練データ正解率 (学習時) : 80.53%
検証データ正解率 (学習時自動検証) : 31.25%
検証誤分類率 (学習時自動検証) : 68.75%## クラス別性能指標
【black_and_white】
再現率: 20.0% + 適合率: 16.7% + F1スコア: 18.2%
【human_hands_detected】
再現率: 33.3% + 適合率: 50.0% + F1スコア: 40.0%
【mouth_open】
再現率: 0.0% + 適合率: 0.0% + F1スコア: 0.0%
【sphynx】
再現率: 75.0% + 適合率: 100.0% + F1スコア: 85.7%

## 混同行列
Actual\Predicted	black_and_white	human_hands_detected	mouth_open	sphynx
black_and_white	1	0	4	0
human_hands_detected	1	1	1	0
mouth_open	3	1	0	0
sphynx	1	0	0	3

## モデルメタデータ
作成者            : akitora
バージョン          : v3