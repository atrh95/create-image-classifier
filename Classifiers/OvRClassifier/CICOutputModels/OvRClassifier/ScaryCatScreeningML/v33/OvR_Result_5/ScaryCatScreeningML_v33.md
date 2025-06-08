# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-06-09 01:05:38 +0900
最大反復回数     : 15
データ拡張       : ImageAugmentationOptions(rawValue: 18)
特徴抽出器       : ScenePrint - Logistic Regressor

## トレーニング設定
使用されたクラスラベル : mouth_open (219枚), human_hands_detected (263枚), sphynx (185枚), black_and_white (297枚), safe (178枚)
## 個別モデルの性能指標
| クラス | 訓練正解率 | 検証正解率 | 再現率 | 適合率 | F1スコア |
|--------|------------|------------|--------|--------|----------|
| sphynx | 98.9% | 100.0% | 100.0% | 100.0% | 1.000 |
| safe | 100.0% | 88.9% | 100.0% | 81.8% | 0.900 |
| human_hands_detected | 97.8% | 100.0% | 100.0% | 100.0% | 1.000 |
| black_and_white | 99.8% | 100.0% | 100.0% | 100.0% | 1.000 |
| mouth_open | 97.6% | 81.8% | 72.7% | 88.9% | 0.800 |

## モデルメタデータ
作成者            : akitora
バージョン          : v33