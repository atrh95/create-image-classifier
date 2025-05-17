# OvO (One-vs-One) トレーニング実行レポート

## 実行概要
モデル群         : OvOモデル群 (One-vs-One)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-18 00:29:49 +0900
総クラス数       : 4
総ペア数         : 6
最大反復回数     : 10 (各ペアモデル共通)
データ拡張       : なし (各ペアモデル共通)
特徴抽出器       : ScenePrint(revision: 1) (各ペアモデル共通)
## 個別ペアモデルのパフォーマンス指標
| ペアモデル名 (Class1 vs Class2) | 検証正解率 |
|---------------------------------|--------------|
| ScaryCatScreeningML_OvO_Sphynx_vs_HumanHandsDetected_v1 (Sphynx_vs_HumanHandsDetected) | 1.00% |
| ScaryCatScreeningML_OvO_Sphynx_vs_BlackAndWhite_v1 (Sphynx_vs_BlackAndWhite) | 0.94% |
| ScaryCatScreeningML_OvO_Sphynx_vs_MouthOpen_v1 (Sphynx_vs_MouthOpen) | 0.91% |
| ScaryCatScreeningML_OvO_HumanHandsDetected_vs_BlackAndWhite_v1 (HumanHandsDetected_vs_BlackAndWhite) | 0.88% |
| ScaryCatScreeningML_OvO_HumanHandsDetected_vs_MouthOpen_v1 (HumanHandsDetected_vs_MouthOpen) | 0.82% |
| ScaryCatScreeningML_OvO_BlackAndWhite_vs_MouthOpen_v1 (BlackAndWhite_vs_MouthOpen) | 0.90% |

## 共通メタデータ
作成者            : akitora
バージョン        : v1