---
title: Scatter plots
menu:
  default:
    identifier: ja-guides-models-app-features-panels-scatter-plot
    parent: panels
weight: 40
---

散布図を使用して複数の run を比較し、実験のパフォーマンスを視覚化しましょう。以下のカスタマイズ可能な機能を追加しました：

1. 最小値、最大値、平均の線をプロット
2. カスタムメタデータのツールチップ
3. ポイントの色をコントロール
4. 軸範囲の設定
5. 軸をログスケールに切り替え

ここでは、数週間にわたる実験で異なるモデルの検証精度の例を示します。ツールチップは、バッチサイズとドロップアウトを含むようにカスタマイズされており、軸上の値も表示されます。また、検証精度の移動平均をプロットした線があります。  
[ライブ例を見る →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

{{< img src="/images/general/scatter-plots-1.png" alt="" >}}