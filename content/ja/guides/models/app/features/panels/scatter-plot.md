---
title: Scatter plots
menu:
  default:
    identifier: ja-guides-models-app-features-panels-scatter-plot
    parent: panels
weight: 40
---

散布図を使用すると、複数の run を比較し、実験のパフォーマンスを視覚化できます。いくつかのカスタマイズ可能な機能を追加しました。

1. 最小値、最大値、平均値に沿って線をプロット
2. カスタム メタデータ ツールチップ
3. コントロール ポイントの色
4. 軸の範囲を設定
5. 軸を対数スケールに切り替え

これは、数週間の Experiments におけるさまざまな Models の検証精度の例です。ツールチップは、軸上の値だけでなく、バッチサイズとドロップアウトを含むようにカスタマイズされています。また、検証精度の移動平均をプロットする線もあります。
[ライブの例を見る →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

{{< img src="/images/general/scatter-plots-1.png" alt="" >}}
