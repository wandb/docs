---
title: 散布図
menu:
  default:
    identifier: ja-guides-models-app-features-panels-scatter-plot
    parent: panels
weight: 40
---

このページでは、W&B で散布図を使う方法を紹介します。

## ユースケース

散布図を使って、複数の Runs を比較し、Experiments のパフォーマンスを可視化できます。

- 最小値、最大値、平均値の線を描画する
- メタデータのツールチップをカスタマイズする
- ポイントの色を調整する
- 軸の範囲を調整する
- 軸をログスケールに設定する

## 例

次の例では、数週間にわたる実験で複数 Models の検証精度を示す散布図を表示しています。ツールチップにはバッチサイズやドロップアウト、軸の値が含まれています。さらに、検証精度の移動平均を示すラインも表示されています。

[ライブの例を見る →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

{{< img src="/images/general/scatter-plots-1.png" alt="検証精度の散布図" >}}

## 散布図を作成する

W&B UI で散布図を作成するには、次の手順を行ってください:

1. **Workspaces** タブに移動します。
2. **Charts** パネルで、アクションメニュー `...` をクリックします。
3. ポップアップメニューから **Add panels** を選択します。
4. **Add panels** メニューで、**Scatter plot** を選択します。
5. `x` 軸と `y` 軸に可視化したいデータを設定します。必要に応じて、軸の最大値・最小値を指定したり、`z` 軸を追加できます。
6. **Apply** をクリックして散布図を作成します。
7. Charts パネルに新しい散布図が表示されます。