---
title: 並列座標
description: 機械学習実験間の結果を比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

パラレル座標グラフは、多数のハイパーパラメーターとモデルメトリクスの関係をひと目で要約できます。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="Parallel coordinates plot" >}}

* **軸**: [`wandb.Run.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) からのさまざまなハイパーパラメーターや、[`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) で記録されたメトリクス。
* **線**: 各線は 1 つの Run を表します。線にマウスオーバーすると、その Run の詳細がツールチップで表示されます。現在のフィルターに一致するすべての線が表示されますが、目のアイコンをオフにするとグレーアウトされます。

## パラレル座標パネルの作成

1. ワークスペースのランディングページに移動します
2. **Add Panels** をクリックします
3. **Parallel coordinates** を選択します

## パネル設定

パネルの設定を行うには、パネル右上の編集ボタンをクリックします。

* **ツールチップ**: ホバーすると、各 Run の情報が凡例として表示されます
* **タイトル**: 軸タイトルを編集して、より読みやすくできます
* **勾配**: 好きなカラーレンジにグラデーションをカスタマイズできます
* **対数スケール**: 各軸ごとに対数スケール表示が設定できます
* **軸の反転**: 軸の向きを切り替えます — 精度（accuracy）と損失（loss）を両方カラムに持たせている場合などに便利です

[パラレル座標パネルをライブで操作する](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)