---
title: 平行座標
description: 機械学習の実験を横断して結果を比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

Parallel coordinates チャートは、多数のハイパーパラメーターと モデル メトリクス の関係を一目で把握できるように要約して表示します。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="平行座標プロット" >}}

* **Axes**: [`wandb.Run.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) からのさまざまなハイパーパラメーターと、[`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) からのメトリクス。
* **Lines**: 各線は 1 つの run を表します。線にマウスオーバーすると、その run の詳細を示すツールチップが表示されます。現在のフィルターに一致するすべての線が表示されますが、目のアイコンをオフにすると、線はグレーアウトされます。

## Parallel coordinates パネルを作成する

1. Workspace のランディングページに移動します
2. Add Panels をクリックします
3. Parallel coordinates を選択します

## パネルの設定

パネルを設定するには、パネル右上の編集ボタンをクリックします。

* **Tooltip**: ホバーすると、各 run の情報を示す凡例が表示されます
* **Titles**: 軸タイトルを編集して、より読みやすくできます
* **Gradient**: 好みの色の範囲にカスタマイズできます
* **Log scale**: 各軸は個別に対数スケール表示に設定できます
* **Flip axis**: 軸の向きを切り替えます — 正解率と損失が両方列にある場合に便利です

[ライブの Parallel coordinates パネルを操作してみる](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)