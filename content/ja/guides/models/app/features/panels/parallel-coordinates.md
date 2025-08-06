---
title: パラレル座標
description: 機械学習実験の結果を比較する
menu:
  default:
    identifier: parallel-coordinates
    parent: panels
weight: 30
---

パラレル座標チャートは、多数のハイパーパラメーターとモデルメトリクスの関係を一目で把握できます。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="Parallel coordinates plot" >}}

* **軸**: [`wandb.Run.config`]({{< relref "/guides/models/track/config.md" >}}) から取得したさまざまなハイパーパラメーターおよび [`wandb.Run.log()`]({{< relref "/guides/models/track/log/" >}}) で記録したメトリクス。
* **線**: 各線は 1 つの Run を表します。 ラインにマウスオーバーすると、その Run の詳細情報がツールチップで表示されます。フィルタ条件に合致するすべての線が表示されますが、目のアイコンをオフにするとその線はグレーアウトされます。

## パラレル座標パネルの作成

1. Workspace のランディングページにアクセス
2. **Add Panels** をクリック
3. **Parallel coordinates** を選択

## パネル設定

パネルの右上にある編集ボタンをクリックして設定を行います。

* **Tooltip**: ホバー時に各 Run の情報が凡例として表示されます
* **Titles**: 軸のタイトルを編集して読みやすくできます
* **Gradient**: 勾配を好きな色の範囲にカスタマイズ可能
* **Log scale**: 各軸を個別に対数スケールで表示可能
* **Flip axis**: 軸の方向を切り替えます ― 精度と損失など、異なる種類の値を持つカラムに便利です

[パラレル座標パネルを実際に操作する](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)