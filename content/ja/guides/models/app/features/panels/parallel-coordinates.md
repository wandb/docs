---
title: Parallel coordinates
description: 機械学習の実験間で結果を比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

大規模なハイパーパラメーターとモデルメトリクスの関係を一目でまとめるのが、並行座標チャートです。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="" >}}

* **Axes**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) からの異なるハイパーパラメーターおよび [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) からのメトリクス。
* **Lines**: 各ラインは単一の run を表します。ラインにマウスオーバーすると、その run の詳細がツールチップとして表示されます。現在のフィルタに一致するすべてのラインが表示されますが、目をオフにするとラインがグレーアウトされます。

## パネル設定

これらの機能はパネル設定で構成されます— パネルの右上隅にある編集ボタンをクリックしてください。

* **Tooltip**: ホバーすると、各 run に関する情報が表示される凡例が表示されます
* **Titles**: 軸のタイトルを編集して、より読みやすく
* **Gradient**: 勾配を任意の色の範囲にカスタマイズ
* **Log scale**: 各軸は独立してログスケールで表示するように設定可能
* **Flip axis**: 軸の方向を切り替えます— これは、精度と損失の両方が列にある場合に便利です

[See it live →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)