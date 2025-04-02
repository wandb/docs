---
title: Parallel coordinates
description: 機械学習 の 実験 における 結果 を比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

並列座標チャートは、多数のハイパーパラメータとモデル の メトリクス の関係を一目で把握できるようにまとめたものです。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="" >}}

*   **軸**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) のさまざまなハイパーパラメータと、[`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) のメトリクス。
*   **線**: 各線は単一の run を表します。線にマウスオーバーすると、run に関する詳細がツールチップに表示されます。現在のフィルタに一致するすべての線が表示されますが、目のアイコンをオフにすると、線はグレー表示になります。

## 並列座標パネルの作成

1.  ワークスペース のランディングページに移動します
2.  **パネルを追加** をクリックします
3.  **並列座標** を選択します

## パネル の 設定

パネル を構成するには、パネル の右上隅にある編集ボタンをクリックします。

*   **ツールチップ**: マウスオーバーすると、各 run の情報を示す凡例が表示されます
*   **タイトル**: 軸のタイトルを編集して、より読みやすくします
*   **勾配**: 好みの色範囲に合わせて 勾配 をカスタマイズします
*   **対数スケール**: 各軸は、対数スケールで個別に表示するように設定できます
*   **軸の反転**: 軸の方向を切り替えます。これは、精度と損失の両方を列として持つ場合に便利です

[ライブ並列座標パネルを操作する](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)
