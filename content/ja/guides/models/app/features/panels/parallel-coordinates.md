---
title: Parallel coordinates
description: 機械学習 の 実験 全体で 結果 を比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

並列座標チャートは、多数のハイパーパラメーター と モデル の メトリクス の関係を一目で把握できるようにまとめたものです。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="" >}}

*   **軸**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) のさまざまな ハイパーパラメーター と、[`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) の メトリクス 。
*   **線**: 各線は単一の run を表します。線にマウスオーバーすると、その run に関する詳細がツールチップに表示されます。現在のフィルターに一致するすべての線が表示されますが、目のアイコンをオフにすると、線はグレー表示になります。

## パネル の 設定

これらの機能は パネル の 設定で構成します。パネル の右上隅にある編集ボタンをクリックしてください。

*   **ツールチップ**: マウスオーバーすると、各 run の 情報 が凡例に表示されます
*   **タイトル**: 軸のタイトルを編集して、より読みやすくします
*   **勾配**: 好みのカラー範囲に合わせて 勾配 をカスタマイズできます
*   **対数スケール**: 各軸は、対数スケールで個別に表示するように設定できます
*   **軸を反転**: 軸の方向を切り替えます。これは、精度と損失の両方を列として使用する場合に便利です

[実際の表示はこちら →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)
