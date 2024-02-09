---
description: Compare results across machine learning experiments
displayed_sidebar: ja
---

# 並行座標

並行座標グラフは、多数のハイパーパラメータとモデルメトリクスの関係を一目で把握できるように要約しています。

![](/images/app_ui/parallel_coordinates.gif)

* **軸**: [`wandb.config`](../../../../guides/track/config.md)からの異なるハイパーパラメータや [`wandb.log`](../../../../guides/track/log/intro.md)からのメトリクスです。
* **線**: 各線は1つのrunを表します。線をマウスオーバすると、runの詳細が表示されるツールチップが表示されます。現在のフィルタに一致するすべての線が表示されますが、目をオフにすると、線がグレーアウトされます。

## パネル設定

パネルの右上隅にある編集ボタンをクリックして、パネル設定でこれらの機能を設定します。

* **ツールチップ**: ホバー時に、各runの情報が表示される凡例が表示されます
* **タイトル**: 軸タイトルをもっと読みやすく編集できます
* **グラデーション**: グラデーションを好きな色範囲にカスタマイズできます
* **対数スケール**: 各軸は独立して対数スケールで表示することができます
* **軸の反転**: 軸の方向を切り替えることができます- これは、正確さと損失の両方を列として持つ場合に便利です

[ライブで見る →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)