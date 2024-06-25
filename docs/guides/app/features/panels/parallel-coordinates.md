---
description: 機械学習実験の結果を比較する
displayed_sidebar: default
---


# 平行座標

平行座標チャートは、多数のハイパーパラメーターとモデルメトリクスの関係を一目でまとめます。

![](/images/app_ui/parallel_coordinates.gif)

* **軸**: [`wandb.config`](../../../../guides/track/config.md) からの異なるハイパーパラメーターと [`wandb.log`](../../../../guides/track/log/intro.md) からのメトリクス。
* **線**: 各線は単一のrunを表します。線の上にマウスを移動すると、runの詳細がツールチップに表示されます。現在のフィルタに一致するすべての線が表示されますが、目のアイコンをオフにすると、線がグレー表示されます。

## パネル設定

パネル設定内でこれらの機能を構成します。パネルの右上隅にある編集ボタンをクリックします。

* **ツールチップ**: ホバーすると、各runに関する情報が表示される凡例が表示されます
* **タイトル**: 軸のタイトルをより読みやすく編集します
* **グラデーション**: グラデーションを任意の色範囲にカスタマイズします
* **対数スケール**: 各軸は対数スケールで独立して表示するように設定できます
* **軸の反転**: 軸の方向を切り替えます— これは、精度と損失の両方を列として持つ場合に便利です

[See it live →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)