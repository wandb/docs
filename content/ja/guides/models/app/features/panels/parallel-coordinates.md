---
title: 並列座標
description: 機械学習実験間で結果を比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

大規模なハイパーパラメーターとモデルメトリクスの関係を一目で要約できるのがパラレル座標チャートです。

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="" >}}

* **軸**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) からのさまざまなハイパーパラメーターと [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) からのメトリクス。
* **ライン**: 各ラインは単一の run を表します。ラインにマウスを合わせると、その run の詳細がツールチップで表示されます。現在のフィルターに一致するすべてのラインが表示されますが、目をオフにすると、ラインはグレー表示されます。

## パラレル座標パネルを作成する

1. ワークスペースのランディングページへ移動
2. **Add Panels** をクリック
3. **Parallel coordinates** を選択

## パネル設定

パネルを設定するには、パネルの右上にある編集ボタンをクリックします。

* **ツールチップ**: ホバーすると、各 run の情報が表示されます
* **タイトル**: 軸のタイトルを編集して、より読みやすくします
* **勾配**: グラデーションを好きな色の範囲にカスタマイズ
* **ログスケール**: 各軸は個別にログスケールで表示するように設定できます
* **軸の反転**: 軸の方向を切り替え—正確性と損失を両方持つカラムがあるときに便利です

[ライブのパラレル座標パネルと対話する](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)