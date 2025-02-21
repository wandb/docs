---
title: 'Tutorial: Use custom charts'
description: W&B UI でのカスタムチャート機能のチュートリアル
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-walkthrough
    parent: custom-charts
---

カスタムチャートを使用すると、パネルにロードするデータとその可視化を制御できます。

## 1. データを W&B に ログ 記録

まず、スクリプトにデータを ログ 記録します。トレーニングの開始時に設定された単一のポイント（ ハイパーパラメーター など）には、[wandb.config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) を使用します。時間の経過に伴う複数のポイントには、[wandb.log()]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を使用し、`wandb.Table()` でカスタム 2D 配列を ログ 記録します。ログ に記録されるキーごとに最大 10,000 データポイントを ログ 記録することをお勧めします。

```python
# データのカスタムテーブルのログ記録
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
  {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

データテーブルを ログ 記録するには、[簡単なノートブックの例](https://bit.ly/custom-charts-colab) を試してください。次のステップでは、カスタムチャートを設定します。結果のチャートが [ライブ レポート](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc) でどのように表示されるかを確認してください。

## 2. クエリを作成する

可視化するデータを ログ 記録したら、プロジェクト ページに移動し、**`+`** ボタンをクリックして新しいパネルを追加し、**カスタムチャート** を選択します。[この ワークスペース](https://app.wandb.ai/demo-team/custom-charts) でフォローできます。

{{< img src="/images/app_ui/create_a_query.png" alt="設定する新しい空白のカスタムチャート" >}}

### クエリを追加する

1. `summary` をクリックし、`historyTable` を選択して、run の履歴からデータをプルする新しいクエリを設定します。
2. `wandb.Table()` を ログ 記録したキーを入力します。上記の コードスニペット では、`my_custom_table` でした。[ノートブック の例](https://bit.ly/custom-charts-colab) では、キーは `pr_curve` と `roc_curve` です。

### Vega フィールドを設定する

クエリがこれらの列にロードされるようになったので、Vega フィールドのドロップダウン メニューで選択するオプションとして使用できます。

{{< img src="/images/app_ui/set_vega_fields.png" alt="クエリ結果から列をプルして Vega フィールドを設定する" >}}

* **x 軸:** runSets_historyTable_r (再現率)
* **y 軸:** runSets_historyTable_p (精度)
* **色:** runSets_historyTable_c (クラス ラベル)

## 3. チャートをカスタマイズする

見た目はかなり良いですが、散布図から折れ線グラフに切り替えたいと思います。**編集** をクリックして、この組み込みチャートの Vega 仕様を変更します。[この ワークスペース](https://app.wandb.ai/demo-team/custom-charts) でフォローできます。

{{< img src="/images/general/custom-charts-1.png" alt="" >}}

Vega 仕様を更新して、可視化をカスタマイズしました。

* プロット、凡例、x 軸、y 軸のタイトルを追加します (各フィールドに「title」を設定します)
* 「mark」の値を「point」から「line」に変更します
* 未使用の「size」フィールドを削除します

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="" >}}

これをこのプロジェクト の他の場所で使用できるプリセットとして保存するには、ページの上部にある **名前を付けて保存** をクリックします。結果は次のようになります。ROC 曲線と合わせて表示されます。

{{< img src="/images/general/custom-charts-2.png" alt="" >}}

## ボーナス: 複合ヒストグラム

ヒストグラムは、数値分布を可視化して、より大きなデータセット を理解するのに役立ちます。複合ヒストグラムは、同じビンに複数の分布を表示し、異なる モデル 間または モデル 内の異なるクラス間で 2 つ以上の メトリクス を比較できます。運転シーンで オブジェクト を検出する セマンティックセグメンテーション モデル の場合、精度と Intersection over Union (IoU) の最適化の効果を比較したり、異なる モデル が車 (データ 内の大きく、一般的な領域) と交通標識 (はるかに小さく、一般的でない領域) をどの程度検出できるかを知りたい場合があります。[デモ Colab](https://bit.ly/custom-charts-colab) では、10 種類の生物のクラスのうち 2 つの信頼性スコアを比較できます。

{{< img src="/images/app_ui/composite_histograms.png" alt="" >}}

カスタム複合ヒストグラム パネルの独自のバージョンを作成するには:

1. ワークスペース または レポート で新しいカスタムチャート パネルを作成します (「カスタムチャート」可視化を追加します)。右上にある「編集」ボタンをクリックして、組み込みのパネル タイプから Vega 仕様を変更します。
2. その組み込みの Vega 仕様を、[Vega の複合ヒストグラムの MVP コード](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21) に置き換えます。この Vega 仕様では、メイン タイトル、軸タイトル、入力ドメイン、その他の詳細を [Vega 構文を使用して](https://vega.github.io/) 直接変更できます (色を変更したり、3 番目のヒストグラムを追加したりすることもできます :)
3. 右側のクエリを変更して、wandb ログ から正しいデータをロードします。フィールド `summaryTable` を追加し、対応する `tableKey` を `class_scores` に設定して、run によって ログ 記録された `wandb.Table` をフェッチします。これにより、`class_scores` として ログ 記録された `wandb.Table` の列を使用して、ドロップダウン メニューから 2 つのヒストグラム ビン セット (`red_bins` と `blue_bins`) を入力できます。私の例では、赤いビンには `animal` クラスの予測スコアを、青いビンには `plant` を選択しました。
4. プレビュー レンダリングに表示されるプロットに満足するまで、Vega 仕様とクエリの変更を続けることができます。完了したら、上部の **名前を付けて保存** をクリックし、カスタム プロットに名前を付けて再利用できるようにします。次に、**パネル ライブラリから適用** をクリックして、プロットを完了します。

簡単な 実験 からの結果は次のようになります。1 つのエポック で 1000 個の例のみでトレーニングすると、ほとんどの画像が植物ではなく、どの画像が動物である可能性があるかについて非常に不確かな モデル が生成されます。

{{< img src="/images/general/custom-charts-3.png" alt="" >}}

{{< img src="/images/general/custom-charts-4.png" alt="" >}}
