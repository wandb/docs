---
title: 'チュートリアル: カスタムチャートの使用'
description: W&B UI でのカスタムチャート機能の使用に関するチュートリアル
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-walkthrough
    parent: custom-charts
---

カスタムチャートを使用して、パネルに読み込むデータとその可視化を制御します。

## 1. データを W&B にログする

まず、スクリプトにデータをログします。ハイパーパラメーターのようなトレーニングの開始時に設定される単一のポイントには [wandb.config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) を使用します。時間の経過に伴う複数のポイントには [wandb.log()]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を使用し、`wandb.Table()` でカスタムの2D配列をログします。ログされたキーごとに最大10,000データポイントのログを推奨します。

```python
# データのカスタムテーブルをログする
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
  {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

データテーブルをログするための[短い例のノートブック](https://bit.ly/custom-charts-colab) を試してみてください。次のステップでカスタムチャートを設定します。生成されたチャートが [ライブレポート](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc) でどのように見えるか確認できます。

## 2. クエリを作成する

データを視覚化するためにログしたら、プロジェクトページに移動し、新しいパネルを追加するために **`+`** ボタンをクリックし、**Custom Chart** を選びます。[このワークスペース](https://app.wandb.ai/demo-team/custom-charts) で案内に従うことができます。

{{< img src="/images/app_ui/create_a_query.png" alt="設定する準備が整った新しいカスタムチャート" >}}

### クエリを追加する

1. `summary` をクリックして `historyTable` を選択し、run 履歴からデータを引き出す新しいクエリを設定します。
2. `wandb.Table()` をログしたキーを入力します。上記のコードスニペットでは `my_custom_table` でした。[例のノートブック](https://bit.ly/custom-charts-colab) では、キーは `pr_curve` と `roc_curve` です。

### Vega フィールドを設定する

これらの列がクエリに読み込まれたので、Vega フィールドのドロップダウンメニューで選択オプションとして利用可能です：

{{< img src="/images/app_ui/set_vega_fields.png" alt="Vega フィールドを設定するためにクエリ結果から列を引き出す" >}}

* **x-axis:** runSets_historyTable_r (recall)
* **y-axis:** runSets_historyTable_p (precision)
* **color:** runSets_historyTable_c (class label)

## 3. チャートをカスタマイズする

見た目はかなり良いですが、散布図から折れ線グラフに切り替えたいと思います。組み込みチャートの Vega スペックを変更するために **Edit** をクリックします。[このワークスペース](https://app.wandb.ai/demo-team/custom-charts) で案内に従うことができます。

{{< img src="/images/general/custom-charts-1.png" alt="" >}}

Vega スペックを更新して可視化をカスタマイズしました：

* プロット、凡例、x-axis、および y-axis のタイトルを追加 (各フィールドに「title」を設定)
* 「mark」の 値を「point」から「line」に変更
* 使用されていない「size」フィールドを削除

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="" >}}

これを別の場所で使用できるプリセットとして保存するには、ページ上部の **Save as** をクリックします。結果は次の通り、ROC 曲線と共に次のようになります：

{{< img src="/images/general/custom-charts-2.png" alt="" >}}

## ボーナス: コンポジットヒストグラム

ヒストグラムは、数値の分布を可視化し、大きなデータセットを理解するのに役立ちます。コンポジットヒストグラムは、同じビンにまたがる複数の分布を示し、異なるモデルまたはモデル内の異なるクラス間で2つ以上のメトリクスを比較することができます。ドライブシーンのオブジェクトを検出するセマンティックセグメンテーションモデルの場合、精度最適化と Intersection over union (IOU) の効果を比較したり、異なるモデルが車（データの大きく一般的な領域）と交通標識（より小さく一般的でない領域）をどれだけよく検出するかを知りたいかもしれません。[デモ Colab](https://bit.ly/custom-charts-colab) では、生命体の10クラスのうち2つのクラスの信頼スコアを比較できます。

{{< img src="/images/app_ui/composite_histograms.png" alt="" >}}

カスタム合成ヒストグラムパネルのバージョンを作成するには：

1. ワークスペース または レポート で新しい Custom Chart パネルを作成します（「Custom Chart」可視化を追加することによって）。右上の「Edit」ボタンを押して、組み込みパネルタイプから始めて Vega スペックを変更します。
2. 組み込みの Vega スペックを私の [Vega におけるコンポジットヒストグラムの MVP コード](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21) に置き換えます。メインタイトル、軸タイトル、入力ドメイン、および Vega syntax](https://vega.github.io/) を使用して、他の詳細を直接変更できます（色を変更したり、3番目のヒストグラムを追加したりできます :) 
3. 正しいデータを wandb ログから読み込むために右側のクエリを修正します。 `summaryTable` フィールドを追加し、対応する 'tableKey' を `class_scores` に設定して、run でログした `wandb.Table` を取得します。これにより、 `wandb.Table` のクラススコアとしてログされた列を使用して、ドロップダウン メニューから2つのヒストグラムビンセット (`red_bins` と `blue_bins`) を埋めることができます。私の例では、`赤ビン` の動物クラスの予測スコアと`青ビン` の植物の予測スコアを選びました。
4. プレビュー表示に表示されるプロットに満足するまで、Vega スペックとクエリを変更し続けることができます。完了したら、上部で **Save as** をクリックしてカスタムプロットに名前を付けて再利用できるようにします。 次に **Apply from panel library** をクリックしてプロットを終了します。

私の非常に短い実験の結果は次のようになりました：1,000エポックで1,000エグゼンプルだけでトレーニングすると、モデルはほとんどの画像が植物でないことに非常に自信を持ち、どの画像が動物かについては非常に不確かです。

{{< img src="/images/general/custom-charts-3.png" alt="" >}}

{{< img src="/images/general/custom-charts-4.png" alt="" >}}