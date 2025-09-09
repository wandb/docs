---
title: 'チュートリアル: カスタム チャートを使用する'
description: W&B UI のカスタム チャート機能の使い方チュートリアル
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-walkthrough
    parent: custom-charts
---

カスタムチャートを使って、パネルに読み込む データ とその 可視化 をコントロールしましょう。

## 1. W&B にデータをログする

まずは スクリプト から データ をログします。トレーニング の開始時に設定する単一のポイント（ハイパーパラメーター など）には [wandb.Run.config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) を使います。時間とともに変化する複数ポイントには [wandb.Run.log()]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を使い、カスタムな 2D 配列は `wandb.Table()` でログします。各 キー につき最大 10,000 個のデータポイントのログを推奨します。

```python
with wandb.init() as run: 

  # データのカスタムテーブルをログする
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

データテーブルをログするための [クイックなサンプルノートブック](https://bit.ly/custom-charts-colab) を試してみてください。次のステップでカスタムチャートを設定します。結果のチャートは [ライブな Report](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc) で確認できます。

## 2. クエリを作成する

可視化したい データ をログしたら、Project ページに移動し、新しいパネルを追加するために **`+`** ボタンをクリックして **Custom Chart** を選びます。[custom charts のデモ Workspace](https://app.wandb.ai/demo-team/custom-charts) を見ながら進められます。

{{< img src="/images/app_ui/create_a_query.png" alt="空のカスタムチャート" >}}

### クエリを追加

1. `summary` をクリックして `historyTable` を選び、run の履歴から データ を取得する新しいクエリを作成します。
2. `wandb.Table()` をログした際の キー を入力します。上の コードスニペット では `my_custom_table` でした。[サンプルノートブック](https://bit.ly/custom-charts-colab) では、キーは `pr_curve` と `roc_curve` です。

### Vega フィールドを設定

このクエリで列が読み込まれたので、Vega フィールドのドロップダウンメニューで選択できるようになりました:

{{< img src="/images/app_ui/set_vega_fields.png" alt="クエリ結果から列を取り込み Vega フィールドを設定" >}}

* **x 軸:** runSets_historyTable_r（再現率）
* **y 軸:** runSets_historyTable_p（適合率）
* **色:** runSets_historyTable_c（クラスラベル）

## 3. チャートをカスタマイズする

いい感じになってきたので、散布図から折れ線グラフに切り替えたいと思います。ビルトインのチャートに対する Vega spec を変更するには **Edit** をクリックします。[custom charts のデモ Workspace](https://app.wandb.ai/demo-team/custom-charts) を見ながら進められます。

{{< img src="/images/general/custom-charts-1.png" alt="カスタムチャートの選択" >}}

可視化をカスタマイズするために Vega spec を更新しました:

* プロット、凡例、x 軸、y 軸にタイトルを追加（各フィールドの “title” を設定）
* “mark” の 値 を “point” から “line” に変更
* 未使用の “size” フィールドを削除

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="PR 曲線の Vega spec" >}}

この設定をこの Project の他の場所でも使えるプリセットとして保存するには、ページ上部の **Save as** をクリックしてください。ROC 曲線とあわせた結果は次のとおりです:

{{< img src="/images/general/custom-charts-2.png" alt="PR 曲線のチャート" >}}

## おまけ: 複合ヒストグラム

ヒストグラムは数値分布を可視化し、大きな データセット を理解する助けになります。複合ヒストグラムは同じビンに対して複数の分布を重ね、異なる モデル 間や、1 つの モデル 内のクラス間で 2 つ以上の メトリクス を比較できます。運転シーンの オブジェクト を検出する セマンティックセグメンテーション モデルでは、精度最適化と Intersection over union (IoU) のどちらが有効かを比較したり、異なる モデル が車（データ 中の大きく一般的な領域）や交通標識（はるかに小さく希少な領域）をどれだけうまく検出できるかを知りたくなるでしょう。[デモ Colab](https://bit.ly/custom-charts-colab) では、10 クラスの生物のうち 2 クラスについて信頼度スコアを比較できます。

{{< img src="/images/app_ui/composite_histograms.png" alt="複合ヒストグラム" >}}

カスタム複合ヒストグラムのパネルを自分で作成するには:

1. Workspace または Report に新しい Custom Chart パネル（「Custom Chart」 可視化を追加）を作成します。右上の「Edit」ボタンを押して、任意のビルトイン パネルタイプを起点に Vega spec を編集します。
2. そのビルトインの Vega spec を、私の [Vega による複合ヒストグラムの MVP コード](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21) に置き換えます。メインタイトル、軸タイトル、入力ドメイン、その他の詳細は、この Vega spec 内で直接 [Vega の構文](https://vega.github.io/) を使って変更できます（色を変えたり、ヒストグラムを 3 つ目まで追加することもできます :)）。
3. 右側のクエリを変更して、wandb のログから正しい データ を読み込みます。`summaryTable` フィールドを追加し、対応する `tableKey` を `class_scores` に設定して、run によってログされた `wandb.Table` を取得します。これにより、ドロップダウンメニューから、`class_scores` としてログされた `wandb.Table` の列を使って、2 つのヒストグラムのビン集合（`red_bins` と `blue_bins`）を設定できるようになります。私の例では、赤いビンに `animal` クラスの 予測 スコア、青いビンに `plant` を選びました。
4. プレビューのレンダリングで見えるプロットに満足いくまで、Vega spec とクエリを引き続き調整できます。完了したら、上部の **Save as** をクリックしてカスタムプロットに名前を付けて再利用できるようにし、最後に **Apply from panel library** をクリックしてプロットを完成させます。

とても短い 実験 の結果はこんな感じです。1000 件のサンプルで 1 エポック だけ トレーニング した モデル は、「ほとんどの画像は植物ではない」と非常に自信を持つ一方で、「どの画像が動物か」についてはとても不確かでした。

{{< img src="/images/general/custom-charts-3.png" alt="チャートの設定" >}}

{{< img src="/images/general/custom-charts-4.png" alt="チャートの結果" >}}