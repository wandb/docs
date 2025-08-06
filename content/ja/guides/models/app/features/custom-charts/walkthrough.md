---
title: チュートリアル：カスタムチャートを使う
description: W&B UI でカスタムチャート機能を使うためのチュートリアル
menu:
  default:
    identifier: walkthrough
    parent: custom-charts
---

カスタムチャートを使うことで、パネルに読み込むデータとその可視化方法を自由にコントロールできます。

## 1. データを W&B にログする

まず、スクリプト内でデータをログします。トレーニング開始時に設定するハイパーパラメーターなどの単一ポイントには [wandb.Run.config]({{< relref "/guides/models/track/config.md" >}}) を使いましょう。時間経過や複数のポイントを記録する場合は [wandb.Run.log()]({{< relref "/guides/models/track/log/" >}}) を利用し、カスタムな 2D 配列は `wandb.Table()` でログできます。1つのキーにつき最大10,000件までのデータポイントのログを推奨します。

```python
with wandb.init() as run: 

  # カスタムテーブルのデータをログ
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

[クイックなサンプルノートブック](https://bit.ly/custom-charts-colab)でデータテーブルのログを体験できます。次のステップではカスタムチャートの設定方法を説明します。結果のチャートは[ライブレポート](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)で確認できます。

## 2. クエリを作成

データの可視化をログしたら、プロジェクトページにアクセスし、**`+`** ボタンを押して新しいパネルを追加し、**Custom Chart** を選択します。[カスタムチャートのデモワークスペース](https://app.wandb.ai/demo-team/custom-charts)でも同じ手順を試せます。

{{< img src="/images/app_ui/create_a_query.png" alt="空のカスタムチャート" >}}

### クエリを追加

1. `summary` をクリックし、`historyTable` を選択して run ヒストリーからデータを取得する新しいクエリをセットします。
2. どのキーに `wandb.Table()` を保存したかを入力します。上記のコードスニペットでは `my_custom_table` です。[サンプルノートブック](https://bit.ly/custom-charts-colab)内では `pr_curve` と `roc_curve` というキーが使われています。

### Vega フィールドを指定

クエリでこれらのカラムが読み込まれると、Vega の各フィールドで選択できるようになります。

{{< img src="/images/app_ui/set_vega_fields.png" alt="クエリ結果からカラムを選択して Vega フィールドをセット" >}}

* **x軸:** runSets_historyTable_r（リコール）
* **y軸:** runSets_historyTable_p（適合率）
* **色分け:** runSets_historyTable_c（クラスラベル）

## 3. チャートをカスタマイズ

ここまでで良い感じですが、散布図から折れ線グラフに切り替えたい場合は、**Edit** をクリックして、この組み込みチャートの Vega スペックを変更します。[カスタムチャートのデモワークスペース](https://app.wandb.ai/demo-team/custom-charts)を見ながら進めましょう。

{{< img src="/images/general/custom-charts-1.png" alt="カスタムチャートの選択" >}}

Vega スペックを更新し、可視化をカスタマイズしました。

* グラフ・凡例・x軸・y軸にタイトルを追加（各フィールドの “title” を設定）
* “mark” の値を “point” から “line” に変更
* 未使用の “size” フィールドを削除

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="PRカーブの Vega スペック" >}}

このチャートを他の場所でも使いたい場合は、ページ上部の **Save as** をクリックしてプリセットとして保存できます。こちらは実際の PR曲線と ROC曲線のチャート例です。

{{< img src="/images/general/custom-charts-2.png" alt="PRカーブチャート" >}}

## おまけ：複合ヒストグラム

ヒストグラムは数値データの分布を可視化し、大規模なデータセットの理解に役立ちます。複合ヒストグラムは複数の分布を同じビンで比較できるため、異なるモデルや異なるクラスでメトリクスを比較できます。例えば、運転シーン中の物体検出を行うセマンティックセグメンテーションモデルでは、精度と交差率（Intersection over union, IOU）の最適化効果を比較したり、異なるモデルで車（データ内で多く占める大きな領域）と交通標識（小さく希少な領域）の検出精度を比較するといった使い方ができます。[デモの Colab](https://bit.ly/custom-charts-colab) では、生き物10クラスのうち2クラスの信頼度スコアを比較しています。

{{< img src="/images/app_ui/composite_histograms.png" alt="複合ヒストグラム" >}}

カスタム複合ヒストグラムパネルを自作したい場合は:

1. Workspace または Report 内で新しい Custom Chart パネルを作成します（“Custom Chart” の可視化を追加）。右上の “Edit” ボタンで、どの組み込みパネルタイプからでも Vega スペックを変更できます。
2. 組み込みの Vega スペックを、私の [Vega 向け複合ヒストグラム MVP コード](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21) に置き換えます。Vega [構文](https://vega.github.io/) でメインタイトルや軸タイトル、入力ドメインなど自由に編集できます（色の変更や3つ目のヒストグラム追加も可能です）。
3. 右側のクエリを修正し、wandb のログから正しいデータを取得します。フィールドに `summaryTable` を追加し、対応する `tableKey` を `class_scores` にセットして run でログした `wandb.Table` を取得します。これで `class_scores` として記録した `wandb.Table` の各カラムを使い、ドロップダウンから2組のヒストグラムビン（`red_bins`と`blue_bins`）を設定できます。例では、赤ビンに `animal` クラスの予測スコア、青ビンには `plant` のスコアを使いました。
4. プレビュー描画で思い通りのグラフになるまで、Vega スペックやクエリは自由に編集可能です。完成したら、上部の **Save as** をクリックして名前を付けて保存し、**Apply from panel library** で画面へ反映します。

たった1,000枚・1エポックだけで学習した極めてシンプルな実験結果を以下に示します。モデルが「植物でない」と強く確信する一方、「動物」であるかどうかはかなり不確かです。

{{< img src="/images/general/custom-charts-3.png" alt="チャート設定" >}}

{{< img src="/images/general/custom-charts-4.png" alt="チャート結果" >}}