---
title: カスタムチャート
cascade:
- url: /ja/guides/app/features/custom-charts/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-_index
    parent: w-b-app-ui-reference
url: /ja/guides/app/features/custom-charts
weight: 2
---

W&Bプロジェクトでカスタムチャートを作成しましょう。任意のデータテーブルをログし、自由に可視化できます。フォント、色、ツールチップの詳細を[Vega](https://vega.github.io/vega/)の力でコントロールしましょう。

* コード: 例の[Colabノートブック](https://tiny.cc/custom-charts)を試してみてください。
* ビデオ: [ウォークスルービデオ](https://www.youtube.com/watch?v=3-N9OV6bkSM)を視聴します。
* 例: KerasとSklearnの[デモノートブック](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

{{< img src="/images/app_ui/supported_charts.png" alt="Supported charts from vega.github.io/vega" max-width="90%" >}}

### 仕組み

1. **データをログする**: スクリプトから、[config]({{< relref path="/guides/models/track/config.md" lang="ja" >}})とサマリーデータをログします。
2. **チャートをカスタマイズする**: [GraphQL](https://graphql.org)クエリを使ってログされたデータを呼び出します。[Vega](https://vega.github.io/vega/)、強力な可視化文法でクエリの結果を可視化します。
3. **チャートをログする**: あなた自身のプリセットをスクリプトから`wandb.plot_table()`で呼び出します。

{{< img src="/images/app_ui/pr_roc.png" alt="" >}}

期待したデータが表示されない場合、選択した Runs に求めている列がログされていない可能性があります。チャートを保存し、Runsテーブルに戻って、選択した Runs を**目のアイコン**で確認してください。

## スクリプトからチャートをログする

### 組み込みプリセット

W&Bにはスクリプトから直接ログできるいくつかの組み込みチャートプリセットがあります。これらには、ラインプロット、スキャッタープロット、バーチャート、ヒストグラム、PR曲線、ROC曲線が含まれます。

{{< tabpane text=true >}}
{{% tab header="Line plot" value="line-plot" %}}

  `wandb.plot.line()`

  カスタムラインプロットをログします — 任意の軸xとy上の接続され順序付けされた点（x,y）のリストです。

  ```python
  data = [[x, y] for (x, y) in zip(x_values, y_values)]
  table = wandb.Table(data=data, columns=["x", "y"])
  wandb.log(
      {
          "my_custom_plot_id": wandb.plot.line(
              table, "x", "y", title="Custom Y vs X Line Plot"
          )
      }
  )
  ```

  ラインプロットは任意の2次元上に曲線をログします。もし2つのlistの値を互いにプロットする場合、listの値の数が完全に一致している必要があります（例えば、各点はxとyを持たなければなりません）。

  {{< img src="/images/app_ui/line_plot.png" alt="" >}}

  [例のレポートを確認](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)するか、[例のGoogle Colabノートブックを試す](https://tiny.cc/custom-charts)ことができます。

{{% /tab %}}

{{% tab header="Scatter plot" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  カスタムスキャッタープロットをログします — 任意の軸xとy上の点（x, y）のリストです。

  ```python
  data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
  table = wandb.Table(data=data, columns=["class_x", "class_y"])
  wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  任意の2次元上にスキャッターポイントをログするためにこれを使うことができます。もし2つのlistの値を互いにプロットする場合、listの値の数が完全に一致している必要があります（例えば、各点はxとyを持たなければなりません）。

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="" >}}

  [例のレポートを確認](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)するか、[例のGoogle Colabノートブックを試す](https://tiny.cc/custom-charts)ことができます。

{{% /tab %}}

{{% tab header="Bar chart" value="bar-chart" %}}

  `wandb.plot.bar()`

  カスタムバーチャートをログします — ラベル付き値のリストをバーとして表示する — 数行でネイティブに:

  ```python
  data = [[label, val] for (label, val) in zip(labels, values)]
  table = wandb.Table(data=data, columns=["label", "value"])
  wandb.log(
      {
          "my_bar_chart_id": wandb.plot.bar(
              table, "label", "value", title="Custom Bar Chart"
          )
      }
  )
  ```

  任意のバーチャートをログするためにこれを使用することができます。list内のラベルと値の数は完全に一致している必要があります（例えば、各データポイントが両方を持つ必要があります）。

  [例のレポートを確認](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)するか、[例のGoogle Colabノートブックを試す](https://tiny.cc/custom-charts)ことができます。
{{% /tab %}}

{{% tab header="Histogram" value="histogram" %}}

  `wandb.plot.histogram()`

  カスタムヒストグラムをログします — いくつかの行で値をカウントまたは出現頻度によってビンにソートします。予測信頼度スコア（`scores`）のリストがあるとしましょう。それらの分布を可視化したいとします。

  ```python
  data = [[s] for s in scores]
  table = wandb.Table(data=data, columns=["scores"])
  wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  任意のヒストグラムをログするためにこれを使用することができます。注意として、 `data` は list of lists であり、2次元配列の行と列をサポートすることを意図しています。

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="" >}}

  [例のレポートを確認](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)するか、[例のGoogle Colabノートブックを試す](https://tiny.cc/custom-charts)ことができます。

{{% /tab %}}

{{% tab header="PR curve" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  [Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を1行で作成します。

  ```python
  plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

  wandb.log({"pr": plot})
  ```

  コードが次にアクセス可能なときにこれをログできます:

  * モデルの予測スコア (`predictions`) の一群の例
  * それらの例の対応する正解ラベル (`ground_truth`)
  * （オプション）ラベルまたはクラス名のリスト (`labels=["cat", "dog", "bird"...]` ラベルインデックス0はcat、1番目はdog、2番目はbird...)
  * （オプション）プロットに可視化するラベルのサブセット（リスト形式のまま）

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="" >}}

  [例のレポートを確認](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)するか、[例のGoogle Colabノートブックを試す](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)ことができます。

{{% /tab %}}

{{% tab header="ROC curve" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) を1行で作成します。

  ```python
  plot = wandb.plot.roc_curve(
      ground_truth, predictions, labels=None, classes_to_plot=None
  )

  wandb.log({"roc": plot})
  ```

  コードが次にアクセス可能なときにこれをログできます:

  * モデルの予測スコア (`predictions`) の一群の例
  * それらの例の対応する正解ラベル (`ground_truth`)
  * （オプション）ラベルまたはクラス名のリスト (`labels=["cat", "dog", "bird"...]` ラベルインデックス0はcat、1番目はdog、2番目はbird...)
  * （オプション）このプロットに可視化するラベルのサブセット（リスト形式のまま）

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="" >}}

  [例のレポートを確認](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)するか、[例のGoogle Colabノートブックを試す](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)ことができます。

{{% /tab %}}
{{< /tabpane >}}

### カスタムプリセット

組み込みプリセットを調整するか新しいプリセットを作成し、チャートを保存します。チャートIDを使ってそのカスタムプリセットに直接スクリプトからデータをログします。[例のGoogle Colabノートブックを試す](https://tiny.cc/custom-charts)。

```python
# プロットする列を持つテーブルを作成します
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列からチャートのフィールドへのマッピング
fields = {"x": "step", "value": "height"}

# 新しいカスタムチャートプリセットを埋めるためにテーブルを使用
# 保存した自身のチャートプリセットを使用するには、vega_spec_nameを変更します
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

{{< img src="/images/app_ui/custom_presets.png" alt="" max-width="90%" >}}

## データをログする

スクリプトから次のデータタイプをログし、カスタムチャートで使用できます。

* **Config**: 実験の初期設定（独立変数）。これは実験の開始時に `wandb.config` にキーとしてログされた名前付きフィールドを含みます。例えば: `wandb.config.learning_rate = 0.0001`
* **Summary**: トレーニング中にログされた単一の値（結果や従属変数）。例えば、`wandb.log({"val_acc" : 0.8})`。トレーニング中に `wandb.log()` を使用してキーに複数回書き込んだ場合、サマリーはそのキーの最終的な値に設定されます。
* **History**: ログされたスカラーの時系列全体は、`history` フィールドを通じてクエリに利用可能です。
* **summaryTable**: 複数の値のリストをログする必要がある場合、`wandb.Table()` を使用してそのデータを保存し、それをカスタムパネルでクエリします。
* **historyTable**: 履歴データを確認したい場合、カスタムチャートパネルで `historyTable` をクエリします。 `wandb.Table()` の呼び出しごとまたはカスタムチャートのログごとに、そのステップにおける履歴に新しいテーブルが作成されます。

### カスタムテーブルをログする方法

`wandb.Table()` を使ってデータを2次元配列としてログします。一般的にこのテーブルの各行は一つのデータポイントを表し、各列はプロットしたい各データポイントの関連フィールド/次元を示しています。カスタムパネルを設定する際、 `wandb.log()` に渡された名前付きキー（以下の `custom_data_table`）を通じてテーブル全体にアクセスでき、個別のフィールドには列の名前（`x`, `y`, `z`）を通じてアクセスできます。実験のさまざまなタイムステップでテーブルをログすることができます。各テーブルの最大サイズは10,000行です。[例のGoogle Colabを試す](https://tiny.cc/custom-charts)。

```python
# データのカスタムテーブルをログする
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## チャートをカスタマイズする

新しいカスタムチャートを追加して開始し、次にクエリを編集して表示可能な Runs からデータを選択します。クエリは[GraphQL](https://graphql.org)を使用して、実行での設定、サマリー、履歴フィールドからデータを取得します。

{{< img src="/images/app_ui/customize_chart.gif" alt="Add a new custom chart, then edit the query" max=width="90%" >}}

### カスタム可視化

右上の**Chart**を選択してデフォルトプリセットから始めましょう。次に、**Chart fields**を選択してクエリから引き出したデータをチャートの対応するフィールドにマッピングします。

次の画像は、メトリックをどのように選択し、それを下のバーチャートフィールドにマッピングするかの一例を示しています。

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="Creating a custom bar chart showing accuracy across runs in a project" max-width="90%" >}}

### Vegaを編集する方法

パネルの上部にある**Edit**をクリックして[Vega](https://vega.github.io/vega/)編集モードに入ります。ここでは、[Vega仕様](https://vega.github.io/vega/docs/specification/)を定義して、UIでインタラクティブなチャートを作成することができます。チャートの任意の面を変更できます。例えば、タイトルを変更したり、異なるカラー スキームを選択したり、曲線を接続された線ではなく一連の点として表示したりできます。また、Vega変換を使用して値の配列をヒストグラムにビン分けするなど、データ自体にも変更を加えることができます。パネルプレビューはインタラクティブに更新されるため、Vega仕様やクエリを編集している間に変更の効果を確認できます。[Vegaのドキュメントとチュートリアルを参照してください](https://vega.github.io/vega/)。

**フィールド参照**

W&Bからチャートにデータを引き込むには、Vega仕様のどこにでも`"${field:<field-name>}"` 形式のテンプレート文字列を追加します。これにより**Chart Fields**エリアにドロップダウンが作成され、ユーザーがクエリ結果の列を選択してVegaにマップできます。

フィールドのデフォルト値を設定するには、この構文を使用します:`"${field:<field-name>:<placeholder text>}"`

### チャートプリセットの保存

モーダルの下部にあるボタンで、特定の可視化パネルに変更を適用します。または、プロジェクト内の他の場所で使用するためにVega仕様を保存できます。使い回しができるチャート定義を保存するには、Vegaエディタの上部にある**Save as**をクリックしてプリセットに名前を付けます。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## 共通のユースケース

* 誤差線のあるバープロットをカスタマイズする
* モデル検証メトリクスの表示（PR曲線のようにカスタムx-y座標が必要なもの）
* 2つの異なるモデル/実験からのデータ分布をヒストグラムとして重ね合わせる
* トレーニング中のスナップショットで複数のポイントにわたるメトリックの変化を示す
* W&Bにまだないユニークな可視化を作成する（そして、できればそれを世界と共有する）