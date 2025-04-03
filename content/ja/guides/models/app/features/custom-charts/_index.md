---
title: Custom charts
cascade:
- url: guides/app/features/custom-charts/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-_index
    parent: w-b-app-ui-reference
url: guides/app/features/custom-charts
weight: 2
---

W&B のプロジェクトでカスタムグラフを作成できます。任意のデータのテーブルを記録し、必要な方法で正確に可視化します。[Vega](https://vega.github.io/vega/) の機能を使用して、フォント、色、ツールチップの詳細を制御します。

* コード: [Colab Colabノートブック](https://tiny.cc/custom-charts) の例をお試しください。
* 動画: [解説動画](https://www.youtube.com/watch?v=3-N9OV6bkSM) をご覧ください。
* 例: KerasとSklearnの[デモ notebook](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

{{< img src="/images/app_ui/supported_charts.png" alt="vega.github.io/vega でサポートされているグラフ" max-width="90%" >}}

### 仕組み

1. **データ の ログ**: スクリプトから、[config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) とサマリーデータを記録します。
2. **グラフのカスタマイズ**: [GraphQL](https://graphql.org) クエリでログに記録されたデータを取得します。強力な可視化文法である [Vega](https://vega.github.io/vega/) を使用して、クエリの結果を可視化します。
3. **グラフのログ**: `wandb.plot_table()` を使用して、スクリプトから独自のプリセットを呼び出します。

{{< img src="/images/app_ui/pr_roc.png" alt="" >}}

期待されるデータが表示されない場合は、探している列が選択された runs に記録されていない可能性があります。グラフを保存し、runs テーブルに戻り、**目** のアイコンを使用して選択した runs を確認します。

## スクリプトからグラフをログ

### 組み込みプリセット

W&B には、スクリプトから直接ログに記録できる多数の組み込みグラフプリセットがあります。これには、折れ線グラフ、散布図、棒グラフ、ヒストグラム、PR曲線、ROC曲線が含まれます。

{{< tabpane text=true >}}
{{% tab header="折れ線グラフ" value="line-plot" %}}

  `wandb.plot.line()`

  任意の軸 x と y 上の接続された順序付けられた点 (x,y) のリストであるカスタム折れ線グラフをログに記録します。

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

  折れ線グラフは、任意の2つの次元で曲線をログに記録します。2つの値のリストを互いにプロットする場合、リスト内の値の数は完全に一致する必要があります (たとえば、各ポイントには x と y が必要です)。

  {{< img src="/images/app_ui/line_plot.png" alt="" >}}

  [レポートの例](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA) を参照するか、[Google Colabノートブックの例](https://tiny.cc/custom-charts) をお試しください。

{{% /tab %}}

{{% tab header="散布図" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  カスタム散布図 (任意の軸 x と y のペア上の点のリスト (x, y)) をログに記録します。

  ```python
  data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
  table = wandb.Table(data=data, columns=["class_x", "class_y"])
  wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  これを使用して、任意の2つの次元で散布点をログに記録できます。2つの値のリストを互いにプロットする場合、リスト内の値の数は完全に一致する必要があることに注意してください (たとえば、各ポイントには x と y が必要です)。

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="" >}}

  [レポートの例](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ) を参照するか、[Google Colabノートブックの例](https://tiny.cc/custom-charts) をお試しください。

{{% /tab %}}

{{% tab header="棒グラフ" value="bar-chart" %}}

  `wandb.plot.bar()`

  カスタム棒グラフ (ラベル付きの値のリストを棒として) を数行でネイティブにログに記録します。

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

  これを使用して、任意の棒グラフをログに記録できます。リスト内のラベルと値の数は完全に一致する必要があることに注意してください (たとえば、各データポイントには両方が必要です)。

  {{< img src="/images/app_ui/line_plot_bar_chart.png" alt="" >}}

  [レポートの例](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk) を参照するか、[Google Colabノートブックの例](https://tiny.cc/custom-charts) をお試しください。
{{% /tab %}}

{{% tab header="ヒストグラム" value="histogram" %}}

  `wandb.plot.histogram()`

  カスタムヒストグラム (値のソートされたリストを、発生のカウント/頻度でビンに分類) を数行でネイティブにログに記録します。予測信頼度スコア (`scores`) のリストがあり、その分布を可視化するとします。

  ```python
  data = [[s] for s in scores]
  table = wandb.Table(data=data, columns=["scores"])
  wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  これを使用して、任意のヒストグラムをログに記録できます。`data` はリストのリストであり、行と列の2D配列をサポートすることを目的としています。

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="" >}}

  [レポートの例](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM) を参照するか、[Google Colabノートブックの例](https://tiny.cc/custom-charts) をお試しください。

{{% /tab %}}

{{% tab header="PR曲線" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  [適合率-再現率曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を1行で作成します。

  ```python
  plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

  wandb.log({"pr": plot})
  ```

  コードが以下にアクセスできる場合は、いつでもこれをログに記録できます。

  * 例のセットに対するモデルの予測スコア (`predictions`)
  * これらの例に対応する正解ラベル (`ground_truth`)
  * (オプション) ラベル/クラス名のリスト (`labels=["cat", "dog", "bird"...]` (ラベルインデックス0がcat、1 = dog、2 = birdなどを意味する場合))
  * (オプション) プロットで可視化するラベルのサブセット (リスト形式のまま)

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="" >}}

  [レポートの例](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY) を参照するか、[Google Colabノートブックの例](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing) をお試しください。

{{% /tab %}}

{{% tab header="ROC曲線" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  [ROC曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) を1行で作成します。

  ```python
  plot = wandb.plot.roc_curve(
      ground_truth, predictions, labels=None, classes_to_plot=None
  )

  wandb.log({"roc": plot})
  ```

  コードが以下にアクセスできる場合は、いつでもこれをログに記録できます。

  * 例のセットに対するモデルの予測スコア (`predictions`)
  * これらの例に対応する正解ラベル (`ground_truth`)
  * (オプション) ラベル/クラス名のリスト (`labels=["cat", "dog", "bird"...]` (ラベルインデックス0がcat、1 = dog、2 = birdなどを意味する場合))
  * (オプション) プロットで可視化するこれらのラベルのサブセット (リスト形式のまま)

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="" >}}

  [レポートの例](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE) を参照するか、[Google Colabノートブックの例](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing) をお試しください。

{{% /tab %}}
{{< /tabpane >}}

### カスタムプリセット

組み込みプリセットを調整するか、新しいプリセットを作成して、グラフを保存します。グラフIDを使用して、スクリプトからそのカスタムプリセットに直接データをログに記録します。[Google Colabノートブックの例](https://tiny.cc/custom-charts) をお試しください。

```python
# プロットする列を含むテーブルを作成します
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列からグラフのフィールドへのマッピング
fields = {"x": "step", "value": "height"}

# テーブルを使用して、新しいカスタムグラフプリセットを作成します
# 独自の保存されたグラフプリセットを使用するには、vega_spec_name を変更します
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

{{< img src="/images/app_ui/custom_presets.png" alt="" max-width="90%" >}}

## データの ログ

スクリプトから次のデータ型をログに記録し、カスタムグラフで使用できます。

* **Config**: experiment の初期設定 (独立変数)。これには、トレーニングの開始時に `wandb.config` のキーとしてログに記録した名前付きフィールドが含まれます。例: `wandb.config.learning_rate = 0.0001`
* **サマリー**: トレーニング中にログに記録された単一の値 (結果または従属変数)。例: `wandb.log({"val_acc" : 0.8})`。`wandb.log()` を介してトレーニング中にこのキーに複数回書き込むと、サマリーはそのキーの最終値に設定されます。
* **履歴**: ログに記録されたスカラーの完全な時系列は、`history` フィールドを介してクエリで使用できます
* **summaryTable**: 複数の値のリストをログに記録する必要がある場合は、`wandb.Table()` を使用してそのデータを保存し、カスタム パネルでクエリを実行します。
* **historyTable**: 履歴データを確認する必要がある場合は、カスタムグラフ パネルで `historyTable` をクエリします。`wandb.Table()` を呼び出すか、カスタムグラフをログに記録するたびに、そのステップの履歴に新しいテーブルが作成されます。

### カスタムテーブルをログする方法

`wandb.Table()` を使用して、データを2D配列としてログに記録します。通常、このテーブルの各行は1つのデータポイントを表し、各列はプロットする各データポイントの関連フィールド/次元を示します。カスタム パネルを構成すると、テーブル全体が `wandb.log()` (`custom_data_table` below) に渡される名前付きキーを介してアクセスできるようになり、個々のフィールドは列名 (`x`、`y`、および `z`) を介してアクセスできるようになります。 experiment 全体で複数のタイムステップでテーブルをログに記録できます。各テーブルの最大サイズは10,000行です。[Google Colab の例](https://tiny.cc/custom-charts) をお試しください。

```python
# データのカスタムテーブルをログ
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## グラフのカスタマイズ

新しいカスタムグラフを追加して開始し、クエリを編集して表示されている runs からデータを選択します。クエリは [GraphQL](https://graphql.org) を使用して、runs の config、サマリー、および履歴フィールドからデータをフェッチします。

{{< img src="/images/app_ui/customize_chart.gif" alt="新しいカスタムグラフを追加し、クエリを編集します" max=width="90%" >}}

### カスタム可視化

右上隅の **グラフ** を選択して、デフォルトのプリセットから開始します。次に、**グラフフィールド** を選択して、クエリから取得しているデータをグラフの対応するフィールドにマッピングします。

次の画像は、メトリクスを選択し、それを下の棒グラフフィールドにマッピングする方法の例を示しています。

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="プロジェクトの runs 全体で精度を示すカスタム棒グラフの作成" max-width="90%" >}}

### Vega を編集する方法

パネルの上部にある **編集** をクリックして、[Vega](https://vega.github.io/vega/) 編集モードに移動します。ここでは、UI でインタラクティブなグラフを作成する [Vega 仕様](https://vega.github.io/vega/docs/specification/) を定義できます。グラフのあらゆる側面を変更できます。たとえば、タイトルを変更したり、別の配色を選択したり、曲線を接続された線としてではなく一連の点として表示したりできます。また、Vega 変換を使用して値の配列をヒストグラムにビン化するなど、データ自体に変更を加えることもできます。パネルプレビューはインタラクティブに更新されるため、Vega 仕様またはクエリを編集するときの変更の効果を確認できます。[Vega のドキュメントとチュートリアル](https://vega.github.io/vega/) を参照してください。

**フィールド参照**

W&B からグラフにデータをプルするには、Vega 仕様の任意の場所に `"${field:<field-name>}"` 形式のテンプレート文字列を追加します。これにより、右側の **グラフフィールド** 領域にドロップダウンが作成され、ユーザーはクエリ結果の列を選択して Vega にマッピングできます。

フィールドのデフォルト値を設定するには、次の構文を使用します: `"${field:<field-name>:<placeholder text>}"`

### グラフプリセットの保存

モーダルの下部にあるボタンを使用して、特定の可視化パネルに変更を適用します。または、Vega 仕様を保存して、プロジェクトの他の場所で使用することもできます。再利用可能なグラフ定義を保存するには、Vega エディターの上部にある **名前を付けて保存** をクリックし、プリセットに名前を付けます。

## 記事とガイド

1. [W&B 機械学習 可視化 IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [カスタムグラフを使用した NLP 注意ベース モデルの可視化](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [カスタムグラフを使用した勾配フローに対する注意の影響の可視化](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [任意の曲線をログ](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## 一般的なユースケース

* エラーバー付きのカスタム棒グラフ
* カスタム x-y 座標を必要とするモデル検証メトリクスを表示する (適合率-再現率曲線など)
* 2つの異なるモデル/ experiment からのデータ分布をヒストグラムとしてオーバーレイする
* トレーニング中の複数のポイントでのスナップショットを介してメトリクスの変化を表示する
* W&B でまだ利用できない独自の可視化を作成する (そして、うまくいけばそれを世界と共有する)
