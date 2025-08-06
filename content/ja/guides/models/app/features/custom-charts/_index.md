---
title: カスタムチャート
menu:
  default:
    identifier: intro
    parent: w-b-app-ui-reference
weight: 2
url: guides/app/features/custom-charts
cascade:
- url: guides/app/features/custom-charts/:filename
---

W&B プロジェクトでカスタムチャートを作成しましょう。任意のデータテーブルをログして、好きなように可視化できます。フォントや色、ツールチップなどの細かいデザインも、[Vega](https://vega.github.io/vega/) のパワフルな機能で自由自在にコントロール可能です。

* コード例: [Colabノートブック](https://tiny.cc/custom-charts) でサンプルをお試しください。
* 動画: [ウォークスルー動画](https://www.youtube.com/watch?v=3-N9OV6bkSM) をご覧ください。
* サンプル: Keras と Sklearn による [デモノートブック](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing) をご参照ください。

{{< img src="/images/app_ui/supported_charts.png" alt="Supported charts from vega.github.io/vega" max-width="90%" >}}

### 仕組み

1. **データをログする**: スクリプトから [config]({{< relref "/guides/models/track/config.md" >}}) やサマリーデータをログします。
2. **チャートをカスタマイズ**: ログしたデータを [GraphQL](https://graphql.org) クエリで取得。強力な可視化記法である [Vega](https://vega.github.io/vega/) を使い、クエリの結果を表現します。
3. **チャートをログする**: `wandb.plot_table()` を使って、独自プリセットをスクリプトから呼び出して保存します。

{{< img src="/images/app_ui/pr_roc.png" alt="PR and ROC curves" >}}

期待するデータが表示されない場合、探しているカラムが選択した Runs でログされていない可能性があります。チャートを保存後、Runs テーブルに戻り、**目のアイコン**で選択中の Run を確認しましょう。

## スクリプトからチャートをログする

### 組み込みプリセット

W&B には、スクリプトからそのままログできる組み込みチャートプリセットが多数用意されています。これには折れ線グラフ、散布図、棒グラフ、ヒストグラム、PR曲線、ROC曲線が含まれます。

{{< tabpane text=true >}}
{{% tab header="Line plot" value="line-plot" %}}

  `wandb.plot.line()`

  任意の x・y 軸上に連結された点 (x, y) のリストから独自の折れ線グラフをログできます。

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {
            "my_custom_plot_id": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )
  ```

  折れ線グラフは任意の2軸間でカーブを表示します。x・y 両方のリストの要素数は必ず等しくしてください（例：各点に x と y 両方が必要）。

  {{< img src="/images/app_ui/line_plot.png" alt="Custom line plot" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA) または [Google Colabノートブック](https://tiny.cc/custom-charts) でお試しください。

{{% /tab %}}

{{% tab header="Scatter plot" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  任意の x・y 軸上の点のリスト (x, y) からカスタム散布図をログできます。

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  任意の2次元で散布図を作成可能です。x, y の値リストは要素数を一致させてください。

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="Scatter plot" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ) または [Google Colabノートブック](https://tiny.cc/custom-charts) でお試しください。

{{% /tab %}}

{{% tab header="Bar chart" value="bar-chart" %}}

  `wandb.plot.bar()`

  ラベル付きの値リスト（棒グラフ）を数行でネイティブにログ可能です。

  ```python
  with wandb.init() as run:
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    run.log(
        {
            "my_bar_chart_id": wandb.plot.bar(
                table, "label", "value", title="Custom Bar Chart"
            )
        }
    )
  ```

  任意の棒グラフに利用可能です。ラベルと値のリストは要素数を一致させてください。

{{< img src="/images/app_ui/demo_bar_plot.png" alt="Demo bar plot" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk) または [Google Colabノートブック](https://tiny.cc/custom-charts) でお試しください。
{{% /tab %}}

{{% tab header="Histogram" value="histogram" %}}

  `wandb.plot.histogram()`

  値のリストをビンごと（出現頻度や個数ごと）に分けたヒストグラムを数行でログ可能です。たとえば予測信頼度スコア (`scores`) の分布を可視化する場合:

  ```python
  with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  任意のヒストグラムを作成できます。`data` はリストのリスト（2次元配列）を想定しています。

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="Custom histogram" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM) または [Google Colabノートブック](https://tiny.cc/custom-charts) でお試しください。

{{% /tab %}}

{{% tab header="PR curve" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  [PR曲線（Precision-Recall curve）](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を1行で作成できます。

  ```python
  with wandb.init() as run:
    plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

    run.log({"pr": plot})
  ```

  コードで以下のデータにアクセスできればいつでもログ可能です:

  * モデルによる予測スコア (`predictions`)
  * 対応する正解ラベル (`ground_truth`)
  * （任意）ラベル/クラス名のリスト（例：`labels=["cat", "dog", "bird"...]`）
  * （任意）プロットしたいラベルのサブセット（リスト形式）

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Precision-recall curves" >}}


  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY) または [Google Colabノートブック](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing) でお試しください。

{{% /tab %}}

{{% tab header="ROC curve" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  [ROC曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) を1行で作成できます。

  ```python
  with wandb.init() as run:
    # ground_truth は正解ラベル、predictions は予測スコアのリスト
    ground_truth = [0, 1, 0, 1, 0, 1]
    predictions = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]

    # ROC曲線用プロットを作成
    # labels はオプションのクラス名リスト、classes_to_plot でそのサブセットを指定可能
    plot = wandb.plot.roc_curve(
        ground_truth, predictions, labels=None, classes_to_plot=None
    )

    run.log({"roc": plot})
  ```

  コードで以下のデータにアクセスできればいつでもログ可能です:

  * モデルによる予測スコア (`predictions`)
  * 対応する正解ラベル (`ground_truth`)
  * （任意）クラス名リスト（例：`labels=["cat", "dog", "bird"...]`）
  * （任意）プロット対象にしたいクラスのサブセット（リスト形式）

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="ROC curve" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE) または [Google Colabノートブック](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing) でお試しください。

{{% /tab %}}
{{< /tabpane >}}

### カスタムプリセット

組み込みプリセットを調整したり、新しく作成したプリセットをチャートとして保存できます。そのチャートIDを使えば、スクリプトからそのプリセットに直接データをログ可能です。[Google Colabノートブックで例を試す](https://tiny.cc/custom-charts)。

```python
# プロットしたいカラムを含むテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルのカラムをチャートのフィールドにマッピング
fields = {"x": "step", "value": "height"}

# 新しいカスタムチャートプリセットにテーブルを反映
# 保存済みのチャートプリセットを使いたい場合、vega_spec_name を変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

{{< img src="/images/app_ui/custom_presets.png" alt="Custom chart presets" max-width="90%" >}}

## データのログ

以下のデータタイプをスクリプトからログして、カスタムチャートで利用できます:

* **Config**: 実験の初期設定（独立変数）。たとえば `wandb.Run.config.learning_rate = 0.0001` のような形で、トレーニング開始時に `wandb.Run.config` に保存した名前付きフィールドが含まれます。
* **Summary**: トレーニング中に記録された単一値（結果や従属変数）。例：`wandb.Run.log({"val_acc" : 0.8})`。トレーニング中に同じキーに複数回書き込むと、そのキーの最終値が summary に残ります。
* **History**: ログされたスカラーの全時系列データは、`history` フィールドからクエリで取得できます。
* **summaryTable**: 複数値をリストで保存したい場合は `wandb.Table()` でデータを保存し、カスタムパネルでクエリしましょう。
* **historyTable**: 履歴のデータを見たい場合は、カスタムチャートパネルで `historyTable` をクエリします。`wandb.Table()` を呼んだりカスタムチャートをログするたび、そのステップごとに新しいテーブルが履歴に作成されます。

### カスタムテーブルをログするには

`wandb.Table()` を使い、2次元配列としてデータをログできます。通常、このテーブルの各行は1つのデータポイントを表し、各列がプロットしたい指標やディメンション（軸）になります。カスタムパネルを設定するときは、テーブル全体が `wandb.Run.log()` で指定したキー（下記例では `custom_data_table`）でアクセスでき、個々のフィールドはカラム名（`x`, `y`, `z`）で利用できます。実験中に何度もテーブルをログできます。各テーブルの最大行数は10,000行です。[Google Colab で例を試す](https://tiny.cc/custom-charts)。

```python
with wandb.init() as run:
  # 任意のデータテーブルをログ
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
      {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

## チャートをカスタマイズ

新しいカスタムチャートを追加し、クエリを編集して可視化したい Runs からデータを選択しましょう。クエリは [GraphQL](https://graphql.org) を使い、各 Run の config や summary、history からデータを取得します。

{{< img src="/images/app_ui/customize_chart.gif" alt="Custom chart creation" max=width="90%" >}}

### カスタム可視化

右上の **Chart** から初期プリセットを選びます。次に **Chart fields** でクエリから取得したデータをチャートのどのフィールドに割り当てるかを指定できます。

下記の画像は、指標を選択し、それを棒グラフのフィールドにマッピングするイメージです。

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="Creating a custom bar chart" max-width="90%" >}}

### Vega の編集方法

パネル上部の **Edit** をクリックして [Vega](https://vega.github.io/vega/)編集モードに入ります。[Vega specification](https://vega.github.io/vega/docs/specification/) を定義することで、UI上にインタラクティブなチャートを作成できます。タイトルの変更、色の変更、カーブをバラバラの点として表示する、またはデータを変換してヒストグラムにするなど、あらゆる面をカスタマイズ可能です。パネルのプレビューは編集内容に即座に反映されるので、Vega仕様やクエリを編集しながら仕上がりを確認できます。詳細は [Vega公式ドキュメントおよびチュートリアル](https://vega.github.io/vega/) をご参照ください。

**フィールド参照方法**

W&B からデータを取得するには、Vega スペック内の任意の場所に `"${field:<field-name>}"` というテンプレート文字列を挿入してください。これにより右側の **Chart Fields** エリアでドロップダウンが表示され、ユーザーがVegaのフィールドにクエリの列をマッピングできるようになります。

あるフィールドのデフォルト値を指定したい場合は、`"${field:<field-name>:<placeholder text>}"` の形式にしてください。

### チャートプリセットを保存する

ビジュアライゼーションパネルに適用した変更は、モーダル下部のボタンで保存できます。また、Vega スペック自体をプロジェクト内で再利用するために保存することも可能です。Vegaエディタ上部の **Save as** をクリックし、プリセット名を付けて保存してください。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)


## 主なユースケース

* 誤差バー付きの棒グラフのカスタマイズ
* PR曲線など x-y座標が独自指標となるモデル検証メトリクスの表示
* 2つの異なるモデルや実験から得た分布をヒストグラムで重ねて表示
* 学習途中の複数ポイントでスナップショットとして指標の変化を可視化
* W&B に未搭載の独自ビジュアライゼーションを作成し、世界中で共有