---
slug: /guides/app/features/custom-charts
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# カスタムチャート

**Custom Charts** を使用して、現在のデフォルトUIでは不可能なチャートを作成します。任意のデータテーブルをログに記録し、希望通りに可視化します。[Vega](https://vega.github.io/vega/) の力を借りてフォント、色、ツールチップの詳細を制御できます。

* **可能なこと**: [launch announcement →](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
* **コード**: ライブ例を試す [hosted notebook →](https://tiny.cc/custom-charts)
* **ビデオ**: クイックな [walkthrough video →](https://www.youtube.com/watch?v=3-N9OV6bkSM)
* **例**: Quick Keras and Sklearn [demo notebook →](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0\_5AoZdoSD?usp=sharing)

![Supported charts from vega.github.io/vega](/images/app_ui/supported_charts.png)

### 仕組み

1. **データをログ**: スクリプトから、[config](../../../../guides/track/config.md) や実験中の通常の方法で summary データをログに記録します。特定の時点で記録された複数の値のリストを可視化するには、カスタム `wandb.Table` を使用します。
2. **チャートをカスタマイズ**: ログに記録されたデータを [GraphQL](https://graphql.org) クエリですぐに取り込みます。[Vega](https://vega.github.io/vega/) を使ってクエリ結果を可視化します。
3. **チャートをログ**: スクリプトから `wandb.plot_table()` を呼び出してカスタムプリセットを実行します。

![](/images/app_ui/pr_roc.png)

## スクリプトからチャートをログ

### 事前設定済みの前提条件

これらの前提条件には、スクリプトから直接チャートをログに素早く記録するための `wandb.plot` メソッドが用意されており、UIで求める正確な可視化がすぐに確認できます。

<Tabs
  defaultValue="line-plot"
  values={[
    {label: 'Line plot', value: 'line-plot'},
    {label: 'Scatter plot', value: 'scatter-plot'},
    {label: 'Bar chart', value: 'bar-chart'},
    {label: 'Histogram', value: 'histogram'},
    {label: 'PR curve', value: 'pr-curve'},
    {label: 'ROC curve', value: 'roc-curve'},
  ]}>
  <TabItem value="line-plot">

`wandb.plot.line()`

カスタムラインプロットをログに記録します。これは任意の x 軸と y 軸上の接続された順序付けられたポイント (x,y) のリストです。

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

これを使用して、任意の2次元上に曲線をログに記録できます。リストの値を対にしてプロットする場合、リストの値の数は正確に一致する必要があります (すなわち、各ポイントにはxとyの両方が必要です)。

![](/images/app_ui/line_plot.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="scatter-plot">

`wandb.plot.scatter()`

カスタム散布図をログに記録します。これは任意の x 軸と y 軸上のポイント (x, y) のリストです。

```python
data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

これを使用して、任意の2次元上に散布ポイントをログに記録できます。リストの値を対にしてプロットする場合、リストの値の数は正確に一致する必要があります (すなわち、各ポイントにはxとyの両方が必要です)。

![](/images/app_ui/demo_scatter_plot.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="bar-chart">

`wandb.plot.bar()`

カスタム棒グラフをログに記録します。これは、バーとしてラベル付けされた値のリストです。以下のように数行でネイティブに扱えます：

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

これを使用して、任意の棒グラフをログに記録できます。ラベルと値のリスト数は正確に一致する必要があります (すなわち、各データポイントには両方が必要です)。

![](@site/static/images/app_ui/line_plot_bar_chart.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

カスタムヒストグラムをログに記録します。以下のように数行でネイティブに扱え、値のリストを頻度に基づいてビンに分類します。例えば、予測信頼スコア (`scores`) のリストがあり、その分布を可視化したい場合：

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
```

これを使用して任意のヒストグラムをログに記録できます。`data` はリストのリストで、行と列の2次元配列をサポートすることを意図しています。

![](/images/app_ui/demo_custom_chart_histogram.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="pr-curve">

`wandb.plot.pr_curve()`

[Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve) を1行で作成します：

```python
plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

wandb.log({"pr": plot})
```

次の場合にこのコードをログに記録できます：

* モデルが予測スコア (`predictions`) を例のセットで持っている場合
* それらの例に対する対応する正解ラベル (`ground_truth`)
* (オプション) ラベルリスト/クラス名 (`labels=["cat", "dog", "bird"...]`) がある場合（ラベルインデックス0が猫、1が犬、2が鳥などの場合）
* (オプション) プロットで可視化するラベルのサブセット（リスト形式）

![](/images/app_ui/demo_average_precision_lines.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[Run the code →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

  </TabItem>
  <TabItem value="roc-curve">

`wandb.plot.roc_curve()`

[ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve) を1行で作成します：

```python
plot = wandb.plot.roc_curve(
    ground_truth, predictions, labels=None, classes_to_plot=None
)

wandb.log({"roc": plot})
```

次の場合にこのコードをログに記録できます：

* モデルが予測スコア (`predictions`) を例のセットで持っている場合
* それらの例に対する対応する正解ラベル (`ground_truth`)
* (オプション) ラベルリスト/クラス名 (`labels=["cat", "dog", "bird"...]`) がある場合（ラベルインデックス0が猫、1が犬、2が鳥などの場合）
* (オプション) プロットで可視化するラベルのサブセット（リスト形式）

![](/images/app_ui/demo_custom_chart_roc_curve.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[Run the code →](https://colab.research.google.com/drive/1\_RMppCqsA8XInV\_jhJz32NCZG6Z5t1RO?usp=sharing)

  </TabItem>
</Tabs>

### カスタムプリセット

事前設定済みのプリセットを調整するか、新しいプリセットを作成してからチャートを保存します。スクリプトから直接そのカスタムプリセットにデータをログするためにチャートIDを使用します。

```python
# プロットする列を含むテーブルを作成します
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列をチャートのフィールドにマップします
fields = {"x": "step", "value": "height"}

# テーブルを使用して新しいカスタムチャートプリセットを作成します
# 自分の保存されたチャートプリセットを使用するためには、vega_spec_name を変更します
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[Run the code →](https://tiny.cc/custom-charts)

![](/images/app_ui/custom_presets.png)

## データをログ

スクリプトからログに記録し、カスタムチャートで使用できるデータタイプは以下の通りです：

* **Config**: 実験の初期設定（独立変数）。これはトレーニングの最初に `wandb.config` にキーとして記録された任意のフィールドが含まれます (例：`wandb.config.learning_rate = 0.0001`)
* **Summary**: トレーニング中にログされた単一の値（結果や従属変数）、例：`wandb.log({"val_acc" : 0.8})`。トレーニング中に `wandb.log()` を使用してこのキーに複数回書き込む場合、summary はそのキーの最終値に設定されます。
* **History**: ログされたスカラーの完全な時系列が `history` フィールドを介してクエリで利用可能
* **summaryTable**: 複数の値をログする必要がある場合は `wandb.Table()` を使用してデータを保存し、カスタムパネルでクエリします。
* **historyTable**: 履歴データを確認する必要がある場合は、カスタムチャートパネルで `historyTable` をクエリします。`wandb.Table()` を呼び出すか、カスタムチャートをログするたびに、そのステップの履歴に新しいテーブルが作成されます。

### カスタムテーブルをログする方法

`wandb.Table()` を使用して、データを2次元配列としてログに記録します。通常、このテーブルの各行は1つのデータポイントを表し、各列はプロットしたい各データポイントの関連フィールド/次元を示します。カスタムパネルを構成する際、`wandb.log()` に渡される名前付きキー ("custom\_data\_table" 以下) を使用してテーブル全体にアクセスでき、個々のフィールドには列名 ("x", "y", "z") を通じてアクセスできます。実験全体で複数のタイムステップでテーブルをログに記録できます。各テーブルの最大サイズは10,000行です。

[Try it in a Google Colab →](https://tiny.cc/custom-charts)

```python
# データのカスタムテーブルをログします
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## チャートをカスタマイズ

新しいカスタムチャートを追加して開始し、クエリを編集して表示される Runs からデータを選択します。このクエリは [GraphQL](https://graphql.org) を使用して、config、summary、 history フィールドからデータを取得します。

![Add a new custom chart, then edit the query](/images/app_ui/customize_chart.gif)

### カスタム可視化

右上の **Chart** を選択してデフォルトプリセットから開始します。次に、**Chart fields** を選択して、クエリからチャートの対応するフィールドにマップするデータを指定します。以下は、クエリからメトリクスを選択し、それを棒グラフフィールドにマップする例です。

![Creating a custom bar chart showing accuracy across runs in a project](/images/app_ui/demo_make_a_custom_chart_bar_chart.gif)

### Vega の編集方法

パネル上部の **Edit** をクリックして [Vega](https://vega.github.io/vega/) 編集モードに切り替えます。ここで[Vega仕様](https://vega.github.io/vega/docs/specification/) を定義し、UIでインタラクティブチャートを作成します。チャートの視覚スタイル（例：タイトルの変更、異なるカラースキームの選択、曲線を連続した線ではなく一連のポイントとして表示する）からデータそのもの（Vega変換を使用して値の配列をヒストグラムに分類するなど）に至るまで、あらゆる側面を変更できます。パネルのプレビューは対話的に更新されるため、Vega仕様やクエリを編集する際に変更の効果を確認できます。[Vegaのドキュメントとチュートリアル](https://vega.github.io/vega/) はインスピレーションの優れたソースです。

**フィールド参照**

W&B からデータをチャートに取り込むには、Vega 仕様の任意の場所に `"${field:<field-name>}"` 形式のテンプレート文字列を追加します。これにより、**Chart Fields** エリアにドロップダウンが作成され、ユーザーはクエリ結果列を選択して Vega にマップできます。

フィールドのデフォルト値を設定するには、次の構文を使用します： `"${field:<field-name>:<placeholder text>}"`

### チャートプリセットの保存

モーダルの下部にあるボタンを使用して、特定の可視化パネルに変更を適用します。あるいは、プロジェクトの他の場所で使用するために Vega 仕様を保存することもできます。Vegaエディタの上部にある **Save as** をクリックしてプリセットに名前を付けて保存します。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves

### カスタムチャートに「ステップスライダー」を表示する方法は？

これはカスタムチャートエディタの「その他の設定」ページで有効にできます。クエリを `summaryTable` ではなく `historyTable` を使うように変更すると、カスタムチャートエディタで「ステップセレクタを表示」するオプションが表示されます。これにより、ステップを選択できるスライダーが表示されます。

### カスタムチャートプリセットを削除する方法は？

カスタムチャートエディタに入り、現在選択されているチャートタイプをクリックすると、すべてのプリセットが表示されるメニューが開きます。削除したいプリセットにマウスをホバリングし、ゴミ箱アイコンをクリックします。

![](/images/app_ui/delete_custome_chart_preset.gif)

### 一般的なユースケース

* エラーバー付きの棒グラフをカスタマイズ
* 特定の x-y 座標（例えば PR曲線）を必要とするモデルの検証メトリクスを表示
* 異なる二つのModelsやExperimentsからのデータ分布をヒストグラムとしてオーバーレイ表示
* トレーニング中の複数のポイントでのスナップショットを使用してメトリクスの変化を表示
* W&B でまだ利用できないユニークな可視化を作成（そしてぜひ世界と共有）