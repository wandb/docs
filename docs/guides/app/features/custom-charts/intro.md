---
slug: /guides/app/features/custom-charts
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Custom Charts

**Custom Charts** を使って、現在のデフォルトのUIでは実現できないチャートを作成しましょう。任意のデータのテーブルをログし、あなたが望むままに可視化します。[Vega](https://vega.github.io/vega/)を使って、フォントや色、ツールチップの詳細を制御できます。

* **できること**: [ローンチ発表を読む →](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
* **コード**: [ホストされたノートブックでライブ例を試す →](https://tiny.cc/custom-charts)
* **ビデオ**: [クイックウォークスルービデオを見る →](https://www.youtube.com/watch?v=3-N9OV6bkSM)
* **例**: KerasとSklearnの[デモノートブック →](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0\_5AoZdoSD?usp=sharing)

![Supported charts from vega.github.io/vega](/images/app_ui/supported_charts.png)

### 仕組み

1. **データをログする**: スクリプトから、[config](../../../../guides/track/config.md)やサマリーデータを通常通りW&Bと共にログします。特定の時点で複数の値をログするには、カスタム `wandb.Table` を使用します。
2. **チャートをカスタマイズする**: ログしたデータを[GraphQL](https://graphql.org)クエリで取得します。[Vega](https://vega.github.io/vega/)を使ってクエリ結果を可視化します。
3. **チャートをログする**: スクリプトから`wandb.plot_table()`を使用してカスタムチャートを呼び出します。

![](/images/app_ui/pr_roc.png)

## スクリプトからチャートをログする

### ビルトインプリセット

これらのプリセットにはビルトインの `wandb.plot` メソッドがあり、スクリプトからチャートを直接ログして、UIで期待する可視化をすばやく確認できます。

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

カスタムの折れ線グラフをログします。任意の軸xとy上の接続された順序付き点のリスト。

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

任意の2次元上のカーブをログするためにこれを使用できます。リストの値を対向させる場合、リストの値の数が正確に一致する必要があります（つまり、各点はxとyを持たなければなりません）。

![](/images/app_ui/line_plot.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="scatter-plot">

`wandb.plot.scatter()`

カスタムの散布図をログします。任意の軸xとy上の点のリスト。

```python
data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

任意の2次元上の散布点をログするためにこれを使用できます。リストの値を対向させる場合、リストの値の数が正確に一致する必要があります（つまり、各点はxとyを持たなければなりません）。

![](/images/app_ui/demo_scatter_plot.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="bar-chart">

`wandb.plot.bar()`

カスタムの棒グラフをログします。ラベル付き値のリストをバーとして表示します。次のように数行でネイティブにログします。

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

任意の棒グラフをログするためにこれを使用できます。リストのラベルと値の数が正確に一致する必要があります（つまり、各データポイントは両方とも持っていなければなりません）。

![](@site/static/images/app_ui/line_plot_bar_chart.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

カスタムヒストグラムをログします。値のリストをカウント/出現頻度でビンに分けます。次のように数行でネイティブにログします。予測信頼度スコア(`scores`)のリストがあり、その分布を可視化したいとします。

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
```

任意のヒストグラムをログするためにこれを使用できます。`data`はリストのリストであり、行と列の2D配列をサポートすることを意図しています。

![](/images/app_ui/demo_custom_chart_histogram.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[Run the code →](https://tiny.cc/custom-charts)

  </TabItem>
    <TabItem value="pr-curve">

`wandb.plot.pr_curve()`

1行で [Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve) を作成します。

```python
plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

wandb.log({"pr": plot})
```

あなたのコードが以下のアクセスができる時にこの機能を使用できます。

* 予測スコア (`predictions`)
* 正解ラベル (`ground_truth`)
* (オプション) ラベル/クラス名のリスト
* (オプション) 可視化したいラベルのサブセット

![](/images/app_ui/demo_average_precision_lines.png)


[See in the app →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[Run the code →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

  </TabItem>
  <TabItem value="roc-curve">

`wandb.plot.roc_curve()`

1行で [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve) を作成します。

```python
plot = wandb.plot.roc_curve(
    ground_truth, predictions, labels=None, classes_to_plot=None
)

wandb.log({"roc": plot})
```

あなたのコードが以下のアクセスができる時にこの機能を使用できます。

* 予測スコア (`predictions`)
* 正解ラベル (`ground_truth`)
* (オプション) ラベル/クラス名のリスト
* (オプション) 可視化したいラベルのサブセット

![](/images/app_ui/demo_custom_chart_roc_curve.png)

[See in the app →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[Run the code →](https://colab.research.google.com/drive/1\_RMppCqsA8XInV\_jhJz32NCZG6Z5t1RO?usp=sharing)

  </TabItem>
</Tabs>

### カスタムプリセット

ビルトインプリセットを微調整するか、新しいプリセットを作成し、チャートを保存します。チャートIDを使用して、スクリプトからそのカスタムプリセットに直接データをログします。

```python
# グラフに表示する列を含むテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列をグラフのフィールドにマップ
fields = {"x": "step", "value": "height"}

# テーブルを使用して新しいカスタムチャートプリセットを作成
# 自分の保存済みチャートプリセットを使用するには、vega_spec_nameを変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[Run the code →](https://tiny.cc/custom-charts)

![](/images/app_ui/custom_presets.png)

## データをログする

以下は、スクリプトからログしてカスタムチャートで使用できるデータタイプです。

* **Config**: 実験の初期設定（独立変数）。トレーニング開始時に`wandb.config`としてログした名前付きフィールドを含む。
* **Summary**: トレーニング中にログされた単一の値（結果や従属変数）。例: `wandb.log({"val_acc" : 0.8})`。トレーニング中にこのキーを複数回書き込む場合、サマリーはそのキーの最終値に設定される。
* **History**: ログされたスカラーの全時系列が`history`フィールドからクエリで取得可能。
* **summaryTable**: 複数の値のリストをログする必要がある場合、`wandb.Table()`を使用してそのデータを保存し、カスタムパネルでクエリする。
* **historyTable**: 履歴データを表示する必要がある場合、カスタムチャートパネルで`historyTable`をクエリする。`wandb.Table()`を呼び出すたび、またはカスタムチャートをログするたびに、そのステップの履歴で新しいテーブルが作成される。

### カスタムテーブルをログする方法

`wandb.Table()`を使用して、データを2D配列としてログします。通常、このテーブルの各行は1つのデータポイントを表し、各列はプロットしたい各データポイントの関連フィールド/次元を示します。カスタムパネルを設定する際に、このテーブル全体は`wandb.log()`に渡される名前付きキー（以下の"custom_data_table"）でアクセス可能となり、個々のフィールドは列名（"x", "y", "z"）でアクセス可能となります。実験全体で複数のタイムステップでテーブルをログできます。各テーブルの最大サイズは10,000行です。

[Google Colabで試す →](https://tiny.cc/custom-charts)

```python
# データのカスタムテーブルをログする
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## チャートをカスタマイズする

新しいカスタムチャートを追加して開始し、その後クエリを編集して表示するランのデータを選択します。クエリは [GraphQL](https://graphql.org) を使って、ランの設定、サマリー、履歴フィールドからデータを取得します。

![Add a new custom chart, then edit the query](/images/app_ui/customize_chart.gif)

### カスタムビジュアライゼーション

右上の **Chart** を選択してデフォルトプリセットからスタートします。次に **Chart fields** からクエリで取得したデータを対応するフィールドにマッピングします。以下は、クエリからメトリクスを選択し、それを棒グラフフィールドにマッピングする例です。

![Creating a custom bar chart showing accuracy across runs in a project](/images/app_ui/demo_make_a_custom_chart_bar_chart.gif)

### Vegaの編集方法

パネル上部の **Edit** をクリックして[Vega](https://vega.github.io/vega/)編集モードに入ります。ここでは、[Vega specification](https://vega.github.io/vega/docs/specification/) を定義し、UIにインタラクティブなチャートを作成できます。ビジュアルスタイル（例：タイトルの変更、異なるカラースキームの選択、曲線をポイントの系列として表示）やデータ自体を変更できます（例：Vegaの変換を使って値の配列をヒストグラムに変換）。パネルプレビューはインタラクティブに更新されるため、クエリやVegaスペックを編集する際に変更の効果を見ることができます。[Vegaのドキュメントとチュートリアル](https://vega.github.io/vega/)は、インスピレーションの素晴らしい源です。

**フィールド参照**

W&Bからチャートにデータを取り込むには、Vegaスペック内の任意の場所に `"${field:<field-name>}"` 形式のテンプレート文字列を追加します。これにより **Chart Fields** エリアの右側にドロップダウンが作成され、ユーザーはクエリ結果の列をVegaにマッピングできます。

フィールドのデフォルト値を設定するには、`"${field:<field-name>:<placeholder text>}"`の構文を使用します。

### チャートプリセットを保存する

特定の可視化パネルに変更を適用するには、モーダルの下部にあるボタンを使用します。あるいは、プロジェクトで他の場所で使用するためにVegaスペックを保存することもできます。再利用可能なチャート定義を保存するには、Vegaエディタの上部にある **Save as** をクリックしてプリセットに名前を付けます。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## よくある質問

### 近日公開

* **ポーリング**: チャート内のデータの自動更新
* **サンプリング**: パネルにロードされるポイントの総数を効率的に調整

### よくある問題

* チャートを編集しているときに期待するデータが表示されない場合があります。それは、選択したランにログされていない列が原因かもしれません。チャートを保存し、ランテーブルに戻って、表示したいランを **目のマーク** で選択します。

### カスタムチャートで「ステップスライダー」を表示する方法は？

これはカスタムチャートエディタの「その他の設定」ページで有効化できます。クエリを`summaryTable`から`historyTable`に変更すると、カスタ