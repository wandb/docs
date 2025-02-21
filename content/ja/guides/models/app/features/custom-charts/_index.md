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

デフォルトのUIでは現在不可能なチャートを作成するには、**カスタムチャート** を使用します。 任意のデータテーブルをログに記録し、必要な方法で正確に可視化します。 [Vega](https://vega.github.io/vega/) の機能を使用して、フォント、色、ツールチップの詳細を制御します。

* **可能なこと**: [発表](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg) をお読みください
* **コード**: [ホストされているノートブック](https://tiny.cc/custom-charts) でライブの例をお試しください
* **動画**: クイック [チュートリアル動画](https://www.youtube.com/watch?v=3-N9OV6bkSM) をご覧ください
* **例**: クイック Keras と Sklearn の [デモノートブック](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

{{< img src="/images/app_ui/supported_charts.png" alt="vega.github.io/vega でサポートされているチャート" max-width="90%" >}}

### 仕組み

1. **データ の ログ**: スクリプトから、W&B で実行する場合と同様に、[config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) とサマリーデータ をログに記録します。 特定の時点でログに記録された複数の値のリストを可視化するには、カスタムの `wandb.Table` を使用します。
2. **チャート をカスタマイズ**: [GraphQL](https://graphql.org) クエリで、このログに記録されたデータ をプルします。 強力な可視化文法である [Vega](https://vega.github.io/vega/) を使用して、クエリの結果を可視化します。
3. **チャート をログに記録**: `wandb.plot_table()` でスクリプトから独自のプリセットを呼び出します。

{{< img src="/images/app_ui/pr_roc.png" alt="" >}}

## スクリプト からチャート をログに記録

### 組み込みプリセット

これらのプリセットには、スクリプト からチャート を直接ログに記録し、UIで探している正確な可視化を確認するための組み込み `wandb.plot` メソッドがあります。

{{< tabpane text=true >}}
{{% tab header="折れ線グラフ" value="line-plot" %}}

  `wandb.plot.line()`

  カスタム折れ線グラフ (任意の軸xおよびy上の接続され順序付けられた点(x、y)のリスト) をログに記録します。

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

  これを使用して、任意の2つの次元で曲線をログに記録できます。 2つの値のリストを互いに対してプロットする場合、リスト内の値の数 は正確に一致する必要があることに注意してください (たとえば、各点にはxとyが必要です)。

  {{< img src="/images/app_ui/line_plot.png" alt="" >}}

  [アプリ で見る](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

  [コードを実行](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="散布図" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  カスタム散布図 (任意の軸xとyのペア上の点(x、y)のリスト) をログに記録します。

  ```python
  data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
  table = wandb.Table(data=data, columns=["class_x", "class_y"])
  wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  これを使用して、任意の2つの次元で散布点をログに記録できます。 2つの値のリストを互いに対してプロットする場合、リスト内の値の数 は正確に一致する必要があることに注意してください (たとえば、各点にはxとyが必要です)。

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="" >}}

  [アプリ で見る](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

  [コードを実行](https://tiny.cc/custom-charts)

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

  これを使用して、任意の棒グラフをログに記録できます。 リスト内のラベルと値の数 は正確に一致する必要があることに注意してください (たとえば、各データポイントには両方が必要です)。

  {{< img src="/images/app_ui/line_plot_bar_chart.png" alt="" >}}

  [アプリ で見る](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

  [コードを実行](https://tiny.cc/custom-charts)
{{% /tab %}}

{{% tab header="ヒストグラム" value="histogram" %}}

  `wandb.plot.histogram()`

  カスタムヒストグラム (値のソートリストを、発生のカウント/頻度でビンに分類) を数行でネイティブにログに記録します。 予測信頼度スコア (`scores`) のリストがあり、その分布を可視化するとします。

  ```python
  data = [[s] for s in scores]
  table = wandb.Table(data=data, columns=["scores"])
  wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  これを使用して、任意のヒストグラムをログに記録できます。 `data` は、行と列の2D配列をサポートすることを目的としたリストのリストであることに注意してください。

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="" >}}

  [アプリ で見る](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

  [コードを実行](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="PR曲線" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  [適合率 - 再現率曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を1行で作成します。

  ```python
  plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

  wandb.log({"pr": plot})
  ```

  コードが以下にアクセスできる場合は、いつでもこれをログに記録できます。

  * 例のセットに関するモデルの予測スコア (`predictions`)
  * それらの例に対応する正解ラベル (`ground_truth`)
  * (オプション) ラベル/クラス名 のリスト (`labels=["cat", "dog", "bird"...]`。ラベルインデックス0がcat、1 = dog、2 = birdなどを意味する場合)
  * (オプション) プロットで可視化するラベルのサブセット (リスト形式のまま)

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="" >}}


  [アプリ で見る](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

  [コードを実行](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

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

  * 例のセットに関するモデルの予測スコア (`predictions`)
  * それらの例に対応する正解ラベル (`ground_truth`)
  * (オプション) ラベル/クラス名 のリスト (`labels=["cat", "dog", "bird"...]`。ラベルインデックス0がcat、1 = dog、2 = birdなどを意味する場合)
  * (オプション) プロットで可視化するこれらのラベルのサブセット (リスト形式のまま)

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="" >}}

  [アプリ で見る](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

  [コードを実行](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)

{{% /tab %}}
{{< /tabpane >}}

### カスタムプリセット

組み込みプリセットを微調整するか、新しいプリセットを作成して、チャート を保存します。 チャート IDを使用して、スクリプト からそのカスタムプリセットにデータを直接ログに記録します。

```python
# プロットする列を含むテーブルを作成する
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列からチャートのフィールドへのマッピング
fields = {"x": "step", "value": "height"}

# テーブルを使用して、新しいカスタムチャートプリセットを設定します
# 独自の保存されたチャートプリセットを使用するには、vega_spec_nameを変更します
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[コードを実行](https://tiny.cc/custom-charts)

{{< img src="/images/app_ui/custom_presets.png" alt="" max-width="90%" >}}

## データ をログに記録

スクリプト からログに記録してカスタムチャートで使用できるデータ型を以下に示します。

* **Config**: 実験 の初期設定 (独立変数)。 これには、トレーニングの開始時に `wandb.config` のキーとしてログに記録した名前付きフィールドが含まれます。 例: `wandb.config.learning_rate = 0.0001`
* **サマリー**: トレーニング中にログに記録された単一の値 (結果または従属変数)。 たとえば、`wandb.log({"val_acc" : 0.8})` などです。 `wandb.log()` を使用してトレーニング中にこのキーに複数回書き込む場合、サマリー はそのキーの最終値に設定されます。
* **履歴**: ログに記録されたスカラーの完全な時系列は、`history` フィールドを介してクエリで使用できます。
* **summaryTable**: 複数の値のリストをログに記録する必要がある場合は、`wandb.Table()` を使用してそのデータを保存し、カスタムパネルでクエリを実行します。
* **historyTable**: 履歴データを表示する必要がある場合は、カスタムチャートパネルで `historyTable` をクエリします。 `wandb.Table()` を呼び出すか、カスタムチャートをログに記録するたびに、そのステップの履歴に新しいテーブルが作成されます。

### カスタムテーブル をログに記録する方法

`wandb.Table()` を使用して、データを2D配列としてログに記録します。 通常、このテーブルの各行は1つのデータポイントを表し、各列はプロットする各データポイントに関連するフィールド/ディメンションを示します。 カスタムパネルを構成すると、テーブル全体が `wandb.log()` (以下の `custom_data_table`) に渡される名前付きキーを介してアクセス可能になり、個々のフィールドは列名 (`x`、`y`、および `z`) を介してアクセス可能になります。 実験 全体を通して、複数のタイムステップでテーブルをログに記録できます。 各テーブルの最大サイズは10,000行です。

[Google Colab で試す](https://tiny.cc/custom-charts)

```python
# データのカスタムテーブルをログに記録する
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## チャート をカスタマイズ

新しいカスタムチャートを追加して開始し、クエリを編集して表示されている run からデータを選択します。 クエリは [GraphQL](https://graphql.org) を使用して、run の config、サマリー、および履歴フィールドからデータを取得します。

{{< img src="/images/app_ui/customize_chart.gif" alt="新しいカスタムチャートを追加してから、クエリを編集します" max=width="90%" >}}

### カスタム可視化

右上隅の **チャート** を選択して、デフォルトのプリセットから開始します。 次に、**チャートフィールド** を選択して、クエリからプルしているデータをチャートの対応するフィールドにマッピングします。 クエリから取得するメトリクスを選択し、それを下の棒グラフフィールドにマッピングする例を次に示します。

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="プロジェクト のrun 全体で精度を示すカスタム棒グラフを作成する" max-width="90%" >}}

### Vega を編集する方法

パネルの上部にある **編集** をクリックして、[Vega](https://vega.github.io/vega/) 編集モードに移動します。 ここでは、UI でインタラクティブなチャートを作成する [Vega 仕様](https://vega.github.io/vega/docs/specification/) を定義できます。 チャート の任意の側面を変更できます。 たとえば、タイトルを変更したり、別の配色を選択したり、曲線を接続された線としてではなく一連の点として表示したりできます。 また、Vega変換を使用して値の配列をヒストグラムにビン化するなど、データ自体に変更を加えることもできます。 パネルプレビューはインタラクティブに更新されるため、Vega仕様またはクエリを編集するときに、変更の効果を確認できます。 [Vega のドキュメントとチュートリアル ](https://vega.github.io/vega/) を参照してください。

**フィールド参照**

W&B からチャート にデータをプルするには、Vega仕様の任意の場所に `"${field:<field-name>}"` 形式のテンプレート文字列を追加します。 これにより、右側の **チャートフィールド** 領域にドロップダウンが作成されます。ユーザーはこれを使用して、クエリ結果の列を選択して Vega にマッピングできます。

フィールドのデフォルト値を設定するには、次の構文を使用します。`"${field:<field-name>:<placeholder text>}"`

### チャートプリセットの保存

モーダルの下部にあるボタンを使用して、特定の可視化パネルに変更を適用します。 または、Vega仕様を保存して、プロジェクト の他の場所で使用することもできます。 再利用可能なチャート定義を保存するには、Vegaエディターの上部にある **名前を付けて保存** をクリックし、プリセットに名前を付けます。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## よくある質問

### まもなく公開

* **ポーリング**: チャート 内のデータの自動更新
* **サンプリング**: 効率を高めるために、パネルにロードされるポイントの総数を動的に調整します

### 注意点

* チャート を編集しているときに、クエリで期待されるデータが表示されない場合、探している列が選択した run に記録されていない可能性があります。 チャート を保存して run テーブルに戻り、**目のアイコン** で可視化する run を選択します。

## 一般的なユースケース

* エラーバー付きのカスタム棒グラフ
* カスタムのx-y座標を必要とするモデル検証メトリクスを表示します (適合率 - 再現率曲線 など)。
* 2つの異なるモデル/実験 からのデータ分布をヒストグラムとしてオーバーレイします
* トレーニング中の複数のポイントでのスナップショットを介してメトリクスの変化を表示します
* W&B でまだ利用できない独自の可視化を作成します (そして、うまくいけば世界と共有します)
