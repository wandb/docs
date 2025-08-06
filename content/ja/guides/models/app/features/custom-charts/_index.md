---
title: カスタムチャート
cascade:
- url: guides/app/features/custom-charts/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-_index
    parent: w-b-app-ui-reference
url: guides/app/features/custom-charts
weight: 2
---

W&B プロジェクトでカスタムチャートを作成しましょう。任意のテーブルデータをログして、思い通りに可視化できます。フォントや色、ツールチップなど細かなカスタマイズも [Vega](https://vega.github.io/vega/) のパワーで実現できます。

* コード例：[Colab ノートブック](https://tiny.cc/custom-charts) で試してみる
* ビデオ：[ウォークスルー動画](https://www.youtube.com/watch?v=3-N9OV6bkSM) を視聴
* サンプル：Keras と Sklearn の [デモノートブック](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)（すぐ試せます）

{{< img src="/images/app_ui/supported_charts.png" alt="Supported charts from vega.github.io/vega" max-width="90%" >}}

### 仕組み

1. **データをログする**: スクリプトから [config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) や summary データをログします。
2. **チャートをカスタマイズする**: ログしたデータを [GraphQL](https://graphql.org) クエリで取得し、[Vega](https://vega.github.io/vega/) を使って可視化します。
3. **チャートをログする**: スクリプトから `wandb.plot_table()` でカスタムプリセットを呼び出します。

{{< img src="/images/app_ui/pr_roc.png" alt="PR and ROC curves" >}}

もし期待したデータが表示されない場合、対象カラムが選択した Runs にログされていない可能性があります。チャートを保存して Runs テーブルに戻り、**目のアイコン**で選択した Runs を確認してください。

## スクリプトからチャートをログする

### 標準プリセット

W&B には標準で多くのチャートプリセットが用意されており、スクリプトから直接ログできます。たとえば、折れ線グラフ、散布図、棒グラフ、ヒストグラム、PR 曲線、ROC 曲線などです。

{{< tabpane text=true >}}
{{% tab header="Line plot" value="line-plot" %}}

  `wandb.plot.line()`

  任意の x, y 軸上に、接続された点列（x, y）を並べたカスタム折れ線グラフをログします。

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

  折れ線グラフは任意の2次元でカーブを描画します。x, y のリストの長さが一致している必要があります（つまり、各点ごとに x と y が必要です）。

  {{< img src="/images/app_ui/line_plot.png" alt="Custom line plot" >}}

  [サンプルレポートを見る](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA) または [Google Colab ノートブックで試す](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="Scatter plot" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  x, y の2つの任意軸上に点列（x, y）で表現されたカスタム散布図をログします。

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  これを使って任意2次元の散布図をログ可能です。2つの値のリストをプロットする場合、長さが完全に一致していることに注意してください（例：各点は x、y を持つ）。

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="Scatter plot" >}}

  [サンプルレポートを見る](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ) または [Google Colab ノートブックで試す](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="Bar chart" value="bar-chart" %}}

  `wandb.plot.bar()`

  ラベル付きの値のリストを棒グラフとしてログします。数行で簡単に書けます。

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

  任意の棒グラフをログできます。リストのラベルと値の数が一致している必要があります（例：各データ点にラベルと値がある）。

{{< img src="/images/app_ui/demo_bar_plot.png" alt="Demo bar plot" >}}

  [サンプルレポートを見る](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk) または [Google Colab ノートブックで試す](https://tiny.cc/custom-charts)
{{% /tab %}}

{{% tab header="Histogram" value="histogram" %}}

  `wandb.plot.histogram()`

  値のリストをカウント／出現頻度ごとにビン分けしてヒストグラムで表示します。たとえば予測信頼度スコア（`scores`）の分布を可視化したい場合：

  ```python
  with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  任意のヒストグラムをログできます。`data` はリストのリスト構造（二次元配列）です。

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="Custom histogram" >}}

  [サンプルレポートを見る](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM) または [Google Colab ノートブックで試す](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="PR curve" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  [PR 曲線（精度 - 再現率 曲線）](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) をワンライナーで作成：

  ```python
  with wandb.init() as run:
    plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

    run.log({"pr": plot})
  ```

  以下が利用可能なときにログできます：

  * モデルによる予測スコア（`predictions`）のリスト
  * 対応する正解ラベル（`ground_truth`）のリスト
  * （オプション）ラベル名のリスト（例：`labels=["cat", "dog", "bird"]` など）
  * （オプション）可視化するラベルのサブセット（リスト形式）

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Precision-recall curves" >}}


  [サンプルレポートを見る](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY) または [Google Colab ノートブックで試す](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

{{% /tab %}}

{{% tab header="ROC curve" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  [ROC 曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) をワンライナーで作成：

  ```python
  with wandb.init() as run:
    # ground_truth は正解ラベルのリスト, predictions は予測スコアのリスト
    ground_truth = [0, 1, 0, 1, 0, 1]
    predictions = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]

    # ROC曲線を作成
    # labels はクラス名リスト（オプション）, classes_to_plot は可視化するラベルのサブセット（オプション）
    plot = wandb.plot.roc_curve(
        ground_truth, predictions, labels=None, classes_to_plot=None
    )

    run.log({"roc": plot})
  ```

  以下が利用可能なときにログできます：

  * モデルによる予測スコア（`predictions`）のリスト
  * 対応する正解ラベル（`ground_truth`）のリスト
  * （オプション）ラベル名のリスト（例：`labels=["cat", "dog", "bird"]` など）
  * （オプション）可視化するラベルのサブセット（リスト形式）

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="ROC curve" >}}

  [サンプルレポートを見る](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE) または [Google Colab ノートブックで試す](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)

{{% /tab %}}
{{< /tabpane >}}

### カスタムプリセット

標準プリセットを調整したり、新規プリセットを作ってチャートとして保存できます。チャートIDを使って、スクリプトからそのカスタムプリセットに直接データをログできます。[Google Colab ノートブックで試す](https://tiny.cc/custom-charts)

```python
# プロットしたいカラムでテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルのカラムをチャートのフィールドにマッピング
fields = {"x": "step", "value": "height"}

# このテーブルを用いて新しいカスタムチャートプリセットを作成
# 独自の保存済みチャートプリセットを利用したい場合は vega_spec_name を変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

{{< img src="/images/app_ui/custom_presets.png" alt="Custom chart presets" max-width="90%" >}}

## データをログする

スクリプトから以下のデータ種別をログし、カスタムチャートで利用することができます：

* **Config**: 実験の初期設定（独立変数）。たとえば `wandb.Run.config.learning_rate = 0.0001` で設定値をキーとしてログ可能です。
* **Summary**: トレーニング中にログされた単一の値（結果や従属変数など）。例：`wandb.Run.log({"val_acc" : 0.8})`。トレーニング途中で何度も同じキーに書き込んだ場合は、そのキーの最終値が summary になります。
* **History**: ログしたスカラー値の全時系列データ。クエリ内で `history` フィールドで参照できます。
* **summaryTable**: 複数の値のリストをログしたい場合は `wandb.Table()` で保存し、カスタムパネルでクエリします。
* **historyTable**: ヒストリーデータを確認したい場合は、カスタムチャートパネルで `historyTable` を参照。`wandb.Table()` を呼び出すごと、もしくはカスタムチャートをログするごとに、該当ステップの履歴テーブルが作成されます。

### カスタムテーブルのログ方法

`wandb.Table()` でデータを2次元配列としてログします。通常、各行が1つのデータ点、各列がプロットしたいフィールドや次元を表します。カスタムパネルを設定する際、テーブル全体は `wandb.Run.log()` で指定したキー（下記例では `custom_data_table`）でアクセスでき、カラム名（`x`、`y`、`z` など）で個別フィールドにアクセスできます。複数のタイムステップでテーブルをログすることも可能です。テーブル1つあたりの最大行数は1万行です。[Google Colab でサンプルを試す](https://tiny.cc/custom-charts)

```python
with wandb.init() as run:
  # データのカスタムテーブルをログ
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
      {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

## チャートのカスタマイズ

新しいカスタムチャートを追加して開始し、次にクエリを編集して可視化したい Runs のデータを選びます。クエリは [GraphQL](https://graphql.org) を利用し、Runs の config, summary, history フィールドからデータを取得します。

{{< img src="/images/app_ui/customize_chart.gif" alt="Custom chart creation" max=width="90%" >}}

### カスタム可視化

右上の **Chart** で既定のプリセットから始められます。次に **Chart fields** を選び、クエリで取得したデータをチャートフィールドにマッピングします。

下図は、指標を選択してそれを棒グラフのフィールドにマッピングするプロセス例です。

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="Creating a custom bar chart" max-width="90%" >}}

### Vega の編集方法

パネル上部の **Edit** をクリックして [Vega](https://vega.github.io/vega/) 編集モードに入ります。ここでは [Vega specification](https://vega.github.io/vega/docs/specification/) を定義して、UI 上で対話型チャートを作成できます。タイトル変更、カラースキーム変更、カーブを点だけで表示する変更など、チャートのあらゆる面をカスタマイズ可能です。また Vega transform で値の配列をヒストグラム化するなど、データへの操作も可能。プレビューは即時反映されるため、Vega spec やクエリを編集しながら結果を確認できます。[Vega のドキュメント・チュートリアル](https://vega.github.io/vega/) も参考にしてください。

**フィールド参照について**

チャートでデータを使うには、Vega spec 内で `"${field:<field-name>}"` というテンプレート文字列を追加します。これにより、右側の **Chart Fields** 項目にドロップダウンが追加され、ユーザーがクエリ結果のカラムを Vega にマッピングできます。

フィールドのデフォルト値を設定したい場合は `"${field:<field-name>:<placeholder text>}"` という構文を使ってください。

### チャートプリセットの保存

個別の可視化パネルに対する変更はモーダル下部のボタンで適用できます。また、Vega spec をプロジェクト内で再利用できるように保存することも可能です。使い回し可能なチャート定義として保存するには、Vega エディタ上部の **Save as** をクリックし、プリセット名を付けてください。

## 記事・ガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## よくあるユースケース

* 誤差バー付きの棒グラフをカスタマイズ
* 特殊な x-y 座標が必要なモデル検証メトリクス（精度-再現率曲線など）を表示
* 2つの異なるモデル・実験のデータ分布をヒストグラムで重ねて比較
* トレーニング中の様々なタイミングでのメトリクス変化をスナップショットで見せる
* W&B にまだ搭載されていないユニークな可視化を作成（ぜひコミュニティでシェアしてください）