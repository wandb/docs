---
title: 実験のプロットを作成して追跡する
description: 機械学習の実験からプロットを作成して追跡する。
menu:
  default:
    identifier: ja-guides-models-track-log-plots
    parent: log-objects-and-media
---

`wandb.plot` の メソッドを使うと、`wandb.Run.log()` で チャートを トラッキング できます。トレーニング 中に時間とともに変化するチャートも含まれます。カスタム チャート フレームワークの詳細は、[custom charts walkthrough]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}}) をご覧ください。

### 基本的なチャート

これらのシンプルなチャートは、メトリクス と 結果 の基本的な 可視化 を簡単に作成できます。

{{< tabpane text=true >}}
    {{% tab header="折れ線" %}}

任意の座標軸上に、順序付きで連結された点のリストとして、カスタム折れ線グラフを ログ します。

```python
import wandb

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

任意の 2 次元で曲線を ログ できます。2 つの 値 のリストを互いにプロットする場合、各リストの要素数は完全に一致している必要があります。たとえば、各点には x と y が必要です。

{{< img src="/images/track/line_plot.png" alt="カスタム折れ線グラフ" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行](https://tiny.cc/custom-charts)   
    {{% /tab %}}
    {{% tab header="散布図" %}}

任意の 2 軸 x と y 上に、点 (x, y) のリストとして、カスタム散布図を ログ します。

```python
import wandb

with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

任意の 2 次元で散布点を ログ できます。2 つの 値 のリストを互いにプロットする場合、各リストの要素数は完全に一致している必要があります。たとえば、各点には x と y が必要です。

{{< img src="/images/track/demo_scatter_plot.png" alt="カスタム散布図" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="棒グラフ" %}}

ラベル付き 値 のリストを棒として表示する、カスタム棒グラフを数行でネイティブに ログ します。

```python
import wandb

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

任意の棒グラフを ログ できます。リスト内のラベルと 値 の数は完全に一致している必要があります。各データポイントには両方が必要です。

{{< img src="/images/track/basic_charts_bar.png" alt="カスタム棒グラフ" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="ヒストグラム" %}}

カスタム ヒストグラムを数行で ログ します。これは 値 のリストを発生回数／頻度でビン分割して集計するものです。たとえば、予測の確信度スコア (`scores`) のリストがあり、その分布を可視化したいとします:

```python
import wandb

with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title="Histogram")})
```

任意のヒストグラムを ログ できます。`data` はリストのリストであり、行と列からなる 2 次元配列を想定している点に注意してください。

{{< img src="/images/track/demo_custom_chart_histogram.png" alt="カスタム ヒストグラム" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行](https://tiny.cc/custom-charts)    
    {{% /tab %}}
    {{% tab header="マルチライン" %}}

1 つの共有 x-y 軸上に、複数の折れ線（複数の x-y 座標ペアのリスト）をプロットします:

```python
import wandb
with wandb.init() as run:
    run.log(
        {
            "my_custom_id": wandb.plot.line_series(
                xs=[0, 1, 2, 3, 4],
                ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
            keys=["metric Y", "metric Z"],
            title="Two Random Metrics",
            xname="x units",
        )
    }
)
```

x と y の点の数は完全に一致している必要があります。複数の y 値リストに対応する 1 つの x 値リストを渡すことも、各 y 値リストごとに別々の x 値リストを渡すこともできます。

{{< img src="/images/track/basic_charts_histogram.png" alt="マルチライン プロット" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU)    
    {{% /tab %}}
{{< /tabpane >}}



### モデルの評価用 チャート

これらのプリセット チャートには `wandb.plot()` の組み込み メソッド があり、スクリプトから直接すばやく簡単にチャートを ログ して、UI で必要な情報をそのまま確認できます。

{{< tabpane text=true >}}
    {{% tab header="PR 曲線" %}}

1 行で [Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を作成します:

```python
import wandb
with wandb.init() as run:
    # ground_truth は真のラベルのリスト、predictions は予測スコアのリスト
    # 例: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})
```

あなたの コード が以下に アクセス できるときはいつでも ログ できます:

* 一連のサンプルに対する モデル の予測スコア（`predictions`）
* そのサンプルに対応する 正解 ラベル（`ground_truth`）
* （オプション）ラベル／クラス名のリスト（例: ラベルのインデックス 0 が cat、1 = dog、2 = bird なら `labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで可視化するラベルの サブセット（リスト形式のまま）

{{< img src="/images/track/model_eval_charts_precision_recall.png" alt="PR 曲線" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)    
    {{% /tab %}}
    {{% tab header="ROC 曲線" %}}

1 行で [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) を作成します:

```python
import wandb

with wandb.init() as run:
    # ground_truth は真のラベルのリスト、predictions は予測スコアのリスト
    # 例: ground_truth = [0, 1, 1, 0], predictions = [0.1, 0.4, 0.35, 0.8]
    ground_truth = [0, 1, 1, 0]
    predictions = [0.1, 0.4, 0.35, 0.8]
    run.log({"roc": wandb.plot.roc_curve(ground_truth, predictions)})
```

あなたの コード が以下に アクセス できるときはいつでも ログ できます:

* 一連のサンプルに対する モデル の予測スコア（`predictions`）
* そのサンプルに対応する 正解 ラベル（`ground_truth`）
* （オプション）ラベル／クラス名のリスト（例: ラベルのインデックス 0 が cat、1 = dog、2 = bird なら `labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで可視化するラベルの サブセット（リスト形式のまま）

{{< img src="/images/track/demo_custom_chart_roc_curve.png" alt="ROC 曲線" >}}

[アプリで見る](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_ROC_Curves_with_W%26B.ipynb)    
    {{% /tab %}}
    {{% tab header="混同行列" %}}

1 行で多クラスの [confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) を作成します:

```python
import wandb

cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=class_names
)

with wandb.init() as run:
    run.log({"conf_mat": cm})
```

あなたの コード が以下に アクセス できるところならどこでも ログ できます:

* 一連のサンプルに対する モデル の予測ラベル（`preds`）または正規化された確率スコア（`probs`）。確率は（サンプル数、クラス数）の形状でなければなりません。確率または予測のいずれか一方のみを渡せます（両方は不可）。
* そのサンプルに対応する 正解 ラベル（`y_true`）
* 文字列の `class_names` として完全なラベル／クラス名リスト。例: インデックス 0 が `cat`、1 が `dog`、2 が `bird` の場合は `class_names=["cat", "dog", "bird"]`

{{< img src="/images/experiments/confusion_matrix.png" alt="混同行列" >}}

​[アプリで見る](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)​

​[コードを実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W%26B.ipynb)    
    {{% /tab %}}
{{< /tabpane >}}


### インタラクティブな カスタム チャート

フルにカスタマイズしたい場合は、組み込みの [Custom Chart のプリセット]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}}) を調整するか新しいプリセットを作成し、チャートを保存します。チャート ID を使って、スクリプトからそのカスタム プリセットへ直接 データ を ログ できます。

```python
import wandb
# プロットする列を持つ Table を作成
table = wandb.Table(data=data, columns=["step", "height"])

# Table の列をチャートのフィールドにマッピング
fields = {"x": "step", "value": "height"}

# 新しいカスタム チャート プリセットを Table で埋める
# 自分の保存済みチャート プリセットを使うには vega_spec_name を変更
# タイトルを編集するには string_fields を変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
    string_fields={"title": "Height Histogram"},
)

with wandb.init() as run:
    # カスタム チャートをログ
    run.log({"my_custom_chart": my_custom_chart})
```

[コードを実行](https://tiny.cc/custom-charts)

### Matplotlib と Plotly のプロット

`wandb.plot()` を使う W&B の [カスタム チャート]({{< relref path="/guides/models/app/features/custom-charts/walkthrough.md" lang="ja" >}}) の代わりに、[matplotlib](https://matplotlib.org/) や [Plotly](https://plotly.com/) で生成したチャートを ログ することもできます。

```python
import wandb
import matplotlib.pyplot as plt

with wandb.init() as run:
    # シンプルな matplotlib プロットを作成
    plt.figure()
    plt.plot([1, 2, 3, 4])
    plt.ylabel("some interesting numbers")
    
    # プロットを W&B にログ
    run.log({"chart": plt})
```

`matplotlib` のプロットまたは Figure オブジェクトを `wandb.Run.log()` に渡すだけです。デフォルトでは、そのプロットを [Plotly](https://plot.ly/) プロットに変換します。プロットを画像として ログ したい場合は、`wandb.Image` に渡してください。Plotly のチャートをそのまま渡すこともできます。

{{% alert %}}
エラー “You attempted to log an empty plot” が出る場合は、`fig = plt.figure()` で Figure をプロットと分けて保持し、`wandb.Run.log()` への呼び出しで `fig` を ログ してください。
{{% /alert %}}

### カスタム HTML を W&B テーブル に ログ

W&B は、Plotly と Bokeh のインタラクティブなチャートを HTML として ログ し、テーブルに追加することをサポートしています。

#### Plotly 図を テーブル に HTML として ログ

Plotly のインタラクティブなチャートを HTML に変換して、wandb テーブルに ログ できます。

```python
import wandb
import plotly.express as px

# 新しい run を初期化
with wandb.init(project="log-plotly-fig-tables", name="plotly_html") as run:

    # Table を作成
    table = wandb.Table(columns=["plotly_figure"])

    # Plotly 図の保存パスを作成
    path_to_plotly_html = "./plotly_figure.html"

    # Plotly 図の例
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # Plotly 図を HTML に書き出す
    # auto_play を False にすると、アニメーションする Plotly チャートが
    # テーブル内で自動再生されなくなります
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Plotly 図を HTML ファイルとして Table に追加
    table.add_data(wandb.Html(path_to_plotly_html))

    # Table をログ
    run.log({"test_table": table})
```

#### Bokeh 図を テーブル に HTML として ログ

Bokeh のインタラクティブなチャートを HTML に変換して、wandb テーブルに ログ できます。

```python
from scipy.signal import spectrogram
import holoviews as hv
import panel as pn
from scipy.io import wavfile
import numpy as np
from bokeh.resources import INLINE

hv.extension("bokeh", logo=False)
import wandb


def save_audio_with_bokeh_plot_to_html(audio_path, html_file_name):
    sr, wav_data = wavfile.read(audio_path)
    duration = len(wav_data) / sr
    f, t, sxx = spectrogram(wav_data, sr)
    spec_gram = hv.Image((t, f, np.log10(sxx)), ["Time (s)", "Frequency (hz)"]).opts(
        width=500, height=150, labelled=[]
    )
    audio = pn.pane.Audio(wav_data, sample_rate=sr, name="Audio", throttle=500)
    slider = pn.widgets.FloatSlider(end=duration, visible=False)
    line = hv.VLine(0).opts(color="white")
    slider.jslink(audio, value="time", bidirectional=True)
    slider.jslink(line, value="glyph.location")
    combined = pn.Row(audio, spec_gram * line, slider).save(html_file_name)


html_file_name = "audio_with_plot.html"
audio_path = "hello.wav"
save_audio_with_bokeh_plot_to_html(audio_path, html_file_name)

wandb_html = wandb.Html(html_file_name)

with wandb.init(project="audio_test") as run:
    my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html]])
    run.log({"audio_table": my_table})
```