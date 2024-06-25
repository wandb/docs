---
description: 機械学習実験データの追跡に関するよくある質問とその回答（W&B Experiments）
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Logging FAQs

<head>
  <title>実験データのログに関するよくある質問</title>
</head>

### W&B UIでログに記録されたチャートやメディアをどのように整理できますか？

W&B UIでは、`/` をログに記録されたパネルを整理するための区切り文字として扱います。デフォルトでは、ログに記録された項目名の `/` 前の部分が「パネルセクション」というパネル群を定義するために使用されます。

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace](../../app/pages/workspaces.md) の設定で、パネルを最初のコンポーネントのみでグループ化するか、`/`で区切られたすべてのコンポーネントでグループ化するかを変更できます。

### エポックやステップごとに画像やメディアを比較するにはどうすればいいですか？

ステップごとに画像をログに記録するたびに、それらをUIに表示するために保存します。画像パネルを展開し、ステップスライダーを使用して異なるステップの画像を確認します。これにより、トレーニング中にモデルの出力がどのように変化するかを簡単に比較できます。

### バッチごとに一部のメトリクスを、エポックごとに他のメトリクスをログに記録するにはどうすればいいですか？

特定のメトリクスを毎バッチログに記録し、プロットを標準化したい場合、メトリクスと共にプロットするX軸の値をログに記録できます。カスタムプロットで「編集」をクリックし、カスタムX軸を選択します。

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```

### 値のリストをログに記録するにはどうすればいいですか？


<Tabs
  defaultValue="dictionary"
  values={[
    {label: 'Using a dictionary', value: 'dictionary'},
    {label: 'As a histogram', value: 'histogram'},
  ]}>
  <TabItem value="dictionary">

```python
wandb.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
  </TabItem>
  <TabItem value="histogram">

```python
wandb.log({"losses": wandb.Histogram(losses)})  # losses をヒストグラムに変換する
```
  </TabItem>
</Tabs>

### 凡例付きで複数の線をプロットする方法は？

マルチラインのカスタムチャートは `wandb.plot.line_series()` を使用して作成できます。[プロジェクトページ](../../app/pages/project-page.md) に移動してラインチャートを確認する必要があります。プロットに凡例を追加するには、`wandb.plot.line_series()` の引数 `keys` を渡します。例えば：

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

マルチラインプロットの詳細は [こちら](../../track/log/plots.md#basic-charts) の「Multi-line」タブをご覧ください。

### Plotly/Bokeh のチャートを Tables に追加するには？

Plotly/Bokeh の図を直接 Tables に追加することはまだサポートされていません。代わりに、図をHTML形式で保存し、そのHTMLを Table に追加します。以下にインタラクティブな Plotly と Bokeh のチャートの例を示します。

<Tabs
  defaultValue="plotly"
  values={[
    {label: 'Using Plotly', value: 'plotly'},
    {label: 'Using Bokeh', value: 'bokeh'},
  ]}>
  <TabItem value="plotly">

```python
import wandb
import plotly.express as px

# 新しい run を初期化する
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# テーブルを作成する
table = wandb.Table(columns=["plotly_figure"])

# Plotly 図のパスを作成する
path_to_plotly_html = "./plotly_figure.html"

# Plotly 図の例
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Plotly 図をHTMLに書き出す
# auto_playをFalseに設定すると、Table内でPlotlyのアニメーションチャートが自動再生されるのを防ぎます
fig.write_html(path_to_plotly_html, auto_play=False)

# Plotly 図をHTMLファイルとして Table に追加する
table.add_data(wandb.Html(path_to_plotly_html))

# Table をログに記録する
run.log({"test_table": table})
wandb.finish()
```

  </TabItem>
  <TabItem value="bokeh">

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
run = wandb.init(project="audio_test")
my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html], [wandb_html]])
run.log({"audio_table": my_table})
run.finish()
```
  </TabItem>
</Tabs>

### グラフにデータが表示されないのはなぜですか？

「No visualization data logged yet」（可視化データがまだログに記録されていません）と表示される場合、それはスクリプトからの最初の `wandb.log` 呼び出しがまだ受信されていないことを意味します。これは、run がステップを完了するのに長い時間がかかるためかもしれません。各エポックの終了時にログを記録している場合は、エポックごとに数回ログを記録してデータをより早く表示できるようにします。

### 同じメトリクスが複数回表示されるのはなぜですか？

同じキーの下で異なる種類のデータをログに記録している場合、データベースでこれらを分割する必要があります。これにより、UIのドロップダウンに同じメトリクス名が複数回表示されることになります。グループ化するタイプは、`number`、`string`、`bool`、`other`（主に配列）、および `wandb` データ型（`Histogram`、`Image`など）です。この挙動を避けるために、各キーには一種類のデータのみを送信してください。

メトリクスは大文字小文字を区別しない形で保存されるため、`"My-Metric"` と `"my-metric"` のように同じ名前のメトリクスが2つ存在しないように注意してください。

### run にログに記録されたデータに直接プログラムからアクセスするにはどうすればいいですか？

history オブジェクトは `wandb.log` によってログに記録されたメトリクスを追跡するために使用されます。[API](../public-api-guide.md) を使用して、`run.history()` で history オブジェクトにアクセスできます。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```

### W&B に対して数百万のステップをログに記録するとどうなりますか？それはブラウザでどのように表示されますか？

送信するポイントが多いほど、UIでグラフを読み込むのに時間がかかります。一行に1000ポイント以上ある場合、バックエンドでデータをブラウザに送信する前に1000ポイントにサンプリングします。このサンプリングは非決定的であるため、ページを更新すると異なるサンプルポイントのセットが表示されます。

**ガイドライン**

1メトリクスあたり1万ポイント以下のログを目指すことをお勧めします。一行に100万ポイント以上ログに記録すると、ページの読み込みに時間がかかります。精度を犠牲にせずにログのフットプリントを減らすための戦略については、[この Colab](http://wandb.me/log-hf-colab) を参照してください。設定およびサマリーメトリクスの列が500を超える場合、テーブルには500のみ表示されます。

### プロジェクトに W&B を統合したいが、画像やメディアをアップロードしたくない場合はどうすればいいですか？

W&B はスカラーのみをログに記録するプロジェクトにも使用できます。アップロードしたいファイルやデータは明示的に指定します。こちらの [PyTorchのクイック例](http://wandb.me/pytorch-colab) は画像をログに記録しません。

### クラス属性を wandb.log() に渡すとどうなりますか？

クラス属性を `wandb.log()` に渡すことは一般的に推奨されません。属性がネットワーク呼び出しが行われる前に変更される可能性があるためです。メトリクスをクラスの属性として保存している場合は、属性をディープコピーして、`wandb.log()` が呼び出された時点の属性の値と一致するようにログに記録されることを確認することをお勧めします。

### ログに記録したデータポイントが予想より少ないのはなぜですか？

X軸に `Step` 以外のものを使用してメトリクスを可視化している場合、予想よりも少ないデータポイントが表示されることがあります。これは、メトリクスを互いにプロットする際に同じ `Step` でログに記録される必要があるためです。これにより、メトリクスが同期されるようにしています。つまり、同じ `Step` でログに記録されたメトリクスのみをサンプリングし、サンプル間で補間します。

**ガイドライン**\
**メトリクスを同じ `log()` 呼び出しにまとめることをお勧めします。コードが次のようになっている場合：**

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

**次のようにログに記録する方が良いでしょう：**

```python
wandb.log({"Precision": precision, "Recall": recall})
```

**あるいは、ステップパラメータを手動で制御し、自分のコードでメトリクスを同期させることもできます：**

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

**2つの `log()` 呼び出しで `step` の値が同じであれば、メトリクスは同じステップの下でログに記録され、まとめてサンプリングされます。各 `log()` 呼び出しで `step` は単調増加する必要があります。そうでない場合、`log()` 呼び出し中に `step` 値は無視されます。**