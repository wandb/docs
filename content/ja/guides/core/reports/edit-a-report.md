---
title: レポートを編集する
description: レポートは、App UI でインタラクティブに編集することも、W&B SDK を使ってプログラムで編集することもできます。
menu:
  default:
    identifier: ja-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

App UI を使ってインタラクティブに、または W&B SDK を使ってプログラム的にレポートを編集できます。

Reports は _blocks_ で構成されています。blocks はレポート本文を構成します。これらの block 内にはテキスト、画像、埋め込み可視化、experiment や run のプロット、panel grid などを追加できます。

_Panel grid_ は、panel と _run set_ を保持する特定の block です。run set は W&B のプロジェクトにログされた run の集合です。panel は run set データの可視化を行います。

{{% alert %}}
プログラムから保存済みワークスペースビューを作成・カスタマイズする手順例については、[Programmatic workspaces チュートリアル]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}

{{% alert %}}
レポートをプログラム的に編集したい場合は、W&B Python SDK に加えて W&B Report および Workspace API `wandb-workspaces` がインストールされていることを確認してください。

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## プロットの追加

各 panel grid には run set のリストと panel のリストがあります。セクション下部の run set が grid 内 panel に表示するデータを制御します。異なる run 集合からデータを引っ張るチャートを追加したい場合は新しい panel grid を作成してください。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

レポート内でスラッシュ（`/`）を入力すると、ドロップダウンメニューが表示されます。**Add panel**（パネル追加）を選択して panel を追加します。W&B でサポートされている任意の panel（折れ線グラフ、散布図、平行座標チャートなど）を追加できます。

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="レポートにチャートを追加" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
SDK でレポートにプログラムからプロットを追加できます。`PanelGrid` Public API クラスの `panels` パラメータに、プロットやチャートオブジェクトのリストを渡してください。各プロット・チャートオブジェクトは対応する Python クラスで作成します。

以下の例では、折れ線グラフと散布図を作成する方法を示しています。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

blocks = [
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="time", y="velocity"),
            wr.ScatterPlot(x="time", y="acceleration"),
        ]
    )
]

report.blocks = blocks
report.save()
```

レポートに追加できるプロット・チャートの詳細は `wr.panels` をご覧ください。

{{% /tab %}}
{{< /tabpane >}}


## Run set の追加

プロジェクトから run set を App UI または W&B SDK 経由でインタラクティブに追加できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

レポート内でスラッシュ（`/`）を入力すると、ドロップダウンメニューが表示されます。そこから **Panel Grid** を選択すると、レポート作成元プロジェクトの run set が自動的にインポートされます。

panel をレポートにインポートすると、run 名はプロジェクトから継承されます。必要に応じて、レポート内で [run 名を変更]({{< relref path="/guides/models/track/runs/#rename-a-run" lang="ja" >}}) して、読者により分かりやすくすることも可能です。run 名の変更はその panel 内だけに反映されます。同じレポート内で panel をコピーした場合は、コピー先 panel の run 名も同様に変更されます。

1. レポート内で鉛筆アイコンをクリックし、レポートエディターを開きます。
1. run set 内でリネームしたい run を探します。レポート名にカーソルを乗せて、縦の三点リーダをクリックし、下記いずれかを選択してフォームを送信します。

    - **Rename run for project**: プロジェクト全体で run 名を変更します。新しいランダム名を生成したい場合は、フィールドを空のまま送信してください。
    - **Rename run for panel grid**: レポート内のみ run 名を変更し、他の文脈では元の名前を保持します。ランダム名の自動生成は対応していません。

1. **Publish report** をクリックします。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

`wr.Runset()` および `wr.PanelGrid` クラスでプロジェクトから run set を追加できます。以下は runset を追加する手順です。

1. `wr.Runset()` オブジェクトを作成。project パラメータに run set を持つプロジェクト名、entity パラメータにそのプロジェクト所有 entity 名を指定します。
2. `wr.PanelGrid()` オブジェクトを作成し、`runsets` パラメータに runset オブジェクトのリストを渡します。
3. 1つ以上の `wr.PanelGrid()` オブジェクトをリストに格納します。
4. レポートインスタンスの blocks 属性に panel grid インスタンスのリストを設定します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")]
)

report.blocks = [panel_grids]
report.save()
```

runset と panel を 1 回の SDK 呼び出しで同時に追加することも可能です。

```python
import wandb

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    panels=[
        wr.LinePlot(
            title="line title",
            x="x",
            y=["y"],
            range_x=[0, 100],
            range_y=[0, 100],
            log_x=True,
            log_y=True,
            title_x="x axis title",
            title_y="y axis title",
            ignore_outliers=True,
            groupby="hyperparam1",
            groupby_aggfunc="mean",
            groupby_rangefunc="minmax",
            smoothing_factor=0.5,
            smoothing_type="gaussian",
            smoothing_show_original=True,
            max_runs_to_show=10,
            plot_type="stacked-area",
            font_size="large",
            legend_position="west",
        ),
        wr.ScatterPlot(
            title="scatter title",
            x="y",
            y="y",
            # z='x',
            range_x=[0, 0.0005],
            range_y=[0, 0.0005],
            # range_z=[0,1],
            log_x=False,
            log_y=False,
            # log_z=True,
            running_ymin=True,
            running_ymean=True,
            running_ymax=True,
            font_size="small",
            regression=True,
        ),
    ],
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")],
)


report.blocks = [panel_grids]
report.save()
``` 

{{% /tab %}}
{{< /tabpane >}}


## Run set を凍結（freeze）する

レポートでは run set が自動で更新され、プロジェクトの最新データが表示されます。run set をレポート内で保持したいときは、*freeze* 操作でその時点の状態を保存できます。

レポート閲覧時に run set を凍結するには、各 panel grid 内の **Filter** ボタン付近にある雪の結晶アイコンをクリックします。

{{< img src="/images/reports/freeze_runset.png" alt="Freeze runset button" >}}

## run set をプログラムでフィルタ

[Workspace and Reports API]({{< relref path="/ref/python/wandb_workspaces/reports" lang="ja" >}}) を利用し、プログラム的に run set を絞り込み、レポートに追加できます。

一般的なフィルタ式の構文は以下です。

```text
Filter('key') operation <value>
```

ここで、`key` はフィルタ名、`operation` は比較演算子（例：`>`, `<`, `==`, `in`, `not in`, `or`, `and`）、`<value>` は比較対象値です。`Filter` は適用したいフィルタタイプです。利用可能なフィルタ／説明は以下の通りです。

| Filter | 説明 | 指定可能キー |
| ---|---| --- |
|`Config('key')` | 設定値でフィルタ | `wandb.init(config=)` の `config` パラメータで指定した値 |
|`SummaryMetric('key')` | サマリーメトリクスでフィルタ | `wandb.Run.log()` で run にログした値 |
|`Tags('key')` | タグ値でフィルタ | プログラム または W&B App で run に追加したタグ値 |
|`Metric('key')` | run プロパティでフィルタ | `tags`, `state`, `displayName`, `jobType` |

フィルタ定義後、レポート作成時に `wr.PanelGrid(runsets=)` へフィルタ済み run set を渡せます。各種要素の追加方法詳細は本ページ各所の **Report and Workspace API** タブを参照してください。

次に、レポート内 run set のフィルタ例をいくつか示します。

### Config フィルタ

1つまたは複数の config 値で runset をフィルタします。config 値は run 設定（`wandb.init(config=)`）で指定したパラメータです。

たとえば下記コードスニペットは `learning_rate` と `batch_size` を config として run を初期化し、その後 `learning_rate` の値を基に run をフィルタしています。

```python
import wandb

config = {
    "learning_rate": 0.01,
    "batch_size": 32,
}

with wandb.init(project="<project>", entity="<entity>", config=config) as run:
    # ここにトレーニングコードを書く
    pass
```

Python スクリプトやノートブック内で、`learning_rate` が `0.01` より大きい run を絞り込むフィルタもこのように書けます。

```python
import wandb_workspaces.reports.v2 as wr

runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Config('learning_rate') > 0.01"
)
```

また、`and` 演算子で複数の条件を組み合わせることもできます。
 
```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Config('learning_rate') > 0.01 and Config('batch_size') == 32"
)
```

このフィルタ済み runset でレポートを作成する場合は以下のようになります。

```python
report = wr.Report(
  entity="your_entity",
  project="your_project",
  title="My Report"
)

report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
      panels=[
          wr.LinePlot(
              x="Step",
              y=["accuracy"],
          )
      ]
  )
]

report.save()
```

### Metric フィルタ

run のタグ（`tags`）、状態（`state`）、名前（`displayName`）、ジョブタイプ（`jobType`）で run set を絞り込めます。

{{% alert %}}
`Metric` フィルタは他と異なる構文を持っています。値のリストをリスト形式で渡してください。

```text
Metric('key') operation [<value>]
```
{{% /alert %}}

例えば、Python スニペットで 3 つの run を作り、それぞれに名前をつける例：

```python
import wandb

with wandb.init(project="<project>", entity="<entity>") as run:
    for i in range(3):
        run.name = f"run{i+1}"
        # ここにトレーニングコードを書く
        pass
```

レポート作成時、display name で run をフィルタしたい場合は以下のように書けます。

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('displayName') in ['run1', 'run2', 'run3']"
)
```

{{% alert %}}
run の名前は W&B App の run の **Overview** ページや、`Api.runs().run.name` でプログラムから確認できます。
{{% /alert %}}

run の状態（`finished`, `crashed`, `running` など）で runset をフィルタする例：

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('state') in ['finished']"
)
```

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('state') not in ['crashed']"
)
```


### SummaryMetric フィルタ

run set を summary metrics でフィルタする例です。summary metrics は `wandb.Run.log()` で run に保存した値です。run を保存後、metric 名は W&B App の **Overview** ページ中 **Summary** セクションで確認できます。

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="SummaryMetric('accuracy') > 0.9"
)
```

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('state') in ['finished'] and SummaryMetric('train/train_loss') < 0.5"
)
```

### Tags フィルタ

runset をタグでフィルタする例。タグはプログラムや W&B App で run に付与できる値です。

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Tags('training') == 'training'"
)
```

## コードブロックの追加

App UI または W&B SDK で、レポートにコードブロックを追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート内でスラッシュ（`/`）を入力し、ドロップダウンから **Code** を選択してください。

コードブロック右側のプルダウンで言語名を選択できます。JavaScript、Python、CSS、JSON、HTML、Markdown、YAML から選択できます。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wr.CodeBlock` クラスを利用し、プログラムでコードブロックを生成できます。language、code パラメータに表示する言語名・コード内容を与えてください。

例えば、YAML のリスト：

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    )
]

report.save()
```

この例は下記のような YAML コードブロックとして描画されます。

```yaml
this:
- is
- a
cool:
- yaml
- file
```

Python のコードブロック：

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これは下記のようなコードブロックとして表示されます。

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Markdown の追加

App UI または W&B SDK からレポートに markdown を追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート内でスラッシュ（`/`）を入力し、ドロップダウンから **Markdown** を選んでください。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.MarkdownBlock` クラスを利用し、プログラムで markdown ブロックを追加可能です。`text` パラメータに文字列を渡してください。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これにより、下記のような markdown ブロックが作成されます。

{{< img src="/images/reports/markdown.png" alt="Rendered markdown block" >}}

{{% /tab %}}

{{% /tabpane %}}


## HTML 要素の追加

App UI または W&B SDK からレポートに HTML 要素を追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポート内でスラッシュ（`/`）を入力後、テキストブロックの種類を選択してください。例えば、H2 見出しを作成するには `Heading 2` オプションを選びます。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.blocks` 属性に HTML 要素（複数可）のリストを渡してください。下記は H1, H2, および箇条書きリストを作成する例です。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
```

この例は以下のように HTML 要素がレンダリングされます。

{{< img src="/images/reports/render_html.png" alt="Rendered HTML elements" >}}

{{% /tab %}}

{{% /tabpane %}}

## リッチメディアリンクの埋め込み

App UI または W&B SDK を使って、レポート内にリッチメディアを埋め込むことができます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

URL をレポートにコピーペーストするだけで、リッチメディアを埋め込めます。下記アニメーションは、Twitter・YouTube・SoundCloud から URL をコピー＆ペーストする様子を示しています。

### Twitter

Tweet のリンク URL をレポート内に貼り付ければ、Tweet の内容をレポートで直接見ることができます。

{{< img src="/images/reports/twitter.gif" alt="Embedding Twitter content" >}}

### Youtube

YouTube 動画の URL リンクを貼り付けることで、レポートに動画を埋め込めます。

{{< img src="/images/reports/youtube.gif" alt="Embedding YouTube videos" >}}

### SoundCloud

SoundCloud のリンクを貼り付けると、音声ファイルをレポートに埋め込むことができます。

{{< img src="/images/reports/soundcloud.gif" alt="Embedding SoundCloud audio" >}}

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.blocks` 属性に一つ以上の埋め込みメディアオブジェクトをリストで指定します。以下は、ビデオと Twitter メディアをレポートへ埋め込む例です。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>\n'
    ),
]
report.save()
```

{{% /tab %}}

{{% /tabpane %}}

## パネルグリッドの複製・削除

レイアウトを再利用したい場合、panel grid セクションを選択してコピー＆ペーストすることで、同じレポート内や別のレポートにも複製できます。

panel grid セクション全体を強調表示したいときは、右上のドラッグハンドルをクリックし、レポート内の panel grid・テキスト・見出し等エリアをドラッグして選択してください。

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="Copying panel grids" >}}

panel grid を選択し、キーボードで `delete` を押すと panel grid を削除できます。

{{< img src="/images/reports/delete_panel_grid.gif" alt="Deleting panel grids" >}}

## 見出しをたたんでレポートを整理する

Report の中で見出しを折りたたむことで、テキストブロック内のコンテンツを一時的に非表示にできます。レポート読み込み時には、開いている見出しの内容のみが表示されます。見出しをたたむことでコンテンツ整理や、過度なデータ読込み防止に役立ちます。下記はその動作を示した gif です。

{{< img src="/images/reports/collapse_headers.gif" alt="Collapsing headers in a report." >}}

## 多次元の関係性を可視化する

多次元の関係性を効果的に可視化するには、変数の一つを色の勾配で表現しましょう。これにより理解が深まり、パターンも読み取りやすくなります。

1. 色のグラデーションで表現する変数（例：ペナルティスコア、learning rate など）を選びます。これにより、reward/side effect（y軸）とtraining time（x軸）上で penalty（色）がどのように作用するか明確になります。
2. 主要な傾向をハイライトします。特定の run グループにカーソルを合わせると、可視化内でそのグループが強調表示されます。