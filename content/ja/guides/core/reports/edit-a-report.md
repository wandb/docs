---
title: レポートを編集する
description: App の UI を使って対話的にレポートを編集するか、W&B SDK を使ってプログラム的に編集できます。
menu:
  default:
    identifier: edit-a-report
    parent: reports
weight: 20
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

レポートは App UI からインタラクティブに、もしくは W&B SDK を使ってプログラムで編集できます。

Reports は _ブロック_ で構成されています。ブロックがレポート本文を作り、テキストや画像、埋め込み可視化、ExperimentsやRunのプロット、パネルグリッドなどを追加できます。

_パネルグリッド_ はパネルと _run set_ を保持する特別なブロックです。run set は W&B のプロジェクトに記録された複数のRunのコレクションを指します。パネルは run set データの可視化です。

{{% alert %}}
保存したワークスペースビューの作成・カスタマイズ方法については [Programmatic workspaces チュートリアル]({{< relref "/tutorials/workspaces.md" >}}) をご覧ください。
{{% /alert %}}

{{% alert %}}
レポートをプログラムで編集したい場合は、W&B Python SDK に加え W&B Report and Workspace API `wandb-workspaces` をインストールしていることをご確認ください。

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## プロットの追加

各パネルグリッドには run set とパネルがあります。グリッド下部の run set が、そのグリッドのパネルに表示されるデータを制御します。別の run set からデータを取得してチャートを追加したい場合は、新しいパネルグリッドを作成します。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

レポート内でスラッシュ（`/`）を入力すると、ドロップダウンメニューが表示されます。**Add panel** を選択してパネルを追加できます。W&B がサポートしている任意のパネル（ラインプロット、散布図、パラレル座標チャートなど）が追加可能です。

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="Add charts to a report" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
SDK を使ってプログラムでレポートにプロットを追加できます。Public API クラス `PanelGrid` の `panels` パラメータにプロットやチャートのオブジェクトリストを渡してください。それぞれのプロットやチャートオブジェクトは専用のPythonクラスで作成します。

次の例はラインプロットと散布図を作る方法です。

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

レポートに追加可能なプロットやチャートについてさらに知りたい方は、`wr.panels` をご覧ください。

{{% /tab %}}
{{< /tabpane >}}

## Run set の追加

Projects から run set を App UI または W&B SDK でインタラクティブに追加できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

レポート内でスラッシュ（`/`）を入力するとドロップダウンが開きます。**Panel Grid** を選ぶと、自動的にレポートが作成された Project から run set がインポートされます。

パネルをレポートにインポートすると、run 名は Project から継承されます。レポート内では[run の名前を変更]({{< relref "/guides/models/track/runs/#rename-a-run" >}})して、読者に分かりやすくすることもできます。run のリネームはそのパネル内だけで反映されます。同じレポート内でパネルを複製した場合も、複製版でリネームが反映されます。

1. レポートで鉛筆アイコンをクリックしてレポートエディタを開きます。
1. run set 内で名前を変更したい run を探し、run 名にホバーして三点リーダーをクリック。以下のオプションから選択しフォームを送信します。

    - **Rename run for project**: プロジェクト全体で run 名を変更します。新しいランダム名にしたければ入力を空欄にします。
    - **Rename run for panel grid**: レポート内だけで run 名を変更し、他の場所での名前は保持します。新しいランダム名の生成は未対応です。

1. **Publish report** を押します。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

Projects から run set を追加するには `wr.Runset()` および `wr.PanelGrid` クラスを利用します。以下の手順で run set を追加します。

1. `wr.Runset()` オブジェクトを生成します。run set を含むproject名を `project` に、該当プロジェクトのEntity名を `entity` に指定します。
2. `wr.PanelGrid()` オブジェクトを生成し、`runsets` パラメータにrunsetオブジェクトのリストを渡します。
3. 1つ以上の `wr.PanelGrid()` インスタンスをリストとして保持します。
4. そのリストを、reportインスタンスの `blocks` 属性に代入します。

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

run set とパネルを1回のSDK呼び出しで同時に追加することも可能です。

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

## Run set をフリーズする

レポートでは自動的に run set が更新され、プロジェクトの最新データが表示されます。*フリーズ* 機能を使えば、その時点のrun setの状態をレポート内で保存できます。

レポート閲覧時にrun setをフリーズするには、そのパネルグリッドの**Filter**ボタン近くの雪の結晶アイコンをクリックします。

{{< img src="/images/reports/freeze_runset.png" alt="Freeze runset button" >}}

## プログラムで run set をフィルタする

[Workspace and Reports API]({{< relref "/ref/python/wandb_workspaces/reports" >}}) を利用して、run set をプログラムでフィルタし、レポートに追加できます。

フィルタ式の一般的な構文は次のとおりです。

```text
Filter('key') 演算子 <値>
```

ここで `key` はフィルタ対象の名前、`演算子` は比較演算子（例: `>`, `<`, `==`, `in`, `not in`, `or`, `and` など）、`<値>` は比較する値です。`Filter` は使いたいフィルタタイプのプレースホルダです。利用可能なフィルタと説明の一覧は以下の通りです。

| フィルタ | 説明 | 利用可能なキー |
| ---|---| --- |
|`Config('key')` | config値でフィルタ | `wandb.init(config=)` で指定した `config` パラメータ値 |
|`SummaryMetric('key')` | summary metrics でフィルタ | `wandb.Run.log()`でrunに記録した値 |
|`Tags('key')` | タグでフィルタ | プログラムあるいは W&B App で run に追加したタグの値 |
|`Metric('key')` | run プロパティでフィルタ | `tags`, `state`, `displayName`, `jobType` |

フィルタを定義したら、`wr.PanelGrid(runsets=)` にフィルタ済み run set を渡してレポートを作成できます。具体的な記法はこのページ内の **Report and Workspace API** タブを参考にしてください。

次の例はレポート内で run set をフィルタする方法です。

### Config フィルタ

1つ以上のconfig値で runset をフィルタします。config値はrun設定（`wandb.init(config=)`）で定義します。

例えば、次のコードスニペットは `learning_rate` と `batch_size` のconfig値を持つrunを初期化し、`learning_rate` の値でレポート内runをフィルタします。

```python
import wandb

config = {
    "learning_rate": 0.01,
    "batch_size": 32,
}

with wandb.init(project="<project>", entity="<entity>", config=config) as run:
    # ここにトレーニングコードを記述します
    pass
```

その後のPythonスクリプトやノートブック内で、learning_rateが `0.01` より大きいrunだけをプログラム的にフィルタできます。

```python
import wandb_workspaces.reports.v2 as wr

runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Config('learning_rate') > 0.01"
)
```

`and` 演算子で複数のconfig値によるフィルタも可能です。

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Config('learning_rate') > 0.01 and Config('batch_size') == 32"
)
```

先の例に続き、フィルタ済みrunsetでレポートを作る場合は下記の通りです。

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

run のタグ (`tags`)、run状態 (`state`)、run名 (`displayName`)、ジョブタイプ (`jobType`) で run setをフィルタします。

{{% alert %}}
`Metric` フィルタは書き方が異なります。値リストはリスト形式で渡してください。

```text
Metric('key') 演算子 [<値>]
```
{{% /alert %}}

例えば、以下のPython例は3つのrunを作り、それぞれに名前をつけます。

```python
import wandb

with wandb.init(project="<project>", entity="<entity>") as run:
    for i in range(3):
        run.name = f"run{i+1}"
        # ここにトレーニングコードを記述します
        pass
```

レポート作成時、display nameでrunをフィルタできます。例えばrun名が `run1`, `run2`, `run3` のrunだけを対象にするには次の通りです。

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('displayName') in ['run1', 'run2', 'run3']"
)
```

{{% alert %}}
run の名前は W&B App の run の **Overview** ページや `Api.runs().run.name` からプログラムでも確認できます。
{{% /alert %}}

run の状態（`finished`、`crashed`、`running`）で runset をフィルタする例は以下の通りです。

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

Summary metrics で run set をフィルタする例です。Summary metrics は `wandb.Run.log()` で run に記録した値です。run ログ後、summary metric の名前は W&B App の run の **Overview** - **Summary** セクションでも確認できます。

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

run set をタグでフィルタする例です。タグはプログラムまたは W&B App で run に追加できます。

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Tags('training') == 'training'"
)
```

## コードブロックを追加する

App UI または W&B SDK を使って、コードブロックをレポートに追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ（`/`）を入力し、**Code** を選択します。

コードブロック右側のプログラミング言語名を選択してドロップダウンを表示し、希望の言語シンタックスを選択できます。Javascript, Python, CSS, JSON, HTML, Markdown, YAML から選べます。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wr.CodeBlock` クラスを使えばプログラムでコードブロックを作成できます。表示したい言語名とコード（language と code パラメータ）を指定します。

以下は YAML ファイルでリストを表示する例です。

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

これにより、以下のようなコードブロックがレンダリングされます。

```yaml
this:
- is
- a
cool:
- yaml
- file
```

次はPythonのコードブロック例です。

```python
report = wr.Report(project="report-editing")

report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

これにより下記のようなコード例が表示されます。

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## マークダウンの追加

App UI または W&B SDK を使って、レポートへマークダウンを追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ（`/`）を入力し、**Markdown** を選択します。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.MarkdownBlock` クラスを利用し、プログラムでマークダウンブロックを作成できます。`text` パラメータへ文字列を渡してください。

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

これにより下記のようなマークダウンブロックが表示されます。

{{< img src="/images/reports/markdown.png" alt="Rendered markdown block" >}}

{{% /tab %}}

{{% /tabpane %}}

## HTML 要素を追加する

App UI または W&B SDK で HTML 要素をレポートに追加できます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートでスラッシュ（`/`）を入力し、テキストブロックの種類を選択します。例えば H2 見出しを作成したい場合は `Heading 2` を選択してください。

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.blocks` に 1つ以上のHTML要素リストを渡します。下記の例は H1、H2、箇条書きリストを作成する方法です。

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

これでHTML要素が次のように表示されます。

{{< img src="/images/reports/render_html.png" alt="Rendered HTML elements" >}}

{{% /tab %}}

{{% /tabpane %}}

## リッチメディアリンクの埋め込み

App UI または W&B SDK で、レポート内にリッチメディアを埋め込めます。

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

レポートへURLをコピーペーストするだけでリッチメディアが埋め込まれます。下記は Twitter、YouTube、SoundCloud からURLを貼り付ける操作例です。

### Twitter

TweetのリンクURLをコピーペーストすると、レポート内にTweetが表示されます。

{{< img src="/images/reports/twitter.gif" alt="Embedding Twitter content" >}}

### YouTube

YouTube動画のURLリンクをコピーペーストして、レポートに動画を埋め込みます。

{{< img src="/images/reports/youtube.gif" alt="Embedding YouTube videos" >}}

### SoundCloud

SoundCloudのリンクを貼り付けると、オーディオファイルがレポート内で再生できます。

{{< img src="/images/reports/soundcloud.gif" alt="Embedding SoundCloud audio" >}}

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.blocks` に 1つ以上の埋め込みメディアオブジェクトリストを渡してください。下記は、ビデオやTwitterメディアをレポートに埋め込む例です。

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

レイアウトを使い回したい場合、パネルグリッドを選択してコピー＆ペーストすることで同じレポートだけでなく他のレポートへも複製できます。

パネルグリッド全体を選択するには右上のドラッグハンドルをクリックしてください。クリック＆ドラッグでパネルグリッドやテキスト、ヘッダーなど任意の範囲を選択できます。

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="Copying panel grids" >}}

パネルグリッドを選択してキーボードの `delete` キーを押すと削除可能です。

{{< img src="/images/reports/delete_panel_grid.gif" alt="Deleting panel grids" >}}

## Report 内を整理するためヘッダーを折りたたむ

Report 内のヘッダーを折りたたむと、テキストブロック内の内容を非表示にできます。レポートを表示する際、折りたたみ解除済みヘッダーのみが展開表示されます。Report 内でヘッダーを折りたたむことでコンテンツを整理し、大量データの読み込みを抑えられます。下記の GIF はその操作例です。

{{< img src="/images/reports/collapse_headers.gif" alt="Collapsing headers in a report." >}}

## 多次元の関係性を可視化する

多次元の関係性を効果的に可視化するには、カラ―勾配を使って一つの変数を表現しましょう。これにより、パターンが明確になり、解釈しやすくなります。

1. カラ―勾配で表す変数を選びます（例: ペナルティスコア、学習率など）。これにより、ペナルティ（色）がトレーニング時間（x軸）と報酬・副作用（y軸）にどう作用するかが分かりやすくなります。
2. 重要な傾向をハイライトします。run グループにホバーすると、可視化内でそのグループが強調表示されます。