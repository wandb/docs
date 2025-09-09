---
title: Reports
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-reports
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}




{{% alert %}}
W&B Report と Workspace API は現在 Public Preview です。
{{% /alert %}}

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
W&B Reports API をプログラムから扱うための Python ライブラリ。 

```python
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
     entity="entity",
     project="project",
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

---



## <kbd>class</kbd> `BarPlot`
2D の棒グラフを表示するパネル オブジェクト。 



**Attributes:**
 
 - `title` (Optional[str]): プロット上部に表示するテキスト。 
 - `metrics` (LList[MetricType]): orientation Literal["v", "h"]: 棒グラフの方向。縦 ("v") または 横 ("h") を指定。デフォルトは横 ("h")。 
 - `range_x` (Tuple[float | None, float | None]): x 軸の範囲を指定するタプル。 
 - `title_x` (Optional[str]): x 軸のラベル。 
 - `title_y` (Optional[str]): y 軸のラベル。 
 - `groupby` (Optional[str]): Report が参照する W&B project にログされた metric に基づいて run をグループ化。 
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した関数で run を集約。`mean`、`min`、`max`、`median`、`sum`、`samples`、`None` から選択。 
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化。`minmax`、`stddev`、`stderr`、`none`、`samples`、`None` から選択。 
 - `max_runs_to_show` (Optional[int]): プロットに表示する run の最大数。 
 - `max_bars_to_show` (Optional[int]): 棒グラフに表示する棒の最大数。 
 - `custom_expressions` (Optional[LList[str]]): 棒グラフで使用するカスタム式のリスト。 
 - `legend_template` (Optional[str]): 凡例のテンプレート。 
 - `font_size` ( Optional[FontSize]): 線グラフのフォントサイズ。`small`、`medium`、`large`、`auto`、`None` から選択。 
 - `line_titles` (Optional[dict]): 線のタイトル。キーは線の名前、値はタイトル。 
 - `line_colors` (Optional[dict]): 線の色。キーは線の名前、値は色。 







---



## <kbd>class</kbd> `BlockQuote`
引用テキストのブロック。 



**Attributes:**
 
 - `text` (str): 引用ブロックのテキスト。 







---



## <kbd>class</kbd> `CalloutBlock`
強調表示の呼びかけテキストのブロック。 



**Attributes:**
 
 - `text` (str): 呼びかけテキスト。 







---



## <kbd>class</kbd> `CheckedList`
チェックボックス付きの項目リスト。`CheckedList` の中に 1 つ以上の `CheckedListItem` を追加します。 



**Attributes:**
 
 - `items` (LList[CheckedListItem]): 1 つ以上の `CheckedListItem` オブジェクトのリスト。 







---



## <kbd>class</kbd> `CheckedListItem`
チェックボックス付きのリスト項目。`CheckedList` の中に 1 つ以上の `CheckedListItem` を追加します。 



**Attributes:**
 
 - `text` (str): リスト項目のテキスト。 
 - `checked` (bool): チェックボックスがオンかどうか。デフォルトは `False`。 







---



## <kbd>class</kbd> `CodeBlock`
コードのブロック。 



**Attributes:**
 
 - `code` (str): ブロック内のコード。 
 - `language` (Optional[Language]): コードの言語。指定した言語でシンタックスハイライトされます。デフォルトは `python`。`javascript`、`python`、`css`、`json`、`html`、`markdown`、`yaml` を指定可能。 







---



## <kbd>class</kbd> `CodeComparer`
2 つの異なる run 間のコードを比較するパネル オブジェクト。 



**Attributes:**
 
 - `diff` `(Literal['split', 'unified'])`: コード差分の表示形式。`split` または `unified`。 







---



## <kbd>class</kbd> `Config`
run の config オブジェクトにログされた Metrics。通常は `wandb.Run.config[name] = ...` のようにログするか、辞書のキーと 値 のペアとして config を渡します。ここで key は metric 名で、value はその値です。 



**Attributes:**
 
 - `name` (str): metric の名前。 







---



## <kbd>class</kbd> `CustomChart`
カスタム チャートを表示するパネル。チャートは Weave のクエリで定義します。 



**Attributes:**
 
 - `query` (dict): カスタム チャートを定義するクエリ。キーがフィールド名、値がクエリ。 
 - `chart_name` (str): カスタム チャートのタイトル。 
 - `chart_fields` (dict): プロットの軸を定義するキーと 値 のペア。キーがラベル、値が metric。 
 - `chart_strings` (dict): チャート内の文字列を定義するキーと 値 のペア。 




---



### <kbd>classmethod</kbd> `from_table`

```python
from_table(
    table_name: str,
    chart_fields: dict = None,
    chart_strings: dict = None
)
```

テーブルからカスタム チャートを作成します。 



**Arguments:**
 
 - `table_name` (str): テーブル名。 
 - `chart_fields` (dict): チャートに表示するフィールド。 
 - `chart_strings` (dict): チャートに表示する文字列。 




---



## <kbd>class</kbd> `Gallery`
Report や URL のギャラリーをレンダリングするブロック。 



**Attributes:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport` と `GalleryURL` オブジェクトのリスト。 







---



## <kbd>class</kbd> `GalleryReport`
ギャラリー内の Report への参照。 



**Attributes:**
 
 - `report_id` (str): Report の ID。 







---



## <kbd>class</kbd> `GalleryURL`
外部リソースへの URL。 



**Attributes:**
 
 - `url` (str): リソースの URL。 
 - `title` (Optional[str]): リソースのタイトル。 
 - `description` (Optional[str]): リソースの説明。 
 - `image_url` (Optional[str]): 表示する画像の URL。 







---



## <kbd>class</kbd> `GradientPoint`
グラデーション内の 1 点。 



**Attributes:**
 
 - `color`: 点の色。 
 - `offset`: グラデーション内での位置。0〜100 の値。 







---



## <kbd>class</kbd> `H1`
指定したテキストの H1 見出し。 



**Attributes:**
 
 - `text` (str): 見出しのテキスト。 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しを折りたたんだときに表示するブロック。 







---



## <kbd>class</kbd> `H2`
指定したテキストの H2 見出し。 



**Attributes:**
 
 - `text` (str): 見出しのテキスト。 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しを折りたたんだときに表示する 1 つ以上のブロック。 







---



## <kbd>class</kbd> `H3`
指定したテキストの H3 見出し。 



**Attributes:**
 
 - `text` (str): 見出しのテキスト。 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しを折りたたんだときに表示する 1 つ以上のブロック。 







---



## <kbd>class</kbd> `Heading`










---



## <kbd>class</kbd> `HorizontalRule`
HTML の水平線。 







---



## <kbd>class</kbd> `Image`
画像をレンダリングするブロック。 



**Attributes:**
 
 - `url` (str): 画像の URL。 
 - `caption` (str): 画像のキャプション。画像の下に表示されます。 







---



## <kbd>class</kbd> `InlineCode`
インライン コード。コードの後に改行は追加されません。 



**Attributes:**
 
 - `text` (str): Report に表示したいコード。 







---



## <kbd>class</kbd> `InlineLatex`
インラインの LaTeX マークダウン。LaTeX マークダウンの後に改行は追加されません。 



**Attributes:**
 
 - `text` (str): Report に表示したい LaTeX マークダウン。 







---



## <kbd>class</kbd> `LatexBlock`
LaTeX テキストのブロック。 



**Attributes:**
 
 - `text` (str): LaTeX テキスト。 







---



## <kbd>class</kbd> `Layout`
Report 内のパネルのレイアウト。パネルのサイズと位置を調整します。 



**Attributes:**
 
 - `x` (int): パネルの x 位置。 
 - `y` (int): パネルの y 位置。 
 - `w` (int): パネルの幅。 
 - `h` (int): パネルの高さ。 







---



## <kbd>class</kbd> `LinePlot`
2D 折れ線グラフを表示するパネル オブジェクト。 



**Attributes:**
 
 - `title` (Optional[str]): プロット上部に表示するテキスト。 
 - `x` (Optional[MetricType]): Report が参照する W&B project にログされた metric 名。x 軸に使用。 
 - `y` (LList[MetricType]): Report が参照する W&B project にログされた 1 つ以上の metric。y 軸に使用。 
 - `range_x` (Tuple[float | `None`, float | `None`]): x 軸の範囲を指定するタプル。 
 - `range_y` (Tuple[float | `None`, float | `None`]): y 軸の範囲を指定するタプル。 
 - `log_x` (Optional[bool]): x 座標を常用対数スケールで表示。 
 - `log_y` (Optional[bool]): y 座標を常用対数スケールで表示。 
 - `title_x` (Optional[str]): x 軸のラベル。 
 - `title_y` (Optional[str]): y 軸のラベル。 
 - `ignore_outliers` (Optional[bool]): `True` の場合、外れ値をプロットしません。 
 - `groupby` (Optional[str]): Report が参照する W&B project にログされた metric に基づいて run をグループ化。 
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した関数で run を集約。`mean`、`min`、`max`、`median`、`sum`、`samples`、`None` から選択。 
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化。`minmax`、`stddev`、`stderr`、`none`、`samples`、`None` から選択。 
 - `smoothing_factor` (Optional[float]): 平滑化タイプに適用する係数。0〜1 の値。 
 - `smoothing_type Optional[SmoothingType]`: 指定した分布に基づくフィルターを適用。`exponentialTimeWeighted`、`exponential`、`gaussian`、`average`、`none` から選択。 
 - `smoothing_show_original` (Optional[bool]): `True` の場合、元データも表示。 
 - `max_runs_to_show` (Optional[int]): 折れ線グラフに表示する run の最大数。 
 - `custom_expressions` (Optional[LList[str]]): データに適用するカスタム式。 
 - `plot_type Optional[LinePlotStyle]`: 生成する線グラフのタイプ。`line`、`stacked-area`、`pct-area` から選択。 
 - `font_size Optional[FontSize]`: 線グラフのフォントサイズ。`small`、`medium`、`large`、`auto`、`None` から選択。 
 - `legend_position Optional[LegendPosition]`: 凡例の配置。`north`、`south`、`east`、`west`、`None` から選択。 
 - `legend_template` (Optional[str]): 凡例のテンプレート。 
 - `aggregate` (Optional[bool]): `True` の場合、データを集約。 
 - `xaxis_expression` (Optional[str]): x 軸の式。 
 - `legend_fields` (Optional[LList[str]]): 凡例に含めるフィールド。 







---



## <kbd>class</kbd> `Link`
URL へのリンク。 



**Attributes:**
 
 - `text` (Union[str, TextWithInlineComments]): リンクのテキスト。 
 - `url` (str): リンク先の URL。 







---



## <kbd>class</kbd> `MarkdownBlock`
Markdown テキストのブロック。一般的な Markdown 構文でテキストを書きたい場合に便利です。 



**Attributes:**
 
 - `text` (str): Markdown テキスト。 







---



## <kbd>class</kbd> `MarkdownPanel`
Markdown をレンダリングするパネル。 



**Attributes:**
 
 - `markdown` (str): Markdown パネルに表示したいテキスト。 







---



## <kbd>class</kbd> `MediaBrowser`
メディア ファイルをグリッド レイアウトで表示するパネル。 



**Attributes:**
 
 - `num_columns` (Optional[int]): グリッドの列数。 
 - `media_keys` (LList[str]): メディア ファイルに対応するメディア キーのリスト。 







---



## <kbd>class</kbd> `Metric`
project にログされ、Report に表示する metric。 



**Attributes:**
 
 - `name` (str): metric 名。 







---



## <kbd>class</kbd> `OrderBy`
並べ替えに使用する metric。 



**Attributes:**
 
 - `name` (str): metric 名。 
 - `ascending` (bool): 昇順でソートするか。デフォルトは `False`。 







---



## <kbd>class</kbd> `OrderedList`
番号付きリストの項目群。 



**Attributes:**
 
 - `items` (LList[str]): 1 つ以上の `OrderedListItem` オブジェクトのリスト。 







---



## <kbd>class</kbd> `OrderedListItem`
番号付きリスト内のリスト項目。 



**Attributes:**
 
 - `text` (str): リスト項目のテキスト。 







---



## <kbd>class</kbd> `P`
段落テキスト。 



**Attributes:**
 
 - `text` (str): 段落のテキスト。 







---



## <kbd>class</kbd> `Panel`
パネル グリッド内で可視化を表示するパネル。 



**Attributes:**
 
 - `layout` (Layout): `Layout` オブジェクト。 







---



## <kbd>class</kbd> `PanelGrid`
runset とパネルで構成されるグリッド。`Runset` と `Panel` オブジェクトで runset とパネルを追加します。 

利用可能なパネル: `LinePlot`、`ScatterPlot`、`BarPlot`、`ScalarChart`、`CodeComparer`、`ParallelCoordinatesPlot`、`ParameterImportancePlot`、`RunComparer`、`MediaBrowser`、`MarkdownPanel`、`CustomChart`、`WeavePanel`、`WeavePanelSummaryTable`、`WeavePanelArtifactVersionedFile`。 





**Attributes:**
 
 - `runsets` (LList["Runset"]): 1 つ以上の `Runset` オブジェクトのリスト。 
 - `panels` (LList["PanelTypes"]): 1 つ以上の `Panel` オブジェクトのリスト。 
 - `active_runset` (int): runset 内で表示する run の数。デフォルトは 0。 
 - `custom_run_colors` (dict): キーが run 名、値が 16 進数で指定した色のキーと 値 のペア。 







---



## <kbd>class</kbd> `ParallelCoordinatesPlot`
パラレルコーディネーツプロットを表示するパネル オブジェクト。 



**Attributes:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]): 1 つ以上の `ParallelCoordinatesPlotColumn` オブジェクトのリスト。 
 - `title` (Optional[str]): プロット上部に表示するテキスト。 
 - `gradient` (Optional[LList[GradientPoint]]): グラデーション ポイントのリスト。 
 - `font_size` (Optional[FontSize]): 線グラフのフォントサイズ。`small`、`medium`、`large`、`auto`、`None` から選択。 







---



## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
パラレルコーディネーツプロット内の列。指定した `metric` の順序が、プロットの平行座標（x 軸）の順序になります。 



**Attributes:**
 
 - `metric` (str | Config | SummaryMetric): Report が参照する W&B project にログされた metric 名。 
 - `display_name` (Optional[str]): metric の表示名。 
 - `inverted` (Optional[bool]): metric を反転するか。 
 - `log` (Optional[bool]): metric に対数変換を適用するか。 







---



## <kbd>class</kbd> `ParameterImportancePlot`
選択した metric を予測するうえで各 hyperparameter がどれだけ重要かを示すパネル。 



**Attributes:**
 
 - `with_respect_to` (str): パラメータの重要度と比較する対象の metric。一般的には loss、accuracy など。指定した metric は Report が参照する project 内にログされている必要があります。 







---



## <kbd>class</kbd> `Report`
W&B Report を表すオブジェクト。返されるオブジェクトの `blocks` 属性で Report をカスタマイズできます。Report オブジェクトは自動保存されません。変更を保存するには `save()` メソッドを使用します。 



**Attributes:**
 
 - `project` (str): 読み込む W&B project 名。指定した project は Report の URL に現れます。 
 - `entity` (str): Report の所有者である W&B entity。entity は Report の URL に現れます。 
 - `title` (str): Report のタイトル。Report の先頭に H1 見出しとして表示されます。 
 - `description` (str): Report の説明。タイトルの下に表示されます。 
 - `blocks` (LList[BlockTypes]): 1 つ以上の HTML タグ、プロット、グリッド、runset などのリスト。 
 - `width` (Literal['readable', 'fixed', 'fluid']): Report の幅。'readable'、'fixed'、'fluid' から選択。 


---

#### <kbd>property</kbd> url

Report がホストされる URL。Report の URL は `https://wandb.ai/{entity}/{project_name}/reports/` の形式です。`{entity}` と `{project_name}` は、それぞれ Report が属する entity と project 名です。 



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

Report を現在の環境に読み込みます。Report がホストされている URL を渡してください。 



**Arguments:**
 
 - `url` (str): Report がホストされている URL。 
 - `as_model` (bool): True の場合、Report オブジェクトではなく model オブジェクトを返します。デフォルトは `False`。 

---



### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

Report オブジェクトに対する変更を永続化します。 

---



### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

この Report を表示する iframe を含む HTML を生成します。主に Python ノートブック内で使用されます。 



**Arguments:**
 
 - `height` (int): iframe の高さ。 
 - `hidden` (bool): True の場合、iframe を非表示にします。デフォルトは `False`。

---



## <kbd>class</kbd> `RunComparer`
Report が参照する project 内の異なる run 間で Metrics を比較するパネル。 



**Attributes:**
 
 - `diff_only` `(Optional[Literal["split", True]])`: project 内の run 間の差分のみを表示。W&B Report の UI でオン / オフを切り替え可能。 







---



## <kbd>class</kbd> `Runset`
パネル グリッドに表示する run の集合。 



**Attributes:**
 
 - `entity` (str): run が保存されている project に対して適切な権限を持つ、または所有する entity。 
 - `project` (str): run が保存されている project 名。 
 - `name` (str): run セットの名前。デフォルトは `Run set`。 
 - `query` (str): run をフィルタするクエリ文字列。 
 - `filters` (Optional[str]): run をフィルタするフィルタ文字列。 
 - `groupby` (LList[str]): グループ化に使用する metric 名のリスト。 
 - `order` (LList[OrderBy]): 並べ替えに使用する `OrderBy` オブジェクトのリスト。 
 - `custom_run_colors` (LList[OrderBy]): run ID と色の対応を表す辞書。 







---



## <kbd>class</kbd> `RunsetGroup`
runset のグループを表示する UI 要素。 



**Attributes:**
 
 - `runset_name` (str): runset の名前。 
 - `keys` (Tuple[RunsetGroupKey, ...]): グループ化のキー。1 つ以上の `RunsetGroupKey` オブジェクトを渡します。 







---



## <kbd>class</kbd> `RunsetGroupKey`
metric の種類と値で runset をグループ化します。`RunsetGroup` の一部です。グループ化する metric の種類と値をキーと 値 のペアとして指定します。 



**Attributes:**
 
 - `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): グループ化に使用する metric の種類。 
 - `value` (str): グループ化に使用する metric の値。 







---



## <kbd>class</kbd> `ScalarChart`
スカラー チャートを表示するパネル オブジェクト。 



**Attributes:**
 
 - `title` (Optional[str]): プロット上部に表示するテキスト。 
 - `metric` (MetricType): Report が参照する W&B project にログされた metric 名。 
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した関数で run を集約。`mean`、`min`、`max`、`median`、`sum`、`samples`、`None` から選択。 
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化。`minmax`、`stddev`、`stderr`、`none`、`samples`、`None` から選択。 
 - `custom_expressions` (Optional[LList[str]]): スカラー チャートで使用するカスタム式のリスト。 
 - `legend_template` (Optional[str]): 凡例のテンプレート。 
 - `font_size Optional[FontSize]`: 線グラフのフォントサイズ。`small`、`medium`、`large`、`auto`、`None` から選択。 







---



## <kbd>class</kbd> `ScatterPlot`
2D または 3D の散布図を表示するパネル オブジェクト。 



**Arguments:**
 
 - `title` (Optional[str]): プロット上部に表示するテキスト。 
 - `x Optional[SummaryOrConfigOnlyMetric]`: Report が参照する W&B project にログされた metric 名。x 軸に使用。 
 - `y Optional[SummaryOrConfigOnlyMetric]`: Report が参照する W&B project にログされた 1 つ以上の metric。y 軸にプロット。 z Optional[SummaryOrConfigOnlyMetric]: 
 - `range_x` (Tuple[float | `None`, float | `None`]): x 軸の範囲を指定するタプル。 
 - `range_y` (Tuple[float | `None`, float | `None`]): y 軸の範囲を指定するタプル。 
 - `range_z` (Tuple[float | `None`, float | `None`]): z 軸の範囲を指定するタプル。 
 - `log_x` (Optional[bool]): x 座標を常用対数スケールで表示。 
 - `log_y` (Optional[bool]): y 座標を常用対数スケールで表示。 
 - `log_z` (Optional[bool]): z 座標を常用対数スケールで表示。 
 - `running_ymin` (Optional[bool]): 移動平均（ローリング平均）を適用。 
 - `running_ymax` (Optional[bool]): 移動平均（ローリング平均）を適用。 
 - `running_ymean` (Optional[bool]): 移動平均（ローリング平均）を適用。 
 - `legend_template` (Optional[str]): 凡例のフォーマットを指定する文字列。 
 - `gradient` (Optional[LList[GradientPoint]]): プロットのカラ―グラデーションを指定するグラデーション ポイントのリスト。 
 - `font_size` (Optional[FontSize]): 線グラフのフォントサイズ。`small`、`medium`、`large`、`auto`、`None` から選択。 
 - `regression` (Optional[bool]): `True` の場合、散布図に回帰直線を描画。 







---



## <kbd>class</kbd> `SoundCloud`
SoundCloud プレーヤーをレンダリングするブロック。 



**Attributes:**
 
 - `html` (str): SoundCloud プレーヤーを埋め込むための HTML コード。 







---



## <kbd>class</kbd> `Spotify`
Spotify プレーヤーをレンダリングするブロック。 



**Attributes:**
 
 - `spotify_id` (str): トラックまたはプレイリストの Spotify ID。 







---



## <kbd>class</kbd> `SummaryMetric`
Report に表示するサマリー metric。 



**Attributes:**
 
 - `name` (str): metric 名。 







---



## <kbd>class</kbd> `TableOfContents`
Report に指定された H1、H2、H3 の HTML ブロックを使って、セクションと小見出しのリストを含むブロック。 







---



## <kbd>class</kbd> `TextWithInlineComments`
インライン コメント付きのテキスト ブロック。 



**Attributes:**
 
 - `text` (str): ブロックのテキスト。 







---



## <kbd>class</kbd> `Twitter`
Twitter フィードを表示するブロック。 



**Attributes:**
 
 - `html` (str): Twitter フィードを表示するための HTML コード。 







---



## <kbd>class</kbd> `UnorderedList`
箇条書きリストの項目群。 



**Attributes:**
 
 - `items` (LList[str]): 1 つ以上の `UnorderedListItem` オブジェクトのリスト。 







---



## <kbd>class</kbd> `UnorderedListItem`
箇条書きリスト内のリスト項目。 



**Attributes:**
 
 - `text` (str): リスト項目のテキスト。 







---



## <kbd>class</kbd> `Video`
動画をレンダリングするブロック。 



**Attributes:**
 
 - `url` (str): 動画の URL。 







---



## <kbd>class</kbd> `WeaveBlockArtifact`
W&B にログされた artifact を表示するブロック。クエリの形式は次のとおりです。 

```python
project('entity', 'project').artifact('artifact-name')
``` 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 



**Attributes:**
 
 - `entity` (str): artifact が保存されている project の所有者、または適切な権限を持つ entity。 
 - `project` (str): artifact が保存されている project。 
 - `artifact` (str): 取得する artifact の名前。 
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: artifact パネルで表示するタブ。 







---



## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
W&B の artifact にログされたバージョン付きファイルを表示するブロック。クエリの形式は次のとおりです。 

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
``` 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 



**Attributes:**
 
 - `entity` (str): artifact が保存されている project の所有者、または適切な権限を持つ entity。 
 - `project` (str): artifact が保存されている project。 
 - `artifact` (str): 取得する artifact の名前。 
 - `version` (str): 取得する artifact のバージョン。 
 - `file` (str): artifact に保存されている取得対象のファイル名。 







---



## <kbd>class</kbd> `WeaveBlockSummaryTable`
W&B Table、pandas DataFrame、プロット、その他 W&B にログされた値を表示するブロック。クエリの形式は次のとおりです。 

```python
project('entity', 'project').runs.summary['value']
``` 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 



**Attributes:**
 
 - `entity` (str): 値がログされている project の所有者、または適切な権限を持つ entity。 
 - `project` (str): 値がログされている project。 
 - `table_name` (str): テーブル、DataFrame、プロット、または値の名前。 







---



## <kbd>class</kbd> `WeavePanel`
クエリを使ってカスタム コンテンツを表示できる空のクエリ パネル。 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 







---



## <kbd>class</kbd> `WeavePanelArtifact`
W&B にログされた artifact を表示するパネル。 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 



**Attributes:**
 
 - `artifact` (str): 取得する artifact の名前。 
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: artifact パネルで表示するタブ。 







---



## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
W&B の artifact にログされたバージョン付きファイルを表示するパネル。 

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
``` 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 



**Attributes:**
 
 - `artifact` (str): 取得する artifact の名前。 
 - `version` (str): 取得する artifact のバージョン。 
 - `file` (str): artifact に保存されている取得対象のファイル名。 







---



## <kbd>class</kbd> `WeavePanelSummaryTable`
W&B Table、pandas DataFrame、プロット、その他 W&B にログされた値を表示するパネル。クエリの形式は次のとおりです。 

```python
runs.summary['value']
``` 

API 名に含まれる "Weave" は、LLM のトラッキングと評価に使用される W&B Weave ツールキットを指すものではありません。 



**Attributes:**
 
 - `table_name` (str): テーブル、DataFrame、プロット、または値の名前。