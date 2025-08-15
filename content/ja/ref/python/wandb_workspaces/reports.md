---
title: レポート
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-reports
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}




{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
W&B Reports API をプログラムから操作するための Python ライブラリ。

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
2 次元の棒グラフを表示するパネルオブジェクト。



**属性:**
 
 - `title` (Optional[str]): グラフ上部に表示されるテキスト。
 - `metrics` (LList[MetricType]): 棒グラフの向きを指定します。`vertical ("v")` または `horizontal ("h")` のいずれかを選択。デフォルトは `horizontal ("h")`。
 - `range_x` (Tuple[float | None, float | None]): x 軸の範囲を指定するタプル。
 - `title_x` (Optional[str]): x 軸のラベル。
 - `title_y` (Optional[str]): y 軸のラベル。
 - `groupby` (Optional[str]): W&B プロジェクトに記録されたメトリクスで run をグループ化。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した集約関数で run を集約。`mean`, `min`, `max`, `median`, `sum`, `samples`, または `None` から選択可能。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲ごとに run をグループ化。`minmax`, `stddev`, `stderr`, `none`, `samples`, `None` から選択可能。
 - `max_runs_to_show` (Optional[int]): グラフに表示される最大 run 数。
 - `max_bars_to_show` (Optional[int]): 棒グラフ内に表示する最大バー数。
 - `custom_expressions` (Optional[LList[str]]): 棒グラフで使用するカスタム式のリスト。
 - `legend_template` (Optional[str]): 凡例のテンプレート。
 - `font_size` ( Optional[FontSize]): 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None` のいずれか。
 - `line_titles` (Optional[dict]): 線のタイトル。キーが線の名前、値がタイトル。
 - `line_colors` (Optional[dict]): 線の色。キーが線の名前、値が色を示します。







---



## <kbd>class</kbd> `BlockQuote`
引用テキストのブロック。



**属性:**
 
 - `text` (str): 引用のテキスト。







---



## <kbd>class</kbd> `CalloutBlock`
強調したいテキストのブロック。



**属性:**
 
 - `text` (str): 強調テキスト。







---



## <kbd>class</kbd> `CheckedList`
チェックボックス付きリスト項目のリスト。 `CheckedList` 内に 1 つ以上の `CheckedListItem` を追加します。



**属性:**
 
 - `items` (LList[CheckedListItem]): 1 つ以上の `CheckedListItem` オブジェクトのリスト。







---



## <kbd>class</kbd> `CheckedListItem`
チェックボックス付きリストの項目。 `CheckedList` 内に 1 つ以上追加します。



**属性:**
 
 - `text` (str): 項目のテキスト。
 - `checked` (bool): チェックボックスがオンかどうか。デフォルトは `False`。







---



## <kbd>class</kbd> `CodeBlock`
コードのブロック。



**属性:**
 
 - `code` (str): ブロック内のコード。
 - `language` (Optional[Language]): コードの言語。ハイライトのために使用。デフォルトは `python`。`javascript`, `python`, `css`, `json`, `html`, `markdown`, `yaml` などが選択可能。







---



## <kbd>class</kbd> `CodeComparer`
2 つの異なる run のコードを比較するパネルオブジェクト。



**属性:**
 
 - `diff` `(Literal['split', 'unified'])`: コードの差分表示方法。`split` または `unified`。







---



## <kbd>class</kbd> `Config`
run の config オブジェクトに記録されたメトリクス。`wandb.Run.config[name] = ...` や、キーがメトリクス名、値がその値の key-value 辞書として記録されることが一般的です。



**属性:**
 
 - `name` (str): メトリクスの名前。







---



## <kbd>class</kbd> `CustomChart`
カスタムチャートを表示するパネル。チャートは weave クエリで定義されます。



**属性:**
 
 - `query` (dict): カスタムチャートを定義するクエリ。キーがフィールド名、値がクエリ。
 - `chart_name` (str): カスタムチャートのタイトル。
 - `chart_fields` (dict): プロットの軸を定義する key-value ペア。キーがラベル、値がメトリクス。
 - `chart_strings` (dict): チャート内の文字列を定義する key-value ペア。




---



### <kbd>classmethod</kbd> `from_table`

```python
from_table(
    table_name: str,
    chart_fields: dict = None,
    chart_strings: dict = None
)
```

テーブルからカスタムチャートを作成します。



**引数:**
 
 - `table_name` (str): テーブル名。
 - `chart_fields` (dict): チャートに表示するフィールド。
 - `chart_strings` (dict): チャートに表示する文字列。




---



## <kbd>class</kbd> `Gallery`
Reports や URL のギャラリーを表示するブロック。



**属性:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport` と `GalleryURL` オブジェクトのリスト。







---



## <kbd>class</kbd> `GalleryReport`
ギャラリー内のレポートへの参照。



**属性:**
 
 - `report_id` (str): レポートの ID。







---



## <kbd>class</kbd> `GalleryURL`
外部リソースへの URL。



**属性:**
 
 - `url` (str): リソースの URL。
 - `title` (Optional[str]): リソースのタイトル。
 - `description` (Optional[str]): リソースの説明。
 - `image_url` (Optional[str]): 表示する画像の URL。







---



## <kbd>class</kbd> `GradientPoint`
勾配の中の 1 点。



**属性:**
 
 - `color`: ポイントの色。
 - `offset`: 勾配上の位置 (0～100 の間の値)。







---



## <kbd>class</kbd> `H1`
指定したテキストの H1 見出し。



**属性:**
 
 - `text` (str): 見出しのテキスト。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示するブロックのリスト。







---



## <kbd>class</kbd> `H2`
指定したテキストの H2 見出し。



**属性:**
 
 - `text` (str): 見出しのテキスト。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示する 1 つ以上のブロック。







---



## <kbd>class</kbd> `H3`
指定したテキストの H3 見出し。



**属性:**
 
 - `text` (str): 見出しのテキスト。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示する 1 つ以上のブロック。







---



## <kbd>class</kbd> `Heading`










---



## <kbd>class</kbd> `HorizontalRule`
HTML の水平線。







---



## <kbd>class</kbd> `Image`
画像を表示するブロック。



**属性:**
 
 - `url` (str): 画像の URL。
 - `caption` (str): 画像のキャプション。画像の下に表示されます。







---



## <kbd>class</kbd> `InlineCode`
インラインコード。コードの後に改行は追加されません。



**属性:**
 
 - `text` (str): レポート内に表示したいコード。







---



## <kbd>class</kbd> `InlineLatex`
インライン LaTeX マークダウン。LaTeX の後ろに改行は入りません。



**属性:**
 
 - `text` (str): レポートに表示する LaTeX マークダウン。







---



## <kbd>class</kbd> `LatexBlock`
LaTeX テキストのブロック。



**属性:**
 
 - `text` (str): LaTeX テキスト。







---



## <kbd>class</kbd> `Layout`
レポート内パネルのレイアウト。パネルのサイズや位置を調整できます。



**属性:**
 
 - `x` (int): パネルの x 座標。
 - `y` (int): パネルの y 座標。
 - `w` (int): パネルの幅。
 - `h` (int): パネルの高さ。







---



## <kbd>class</kbd> `LinePlot`
2 次元折れ線グラフのパネルオブジェクト。



**属性:**
 
 - `title` (Optional[str]): グラフ上部に表示されるテキスト。
 - `x` (Optional[MetricType]): W&B プロジェクトで記録したメトリクス名。x 軸として使用。
 - `y` (LList[MetricType]): W&B プロジェクトで記録した 1 つ以上のメトリクス。y 軸として使用。
 - `range_x` (Tuple[float | `None`, float | `None`]): x 軸の範囲を指定するタプル。
 - `range_y` (Tuple[float | `None`, float | `None`]): y 軸の範囲を指定するタプル。
 - `log_x` (Optional[bool]): x 座標を底 10 の対数スケールで表示。
 - `log_y` (Optional[bool]): y 座標を底 10 の対数スケールで表示。
 - `title_x` (Optional[str]): x 軸のラベル。
 - `title_y` (Optional[str]): y 軸のラベル。
 - `ignore_outliers` (Optional[bool]): `True` なら外れ値をプロットしません。
 - `groupby` (Optional[str]): W&B プロジェクト内のメトリクスで run をグループ化。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定の集約関数で run を集約。`mean`, `min`, `max`, `median`, `sum`, `samples`, `None`。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲ごとに run をグループ化。`minmax`, `stddev`, `stderr`, `none`, `samples`, `None`。
 - `smoothing_factor` (Optional[float]): 平滑化に使う平滑化係数。0 から 1 の範囲。
 - `smoothing_type Optional[SmoothingType]`: 指定の分布に基づきフィルタ適用。`exponentialTimeWeighted`, `exponential`, `gaussian`, `average`, `none`。
 - `smoothing_show_original` (Optional[bool]): `True` なら元データも表示。
 - `max_runs_to_show` (Optional[int]): 折れ線グラフで表示する最大 run 数。
 - `custom_expressions` (Optional[LList[str]]): データに適用するカスタム式。
 - `plot_type Optional[LinePlotStyle]`: 線グラフの種類を指定。`line`, `stacked-area`, `pct-area`。
 - `font_size Optional[FontSize]`: 折れ線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None`。
 - `legend_position Optional[LegendPosition]`: 凡例の位置。`north`, `south`, `east`, `west`, `None`。
 - `legend_template` (Optional[str]): 凡例テンプレート。
 - `aggregate` (Optional[bool]): `True` ならデータを集約。
 - `xaxis_expression` (Optional[str]): x 軸の式。
 - `legend_fields` (Optional[LList[str]]): 凡例に表示するフィールド。







---



## <kbd>class</kbd> `Link`
URL へのリンク。



**属性:**
 
 - `text` (Union[str, TextWithInlineComments]): リンクのテキスト。
 - `url` (str): リンク先の URL。







---



## <kbd>class</kbd> `MarkdownBlock`
マークダウンテキストのブロック。一般的なマークダウン記法を使いたい場合に便利です。



**属性:**
 
 - `text` (str): マークダウンテキスト。







---



## <kbd>class</kbd> `MarkdownPanel`
マークダウンを表示するパネル。



**属性:**
 
 - `markdown` (str): パネル内に表示するマークダウンテキスト。







---



## <kbd>class</kbd> `MediaBrowser`
メディアファイルをグリッドレイアウトで表示するパネル。



**属性:**
 
 - `num_columns` (Optional[int]): グリッドのカラム数。
 - `media_keys` (LList[str]): メディアファイルに対応するメディアキーのリスト。







---



## <kbd>class</kbd> `Metric`
プロジェクトに記録され、レポート内に表示されるメトリクス。



**属性:**
 
 - `name` (str): メトリクスの名前。







---



## <kbd>class</kbd> `OrderBy`
ソートに使うメトリクス。



**属性:**
 
 - `name` (str): メトリクスの名前。
 - `ascending` (bool): 昇順でソートするかどうか。デフォルトは `False`。







---



## <kbd>class</kbd> `OrderedList`
番号付きリストの項目一覧。



**属性:**
 
 - `items` (LList[str]): 1つ以上の `OrderedListItem` オブジェクトのリスト。







---



## <kbd>class</kbd> `OrderedListItem`
番号付きリストの項目。



**属性:**
 
 - `text` (str): 項目のテキスト。







---



## <kbd>class</kbd> `P`
テキストの段落。



**属性:**
 
 - `text` (str): 段落のテキスト。







---



## <kbd>class</kbd> `Panel`
パネルグリッド上で可視化を表示するパネル。



**属性:**
 
 - `layout` (Layout): `Layout` オブジェクト。







---



## <kbd>class</kbd> `PanelGrid`
runset とパネルで構成されるグリッド。`Runset` と `Panel` オブジェクトを追加してください。

利用できるパネルは、`LinePlot`、`ScatterPlot`、`BarPlot`、`ScalarChart`、`CodeComparer`、`ParallelCoordinatesPlot`、`ParameterImportancePlot`、`RunComparer`、`MediaBrowser`、`MarkdownPanel`、`CustomChart`、`WeavePanel`、`WeavePanelSummaryTable`、`WeavePanelArtifactVersionedFile` などです。





**属性:**
 
 - `runsets` (LList["Runset"]): 1つ以上の `Runset` オブジェクトのリスト。
 - `panels` (LList["PanelTypes"]): 1つ以上の `Panel` オブジェクトのリスト。
 - `active_runset` (int): runset 内に表示する run の数。デフォルトは 0。
 - `custom_run_colors` (dict): run の名前をキー、16 進カラー値を値とするカラーマッピング。







---



## <kbd>class</kbd> `ParallelCoordinatesPlot`
パラレル座標プロットを表示するパネルオブジェクト。



**属性:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]): 1 つ以上の `ParallelCoordinatesPlotColumn` オブジェクトのリスト。
 - `title` (Optional[str]): グラフ上部に表示されるテキスト。
 - `gradient` (Optional[LList[GradientPoint]]): 勾配ポイントのリスト。
 - `font_size` (Optional[FontSize]): 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None` から選択。







---



## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
パラレル座標プロット内のカラム。指定するメトリクスの順序が、パラレル座標プロットの横軸（x 軸）の順序を決めます。



**属性:**
 
 - `metric` (str | Config | SummaryMetric): W&B プロジェクトに記録されたメトリクス名。
 - `display_name` (Optional[str]): メトリクスの表示名。
 - `inverted` (Optional[bool]): メトリクスを反転するかどうか。
 - `log` (Optional[bool]): メトリクスに対して対数変換を行うかどうか。







---



## <kbd>class</kbd> `ParameterImportancePlot`
どのハイパーパラメーターが指定したメトリクスの予測にどれだけ重要かを示すパネル。



**属性:**
 
 - `with_respect_to` (str): パラメータの重要度を比較したいメトリクス。一般的には loss や accuracy など。指定するメトリクスは、そのプロジェクトでレポートが参照できる必要があります。







---



## <kbd>class</kbd> `Report`
W&B の Report を表すオブジェクト。戻り値のオブジェクトの `blocks` 属性でレポートをカスタマイズできます。Report オブジェクトは自動で保存されません。変更を反映するには `save()` メソッドを使ってください。



**属性:**
 
 - `project` (str): レポートで表示したい W&B プロジェクトの名前。指定したプロジェクトはレポートの URL に表示されます。
 - `entity` (str): レポートの所有者である W&B entity。entity はレポートの URL にも表示されます。
 - `title` (str): レポートのタイトル。タイトルは Report のトップに H1 見出しとして表示されます。
 - `description` (str): レポートの説明。説明はタイトルの下に表示されます。
 - `blocks` (LList[BlockTypes]): 1 つ以上の HTML タグ、プロット、グリッド、runset などのリスト。
 - `width` (Literal['readable', 'fixed', 'fluid']): レポートの横幅。'readable', 'fixed', 'fluid' から選択。


---

#### <kbd>property</kbd> url

レポートがホストされている URL。URL の形式は `https://wandb.ai/{entity}/{project_name}/reports/` です。`{entity}` と `{project_name}` には、そのレポートが属する entity とプロジェクト名が設定されます。



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

レポートを現在の環境にロードします。レポートがホストされている URL を渡してください。



**引数:**
 
 - `url` (str): レポートがホストされている URL。
 - `as_model` (bool): True の場合、Report オブジェクトではなく Model オブジェクトを返します。デフォルトは `False`。

---



### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

Report オブジェクトへの変更を保存します。

---



### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

このレポートを表示する iframe を含む HTML を生成します。主に Python ノートブック内でよく使われます。



**引数:**
 
 - `height` (int): iframe の高さ。
 - `hidden` (bool): True なら iframe を非表示に。デフォルトは `False`。

---



## <kbd>class</kbd> `RunComparer`
プロジェクト内の複数の run 間でメトリクスを比較するパネル。



**属性:**
 
 - `diff_only` `(Optional[Literal["split", True]])`: プロジェクト内 run の差分のみを表示します。この機能は W&B Report UI でオン/オフ可能です。







---



## <kbd>class</kbd> `Runset`
パネルグリッド内で表示される run のセット。



**属性:**
 
 - `entity` (str): run が保存されているプロジェクトの所有者または権限を持つ entity。
 - `project` (str): run が保存されているプロジェクト名。
 - `name` (str): run セットの名前。デフォルトは `Run set`。
 - `query` (str): run のフィルタリングに使うクエリ文字列。
 - `filters` (Optional[str]): run のフィルター文字列。
 - `groupby` (LList[str]): グループ化するメトリクス名のリスト。
 - `order` (LList[OrderBy]): 並び替えに使用する `OrderBy` オブジェクトのリスト。
 - `custom_run_colors` (LList[OrderBy]): run ID とカラーのマッピング辞書。







---



## <kbd>class</kbd> `RunsetGroup`
runset のグループを表示する UI 要素。



**属性:**
 
 - `runset_name` (str): runset の名前。
 - `keys` (Tuple[RunsetGroupKey, ...]): グループ化に使う `RunsetGroupKey` オブジェクトのタプル。







---



## <kbd>class</kbd> `RunsetGroupKey`
特定のメトリクス種別や値で runset をグループ化します。`RunsetGroup` の一部として、グループ化のメトリクス種別と値を key-value で指定します。



**属性:**
 
 - `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): グループ化するメトリクス種別。
 - `value` (str): グループ化に使うメトリクス値。







---



## <kbd>class</kbd> `ScalarChart`
スカラー値チャートを表示するパネルオブジェクト。



**属性:**
 
 - `title` (Optional[str]): グラフ上部に表示するテキスト。
 - `metric` (MetricType): W&B プロジェクトで記録したメトリクス名。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した集約関数で run を集約。`mean`, `min`, `max`, `median`, `sum`, `samples`, `None`。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲ごとに run をグループ化。`minmax`, `stddev`, `stderr`, `none`, `samples`, `None`。
 - `custom_expressions` (Optional[LList[str]]): スカラーチャートで使用するカスタム式のリスト。
 - `legend_template` (Optional[str]): 凡例のテンプレート。
 - `font_size Optional[FontSize]`: 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None`。







---



## <kbd>class</kbd> `ScatterPlot`
2次元または 3 次元散布図を表示するパネルオブジェクト。



**引数:**
 
 - `title` (Optional[str]): グラフ上部に表示するテキスト。
 - `x Optional[SummaryOrConfigOnlyMetric]`: W&B プロジェクトで記録したメトリクス名。x 軸として使用。
 - `y Optional[SummaryOrConfigOnlyMetric]`: W&B プロジェクトに記録された 1 つ以上のメトリクス。y 軸にプロットされます。z Optional[SummaryOrConfigOnlyMetric]:
 - `range_x` (Tuple[float | `None`, float | `None`]): x 軸の範囲を指定するタプル。
 - `range_y` (Tuple[float | `None`, float | `None`]): y 軸の範囲を指定するタプル。
 - `range_z` (Tuple[float | `None`, float | `None`]): z 軸の範囲を指定するタプル。
 - `log_x` (Optional[bool]): x 座標を底 10 の対数スケールで表示。
 - `log_y` (Optional[bool]): y 座標を底 10 の対数スケールで表示。
 - `log_z` (Optional[bool]): z 座標を底 10 の対数スケールで表示。
 - `running_ymin` (Optional[bool]): 移動平均またはローリング平均を適用。
 - `running_ymax` (Optional[bool]): 移動平均またはローリング平均を適用。
 - `running_ymean` (Optional[bool]): 移動平均またはローリング平均を適用。
 - `legend_template` (Optional[str]): 凡例の書式となる文字列。
 - `gradient` (Optional[LList[GradientPoint]]): プロットのカラ―グラデーションを指定する勾配ポイントのリスト。
 - `font_size` (Optional[FontSize]): 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None` から選択。
 - `regression` (Optional[bool]): `True` なら散布図に回帰直線を表示。







---



## <kbd>class</kbd> `SoundCloud`
SoundCloud プレーヤーを表示するブロック。



**属性:**
 
 - `html` (str): SoundCloud プレーヤーを埋め込む HTML コード。







---



## <kbd>class</kbd> `Spotify`
Spotify プレーヤーを表示するブロック。



**属性:**
 
 - `spotify_id` (str): トラックやプレイリストの Spotify ID。







---



## <kbd>class</kbd> `SummaryMetric`
レポートに表示するサマリーメトリクス。



**属性:**
 
 - `name` (str): メトリクスの名前。







---



## <kbd>class</kbd> `TableOfContents`
Report 内に指定された H1/H2/H3 HTML ブロックを用いて、セクションやサブセクションのリストを表示するブロック。







---



## <kbd>class</kbd> `TextWithInlineComments`
インラインコメント付きテキストブロック。



**属性:**
 
 - `text` (str): ブロックのテキスト。







---



## <kbd>class</kbd> `Twitter`
Twitter フィードを表示するブロック。



**属性:**
 
 - `html` (str): Twitter フィードを表示する HTML コード。







---



## <kbd>class</kbd> `UnorderedList`
箇条書きリストのアイテム一覧。



**属性:**
 
 - `items` (LList[str]): 1つ以上の `UnorderedListItem` オブジェクトのリスト。







---



## <kbd>class</kbd> `UnorderedListItem`
箇条書きリストのアイテム。



**属性:**
 
 - `text` (str): 項目のテキスト。







---



## <kbd>class</kbd> `Video`
動画を表示するブロック。



**属性:**
 
 - `url` (str): 動画の URL。







---



## <kbd>class</kbd> `WeaveBlockArtifact`
W&B に記録された artifact を表示するブロック。クエリの形式は

```python
project('entity', 'project').artifact('artifact-name')
```

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。



**属性:**
 
 - `entity` (str): artifact が保管されているプロジェクトの所有者、または十分な権限を持つ entity。
 - `project` (str): artifact が保存されているプロジェクト名。
 - `artifact` (str): 取得したい artifact の名前。
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: artifact パネルで表示するタブ。







---



## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
W&B artifact に記録されたバージョン付きファイルを表示するブロック。クエリの形式は

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。



**属性:**
 
 - `entity` (str): artifact が保管されているプロジェクトの所有者、または十分な権限を持つ entity。
 - `project` (str): artifact が保存されているプロジェクト名。
 - `artifact` (str): 取得したい artifact の名前。
 - `version` (str): 取得したい artifact のバージョン。
 - `file` (str): artifact 内に保存されているファイル名。







---



## <kbd>class</kbd> `WeaveBlockSummaryTable`
W&B Table、pandas DataFrame、プロット、その他の値を表示するブロック。クエリ形式は

```python
project('entity', 'project').runs.summary['value']
```

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。



**属性:**
 
 - `entity` (str): 値が記録されているプロジェクトの所有者、または十分な権限を持つ entity。
 - `project` (str): 値が記録されているプロジェクト名。
 - `table_name` (str): テーブル名、DataFrame 名、プロット名、または値の名前。







---



## <kbd>class</kbd> `WeavePanel`
クエリを指定してカスタム内容を表示できる空のパネル。

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。







---



## <kbd>class</kbd> `WeavePanelArtifact`
W&B に記録された artifact を表示するパネル。

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。



**属性:**
 
 - `artifact` (str): 取得したい artifact の名前。
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: artifact パネルで表示するタブ。







---



## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
W&B artifact に記録されたバージョン付きファイルを表示するパネル。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。



**属性:**
 
 - `artifact` (str): 取得したい artifact の名前。
 - `version` (str): 取得したい artifact のバージョン。
 - `file` (str): artifact 内に保存されているファイル名。







---



## <kbd>class</kbd> `WeavePanelSummaryTable`
W&B Table、pandas DataFrame、プロット、または W&B に記録した他の値を表示するパネル。クエリの例：

```python
runs.summary['value']
```

API 名中の "Weave" は LLM のトラッキング・評価用の W&B Weave ツールキットを指しません。



**属性:**
 
 - `table_name` (str): テーブル名、DataFrame 名、プロット名、または値の名前。
