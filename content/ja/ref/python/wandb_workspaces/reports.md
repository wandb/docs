---
title: レポート
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}




{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
W&B Reports API をプログラムで操作するための Python ライブラリです。

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
2D バープロットを表示するパネルオブジェクトです。



**属性:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキストです。
 - `metrics` (LList[MetricType]): orientation Literal["v", "h"]: バープロットの向きです。縦（"v"）または横（"h"）が指定できます。デフォルトは横（"h"）です。
 - `range_x` (Tuple[float | None, float | None]): x軸の範囲を指定するタプルです。
 - `title_x` (Optional[str]): x軸のラベルです。
 - `title_y` (Optional[str]): y軸のラベルです。
 - `groupby` (Optional[str]): W&B Project にログされたメトリクスに基づいて run をグループ化します。Report はこの情報を取得します。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した関数で run を集約します。`mean`, `min`, `max`, `median`, `sum`, `samples`, または `None` から選択できます。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。`minmax`, `stddev`, `stderr`, `none`, `samples`, または `None` から選択できます。
 - `max_runs_to_show` (Optional[int]): プロット上に表示する run の最大数です。
 - `max_bars_to_show` (Optional[int]): バープロットで表示できるバーの最大数です。
 - `custom_expressions` (Optional[LList[str]]): バープロットで利用するカスタム式のリストです。
 - `legend_template` (Optional[str]): 凡例のテンプレートです。
 - `font_size` ( Optional[FontSize]): 線グラフのフォントサイズです。`small`, `medium`, `large`, `auto`, または `None` から選択できます。
 - `line_titles` (Optional[dict]): 線のタイトルです。キーが線の名前、値がタイトルです。
 - `line_colors` (Optional[dict]): 線の色です。キーが線の名前、値が色です。







---



## <kbd>class</kbd> `BlockQuote`
引用テキストのブロックです。



**属性:**
 
 - `text` (str): ブロック引用文のテキストです。







---



## <kbd>class</kbd> `CalloutBlock`
注記テキストのブロックです。



**属性:**
 
 - `text` (str): コールアウトとなるテキストです。







---



## <kbd>class</kbd> `CheckedList`
チェックボックス付き項目のリストです。`CheckedList` の中に 1 つ以上の `CheckedListItem` を追加します。



**属性:**
 
 - `items` (LList[CheckedListItem]): 1つ以上の `CheckedListItem` オブジェクトのリストです。







---



## <kbd>class</kbd> `CheckedListItem`
チェックボックス付きリスト項目です。`CheckedList` の中に 1 つ以上追加します。



**属性:**
 
 - `text` (str): リスト項目のテキストです。
 - `checked` (bool): チェックありかどうか。デフォルトは `False` です。







---



## <kbd>class</kbd> `CodeBlock`
コードブロックです。



**属性:**
 
 - `code` (str): このブロック内のコードです。
 - `language` (Optional[Language]): コードの言語。指定した言語はシンタックスハイライトに使用されます。デフォルト値は `python`。`javascript`, `python`, `css`, `json`, `html`, `markdown`, `yaml` から選べます。







---



## <kbd>class</kbd> `CodeComparer`
2つの異なる run 間のコードを比較するパネルオブジェクトです。



**属性:**
 
 - `diff` `(Literal['split', 'unified'])`: コード差分の表示方法。`split` または `unified` から選べます。







---



## <kbd>class</kbd> `Config`
run の config オブジェクトへログされたメトリクスです。`wandb.Run.config[name] = ...` としてログしたり、key-value ペアの辞書として config を渡すのが一般的です。key はメトリクス名、value はその値です。



**属性:**
 
 - `name` (str): メトリクスの名前です。







---



## <kbd>class</kbd> `CustomChart`
カスタムチャートを表示するパネルです。チャートは weave クエリで定義します。



**属性:**
 
 - `query` (dict): カスタムチャートを定義するクエリ。キーがフィールド名、値がクエリです。
 - `chart_name` (str): カスタムチャートのタイトルです。
 - `chart_fields` (dict): プロットの軸を定義する key-value ペア。key がラベル、value がメトリクスです。
 - `chart_strings` (dict): チャート内の文字列を定義する key-value ペアです。




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
 - `chart_fields` (dict): チャートで表示するフィールド。
 - `chart_strings` (dict): チャートで表示する文字列。




---



## <kbd>class</kbd> `Gallery`
Reports や URL のギャラリーを表示するブロックです。



**属性:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport` および `GalleryURL` オブジェクトのリストです。







---



## <kbd>class</kbd> `GalleryReport`
ギャラリーにあるレポートへの参照です。



**属性:**
 
 - `report_id` (str): レポートの ID です。







---



## <kbd>class</kbd> `GalleryURL`
外部リソースへの URL です。



**属性:**
 
 - `url` (str): リソースの URL です。
 - `title` (Optional[str]): リソースのタイトル。
 - `description` (Optional[str]): リソースの説明文。
 - `image_url` (Optional[str]): 表示する画像の URL。







---



## <kbd>class</kbd> `GradientPoint`
勾配上の1点。



**属性:**
 
 - `color`: この点の色。
 - `offset`: 勾配上での位置（0～100で指定）。







---



## <kbd>class</kbd> `H1`
指定したテキストの H1 見出しです。



**属性:**
 
 - `text` (str): 見出しのテキストです。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれているときに表示するブロック。







---



## <kbd>class</kbd> `H2`
指定したテキストの H2 見出しです。



**属性:**
 
 - `text` (str): 見出しのテキストです。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示するブロック。







---



## <kbd>class</kbd> `H3`
指定したテキストの H3 見出しです。



**属性:**
 
 - `text` (str): 見出しのテキストです。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示するブロック。







---



## <kbd>class</kbd> `Heading`










---



## <kbd>class</kbd> `HorizontalRule`
HTML の水平線です。







---



## <kbd>class</kbd> `Image`
画像を表示するブロックです。



**属性:**
 
 - `url` (str): 画像の URL。
 - `caption` (str): 画像のキャプション。画像の下に表示されます。







---



## <kbd>class</kbd> `InlineCode`
インラインコード。コードのあとに改行は入りません。



**属性:**
 
 - `text` (str): レポート内に表示したいコード。







---



## <kbd>class</kbd> `InlineLatex`
インラインで LaTeX マークダウンを記述します。コードのあとに改行は入りません。



**属性:**
 
 - `text` (str): レポート内に表示したい LaTeX マークダウン。







---



## <kbd>class</kbd> `LatexBlock`
LaTeX テキストのブロックです。



**属性:**
 
 - `text` (str): LaTeX テキスト。







---



## <kbd>class</kbd> `Layout`
レポート内パネルのレイアウトです。パネルのサイズや位置を調整します。



**属性:**
 
 - `x` (int): パネルの x 位置。
 - `y` (int): パネルの y 位置。
 - `w` (int): パネルの幅。
 - `h` (int): パネルの高さ。







---



## <kbd>class</kbd> `LinePlot`
2D 線グラフのパネルオブジェクト。



**属性:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `x` (Optional[MetricType]): W&B Project にログされたメトリクス名。Report はこの情報を x軸として取得します。
 - `y` (LList[MetricType]): W&B Project にログされた1つ以上のメトリクス名。Report はこの情報を y軸として取得します。
 - `range_x` (Tuple[float | `None`, float | `None`]): x軸の範囲指定タプル。
 - `range_y` (Tuple[float | `None`, float | `None`]): y軸の範囲指定タプル。
 - `log_x` (Optional[bool]): x座標を常用対数スケールで表示します。
 - `log_y` (Optional[bool]): y座標を常用対数スケールで表示します。
 - `title_x` (Optional[str]): x軸のラベル。
 - `title_y` (Optional[str]): y軸のラベル。
 - `ignore_outliers` (Optional[bool]): `True` の場合、外れ値をプロットしません。
 - `groupby` (Optional[str]): W&B Project にログされたメトリクスに基づいて run をグループ化します。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定関数で run を集約。`mean`, `min`, `max`, `median`, `sum`, `samples`, `None` が利用可能です。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化。`minmax`, `stddev`, `stderr`, `none`, `samples`, `None` など。
 - `smoothing_factor` (Optional[float]): 平滑化に適用する係数。0～1の間で指定。
 - `smoothing_type Optional[SmoothingType]`: 指定した分布でフィルターを適用。`exponentialTimeWeighted`, `exponential`, `gaussian`, `average`, `none` から選択。
 - `smoothing_show_original` (Optional[bool]): `True` の場合、元データも表示。
 - `max_runs_to_show` (Optional[int]): 線グラフで表示する run の最大数。
 - `custom_expressions` (Optional[LList[str]]): データに適用するカスタム式。
 - `plot_type Optional[LinePlotStyle]`: 生成する線グラフのタイプ。`line`, `stacked-area`, `pct-area` から選べます。
 - `font_size Optional[FontSize]`: 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None` から選択。
 - `legend_position Optional[LegendPosition]`: 凡例の配置。`north`, `south`, `east`, `west` など。
 - `legend_template` (Optional[str]): 凡例テンプレート。
 - `aggregate` (Optional[bool]): `True` ならデータを集約。
 - `xaxis_expression` (Optional[str]): x軸の式。
 - `legend_fields` (Optional[LList[str]]): 凡例に含めるフィールド群。







---



## <kbd>class</kbd> `Link`
URL へのリンクです。



**属性:**
 
 - `text` (Union[str, TextWithInlineComments]): リンクテキスト。
 - `url` (str): リンク先の URL。







---



## <kbd>class</kbd> `MarkdownBlock`
マークダウンテキストのブロックです。一般的なマークダウンシンタックスを使いたいときに便利です。



**属性:**
 
 - `text` (str): マークダウンテキスト。







---



## <kbd>class</kbd> `MarkdownPanel`
マークダウンを表示するパネルです。



**属性:**
 
 - `markdown` (str): マークダウンパネルに表示したいテキスト。







---



## <kbd>class</kbd> `MediaBrowser`
メディアファイルをグリッド状に表示するパネルです。



**属性:**
 
 - `num_columns` (Optional[int]): グリッドの列数。
 - `media_keys` (LList[str]): メディアファイルに対応する media key のリスト。







---



## <kbd>class</kbd> `Metric`
プロジェクト内でログされたメトリクスをレポートで表示します。



**属性:**
 
 - `name` (str): メトリクスの名前。







---



## <kbd>class</kbd> `OrderBy`
並び替え用のメトリクスです。



**属性:**
 
 - `name` (str): メトリクスの名前。
 - `ascending` (bool): 昇順で並べるか。デフォルトは `False`。







---



## <kbd>class</kbd> `OrderedList`
番号付きリストの項目リストです。



**属性:**
 
 - `items` (LList[str]): 1つ以上の `OrderedListItem` オブジェクトのリスト。







---



## <kbd>class</kbd> `OrderedListItem`
番号付きリストの項目です。



**属性:**
 
 - `text` (str): リスト項目のテキスト。







---



## <kbd>class</kbd> `P`
段落テキストです。



**属性:**
 
 - `text` (str): 段落のテキスト。







---



## <kbd>class</kbd> `Panel`
パネルグリッド内で可視化を表示するパネルです。



**属性:**
 
 - `layout` (Layout): `Layout` オブジェクトです。







---



## <kbd>class</kbd> `PanelGrid`
runset とパネルで構成されるグリッドです。それぞれ `Runset` および `Panel` オブジェクトで追加します。

利用可能なパネル: `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`。





**属性:**
 
 - `runsets` (LList["Runset"]): 1つ以上の `Runset` オブジェクトのリスト。
 - `panels` (LList["PanelTypes"]): 1つ以上の `Panel` オブジェクトのリスト。
 - `active_runset` (int): 1つの runset 内で表示したい run の数。デフォルトは 0 です。
 - `custom_run_colors` (dict): run 名を key、色（16進コード）を value にしたカラー指定辞書。







---



## <kbd>class</kbd> `ParallelCoordinatesPlot`
パラレル座標プロットを表示するパネルオブジェクトです。



**属性:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]): 1つ以上の `ParallelCoordinatesPlotColumn` オブジェクトのリスト。
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `gradient` (Optional[LList[GradientPoint]]): GradientPoint のリスト。
 - `font_size` (Optional[FontSize]): 線グラフのフォントサイズ（`small`, `medium`, `large`, `auto`, `None`）。







---



## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
パラレル座標プロット内の1列。指定した `metric` の順番がパラレル座標プロット（x軸）での軸順になります。



**属性:**
 
 - `metric` (str | Config | SummaryMetric): W&B Project にログされたメトリクス名。
 - `display_name` (Optional[str]): メトリクスの表示名。
 - `inverted` (Optional[bool]): メトリクスを反転するかどうか。
 - `log` (Optional[bool]): メトリクスの対数変換を適用するか。







---



## <kbd>class</kbd> `ParameterImportancePlot`
選択したメトリクスの予測における各ハイパーパラメータの重要度を示すパネルです。



**属性:**
 
 - `with_respect_to` (str): パラメータの重要度を比較する対象メトリクス。例えば損失・精度など。指定するメトリクスは Report が情報を取得するプロジェクトにログされている必要があります。







---



## <kbd>class</kbd> `Report`
W&B Report を表すオブジェクトです。返されるオブジェクトの `blocks` 属性を使ってレポートをカスタマイズできます。Report オブジェクトは自動で保存されないので、変更を保持するには `save()` メソッドを使用してください。



**属性:**
 
 - `project` (str): レポートで読み込む W&B プロジェクト名。Report の URL に表示されます。
 - `entity` (str): Report を所有する W&B Entity。レポートの URL に表示されます。
 - `title` (str): レポートのタイトル。レポート最上部 H1 見出しとして表示されます。
 - `description` (str): レポートの説明文。レポートタイトルの下部に表示されます。
 - `blocks` (LList[BlockTypes]): HTMLタグ、プロット、グリッド、runset など1つ以上のリスト。
 - `width` (Literal['readable', 'fixed', 'fluid']): レポートの横幅。'readable', 'fixed', 'fluid' から選択できます。


---

#### <kbd>property</kbd> url

レポートがホストされている URL です。レポート URL は `https://wandb.ai/{entity}/{project_name}/reports/` の形式です。`{entity}` と `{project_name}` にはレポートの属する Entity とプロジェクト名が入ります。



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

指定した URL でレポートを現在の環境に読み込みます。レポートがホストされている URL を渡してください。



**引数:**
 
 - `url` (str): レポートがホストされている URL。
 - `as_model` (bool): True の場合 Report オブジェクトの代わりに model オブジェクトを返します。デフォルトは `False`。

---



### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

Report オブジェクトに加えた変更を永続化します。

---



### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

このレポートを表示する iframe を含む HTML を生成します。主に Python ノートブック等で利用されます。



**引数:**
 
 - `height` (int): iframe の高さ。
 - `hidden` (bool): True の場合 iframe を隠します。デフォルトは `False`。

---



## <kbd>class</kbd> `RunComparer`
プロジェクトに保存されている run 間でメトリクスを比較するパネルです。Report で情報を取得します。



**属性:**
 
 - `diff_only` `(Optional[Literal["split", True]])`: プロジェクト内 run 間の差分のみを表示します。この機能は W&B Report UI でオン・オフ切替可能です。







---



## <kbd>class</kbd> `Runset`
パネルグリッド内で表示する run のセットです。



**属性:**
 
 - `entity` (str): run が保存されているプロジェクトの所有者（または正しい権限を持つ Entity）。
 - `project` (str): run が保存されているプロジェクト名。
 - `name` (str): runset の名前。デフォルトは `Run set`。
 - `query` (str): run をフィルタリングするクエリ文字列。
 - `filters` (Optional[str]): run をフィルタリングするためのフィルタ文字列。
 - `groupby` (LList[str]): グループ化するメトリクス名のリスト。
 - `order` (LList[OrderBy]): ソート用 `OrderBy` オブジェクトのリスト。
 - `custom_run_colors` (LList[OrderBy]): run ID と色をマッピングする辞書。







---



## <kbd>class</kbd> `RunsetGroup`
複数 runset のグループを表示する UI 要素です。



**属性:**
 
 - `runset_name` (str): runset の名前。
 - `keys` (Tuple[RunsetGroupKey, ...]): グループ化する際に使用する `RunsetGroupKey` オブジェクト群。







---



## <kbd>class</kbd> `RunsetGroupKey`
`RunsetGroup` の一部で、メトリクスタイプと値で runset をまとめます。グループ化したいメトリクスタイプと値を key-value ペアで指定します。



**属性:**
 
 - `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): グループ化対象のメトリクスタイプ。
 - `value` (str): グループ化対象メトリクスの値。







---



## <kbd>class</kbd> `ScalarChart`
スカラー値のチャートを表示するパネルオブジェクトです。



**属性:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `metric` (MetricType): W&B Project にログされたメトリクス名。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定した関数で run を集約します。`mean`, `min`, `max`, `median`, `sum`, `samples`, または `None`。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲で run をグループ化します。`minmax`, `stddev`, `stderr`, `none`, `samples`, または `None`。
 - `custom_expressions` (Optional[LList[str]]): スカラーチャートで使用するカスタム式のリスト。
 - `legend_template` (Optional[str]): 凡例のテンプレート。
 - `font_size Optional[FontSize]`: 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None` から選択。







---



## <kbd>class</kbd> `ScatterPlot`
2D/3D 散布図を表示するパネルオブジェクトです。



**引数:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `x Optional[SummaryOrConfigOnlyMetric]`: W&B Project にログされたメトリクス名（x軸に使用）。
 - `y Optional[SummaryOrConfigOnlyMetric]`: W&B Project にログされた1つ以上のメトリクス名（y軸にプロット）。
 - z Optional[SummaryOrConfigOnlyMetric]:
 - `range_x` (Tuple[float | `None`, float | `None`]): x軸の範囲指定タプル。
 - `range_y` (Tuple[float | `None`, float | `None`]): y軸の範囲指定タプル。
 - `range_z` (Tuple[float | `None`, float | `None`]): z軸の範囲指定タプル。
 - `log_x` (Optional[bool]): x座標を常用対数スケールで表示。
 - `log_y` (Optional[bool]): y座標を常用対数スケールで表示。
 - `log_z` (Optional[bool]): z座標を常用対数スケールで表示。
 - `running_ymin` (Optional[bool]): 移動平均やローリング平均を適用。
 - `running_ymax` (Optional[bool]): 移動平均やローリング平均を適用。
 - `running_ymean` (Optional[bool]): 移動平均やローリング平均を適用。
 - `legend_template` (Optional[str]): 凡例の書式設定文字列。
 - `gradient` (Optional[LList[GradientPoint]]): グラデーションカラー指定用の GradientPoint リスト。
 - `font_size` (Optional[FontSize]): 線グラフのフォントサイズ。`small`, `medium`, `large`, `auto`, `None` から選択。
 - `regression` (Optional[bool]): `True` の場合、散布図に回帰直線を描画します。







---



## <kbd>class</kbd> `SoundCloud`
SoundCloud プレイヤーを表示するブロックです。



**属性:**
 
 - `html` (str): SoundCloud プレイヤー埋め込み用 HTML コード。







---



## <kbd>class</kbd> `Spotify`
Spotify プレイヤーを表示するブロックです。



**属性:**
 
 - `spotify_id` (str): トラックまたはプレイリストの Spotify ID。







---



## <kbd>class</kbd> `SummaryMetric`
レポートで表示するためのサマリーメトリクスです。



**属性:**
 
 - `name` (str): メトリクス名。







---



## <kbd>class</kbd> `TableOfContents`
H1、H2、H3 の HTML ブロックで指定されたセクションやサブセクションのリストを含むブロックです。







---



## <kbd>class</kbd> `TextWithInlineComments`
インラインコメント付きテキストのブロックです。



**属性:**
 
 - `text` (str): ブロックのテキスト。







---



## <kbd>class</kbd> `Twitter`
Twitter フィードを表示するブロックです。



**属性:**
 
 - `html` (str): Twitter フィード表示用の HTML コード。







---



## <kbd>class</kbd> `UnorderedList`
箇条書きリストの項目リストです。



**属性:**
 
 - `items` (LList[str]): 1つ以上の `UnorderedListItem` オブジェクトのリスト。







---



## <kbd>class</kbd> `UnorderedListItem`
箇条書きリストの項目です。



**属性:**
 
 - `text` (str): リスト項目のテキスト。







---



## <kbd>class</kbd> `Video`
動画を表示するブロックです。



**属性:**
 
 - `url` (str): 動画の URL。







---



## <kbd>class</kbd> `WeaveBlockArtifact`
W&B にログされたアーティファクトを表示するブロックです。クエリは下記のような形式です。

```python
project('entity', 'project').artifact('artifact-name')
```

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。



**属性:**
 
 - `entity` (str): アーティファクトが保存されているプロジェクトの正しい権限を持つ Entity。
 - `project` (str): アーティファクトを保存しているプロジェクト。
 - `artifact` (str): 取得したいアーティファクト名。
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: アーティファクトパネルで表示するタブ。







---



## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
W&B アーティファクトにログされたバージョン付きファイルを表示するブロックです。クエリは下記のような形式です。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。



**属性:**
 
 - `entity` (str): アーティファクトの保存場所にアクセス可能な Entity。
 - `project` (str): アーティファクトを保存しているプロジェクト名。
 - `artifact` (str): 取得するアーティファクト名。
 - `version` (str): 取得するアーティファクトのバージョン。
 - `file` (str): アーティファクト内で取得するファイル名。







---



## <kbd>class</kbd> `WeaveBlockSummaryTable`
W&B Table、pandas DataFrame、プロット、その他値等を表示するブロックです。クエリは下記の形式です。

```python
project('entity', 'project').runs.summary['value']
```

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。



**属性:**
 
 - `entity` (str): 値がログされたプロジェクトにアクセスできる Entity。
 - `project` (str): 値がログされたプロジェクト。
 - `table_name` (str): テーブル、DataFrame、プロット、その他値の名前。







---



## <kbd>class</kbd> `WeavePanel`
クエリを使ってカスタムコンテンツを表示できる空のクエリパネルです。

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。







---



## <kbd>class</kbd> `WeavePanelArtifact`
W&B にログされたアーティファクトを表示するパネルです。

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。



**属性:**
 
 - `artifact` (str): 取得したいアーティファクト名。
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: アーティファクトパネルで表示するタブ。







---



## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
W&B アーティファクトにログされたバージョン付きファイルを表示するパネルです。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。



**属性:**
 
 - `artifact` (str): 取得したいアーティファクト名。
 - `version` (str): 取得したいアーティファクトのバージョン。
 - `file` (str): アーティファクト内で取得したいファイル名。







---



## <kbd>class</kbd> `WeavePanelSummaryTable`
W&B Table、pandas DataFrame、プロット、その他値等を表示するパネルです。クエリは下記の構文です。

```python
runs.summary['value']
```

API 名の「Weave」は、LLM のトラッキングや評価用に使われる W&B Weave ツールキットを指すものではありません。



**属性:**
 
 - `table_name` (str): テーブル、DataFrame、プロット、または値の名前。