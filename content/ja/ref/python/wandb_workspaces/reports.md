---
title: Reports
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-reports
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}






# <kbd>module</kbd> `wandb_workspaces.reports.v2`
W&B　Reports APIをプログラムで使用するためのPython ライブラリ。

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
2D棒グラフを表示する パネル オブジェクト。

**Attributes:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `metrics` (LList[MetricType]): orientation Literal["v", "h"]: 棒グラフの向き。縦 ("v") または横 ("h") に設定します。デフォルトは横 ("h") です。
 - `range_x` (Tuple[float | None, float | None]): x軸の範囲を指定するタプル。
 - `title_x` (Optional[str]): x軸のラベル。
 - `title_y` (Optional[str]): y軸のラベル。
 - `groupby` (Optional[str]): レポートが情報を取得するW&B プロジェクトに 記録されたメトリクスに基づいて run をグループ化します。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定された関数で run を集計します。オプションには、`mean`、`min`、`max`、`median`、`sum`、`samples`、または `None` があります。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。オプションには、`minmax`、`stddev`、`stderr`、`none`、=`samples`、または `None` があります。
 - `max_runs_to_show` (Optional[int]): プロットに表示する run の最大数。
 - `max_bars_to_show` (Optional[int]): 棒グラフに表示するバーの最大数。
 - `custom_expressions` (Optional[LList[str]]): 棒グラフで使用するカスタム式 のリスト。
 - `legend_template` (Optional[str]): 凡例のテンプレート。
 - `font_size` ( Optional[FontSize]): 折れ線グラフのフォントのサイズ。オプションには、`small`、`medium`、`large`、`auto`、または `None` があります。
 - `line_titles` (Optional[dict]): 線のタイトル。キーは線の名前、値はタイトルです。
 - `line_colors` (Optional[dict]): 線の色。キーは線の名前、値は色です。

---

## <kbd>class</kbd> `BlockQuote`
引用テキストのブロック。

**Attributes:**
 
 - `text` (str): 引用ブロックのテキスト。

---

## <kbd>class</kbd> `CalloutBlock`
注意書きテキストのブロック。

**Attributes:**
 
 - `text` (str): 注意書きのテキスト。

---

## <kbd>class</kbd> `CheckedList`
チェックボックス付きのアイテムのリスト。`CheckedList` 内に1つまたは複数の `CheckedListItem` を追加します。

**Attributes:**
 
 - `items` (LList[CheckedListItem]): 1つまたは複数の `CheckedListItem` オブジェクトのリスト。

---

## <kbd>class</kbd> `CheckedListItem`
チェックボックス付きのリストアイテム。`CheckedList` 内に1つまたは複数の `CheckedListItem` を追加します。

**Attributes:**
 
 - `text` (str): リストアイテムのテキスト。
 - `checked` (bool): チェックボックスがオンになっているかどうか。デフォルトでは `False` に設定されています。

---

## <kbd>class</kbd> `CodeBlock`
コードのブロック。

**Attributes:**
 
 - `code` (str): ブロック内のコード。
 - `language` (Optional[Language]): コードの言語。指定された言語は、構文の強調表示に使用されます。デフォルトでは `python` に設定されています。オプションには、`javascript`、`python`、`css`、`json`、`html`、`markdown`、`yaml` があります。

---

## <kbd>class</kbd> `CodeComparer`
2つの異なる run 間でコードを比較する パネル オブジェクト。

**Attributes:**
 
 - `diff` `(Literal['split', 'unified'])`: コードの差分をどのように表示するか。オプションには `split` と `unified` があります。

---

## <kbd>class</kbd> `Config`
run の config オブジェクトに記録されたメトリクス。Config オブジェクトは通常、`run.config[name] = ...` を使用するか、キーと値のペアの辞書として config を渡すことによって記録されます。キーはメトリクスの名前、値はそのメトリクスの値です。

**Attributes:**
 
 - `name` (str): メトリクスの名前。

---

## <kbd>class</kbd> `CustomChart`
カスタム チャートを表示する パネル。チャートは Weave クエリによって定義されます。

**Attributes:**
 
 - `query` (dict): カスタム チャートを定義するクエリ。キーはフィールドの名前、値はクエリです。
 - `chart_name` (str): カスタム チャートのタイトル。
 - `chart_fields` (dict): プロットの軸を定義するキーと値のペア。キーはラベル、値はメトリクスです。
 - `chart_strings` (dict): チャート内の文字列を定義するキーと値のペア。

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
 
 - `table_name` (str): テーブルの名前。
 - `chart_fields` (dict): チャートに表示するフィールド。
 - `chart_strings` (dict): チャートに表示する文字列。

---

## <kbd>class</kbd> `Gallery`
ReportsとURLのギャラリーをレンダリングするブロック。

**Attributes:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport` および `GalleryURL` オブジェクトのリスト。

---

## <kbd>class</kbd> `GalleryReport`
ギャラリー内の report への参照。

**Attributes:**
 
 - `report_id` (str): report のID。

---

## <kbd>class</kbd> `GalleryURL`
外部リソースへのURL。

**Attributes:**
 
 - `url` (str): リソースのURL。
 - `title` (Optional[str]): リソースのタイトル。
 - `description` (Optional[str]): リソースの説明。
 - `image_url` (Optional[str]): 表示する画像のURL。

---

## <kbd>class</kbd> `GradientPoint`
勾配内の点。

**Attributes:**
 
 - `color`: 点の色。
 - `offset`: 勾配内の点の位置。値は0から100の間である必要があります。

---

## <kbd>class</kbd> `H1`
指定されたテキストを持つH1見出し。

**Attributes:**
 
 - `text` (str): 見出しのテキスト。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示するブロック。

---

## <kbd>class</kbd> `H2`
指定されたテキストを持つH2見出し。

**Attributes:**
 
 - `text` (str): 見出しのテキスト。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示する1つまたは複数のブロック。

---

## <kbd>class</kbd> `H3`
指定されたテキストを持つH3見出し。

**Attributes:**
 
 - `text` (str): 見出しのテキスト。
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 見出しが折りたたまれたときに表示する1つまたは複数のブロック。

---

## <kbd>class</kbd> `Heading`

---

## <kbd>class</kbd> `HorizontalRule`
HTML水平線。

---

## <kbd>class</kbd> `Image`
画像を表示するブロック。

**Attributes:**
 
 - `url` (str): 画像のURL。
 - `caption` (str): 画像のキャプション。キャプションは画像の下に表示されます。

---

## <kbd>class</kbd> `InlineCode`
インラインコード。コードの後に改行文字を追加しません。

**Attributes:**
 
 - `text` (str): reportに表示するコード。

---

## <kbd>class</kbd> `InlineLatex`
インラインLaTeXマークダウン。LaTeXマークダウンの後に改行文字を追加しません。

**Attributes:**
 
 - `text` (str): reportに表示するLaTeXマークダウン。

---

## <kbd>class</kbd> `LatexBlock`
LaTeXテキストのブロック。

**Attributes:**
 
 - `text` (str): LaTeXテキスト。

---

## <kbd>class</kbd> `Layout`
report内のパネルのレイアウト。パネルのサイズと位置を調整します。

**Attributes:**
 
 - `x` (int): パネルのx位置。
 - `y` (int): パネルのy位置。
 - `w` (int): パネルの幅。
 - `h` (int): パネルの高さ。

---

## <kbd>class</kbd> `LinePlot`
2D折れ線グラフを持つ パネル オブジェクト。

**Attributes:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `x` (Optional[MetricType]): レポートが情報を取得するW&B プロジェクトに記録されたメトリクスの名前。指定されたメトリクスはx軸に使用されます。
 - `y` (LList[MetricType]): レポートが情報を取得するW&B プロジェクトに記録された1つまたは複数のメトリクス。指定されたメトリクスはy軸に使用されます。
 - `range_x` (Tuple[float | `None`, float | `None`]): x軸の範囲を指定するタプル。
 - `range_y` (Tuple[float | `None`, float | `None`]): y軸の範囲を指定するタプル。
 - `log_x` (Optional[bool]): x座標を底10の対数スケールを使用してプロットします。
 - `log_y` (Optional[bool]): y座標を底10の対数スケールを使用してプロットします。
 - `title_x` (Optional[str]): x軸のラベル。
 - `title_y` (Optional[str]): y軸のラベル。
 - `ignore_outliers` (Optional[bool]): `True` に設定すると、外れ値をプロットしません。
 - `groupby` (Optional[str]): レポートが情報を取得するW&B プロジェクトに 記録されたメトリクスに基づいて run をグループ化します。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定された関数で run を集計します。オプションには、`mean`、`min`、`max`、`median`、`sum`、`samples`、または `None` があります。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。オプションには、`minmax`、`stddev`、`stderr`、`none`、`samples`、または `None` があります。
 - `smoothing_factor` (Optional[float]): スムージングタイプに適用するスムージング係数。許容値の範囲は0〜1です。
 - `smoothing_type Optional[SmoothingType]`: 指定された分布に基づいてフィルターを適用します。オプションには、`exponentialTimeWeighted`、`exponential`、`gaussian`、`average`、または `none` があります。
 - `smoothing_show_original` (Optional[bool]): `True` に設定すると、元のデータを表示します。
 - `max_runs_to_show` (Optional[int]): 折れ線グラフに表示する run の最大数。
 - `custom_expressions` (Optional[LList[str]]): データに適用するカスタム式。
 - `plot_type Optional[LinePlotStyle]`: 生成する折れ線グラフのタイプ。オプションには、`line`、`stacked-area`、または `pct-area` があります。
 - `font_size Optional[FontSize]`: 折れ線グラフのフォントのサイズ。オプションには、`small`、`medium`、`large`、`auto`、または `None` があります。
 - `legend_position Optional[LegendPosition]`: 凡例を配置する場所。オプションには、`north`、`south`、`east`、`west`、または `None` があります。
 - `legend_template` (Optional[str]): 凡例のテンプレート。
 - `aggregate` (Optional[bool]): `True` に設定すると、データを集計します。
 - `xaxis_expression` (Optional[str]): x軸の式。
 - `legend_fields` (Optional[LList[str]]): 凡例に含めるフィールド。

---

## <kbd>class</kbd> `Link`
URLへのリンク。

**Attributes:**
 
 - `text` (Union[str, TextWithInlineComments]): リンクのテキスト。
 - `url` (str): リンクが指すURL。

---

## <kbd>class</kbd> `MarkdownBlock`
マークダウンテキストのブロック。一般的なマークダウン構文を使用するテキストを書きたい場合に便利です。

**Attributes:**
 
 - `text` (str): マークダウンテキスト。

---

## <kbd>class</kbd> `MarkdownPanel`
マークダウンをレンダリングする パネル。

**Attributes:**
 
 - `markdown` (str): マークダウン パネルに表示するテキスト。

---

## <kbd>class</kbd> `MediaBrowser`
メディアファイルをグリッドレイアウトで表示する パネル。

**Attributes:**
 
 - `num_columns` (Optional[int]): グリッド内の列数。
 - `media_keys` (LList[str]): メディアファイルに対応するメディアキーのリスト。

---

## <kbd>class</kbd> `Metric`
プロジェクトに記録された report に表示するメトリクス。

**Attributes:**
 
 - `name` (str): メトリクスの名前。

---

## <kbd>class</kbd> `OrderBy`
並べ替えの基準となるメトリクス。

**Attributes:**
 
 - `name` (str): メトリクスの名前。
 - `ascending` (bool): 昇順でソートするかどうか。デフォルトでは `False` に設定されています。

---

## <kbd>class</kbd> `OrderedList`
番号付きリストのアイテムのリスト。

**Attributes:**
 
 - `items` (LList[str]): 1つまたは複数の `OrderedListItem` オブジェクトのリスト。

---

## <kbd>class</kbd> `OrderedListItem`
順序付きリストのリストアイテム。

**Attributes:**
 
 - `text` (str): リストアイテムのテキスト。

---

## <kbd>class</kbd> `P`
テキストの段落。

**Attributes:**
 
 - `text` (str): 段落のテキスト。

---

## <kbd>class</kbd> `Panel`
パネルグリッドに可視化を表示する パネル。

**Attributes:**
 
 - `layout` (Layout): `Layout` オブジェクト。

---

## <kbd>class</kbd> `PanelGrid`
runset と パネル で構成されるグリッド。`Runset` オブジェクトと `Panel` オブジェクトをそれぞれ使用して、runset と パネル を追加します。

利用可能な パネル には、`LinePlot`、`ScatterPlot`、`BarPlot`、`ScalarChart`、`CodeComparer`、`ParallelCoordinatesPlot`、`ParameterImportancePlot`、`RunComparer`、`MediaBrowser`、`MarkdownPanel`、`CustomChart`、`WeavePanel`、`WeavePanelSummaryTable`、`WeavePanelArtifactVersionedFile` があります。

**Attributes:**
 
 - `runsets` (LList["Runset"]): 1つまたは複数の `Runset` オブジェクトのリスト。
 - `panels` (LList["PanelTypes"]): 1つまたは複数の `Panel` オブジェクトのリスト。
 - `active_runset` (int): runset 内に表示する run の数。デフォルトでは0に設定されています。
 - `custom_run_colors` (dict): キーが run の名前、値が16進値で指定された色のキーと値のペア。

---

## <kbd>class</kbd> `ParallelCoordinatesPlot`
平行座標プロットを表示する パネル オブジェクト。

**Attributes:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]): 1つまたは複数の `ParallelCoordinatesPlotColumn` オブジェクトのリスト。
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `gradient` (Optional[LList[GradientPoint]]): 勾配点のリスト。
 - `font_size` (Optional[FontSize]): 折れ線グラフのフォントのサイズ。オプションには、`small`、`medium`、`large`、`auto`、または `None` があります。

---

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
平行座標プロット内の列。指定された `metric` の順序によって、平行座標プロットの平行軸（x軸）の順序が決まります。

**Attributes:**
 
 - `metric` (str | Config | SummaryMetric): レポートが情報を取得するW&B プロジェクトに 記録されたメトリクスの名前。
 - `display_name` (Optional[str]): メトリクスの名前
 - `inverted` (Optional[bool]): メトリクスを反転するかどうか。
 - `log` (Optional[bool]): メトリクスにログ変換を適用するかどうか。

---

## <kbd>class</kbd> `ParameterImportancePlot`
選択したメトリクスを予測する上で、各ハイパーパラメータがどれだけ重要かを示す パネル。

**Attributes:**
 
 - `with_respect_to` (str): パラメータの重要度を比較する対象のメトリクス。一般的なメトリクスには、損失、精度などがあります。指定するメトリクスは、レポートが情報を取得するプロジェクト内に記録されている必要があります。

---

## <kbd>class</kbd> `Report`
W&B　Reportを表すオブジェクト。返されたオブジェクトの `blocks` 属性を使用して、 report をカスタマイズします。Reportオブジェクトは自動的に保存されません。変更を永続化するには、`save()` メソッドを使用します。

**Attributes:**
 
 - `project` (str): ロードするW&B プロジェクトの名前。指定された プロジェクト は、 report のURLに表示されます。
 - `entity` (str): report を所有するW&B　Entity。Entityは report のURLに表示されます。
 - `title` (str): report のタイトル。タイトルは report の上部にH1見出しとして表示されます。
 - `description` (str): report の説明。説明は report のタイトルの下に表示されます。
 - `blocks` (LList[BlockTypes]): 1つまたは複数のHTMLタグ、プロット、グリッド、runsetなどのリスト。
 - `width` (Literal['readable', 'fixed', 'fluid']): report の幅。オプションには、'readable'、'fixed'、'fluid'があります。

---

#### <kbd>property</kbd> url

report がホストされているURL。report のURLは `https://wandb.ai/{entity}/{project_name}/reports/` で構成されます。`{entity}` と `{project_name}` は、 report が属する Entity と プロジェクト の名前で構成されます。

---

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

report を現在の環境にロードします。report がホストされているURLを渡します。

**Arguments:**
 
 - `url` (str): report がホストされているURL。
 - `as_model` (bool): Trueの場合、Reportオブジェクトではなく model オブジェクトを返します。デフォルトでは `False` に設定されています。

---

### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

report オブジェクトに加えられた変更を永続化します。

---

### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

この report を表示する iframe を含むHTMLを生成します。Python ノートブック内でよく使用されます。

**Arguments:**
 
 - `height` (int): iframeの高さ。
 - `hidden` (bool): Trueの場合、iframeを非表示にします。デフォルトは `False` に設定されています。

---

## <kbd>class</kbd> `RunComparer`
レポートが情報を取得する プロジェクト のさまざまな run にわたってメトリクスを比較する パネル。

**Attributes:**
 
 - `diff_only` `(Optional[Literal["split", True]])`: プロジェクト 内の run 間の差のみを表示します。この機能は、W&B Report UIでオン/オフを切り替えることができます。

---

## <kbd>class</kbd> `Runset`
パネルグリッドに表示する run のセット。

**Attributes:**
 
 - `entity` (str): run が保存されている プロジェクト を所有しているか、または適切な権限を持っているエンティティ。
 - `project` (str): run が保存されている プロジェクト の名前。
 - `name` (str): run set の名前。デフォルトでは `Run set` に設定されています。
 - `query` (str): run をフィルタリングするクエリ文字列。
 - `filters` (Optional[str]): run をフィルタリングするフィルタ文字列。
 - `groupby` (LList[str]): グループ化するメトリクス名のリスト。
 - `order` (LList[OrderBy]): 並べ替えを行う `OrderBy` オブジェクトのリスト。
 - `custom_run_colors` (LList[OrderBy]): run IDを色にマッピングする辞書。

---

## <kbd>class</kbd> `RunsetGroup`
runset のグループを表示するUI要素。

**Attributes:**
 
 - `runset_name` (str): runset の名前。
 - `keys` (Tuple[RunsetGroupKey, ...]): グループ化するキー。グループ化するには、1つまたは複数の `RunsetGroupKey` オブジェクトを渡します。

---

## <kbd>class</kbd> `RunsetGroupKey`
メトリクスタイプと値でrunsetをグループ化します。`RunsetGroup` の一部。グループ化するメトリクスタイプと値をキーと値のペアとして指定します。

**Attributes:**
 
 - `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): グループ化するメトリクスタイプ。
 - `value` (str): グループ化するメトリクスの値。

---

## <kbd>class</kbd> `ScalarChart`
スカラーチャートを表示する パネル オブジェクト。

**Attributes:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `metric` (MetricType): レポートが情報を取得するW&B プロジェクトに記録されたメトリクスの名前。
 - `groupby_aggfunc` (Optional[GroupAgg]): 指定された関数で run を集計します。オプションには、`mean`、`min`、`max`、`median`、`sum`、`samples`、または `None` があります。
 - `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。オプションには、`minmax`、`stddev`、`stderr`、`none`、`samples`、または `None` があります。
 - `custom_expressions` (Optional[LList[str]]): スカラーチャートで使用するカスタム式のリスト。
 - `legend_template` (Optional[str]): 凡例のテンプレート。
 - `font_size Optional[FontSize]`: 折れ線グラフのフォントのサイズ。オプションには、`small`、`medium`、`large`、`auto`、または `None` があります。

---

## <kbd>class</kbd> `ScatterPlot`
2Dまたは3D散布図を表示する パネル オブジェクト。

**Arguments:**
 
 - `title` (Optional[str]): プロット上部に表示されるテキスト。
 - `x Optional[SummaryOrConfigOnlyMetric]`: レポートが情報を取得するW&B プロジェクトに記録されたメトリクスの名前。指定されたメトリクスはx軸に使用されます。
 - `y Optional[SummaryOrConfigOnlyMetric]`: レポートが情報を取得するW&B プロジェクトに記録された1つまたは複数のメトリクス。指定されたメトリクスはy軸内にプロットされます。z Optional[SummaryOrConfigOnlyMetric]:
 - `range_x` (Tuple[float | `None`, float | `None`]): x軸の範囲を指定するタプル。
 - `range_y` (Tuple[float | `None`, float | `None`]): y軸の範囲を指定するタプル。
 - `range_z` (Tuple[float | `None`, float | `None`]): z軸の範囲を指定するタプル。
 - `log_x` (Optional[bool]): x座標を底10の対数スケールを使用してプロットします。
 - `log_y` (Optional[bool]): y座標を底10の対数スケールを使用してプロットします。
 - `log_z` (Optional[bool]): z座標を底10の対数スケールを使用してプロットします。
 - `running_ymin` (Optional[bool]): 移動平均またはローリング平均を適用します。
 - `running_ymax` (Optional[bool]): 移動平均またはローリング平均を適用します。
 - `running_ymean` (Optional[bool]): 移動平均またはローリング平均を適用します。
 - `legend_template` (Optional[str]): 凡例の形式を指定する文字列。
 - `gradient` (Optional[LList[GradientPoint]]): プロットの色勾配を指定する勾配点のリスト。
 - `font_size` (Optional[FontSize]): 折れ線グラフのフォントのサイズ。オプションには、`small`、`medium`、`large`、`auto`、または `None` があります。
 - `regression` (Optional[bool]): `True`の場合、回帰線が散布図にプロットされます。

---

## <kbd>class</kbd> `SoundCloud`
SoundCloudプレーヤーを表示するブロック。

**Attributes:**
 
 - `html` (str): SoundCloudプレーヤーを埋め込むためのHTMLコード。

---

## <kbd>class</kbd> `Spotify`
Spotifyプレーヤーを表示するブロック。

**Attributes:**
 
 - `spotify_id` (str): トラックまたはプレイリストのSpotify ID。

---

## <kbd>class</kbd> `SummaryMetric`
レポートに表示するサマリーメトリクス。

**Attributes:**
 
 - `name` (str): メトリクスの名前。

---

## <kbd>class</kbd> `TableOfContents`
レポートで指定されたH1、H2、およびH3 HTMLブロックを使用して、セクションとサブセクションのリストを含むブロック。

---

## <kbd>class</kbd> `TextWithInlineComments`
インラインコメント付きのテキストのブロック。

**Attributes:**
 
 - `text` (str): ブロックのテキスト。

---

## <kbd>class</kbd> `Twitter`
Twitterフィードを表示するブロック。

**Attributes:**
 
 - `html` (str): Twitterフィードを表示するためのHTMLコード。

---

## <kbd>class</kbd> `UnorderedList`
箇条書きリストのアイテムのリスト。

**Attributes:**
 
 - `items` (LList[str]): 1つまたは複数の `UnorderedListItem` オブジェクトのリスト。

---

## <kbd>class</kbd> `UnorderedListItem`
順序なしリストのリスト項目。

**Attributes:**
 
 - `text` (str): リストアイテムのテキスト。

---

## <kbd>class</kbd> `Video`
ビデオを表示するブロック。

**Attributes:**
 
 - `url` (str): ビデオのURL。

---

## <kbd>class</kbd> `WeaveBlockArtifact`
W&Bに記録されたアーティファクトを表示するブロック。クエリは次の形式を取ります。

```python
project('entity', 'project').artifact('artifact-name')
``` 

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

**Attributes:**
 
 - `entity` (str): アーティファクトが保存されている プロジェクト を所有しているか、または適切な権限を持っているエンティティ。
 - `project` (str): アーティファクトが保存されている プロジェクト 。
 - `artifact` (str): 取得するアーティファクトの名前。
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: アーティファクト パネルに表示するタブ。

---

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
W&Bアーティファクトに記録されたバージョン管理されたファイルを表示するブロック。クエリは次の形式を取ります。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
``` 

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

**Attributes:**
 
 - `entity` (str): アーティファクトが保存されている プロジェクト を所有しているか、または適切な権限を持っているエンティティ。
 - `project` (str): アーティファクトが保存されている プロジェクト 。
 - `artifact` (str): 取得するアーティファクトの名前。
 - `version` (str): 取得するアーティファクトのバージョン。
 - `file` (str): 取得するアーティファクトに保存されているファイルの名前。

---

## <kbd>class</kbd> `WeaveBlockSummaryTable`
W&B Table、pandas DataFrame、プロット、またはW&Bに記録されたその他の値を表示するブロック。クエリは次の形式を取ります。

```python
project('entity', 'project').runs.summary['value']
``` 

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

**Attributes:**
 
 - `entity` (str): 値が記録されている プロジェクト を所有しているか、または適切な権限を持っているエンティティ。
 - `project` (str): 値が記録されている プロジェクト 。
 - `table_name` (str): テーブル、DataFrame、プロット、または値の名前。

---

## <kbd>class</kbd> `WeavePanel`
クエリを使用してカスタムコンテンツを表示するために使用できる空のクエリ パネル。

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

---

## <kbd>class</kbd> `WeavePanelArtifact`
W&Bに記録されたアーティファクトを表示する パネル。

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

**Attributes:**
 
 - `artifact` (str): 取得するアーティファクトの名前。
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: アーティファクト パネルに表示するタブ。

---

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
W&Bアーティファクトに記録されたバージョン管理されたファイルを表示する パネル。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
``` 

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

**Attributes:**
 
 - `artifact` (str): 取得するアーティファクトの名前。
 - `version` (str): 取得するアーティファクトのバージョン。
 - `file` (str): アーティファクトに保存されているファイルの名前。

---

## <kbd>class</kbd> `WeavePanelSummaryTable`
W&B Table、pandas DataFrame、プロット、またはW&Bに記録されたその他の値を表示する パネル。クエリは次の形式を取ります。

```python
runs.summary['value']
``` 

API名 "Weave" は、LLMの追跡と評価に使用されるW&B Weaveツールキットを指すものではありません。

**Attributes:**
 
 - `table_name` (str): テーブル、DataFrame、プロット、または値の名前。
