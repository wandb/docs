---
title: レポート
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-reports
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
プログラムで W&B レポート API を操作するための Python ライブラリ。

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
2Dバープロットを表示するパネルオブジェクト。

**Attributes:**
 
- `title` (Optional[str]): プロットの上部に表示されるテキスト。
- `metrics` (LList[MetricType]): orientation Literal["v", "h"]: バープロットの向き。垂直 ("v") または水平 ("h") を選択します。デフォルトは水平 ("h")。
- `range_x` (Tuple[float | None, float | None]): x軸の範囲を指定するタプル。
- `title_x` (Optional[str]): x軸のラベル。
- `title_y` (Optional[str]): y軸のラベル。
- `groupby` (Optional[str]): W&Bプロジェクトにログされたメトリクスに基づいて run をグループ化します。このレポートは情報を取得します。
- `groupby_aggfunc` (Optional[GroupAgg]): 指定された関数で run を集計します。オプションには `mean`, `min`, `max`, `median`, `sum`, `samples`, または `None` が含まれます。
- `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。オプションには `minmax`, `stddev`, `stderr`, `none`, `samples`, または `None` が含まれます。
- `max_runs_to_show` (Optional[int]): プロットに表示する最大 run 数。
- `max_bars_to_show` (Optional[int]): バープロットに表示する最大バー数。
- `custom_expressions` (Optional[LList[str]]): バープロットで使用されるカスタム式のリスト。
- `legend_template` (Optional[str]): 凡例のテンプレート。
- `font_size` (Optional[FontSize]): ラインプロットのフォントサイズ。オプションには `small`, `medium`, `large`, `auto`, または `None` が含まれます。
- `line_titles` (Optional[dict]): ラインのタイトル。キーがライン名で、値がタイトルです。
- `line_colors` (Optional[dict]): ラインの色。キーがライン名で、値が色です。

---

## <kbd>class</kbd> `BlockQuote`
引用されたテキストのブロック。

**Attributes:**

- `text` (str): 引用ブロックのテキスト。

---

## <kbd>class</kbd> `CalloutBlock`
強調されたテキストのブロック。

**Attributes:**

- `text` (str): 強調テキスト。

---

## <kbd>class</kbd> `CheckedList`
チェックボックス付きの項目リスト。`CheckedListItem` を `CheckedList` 内に1つ以上追加します。

**Attributes:**

- `items` (LList[CheckedListItem]): `CheckedListItem` オブジェクトのリスト。

---

## <kbd>class</kbd> `CheckedListItem`
チェックボックス付きのリストアイテム。`CheckedList` 内に1つ以上の `CheckedListItem` を追加します。

**Attributes:**

- `text` (str): リストアイテムのテキスト。
- `checked` (bool): チェックボックスがチェックされているかどうか。デフォルトは `False`。

---

## <kbd>class</kbd> `CodeBlock`
コードのブロック。

**Attributes:**

- `code` (str): ブロック内のコード。
- `language` (Optional[Language]): コードの言語。指定された言語は構文強調表示に使われます。デフォルトは `python`。オプションには `javascript`, `python`, `css`, `json`, `html`, `markdown`, `yaml` が含まれます。

---

## <kbd>class</kbd> `CodeComparer`
異なる2つの run 間のコードを比較するパネルオブジェクト。

**Attributes:**

- `diff` (Literal['split', 'unified']): コードの差異を表示する方法。オプションには `split` と `unified` が含まれます。

---

## <kbd>class</kbd> `Config`
run の設定オブジェクトにログされたメトリクス。設定オブジェクトは通常、`run.config[name] = ...` を使用するか、キーと値のペアを持つ設定として渡されてログされます。ここでキーがメトリクスの名前、値がメトリクスの値です。

**Attributes:**

- `name` (str): メトリクスの名前。

---

## <kbd>class</kbd> `CustomChart`
カスタムチャートを表示するパネル。チャートは Weave クエリによって定義されます。

**Attributes:**

- `query` (dict): カスタムチャートを定義するクエリ。キーがフィールドの名前で、値がクエリです。
- `chart_name` (str): カスタムチャートのタイトル。
- `chart_fields` (dict): プロットの軸を定義するキーと値のペア。ここでキーはラベル、値はメトリクスです。
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

テーブルからカスタムチャートを作成します。

**Arguments:**

- `table_name` (str): テーブルの名前。
- `chart_fields` (dict): チャートに表示するフィールド。
- `chart_strings` (dict): チャートに表示する文字列。

---

## <kbd>class</kbd> `Gallery`
レポートと URL のギャラリーをレンダリングするブロック。

**Attributes:**

- `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport` と `GalleryURL` オブジェクトのリスト。

---

## <kbd>class</kbd> `GalleryReport`
ギャラリー内のレポートへの参照。

**Attributes:**

- `report_id` (str): レポートの ID。

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
勾配内の点。

**Attributes:**

- `color`: 点の色。
- `offset`: 勾配内の点の位置。値は 0 から 100 の範囲であるべきです。

---

## <kbd>class</kbd> `H1`
指定されたテキストを持つ H1 ヘッディング。

**Attributes:**

- `text` (str): ヘッディングのテキスト。
- `collapsed_blocks` (Optional[LList["BlockTypes"]]): ヘッディングを折りたたんだときに表示されるブロック。

---

## <kbd>class</kbd> `H2`
指定されたテキストを持つ H2 ヘッディング。

**Attributes:**

- `text` (str): ヘッディングのテキスト。
- `collapsed_blocks` (Optional[LList["BlockTypes"]]): ヘッディングを折りたたんだときに表示される1つ以上のブロック。

---

## <kbd>class</kbd> `H3`
指定されたテキストを持つ H3 ヘッディング。

**Attributes:**

- `text` (str): ヘッディングのテキスト。
- `collapsed_blocks` (Optional[LList["BlockTypes"]]): ヘッディングを折りたたんだときに表示される1つ以上のブロック。

---

## <kbd>class</kbd> `Heading`

---

## <kbd>class</kbd> `HorizontalRule`
HTML の水平ライン。

---

## <kbd>class</kbd> `Image`
画像をレンダリングするブロック。

**Attributes:**

- `url` (str): 画像の URL。
- `caption` (str): 画像のキャプション。キャプションは画像の下に表示されます。

---

## <kbd>class</kbd> `InlineCode`
インラインコード。コードの後に改行文字を加えません。

**Attributes:**

- `text` (str): レポートに表示したいコード。

---

## <kbd>class</kbd> `InlineLatex`
インライン LaTeX マークダウン。LaTeX マークダウンの後に改行文字を加えません。

**Attributes:**

- `text` (str): レポートに表示したい LaTeX マークダウン。

---

## <kbd>class</kbd> `LatexBlock`
LaTeX テキストのブロック。

**Attributes:**

- `text` (str): LaTeX テキスト。

---

## <kbd>class</kbd> `Layout`
レポート内のパネルのレイアウト。パネルのサイズと位置を調整します。

**Attributes:**

- `x` (int): パネルの x 位置。
- `y` (int): パネルの y 位置。
- `w` (int): パネルの幅。
- `h` (int): パネルの高さ。

---

## <kbd>class</kbd> `LinePlot`
2D ラインプロットを持つパネルオブジェクト。

**Attributes:**

- `title` (Optional[str]): プロットの上部に表示されるテキスト。
- `x` (Optional[MetricType]): W&B プロジェクトにログされたメトリクスの名前。このレポートは情報を取得します。指定されたメトリクスは x 軸に使用されます。
- `y` (LList[MetricType]): W&B プロジェクトにログされた1つ以上のメトリクス。このレポートは情報を取得します。指定されたメトリクスは y 軸に使用されます。
- `range_x` (Tuple[float | None, float | None]): x軸の範囲を指定するタプル。
- `range_y` (Tuple[float | None, float | None]): y軸の範囲を指定するタプル。
- `log_x` (Optional[bool]): x 座標を底 10 の対数スケールでプロットします。
- `log_y` (Optional[bool]): y 座標を底 10 の対数スケールでプロットします。
- `title_x` (Optional[str]): x軸のラベル。
- `title_y` (Optional[str]): y軸のラベル。
- `ignore_outliers` (Optional[bool]): `True` に設定すると、外れ値をプロットしません。
- `groupby` (Optional[str]): W&B プロジェクトにログされたメトリクスに基づいて run をグループ化します。このレポートは情報を取得します。
- `groupby_aggfunc` (Optional[GroupAgg]): 指定された関数で run を集計します。オプションには `mean`, `min`, `max`, `median`, `sum`, `samples`, または `None` が含まれます。
- `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。オプションには `minmax`, `stddev`, `stderr`, `none`, `samples`, または `None` が含まれます。
- `smoothing_factor` (Optional[float]): 平滑化タイプに適用する平滑化係数。許容する値は 0 から 1 の範囲です。
- `smoothing_type` (Optional[SmoothingType]): 指定された分布に基づいてフィルターを適用します。オプションには `exponentialTimeWeighted`, `exponential`, `gaussian`, `average`, または `none` が含まれます。
- `smoothing_show_original` (Optional[bool]): `True` に設定すると、元のデータを表示します。
- `max_runs_to_show` (Optional[int]): ラインプロットに表示する最大 run 数。
- `custom_expressions` (Optional[LList[str]]): データに適用するカスタム式。
- `plot_type` (Optional[LinePlotStyle]): 生成するラインプロットのタイプ。オプションには `line`, `stacked-area`, または `pct-area` が含まれます。
- `font_size` (Optional[FontSize]): ラインプロットのフォントサイズ。オプションには `small`, `medium`, `large`, `auto`, または `None` が含まれます。
- `legend_position` (Optional[LegendPosition]): 凡例を配置する場所。オプションには `north`, `south`, `east`, `west`, または `None` が含まれます。
- `legend_template` (Optional[str]): 凡例のテンプレート。
- `aggregate` (Optional[bool]): `True` に設定すると、データを集計します。
- `xaxis_expression` (Optional[str]): x軸の表現。
- `legend_fields` (Optional[LList[str]]): 凡例に含めるフィールド。

---

## <kbd>class</kbd> `Link`
URL へのリンク。

**Attributes:**

- `text` (Union[str, TextWithInlineComments]): リンクのテキスト。
- `url` (str): リンクが指す URL。

---

## <kbd>class</kbd> `MarkdownBlock`
マークダウンテキストのブロック。一般的なマークダウンサクジを使用してテキストを書くのに便利です。

**Attributes:**

- `text` (str): マークダウンテキスト。

---

## <kbd>class</kbd> `MarkdownPanel`
マークダウンをレンダリングするパネル。

**Attributes:**

- `markdown` (str): マークダウンパネルに表示したいテキスト。

---

## <kbd>class</kbd> `MediaBrowser`
メディアファイルをグリッドレイアウトで表示するパネル。

**Attributes:**

- `num_columns` (Optional[int]): グリッドの列数。
- `media_keys` (LList[str]): メディアファイルに対応するメディアキーのリスト。

---

## <kbd>class</kbd> `Metric`
プロジェクトにログされたメトリクスをレポートに表示する。

**Attributes:**

- `name` (str): メトリクスの名前。

---

## <kbd>class</kbd> `OrderBy`
並び替えに使用するメトリクス。

**Attributes:**

- `name` (str): メトリクスの名前。
- `ascending` (bool): 昇順にソートするかどうか。デフォルトは `False` に設定されています。

---

## <kbd>class</kbd> `OrderedList`
番号付きリストの項目リスト。

**Attributes:**

- `items` (LList[str]): `OrderedListItem` オブジェクトのリスト。

---

## <kbd>class</kbd> `OrderedListItem`
順序付きリストの項目。

**Attributes:**

- `text` (str): リストアイテムのテキスト。

---

## <kbd>class</kbd> `P`
テキストの段落。

**Attributes:**

- `text` (str): 段落のテキスト。

---

## <kbd>class</kbd> `Panel`
パネルグリッドで可視化を表示するパネル。

**Attributes:**

- `layout` (Layout): `Layout` オブジェクト。

---

## <kbd>class</kbd> `PanelGrid`
runset とパネルで構成されるグリッド。runset とパネルはそれぞれ `Runset` と `Panel` オブジェクトで追加します。

**利用可能なパネル:**

- `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`

**Attributes:**

- `runsets` (LList["Runset"]): `Runset` オブジェクトのリスト。
- `panels` (LList["PanelTypes"]): `Panel` オブジェクトのリスト。
- `active_runset` (int): runset 内で表示したい run の数。デフォルトは 0 に設定されています。
- `custom_run_colors` (dict): run の名前をキーに指定し、16進値の色を値として指定するキーと値のペア。

---

## <kbd>class</kbd> `ParallelCoordinatesPlot`
並列座標プロットを表示するパネルオブジェクト。

**Attributes:**

- `columns` (LList[ParallelCoordinatesPlotColumn]): `ParallelCoordinatesPlotColumn` オブジェクトのリスト。
- `title` (Optional[str]): プロットの上部に表示されるテキスト。
- `gradient` (Optional[LList[GradientPoint]]): 勾配ポイントのリスト。
- `font_size` (Optional[FontSize]): ラインプロットのフォントサイズ。オプションには `small`, `medium`, `large`, `auto`, または `None` が含まれます。

---

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
並列座標プロット内の列。指定された `metric` の順序が並列軸 (x軸) の順序を決定します。

**Attributes:**

- `metric` (str | Config | SummaryMetric): W&B プロジェクトにログされたメトリクスの名前。このレポートは情報を取得します。
- `display_name` (Optional[str]): メトリクスの表示名。
- `inverted` (Optional[bool]): メトリクスを反転するかどうか。
- `log` (Optional[bool]): メトリクスに対数変換を適用するかどうか。

---

## <kbd>class</kbd> `ParameterImportancePlot`
各ハイパーパラメーターが選択されたメトリクスの予測にどれほど重要かを示すパネル。

**Attributes:**

- `with_respect_to` (str): パラメータの重要度を比較したいメトリクス。一般的なメトリクスにはロス、精度などが含まれます。指定されたメトリクスはプロジェクト内でログされる必要があります。このレポートは情報を取得します。

---

## <kbd>class</kbd> `Report`
W&B レポートを表すオブジェクト。返されたオブジェクトの `blocks` 属性を使用してレポートをカスタマイズします。レポートオブジェクトは自動的に保存されません。`save()` メソッドを使用して変更を保存してください。

**Attributes:**

- `project` (str): 読み込む W&B プロジェクトの名前。指定されたプロジェクトはレポートの URL に表示されます。
- `entity` (str): レポートを所有する W&B エンティティ。エンティティはレポートの URL に表示されます。
- `title` (str): レポートのタイトル。タイトルはレポートのトップに H1 ヘッディングとして表示されます。
- `description` (str): レポートの説明。説明はレポートのタイトルの下に表示されます。
- `blocks` (LList[BlockTypes]): HTML タグ、プロット、グリッド、runset などのリスト。
- `width` (Literal['readable', 'fixed', 'fluid']): レポートの幅。オプションには 'readable', 'fixed', 'fluid' が含まれます。

---

#### <kbd>property</kbd> url

レポートがホストされている URL。レポート URL は `https://wandb.ai/{entity}/{project_name}/reports/` で構成されます。ここで `{entity}` と `{project_name}` はそれぞれレポートが所属するエンティティとプロジェクトの名前です。

---

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

現在の環境にレポートを読み込みます。レポートがホストされている URL を渡します。

**Arguments:**

- `url` (str): レポートがホストされている URL。
- `as_model` (bool): `True` に設定すると、レポートオブジェクトの代わりにモデルオブジェクトが返されます。デフォルトは `False` に設定されています。

---

### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

レポートオブジェクトに加えた変更を保存します。

---

### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

このレポートを表示する iframe を含む HTML を生成します。通常、Python ノートブック内で使用されます。

**Arguments:**

- `height` (int): iframe の高さ。
- `hidden` (bool): `True` に設定すると、iframe を非表示にします。デフォルトは `False` に設定されています。

---

## <kbd>class</kbd> `RunComparer`
プロジェクトから引き出された情報で、異なる run 間のメトリクスを比較するパネル。

**Attributes:**

- `diff_only` (Optional[Literal["split", True]]): プロジェクト内の run 間の差異のみを表示します。W&B レポート UI ではこの機能のオン/オフを切り替えることができます。

---

## <kbd>class</kbd> `Runset`
パネルグリッドに表示する run のセット。

**Attributes:**

- `entity` (str): run が保存されているプロジェクトを所有したり、正しい権限を持つエンティティ。
- `project` (str): run が保存されているプロジェクトの名前。
- `name` (str): run セットの名前。デフォルトで `Run set` に設定されています。
- `query` (str): run をフィルタリングするためのクエリ文字列。
- `filters` (Optional[str]): run をフィルタリングするためのフィルタ文字列。
- `groupby` (LList[str]): グループ化するメトリクス名のリスト。
- `order` (LList[OrderBy]): ソートするための `OrderBy` オブジェクトのリスト。
- `custom_run_colors` (LList[OrderBy]): run ID を色にマッピングする辞書。

---

## <kbd>class</kbd> `RunsetGroup`
runset のグループを表示する UI エレメント。

**Attributes:**

- `runset_name` (str): runset の名前。
- `keys` (Tuple[RunsetGroupKey, ...]): グループ化するためのキー。グループ化するために1つ以上の `RunsetGroupKey` オブジェクトを渡します。

---

## <kbd>class</kbd> `RunsetGroupKey`
メトリクスタイプと値によって runset をグループ化します。`RunsetGroup` の一部として動作します。メトリクスタイプとグループ化する値をキーと値のペアとして指定します。

**Attributes:**

- `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): グループ化するメトリクスタイプ。
- `value` (str): グループ化するメトリクスの値。

---

## <kbd>class</kbd> `ScalarChart`
スカラーグラフを表示するパネルオブジェクト。

**Attributes:**

- `title` (Optional[str]): プロットの上部に表示されるテキスト。
- `metric` (MetricType): W&B プロジェクトにログされたメトリクスの名前。このレポートは情報を取得します。
- `groupby_aggfunc` (Optional[GroupAgg]): 指定された関数で run を集計します。オプションには `mean`, `min`, `max`, `median`, `sum`, `samples`, または `None` が含まれます。
- `groupby_rangefunc` (Optional[GroupArea]): 範囲に基づいて run をグループ化します。オプションには `minmax`, `stddev`, `stderr`, `none`, `samples`, または `None` が含まれます。
- `custom_expressions` (Optional[LList[str]]): スカラーチャートで使用されるカスタム式のリスト。
- `legend_template` (Optional[str]): 凡例のテンプレート。
- `font_size` (Optional[FontSize]): ラインプロットのフォントサイズ。オプションには `small`, `medium`, `large`, `auto`, または `None` が含まれます。

---

## <kbd>class</kbd> `ScatterPlot`
2D または 3D 散布図を表示するパネルオブジェクト。

**Arguments:**

- `title` (Optional[str]): プロットの上部に表示されるテキスト。
- `x` (Optional[SummaryOrConfigOnlyMetric]): W&B プロジェクトにログされたメトリクスの名前。このレポートは情報を取得します。指定されたメトリクスは x 軸に使用されます。
- `y` (Optional[SummaryOrConfigOnlyMetric]): W&B プロジェクトにログされた1つ以上のメトリクス。このレポートは情報を取得します。指定されたメトリクスは y 軸にプロットされます。
- `range_x` (Tuple[float | None, float | None]): x軸の範囲を指定するタプル。
- `range_y` (Tuple[float | None, float | None]): y軸の範囲を指定するタプル。
- `log_x` (Optional[bool]): x 座標を底 10 の対数スケールでプロットします。
- `log_y` (Optional[bool]): y 座標を底 10 の対数スケールでプロットします。
- `legend_template` (Optional[str]): 凡例の形式を指定する文字列。
- `gradient` (Optional[LList[GradientPoint]]): プロットの色勾配を指定する勾配ポイントのリスト。
- `font_size` (Optional[FontSize]): ラインプロットのフォントサイズ。オプションには `small`, `medium`, `large`, `auto`, または `None` が含まれます。
- `regression` (Optional[bool]): `True` に設定すると、散布図に回帰直線をプロットします。

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
レポート内に表示するサマリメトリクス。

**Attributes:**

- `name` (str): メトリクスの名前。

---

## <kbd>class</kbd> `TableOfContents`
H1, H2, H3 の HTML ブロックを使用して指定されたセクションとサブセクションのリストを含むブロック。

---

## <kbd>class</kbd> `TextWithInlineComments`
インラインコメント付きのテキストブロック。

**Attributes:**

- `text` (str): テキストブロックのテキスト。

---

## <kbd>class</kbd> `Twitter`
Twitter フィードを表示するブロック。

**Attributes:**

- `html` (str): Twitter フィードを表示するための HTML コード。

---

## <kbd>class</kbd> `UnorderedList`
箇条書きリストの項目リスト。

**Attributes:**

- `items` (LList[str]): `UnorderedListItem` オブジェクトのリスト。

---

## <kbd>class</kbd> `UnorderedListItem`
順序のないリストの項目。

**Attributes:**

- `text` (str): リストアイテムのテキスト。

---

## <kbd>class</kbd> `Video`
ビデオをレンダリングするブロック。

**Attributes:**

- `url` (str): ビデオの URL。

---

## <kbd>class</kbd> `WeaveBlockArtifact`
W&B にログされたアーティファクトを示すブロック。クエリは次の形式を取ります。

```python
project('entity', 'project').artifact('artifact-name')
```

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

**Attributes:**

- `entity` (str): アーティファクトが保存されているプロジェクトを所有するか、適切な権限を持つエンティティ。
- `project` (str): アーティファクトが保存されているプロジェクト。
- `artifact` (str): 取得するアーティファクトの名前。
- `tab` (Literal["overview", "metadata", "usage", "files", "lineage"]): アーティファクトパネルに表示するタブ。

---

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
バージョン化されたファイルを W&B アーティファクトにログしたことを示すブロック。クエリは次の形式を取ります。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

**Attributes:**

- `entity` (str): アーティファクトが保存されているプロジェクトを所有するか、適切な権限を持つエンティティ。
- `project` (str): アーティファクトが保存されているプロジェクト。
- `artifact` (str): 取得するアーティファクトの名前。
- `version` (str): 取得するアーティファクトのバージョン。
- `file` (str): 取得するアーティファクト内に保存されているファイルの名前。

---

## <kbd>class</kbd> `WeaveBlockSummaryTable`
W&B にログされた W&B テーブル、pandas DataFrame、プロット、またはその他の値を表示するブロック。クエリは次の形式を取ります。

```python
project('entity', 'project').runs.summary['value']
```

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

**Attributes:**

- `entity` (str): 値がログされたプロジェクトを所有するか、適切な権限を持つエンティティ。
- `project` (str): 値がログされたプロジェクト。
- `table_name` (str): テーブル、DataFrame、プロット、または値の名前。

---

## <kbd>class</kbd> `WeavePanel`
クエリを使用してカスタムコンテンツを表示するための空のクエリパネル。

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

---

## <kbd>class</kbd> `WeavePanelArtifact`
W&B にログされたアーティファクトを示すパネル。

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

**Attributes:**

- `artifact` (str): 取得するアーティファクトの名前。
- `tab` (Literal["overview", "metadata", "usage", "files", "lineage"]): アーティファクトパネルに表示するタブ。

---

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
バージョンのあるファイルを W&B アーティファクトにログしたことを示すパネル。

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

**Attributes:**

- `artifact` (str): 取得するアーティファクトの名前。
- `version` (str): 取得するアーティファクトのバージョン。
- `file` (str): 取得するアーティファクト内に保存されているファイルの名前。

---

## <kbd>class</kbd> `WeavePanelSummaryTable`
W&B にログされた W&B テーブル、pandas DataFrame、プロット、またはその他の値を表示するパネル。クエリは次の形式を取ります。

```python
runs.summary['value']
```

API 名内の "Weave" の用語は、LLM を追跡および評価するために使用される W&B Weave ツールキットを指していません。

**Attributes:**

- `table_name` (str): テーブル、DataFrame、プロット、または値の名前。