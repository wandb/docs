---
title: ワークスペース
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-workspaces
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}




{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}


# <kbd>module</kbd> `wandb_workspaces.workspaces`
W&B Workspace API をプログラムから操作するための Python ライブラリです。

```python
# インポート方法
import wandb_workspaces.workspaces as ws

# Workspace の作成例
ws.Workspace(
     name="Example W&B Workspace",
     entity="entity", # Workspace を所有する entity
     project="project", # Workspace に関連付ける project
     sections=[
         ws.Section(
             name="Validation Metrics",
             panels=[
                 wr.LinePlot(x="Step", y=["val_loss"]),
                 wr.BarPlot(metrics=["val_accuracy"]),
                 wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
             ],
             is_open=True,
         ),
     ],
)
workspace.save()
```

---



## <kbd>class</kbd> `RunSettings`
runset（左側バー）内の run に対する設定です。



**属性:**
 
 - `color` (str): UI 上で表示される run の色。hex（#ff0000）、css カラー（red）、または rgb（rgb(255, 0, 0)）が利用可能です。
 - `disabled` (bool): run が非アクティブかどうか（UI で目のアイコンが閉じている状態）。デフォルトは `False` です。







---



## <kbd>class</kbd> `RunsetSettings`
Workspace の中の runset（run を含む左サイドバー）の設定。



**属性:**
 
 - `query` (str): runset をフィルタするためのクエリ（正規表現も可、詳細は次のパラメータを参照）。
 - `regex_query` (bool): 上記のクエリを正規表現として扱うかどうか。デフォルトは `False`。
 - `filters` `(LList[expr.FilterExpr])`: runset に適用するフィルタのリスト。フィルタはすべて AND されます。フィルタ作成の詳細は FilterExpr を参照してください。
 - `groupby` `(LList[expr.MetricType])`: runset 内でグループ化するメトリクスのリスト。`Metric`、`Summary`、`Config`、`Tags`、または `KeysInfo` に設定可能です。
 - `order` `(LList[expr.Ordering])`: runset に適用するメトリクスおよびソート順のリスト。
 - `run_settings` `(Dict[str, RunSettings])`: run の ID をキー、RunSettings オブジェクトを値とする run の設定の辞書。







---



## <kbd>class</kbd> `Section`
Workspace 内のセクションを表します。



**属性:**
 
 - `name` (str): セクションの名前またはタイトル。
 - `panels` `(LList[PanelTypes])`: セクション内のパネルを順序付きで格納したリスト。デフォルトでは最初が左上、最後が右下になります。
 - `is_open` (bool): セクションが開いているか閉じているか。デフォルトは閉じています。
 - `layout_settings` `(Literal[`standard`, `custom`])`: セクション内のパネルレイアウトに関する設定。
 - `panel_settings`: セクション内すべてのパネルへ適用されるパネルレベルの設定（`Section` 用の `WorkspaceSettings` に類似）。







---



## <kbd>class</kbd> `SectionLayoutSettings`
セクションのパネルレイアウト設定。通常、W&B App Workspace UI のセクション右上で見られます。



**属性:**
 
 - `layout` `(Literal[`standard`, `custom`])`: セクション内のパネルレイアウト。`standard` だとデフォルトのグリッド、`custom` だと個々のパネル設定によるカスタムレイアウトが可能です。
 - `columns` (int): 標準レイアウト時の列数。デフォルトは3。
 - `rows` (int): 標準レイアウト時の行数。デフォルトは2。







---



## <kbd>class</kbd> `SectionPanelSettings`
セクション内のパネル設定。セクション用の `WorkspaceSettings` に相当。



ここで指定した設定は、より細かいパネルの設定で上書き可能です。優先順位はセクション < パネル の順です。



**属性:**
 
 - `x_axis` (str): X軸となるメトリクス名の設定。デフォルトは `Step`。
 - `x_min Optional[float]`: X軸の最小値。
 - `x_max Optional[float]`: X軸の最大値。
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): 全パネルに適用されるスムージングタイプ。
 - `smoothing_weight` (int): 全パネルに適用されるスムージング重み。







---



## <kbd>class</kbd> `Workspace`
セクション・設定・runset の構成など、W&B workspace を表現します。



**属性:**
 
 - `entity` (str): この workspace を保存するエンティティ名（通常は user や team の名前）。
 - `project` (str): この workspace を保存するプロジェクト名。
 - `name`: Workspace の名称。
 - `sections` `(LList[Section])`: Workspace 内にあるセクションの順序付きリスト。最初のセクションが workspace の一番上に表示されます。
 - `settings` `(WorkspaceSettings)`: Workspace の設定。通常、UI の workspace 上部で見られます。
 - `runset_settings` `(RunsetSettings)`: Workspace の runset（左サイドバー）の設定。


---

#### <kbd>property</kbd> url

W&B アプリ内での Workspace の URL。



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str)
```

指定した URL から workspace を取得します。

---



### <kbd>method</kbd> `save`

```python
save()
```

現在の workspace を W&B に保存します。



**戻り値:**
 
 - `Workspace`: 保存された内部名と ID を含む更新済み workspace。

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

現在の workspace を新規ビューとして W&B に保存します。



**戻り値:**
 
 - `Workspace`: 保存された内部名と ID を含む更新済み workspace。

---



## <kbd>class</kbd> `WorkspaceSettings`
Workspace 全体の設定。通常 UI の workspace 上部に表示されるものです。

このオブジェクトには x 軸、スムージング、外れ値、パネル、ツールチップ、run、パネルクエリバーなどの設定が含まれます。

ここで適用された設定はより細かい Section や Panel の設定で上書き可能です。優先順位は Workspace < Section < Panel です。



**属性:**
 
 - `x_axis` (str): X軸メトリクス名の設定。
 - `x_min` `(Optional[float])`: X軸の最小値。
 - `x_max` `(Optional[float])`: X軸の最大値。
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: 全パネルに適用されるスムージングタイプ。
 - `smoothing_weight` (int): 全パネルに適用されるスムージング重み。
 - `ignore_outliers` (bool): すべてのパネルで外れ値を無視するかどうか。
 - `sort_panels_alphabetically` (bool): すべてのセクション内のパネルをアルファベット順でソート。
 - `group_by_prefix` `(Literal[`first`, `last`])`: パネルを最初または最後のプレフィックスでグループ化（first または last）。デフォルトは `last`。
 - `remove_legends_from_panels` (bool): すべてのパネルから凡例を削除。
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: ツールチップに表示する run の数。
 - `tooltip_color_run_names` (bool): ツールチップ内で run 名を runset と同じ色で表示するかどうか。有効時（True）は runset と同じ色、デフォルトは `True`。
 - `max_runs` (int): 各パネルで表示する最大 run 数（runset の最初の10件）。
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: ポイントの可視化方法。
 - `panel_search_query` (str): パネル検索バー用のクエリ（正規表現可）。
 - `auto_expand_panel_search_results` (bool): パネル検索結果を自動的に展開するか。