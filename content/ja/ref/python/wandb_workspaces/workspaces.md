---
title: Workspaces
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-workspaces
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}






# <kbd>module</kbd> `wandb_workspaces.workspaces`
W&B Workspace API をプログラムで使用するための Python ライブラリです。

```python
# インポート方法
import wandb_workspaces.workspaces as ws

# ワークスペース作成例
ws.Workspace(
     name="Example W&B Workspace",
     entity="entity", # ワークスペースを所有する entity
     project="project", # ワークスペースが関連付けられている project
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
runset (左側のバー) 内の run の設定。

**属性:**
 
 - `color` (str): UI での run の色。16 進数 (#ff0000)、CSS カラー (red)、または rgb (rgb(255, 0, 0)) を指定できます。
 - `disabled` (bool): run が非アクティブ化されているかどうか (UI で目が閉じている状態)。デフォルトは `False` に設定されています。







---



## <kbd>class</kbd> `RunsetSettings`
ワークスペース内の runset (run を含む左側のバー) の設定。

**属性:**
 
 - `query` (str): runset をフィルタリングするためのクエリ (正規表現も可。次のパラメーターを参照)。
 - `regex_query` (bool): クエリ (上記) が正規表現であるかどうかを制御します。デフォルトは `False` に設定されています。
 - `filters` `(LList[expr.FilterExpr])`: runset に適用するフィルタのリスト。フィルタは AND で結合されます。フィルタの作成方法の詳細については、FilterExpr を参照してください。
 - `groupby` `(LList[expr.MetricType])`: runset でグループ化するメトリクスのリスト。`Metric`、`Summary`、`Config`、`Tags`、または `KeysInfo` に設定します。
 - `order` `(LList[expr.Ordering])`: runset に適用するメトリクスと順序のリスト。
 - `run_settings` `(Dict[str, RunSettings])`: run 設定の辞書。キーは run の ID で、値は RunSettings オブジェクトです。







---



## <kbd>class</kbd> `Section`
ワークスペース内のセクションを表します。

**属性:**
 
 - `name` (str): セクションの名前/タイトル。
 - `panels` `(LList[PanelTypes])`: セクション内のパネルの順序付きリスト。デフォルトでは、最初が左上、最後が右下になります。
 - `is_open` (bool): セクションが開いているか閉じているか。デフォルトは閉じています。
 - `layout_settings` `(Literal[`standard`, `custom`])`: セクション内のパネルレイアウトの設定。
 - `panel_settings`: セクションの `WorkspaceSettings` と同様に、セクション内のすべてのパネルに適用されるパネルレベルの設定。







---



## <kbd>class</kbd> `SectionLayoutSettings`
セクションのパネルレイアウト設定。通常、W&B App Workspace UI のセクションの右上に見られます。

**属性:**
 
 - `layout` `(Literal[`standard`, `custom`])`: セクション内のパネルのレイアウト。`standard` はデフォルトのグリッドレイアウトに従い、`custom` は個々のパネル設定によって制御されるパネルごとのレイアウトを可能にします。
 - `columns` (int): 標準レイアウトでのレイアウト内の列数。デフォルトは 3 です。
 - `rows` (int): 標準レイアウトでのレイアウト内の行数。デフォルトは 2 です。







---



## <kbd>class</kbd> `SectionPanelSettings`
セクションのパネル設定。セクションの `WorkspaceSettings` と同様です。

ここで適用される設定は、次の優先順位で、より詳細なパネル設定によってオーバーライドできます: Section < Panel。

**属性:**
 
 - `x_axis` (str): X 軸のメトリクス名設定。デフォルトでは、`Step` に設定されています。
 - `x_min Optional[float]`: X 軸の最小値。
 - `x_max Optional[float]`: X 軸の最大値。
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): すべてのパネルに適用されるスムージングタイプ。
 - `smoothing_weight` (int): すべてのパネルに適用されるスムージングの重み。







---



## <kbd>class</kbd> `Workspace`
セクション、設定、および run set の構成を含む、W&B ワークスペースを表します。

**属性:**
 
 - `entity` (str): このワークスペースが保存される entity (通常は user または Team 名)。
 - `project` (str): このワークスペースが保存される project。
 - `name`: ワークスペースの名前。
 - `sections` `(LList[Section])`: ワークスペース内のセクションの順序付きリスト。最初のセクションはワークスペースの上部にあります。
 - `settings` `(WorkspaceSettings)`: ワークスペースの設定。通常、UI のワークスペースの上部に表示されます。
 - `runset_settings` `(RunsetSettings)`: ワークスペース内の runset (run を含む左側のバー) の設定。


---

#### <kbd>property</kbd> url

W&B アプリのワークスペースへの URL。

---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str)
```

URL からワークスペースを取得します。

---



### <kbd>method</kbd> `save`

```python
save()
```

現在のワークスペースを W&B に保存します。

**戻り値:**
 
 - `Workspace`: 保存された内部名と ID を持つ、更新されたワークスペース。

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

現在のワークスペースを新しいビューとして W&B に保存します。

**戻り値:**
 
 - `Workspace`: 保存された内部名と ID を持つ、更新されたワークスペース。

---



## <kbd>class</kbd> `WorkspaceSettings`
ワークスペースの設定。通常、UI のワークスペースの上部に表示されます。

このオブジェクトには、X 軸、スムージング、外れ値、パネル、ツールチップ、run、およびパネルクエリバーの設定が含まれています。

ここで適用される設定は、次の優先順位で、より詳細なセクションおよびパネル設定によってオーバーライドできます: Workspace < Section < Panel

**属性:**
 
 - `x_axis` (str): X 軸のメトリクス名設定。
 - `x_min` `(Optional[float])`: X 軸の最小値。
 - `x_max` `(Optional[float])`: X 軸の最大値。
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: すべてのパネルに適用されるスムージングタイプ。
 - `smoothing_weight` (int): すべてのパネルに適用されるスムージングの重み。
 - `ignore_outliers` (bool): すべてのパネルで外れ値を無視します。
 - `sort_panels_alphabetically` (bool): すべてのセクションでパネルをアルファベット順にソートします。
 - `group_by_prefix` `(Literal[`first`, `last`])`: パネルを最初のプレフィックスまたは最後のプレフィックス (最初または最後) でグループ化します。デフォルトは `last` に設定されています。
 - `remove_legends_from_panels` (bool): すべてのパネルから凡例を削除します。
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: ツールチップに表示する run の数。
 - `tooltip_color_run_names` (bool): ツールチップで run 名を runset に一致するように色分けするかどうか (True) しないか (False)。デフォルトは `True` に設定されています。
 - `max_runs` (int): パネルごとに表示する run の最大数 (これは runset の最初の 10 個の run になります)。
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: ポイントの可視化 method。
 - `panel_search_query` (str): パネル検索バーのクエリ (正規表現も可)。
 - `auto_expand_panel_search_results` (bool): パネル検索 result を自動的に展開するかどうか。
