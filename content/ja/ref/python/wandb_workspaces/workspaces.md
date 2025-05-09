---
title: ワークスペース
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-workspaces
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}

# <kbd>module</kbd> `wandb_workspaces.workspaces`
プログラムで W&B Workspace API を操作するための Python ライブラリ。

```python
# インポート方法
import wandb_workspaces.workspaces as ws

# ワークスペースを作成する例
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
runset （左側のバー）内の run の設定。

**属性:**
 
 - `color` (str): UI での run の色。hex (#ff0000), css color (red), または rgb (rgb(255, 0, 0)) を指定できる。
 - `disabled` (bool): run が非アクティブであるかどうか（UI で目が閉じている）。デフォルトは `False` に設定されている。

---

## <kbd>class</kbd> `RunsetSettings`
ワークスペース内の runset（run を含む左側のバー）の設定。

**属性:**
 
 - `query` (str): runset をフィルターするためのクエリ（regex 式である可能性あり、次のパラメータを参照）。 
 - `regex_query` (bool): 上記のクエリが regex 式であるかどうかを制御する。デフォルトは `False` に設定されている。 
 - `filters` `(LList[expr.FilterExpr])`: runset に適用するフィルターのリスト。フィルターは AND で結合される。フィルターの作成については FilterExpr を参照。 
 - `groupby` `(LList[expr.MetricType])`: runset でグループ化するメトリクスのリスト。 `Metric`, `Summary`, `Config`, `Tags`, または `KeysInfo` に設定。 
 - `order` `(LList[expr.Ordering])`: runset に適用するメトリクスと順序のリスト。 
 - `run_settings` `(Dict[str, RunSettings])`: run の設定の辞書。キーは run の ID で、値は RunSettings オブジェクト。

---

## <kbd>class</kbd> `Section`
ワークスペース内のセクションを表す。

**属性:**
 
 - `name` (str): セクションの名前またはタイトル。
 - `panels` `(LList[PanelTypes])`: セクション内のパネルの順序付きリスト。デフォルトでは、最初が左上で最後が右下。
 - `is_open` (bool): セクションが開いているか閉じているか。デフォルトは閉じている。
 - `layout_settings` `(Literal[`standard`, `custom`])`: セクション内のパネルレイアウトの設定。
 - `panel_settings`: セクション内のすべてのパネルに適用されるパネルレベルの設定。 `WorkspaceSettings` の `Section` に似ている。

---

## <kbd>class</kbd> `SectionLayoutSettings`
セクションのパネルレイアウト設定。通常、W&B App Workspace UI のセクションの右上に表示される。

**属性:**
 
 - `layout` `(Literal[`standard`, `custom`])`: セクション内のパネルのレイアウト。 `standard` はデフォルトのグリッドレイアウトに従い、`custom` は個々のパネル設定で制御されるカスタムレイアウトを許可する。
 - `columns` (int): 標準レイアウトの場合、レイアウト内の列数。デフォルトは 3。
 - `rows` (int): 標準レイアウトの場合、レイアウト内の行数。デフォルトは 2。

---

## <kbd>class</kbd> `SectionPanelSettings`
セクションのパネル設定。セクションの `WorkspaceSettings` に似ている。

ここで適用される設定は、より詳細なパネル設定で上書きされることがある。優先順位は: Section < Panel。

**属性:**
 
 - `x_axis` (str): X 軸メトリック名の設定。デフォルトでは `Step` に設定。
 - `x_min Optional[float]`: X 軸の最小値。
 - `x_max Optional[float]`: X 軸の最大値。
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): すべてのパネルに適用されるスムージングタイプ。
 - `smoothing_weight` (int): すべてのパネルに適用されるスムージングウエイト。

---

## <kbd>class</kbd> `Workspace`
W&B ワークスペースを表し、セクション、設定、run セットの構成を含む。

**属性:**
 
 - `entity` (str): このワークスペースが保存される entity（通常、ユーザーまたはチーム名）。
 - `project` (str): このワークスペースが保存されるプロジェクト。
 - `name`: ワークスペースの名前。
 - `sections` `(LList[Section])`: ワークスペース内のセクションの順序付きリスト。最初のセクションはワークスペースの上部にある。
 - `settings` `(WorkspaceSettings)`: ワークスペースの設定。通常、UI のワークスペースの上部に表示される。
 - `runset_settings` `(RunsetSettings)`: ワークスペース内の run セット（run を含む左側のバー）の設定。

---

#### <kbd>property</kbd> url

W&B アプリ内のワークスペースへの URL。

---

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str)
```

URL からワークスペースを取得。

---

### <kbd>method</kbd> `save`

```python
save()
```

現在のワークスペースを W&B に保存。

**戻り値:**
 
 - `Workspace`: 保存された内部名と ID を持つ更新されたワークスペース。

---

### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

現在のワークスペースを W&B に新しいビューとして保存。

**戻り値:**
 
 - `Workspace`: 保存された内部名と ID を持つ更新されたワークスペース。

---

## <kbd>class</kbd> `WorkspaceSettings`
ワークスペースの設定。通常、UI のワークスペースの上部に表示される。

このオブジェクトには、x 軸、スムージング、外れ値、パネル、ツールチップ、run、パネルクエリバーの設定が含まれる。

ここで適用される設定は、より詳細なセクションおよびパネル設定で上書きされることがある。優先順位は: Workspace < Section < Panel。

**属性:**
 
 - `x_axis` (str): X 軸メトリック名の設定。
 - `x_min` `(Optional[float])`: X 軸の最小値。
 - `x_max` `(Optional[float])`: X 軸の最大値。
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: すべてのパネルに適用されるスムージングタイプ。
 - `smoothing_weight` (int): すべてのパネルに適用されるスムージングウエイト。
 - `ignore_outliers` (bool): すべてのパネルで外れ値を無視する。
 - `sort_panels_alphabetically` (bool): すべてのセクションでパネルをアルファベット順にソート。
 - `group_by_prefix` `(Literal[`first`, `last`])`: 最初または最大最後までのプレフィックスでパネルをグループ化する（first または last）。デフォルトは last に設定。
 - `remove_legends_from_panels` (bool): すべてのパネルから凡例を削除。
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: ツールチップに表示する run の数。
 - `tooltip_color_run_names` (bool): ツールチップで run 名を run セットに合わせて色付けするかどうか（True）あるいはしないか（False）。デフォルトは True に設定。
 - `max_runs` (int): パネルごとに表示される run の最大数（run セットの最初の 10 件の run になる）。
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: 点の可視化メソッド。
 - `panel_search_query` (str): パネル検索バーのクエリ（正規表現式である可能性あり）。
 - `auto_expand_panel_search_results` (bool): パネル検索結果を自動拡張するかどうか。