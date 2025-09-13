---
title: Workspaces
menu:
  reference:
    identifier: ja-ref-python-wandb_workspaces-workspaces
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}




{{% alert %}}
W&B Report と Workspace API はパブリックプレビューです。
{{% /alert %}}


# <kbd>module</kbd> `wandb_workspaces.workspaces`
W&B Workspace API をプログラムから操作するための Python ライブラリ。 

```python
# インポート方法
import wandb_workspaces.workspaces as ws

# Workspace を作成する例
ws.Workspace(
     name="Example W&B Workspace",
     entity="entity", # Workspace の所有者となる Entity
     project="project", # Workspace が紐づく Project
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
runset（左側バー）内の run の設定。 



**属性:**
 
 - `color` (str): UI における run の色。hex（#ff0000）、CSS カラー（red）、または RGB（rgb(255, 0, 0)）が使用可能。 
 - `disabled` (bool): run を非表示（UI の目アイコンが閉じた状態）にするかどうか。デフォルトは `False`。 







---



## <kbd>class</kbd> `RunsetSettings`
Workspace 内の runset（左側の Runs を含むバー）の設定。 



**属性:**
 
 - `query` (str): runset を絞り込むためのクエリ（正規表現を使用可能。次のパラメータ参照）。 
 - `regex_query` (bool): 上記の `query` を正規表現として扱うかどうか。デフォルトは `False`。 
 - `filters` `(LList[expr.FilterExpr])`: runset に適用するフィルターのリスト。フィルターは AND で結合されます。FilterExpr を参照。 
 - `groupby` `(LList[expr.MetricType])`: runset でグループ化に使うメトリクスのリスト。`Metric`、`Summary`、`Config`、`Tags`、`KeysInfo` のいずれか。 
 - `order` `(LList[expr.Ordering])`: runset に適用するメトリクスと並び順のリスト。 
 - `run_settings` `(Dict[str, RunSettings])`: run の設定の辞書。キーは run の ID、値は RunSettings オブジェクト。 







---



## <kbd>class</kbd> `Section`
Workspace 内のセクションを表します。 



**属性:**
 
 - `name` (str): セクションの名前/タイトル。 
 - `panels` `(LList[PanelTypes])`: セクション内のパネルの順序付きリスト。デフォルトでは、先頭が左上、末尾が右下。 
 - `is_open` (bool): セクションが開いているかどうか。デフォルトは閉じています。 
 - `layout_settings` `(Literal[`standard`, `custom`])`: セクション内のパネル レイアウトの設定。 
 - `panel_settings`: セクション内のすべてのパネルに適用されるパネル レベルの設定。`Section` に対する `WorkspaceSettings` に類似。 







---



## <kbd>class</kbd> `SectionLayoutSettings`
セクション用のパネル レイアウト設定。W&B アプリの Workspace UI のセクション右上に表示される項目です。 



**属性:**
 
 - `layout` `(Literal[`standard`, `custom`])`: セクション内のパネルのレイアウト。`standard` はデフォルトのグリッド レイアウト、`custom` は各パネルの設定で制御されるパネル個別のレイアウトを許可します。 
 - `columns` (int): standard レイアウトでの列数。デフォルトは 3。 
 - `rows` (int): standard レイアウトでの行数。デフォルトは 2。 







---



## <kbd>class</kbd> `SectionPanelSettings`
セクション用のパネル設定。セクションに対する `WorkspaceSettings` に類似。 

ここでの設定は、より粒度の細かいパネルの設定で上書きされます。優先度は Section < Panel。 



**属性:**
 
 - `x_axis` (str): X 軸のメトリクス名の設定。デフォルトは `Step`。 
 - `x_min Optional[float]`: X 軸の最小値。 
 - `x_max Optional[float]`: X 軸の最大値。 
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): すべてのパネルに適用されるスムージング種別。 
 - `smoothing_weight` (int): すべてのパネルに適用されるスムージングの重み。 







---



## <kbd>class</kbd> `Workspace`
セクション、設定、runset の構成を含む W&B Workspace を表します。 



**属性:**
 
 - `entity` (str): この Workspace を保存する Entity（通常は User または Team の名前）。 
 - `project` (str): この Workspace を保存する Project。 
 - `name`: Workspace の名前。 
 - `sections` `(LList[Section])`: Workspace 内のセクションの順序付きリスト。最初のセクションは Workspace の最上部に表示されます。 
 - `settings` `(WorkspaceSettings)`: Workspace 用の設定。通常は UI の Workspace 上部に表示されます。 
 - `runset_settings` `(RunsetSettings)`: Workspace の runset（Runs を含む左側バー）の設定。 


---

#### <kbd>property</kbd> url

W&B アプリ内の Workspace への URL。 



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str)
```

URL から Workspace を取得します。 

---



### <kbd>method</kbd> `save`

```python
save()
```

現在の Workspace を W&B に保存します。 



**戻り値:**
 
 - `Workspace`: 保存済みの内部名と ID を持つ、更新された Workspace。 

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

現在の Workspace を新しいビューとして W&B に保存します。 



**戻り値:**
 
 - `Workspace`: 保存済みの内部名と ID を持つ、更新された Workspace。

---



## <kbd>class</kbd> `WorkspaceSettings`
Workspace 用の設定。通常は UI の Workspace 上部に表示されます。 

このオブジェクトには、X 軸、スムージング、外れ値、パネル、ツールチップ、Runs、パネルのクエリ バーに関する設定が含まれます。 

ここでの設定は、より粒度の細かい Section および Panel の設定で上書きされます。優先度は Workspace < Section < Panel 



**属性:**
 
 - `x_axis` (str): X 軸のメトリクス名の設定。 
 - `x_min` `(Optional[float])`: X 軸の最小値。 
 - `x_max` `(Optional[float])`: X 軸の最大値。 
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: すべてのパネルに適用されるスムージング種別。 
 - `smoothing_weight` (int): すべてのパネルに適用されるスムージングの重み。 
 - `ignore_outliers` (bool): すべてのパネルで外れ値を無視。 
 - `sort_panels_alphabetically` (bool): すべてのセクションでパネルをアルファベット順にソート。 
 - `group_by_prefix` `(Literal[`first`, `last`])`: プレフィックスの先頭または末尾でパネルをグループ化（`first` または `last`）。デフォルトは `last`。 
 - `remove_legends_from_panels` (bool): すべてのパネルから凡例を削除。 
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: ツールチップに表示する run の数。 
 - `tooltip_color_run_names` (bool): ツールチップ内の run 名を runset に合わせた色で表示するかどうか（`True`）、しないか（`False`）。デフォルトは `True`。 
 - `max_runs` (int): パネルごとに表示する run の最大数（runset の先頭 10 個の run が対象）。 
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: ポイントの可視化方法。 
 - `panel_search_query` (str): パネル検索バーのクエリ（正規表現可）。 
 - `auto_expand_panel_search_results` (bool): パネル検索結果を自動展開するかどうか。