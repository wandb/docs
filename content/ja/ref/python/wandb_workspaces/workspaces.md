---
title: ワークスペース
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

# Workspace を作成する例
ws.Workspace(
     name="Example W&B Workspace",
     entity="entity", # Workspace を所有する entity
     project="project", # Workspace が紐付く project
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
runset（左側バー）の中の run の設定を管理するクラスです。



**属性:**
 
 - `color` (str): UI 上で run に割り当てられる色。hex（例: #ff0000）、CSS カラー（例: red）、または rgb（例: rgb(255, 0, 0)）が使用可能です。
 - `disabled` (bool): run が非アクティブ（UI で目のアイコンが閉じている状態）かどうか。デフォルトは `False` です。







---



## <kbd>class</kbd> `RunsetSettings`
ワークスペース内の runset（run が並ぶ左側のバー）の設定です。



**属性:**
 
 - `query` (str): runset に対して適用するクエリ（正規表現も利用可、下記参照）。
 - `regex_query` (bool): 上記クエリが正規表現かどうかを制御します。デフォルトは `False`。
 - `filters` `(LList[expr.FilterExpr])`: runset に適用するフィルタのリストです。フィルタは全て AND 条件で適用されます。フィルタ作成については FilterExpr を参照してください。
 - `groupby` `(LList[expr.MetricType])`: runset でグループ化するメトリクスのリスト。`Metric`, `Summary`, `Config`, `Tags`, `KeysInfo` のいずれかを指定。
 - `order` `(LList[expr.Ordering])`: runset で適用する、並び順やメトリクスのリスト。
 - `run_settings` `(Dict[str, RunSettings])`: run の ID をキー、RunSettings オブジェクトを値とした run の設定の辞書。







---



## <kbd>class</kbd> `Section`
ワークスペース内のセクションを表します。



**属性:**
 
 - `name` (str): セクションの名称／タイトル。
 - `panels` `(LList[PanelTypes])`: セクション内に並ぶパネルの順序付きリスト。デフォルトではリストの最初が左上、最後が右下に表示されます。
 - `is_open` (bool): セクションが開いているか閉じているかを指定します。デフォルトは閉じています。
 - `layout_settings` `(Literal[`standard`, `custom`])`: セクション内のパネルレイアウト設定です。
 - `panel_settings`: セクション全体に適用されるパネル設定。`Section` に対する `WorkspaceSettings` のような役割です。







---



## <kbd>class</kbd> `SectionLayoutSettings`
セクション内のパネルレイアウト設定です。W&B アプリ Workspace 画面右上などでよく見られます。



**属性:**
 
 - `layout` `(Literal[`standard`, `custom`])`: セクション内のパネル配置。`standard` はデフォルトのグリッド配置、`custom` は各パネルごとに独自配置を許可します（パネルごとの設定により制御）。
 - `columns` (int): standard レイアウト時における列数。デフォルトは 3。
 - `rows` (int): standard レイアウト時における行数。デフォルトは 2。







---



## <kbd>class</kbd> `SectionPanelSettings`
セクション全体に適用されるパネル設定。`Section` 向けの `WorkspaceSettings` に相当します。

ここで設定した内容は、より詳細なパネル個別設定（優先度：Section < Panel）で上書きできます。



**属性:**
 
 - `x_axis` (str): X 軸に使用するメトリクス名。デフォルトは `Step`。
 - `x_min Optional[float]`: X 軸の最小値。
 - `x_max Optional[float]`: X 軸の最大値。
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): 全パネルに適用される平滑化タイプ。
 - `smoothing_weight` (int): 全パネルに適用される平滑化係数。







---



## <kbd>class</kbd> `Workspace`
W&B Workspace を表現するクラスです。sections, 設定, runset の構成情報を持ちます。



**属性:**
 
 - `entity` (str): Workspace が保存される entity（通常はユーザー名やチーム名）。
 - `project` (str): Workspace が保存される project。
 - `name`: Workspace の名称。
 - `sections` `(LList[Section])`: workspace 内のセクションの順序付きリスト。最初のセクションが画面上部に表示されます。
 - `settings` `(WorkspaceSettings)`: workspace の設定。通常は UI 上の画面上部に表示される内容を管理します。
 - `runset_settings` `(RunsetSettings)`: ワークスペース内 runset（run 一覧が並ぶ左側バー）の設定。


---

#### <kbd>property</kbd> url

W&B アプリ内でワークスペースへアクセスする URL。



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



**返り値:**
 
 - `Workspace`: 保存された内部名と ID を含む更新済み Workspace オブジェクト。

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

現在のワークスペースを新しいビューとして W&B に保存します。



**返り値:**
 
 - `Workspace`: 保存された内部名と ID を含む更新済み Workspace オブジェクト。

---



## <kbd>class</kbd> `WorkspaceSettings`
ワークスペースの設定。通常 UI のワークスペース画面上部で見られます。

このオブジェクトには x 軸設定・平滑化・外れ値・パネル・ツールチップ・run・パネル検索バー関連の設定を含みます。

ここで設定した内容は、より詳細な Section や Panel の設定（優先度: Workspace < Section < Panel）で上書きすることができます。



**属性:**
 
 - `x_axis` (str): X 軸に使用するメトリクス名の設定。
 - `x_min` `(Optional[float])`: X 軸の最小値。
 - `x_max` `(Optional[float])`: X 軸の最大値。
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: 全パネルに適用する平滑化タイプ。
 - `smoothing_weight` (int): 全パネルに適用する平滑化係数。
 - `ignore_outliers` (bool): すべてのパネルで外れ値を無視するかどうか。
 - `sort_panels_alphabetically` (bool): すべてのセクションにおけるパネルをアルファベット順に並べ替えます。
 - `group_by_prefix` `(Literal[`first`, `last`])`: パネル名の先頭または末尾のプレフィックスでグルーピングします（first か last）。デフォルトは `last`。
 - `remove_legends_from_panels` (bool): すべてのパネルから凡例を非表示にします。
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: ツールチップに表示する run 数。
 - `tooltip_color_run_names` (bool): ツールチップ上の run 名を runset の色で表示するかどうか（True で色付け、False で色付けなし）。デフォルトは `True`。
 - `max_runs` (int): 1 パネルあたり表示する最大 run 数（runset の先頭 10 件など）。
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: ポイントの可視化方法。
 - `panel_search_query` (str): パネル検索バーのクエリ（正規表現を使用可能）。
 - `auto_expand_panel_search_results` (bool): パネル検索結果を自動的に展開するかどうか。