# Api

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示する](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L276-L968)

wandbサーバーを照会するために使用されます。

```python
Api(
 overrides=None,
 timeout: Optional[int] = None,
 api_key: Optional[str] = None
) -> None
```

#### 例:

最も一般的な初期化方法
```
 wandb.Api()
```
| 引数 |  |
| :--- | :--- |
| `overrides` | (dict) https://api.wandb.ai 以外の wandb サーバーを使用している場合は、`base_url` を設定できます。また、`entity`、`project`、および `run` のデフォルトも設定できます。 |

| 属性 |  |
| :--- | :--- |

## メソッド

### `artifact`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L939-L962)

```python
artifact(
 name, type=None
)
```

`entity/project/run_id` の形式でパスを解析して、単一のアーティファクトを返します。

| 引数 | |
| :--- | :--- |
| `name` | (str) アーティファクトの名前。エンティティ/プロジェクトでプレフィックスがつく可能性があります。有効な名前は以下の形式であることができます: name:version name:alias digest |
| `type` | (str, 任意) 取得するアーティファクトのタイプ。 |



| 返り値 | |
| :--- | :--- |
| `Artifact` オブジェクト。|



### `artifact_type`



[ソースの表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L928-L931)

```python
artifact_type(
 type_name, project=None
)
```




### `artifact_types`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L923-L926)

```python
artifact_types(
 project=None
)
```




### `artifact_versions`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L933-L937)

```python
artifact_versions(
 type_name, name, per_page=50
)
```




### `create_project`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L462-L463)

```python
create_project(
 name: str,
 entity: str
)
```

### `create_report`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L445-L460)

```python
create_report(
 project: str,
 entity: str = "",
 title: Optional[str] = "無題のレポート",
 description: Optional[str] = "",
 width: Optional[str] = "readable",
 blocks: Optional['wandb.apis.reports.util.Block'] = None
) -> "wandb.apis.reports.Report"
```
### `create_run`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L439-L443)

```python
create_run(
 **kwargs
)
```

新しいrunを作成します。

### `create_team`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L737-L747)

```python
create_team(
 team, admin_username=None
)
```
新しいチームを作成します。

| 引数 |  |
| :--- | :--- |
| `team` | (str) チームの名前 |
| `admin_username` | (str) オプション: チームの管理ユーザーのユーザー名。デフォルトは現在のユーザーです。 |


| 戻り値 |  |
| :--- | :--- |
| `Team` オブジェクト |


### `create_user`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L481-L491)

```python
create_user(
 email, admin=(False)
)
```

新しいユーザーを作成します。
| 引数 | |
| :--- | :--- |
| `email` | (str) チーム名 |
| `admin` | (bool) このユーザーがグローバルインスタンスの管理者であるかどうか |



| 戻り値 | |
| :--- | :--- |
| `User` オブジェクト |



### `flush`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L555-L562)

```python
flush()
```

ローカルキャッシュをフラッシュします。

apiオブジェクトはrunsのローカルキャッシュを保持しているため、スクリプトの実行中にrunの状態が変更される可能性がある場合は、`api.flush()`でローカルキャッシュをクリアして、runに関連する最新の値を取得する必要があります。
### `from_path`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L564-L618)

```python
from_path(
 path
)
```

パスからrun、スイープ、プロジェクト、またはレポートを返します。

#### 例:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| 引数 | |
| :--- | :--- |
| `path` | (str) プロジェクト、run、スイープ、またはレポートへのパス |

| Returns | |
| :--- | :--- |
| `Project`、`Run`、`Sweep`、または `BetaReport` のインスタンス。 |



| Raises | |
| :--- | :--- |
| パスが無効であるか、オブジェクトが存在しない場合は wandb.Error |



### `job`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L964-L968)

```python
job(
 name, path=None
)
```
### `load_report`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L465-L479)

```python
load_report(
 path: str
) -> "wandb.apis.reports.Report"
```

指定したパスのレポートを取得します。

| 引数 | |
| :--- | :--- |
| `path` | (str) 対象のレポートへのパス。形式は `entity/project/reports/reportId` です。これは、wandbのURLの後にURLをコピーして貼り付けることで取得できます。例： `megatruong/report-editing/reports/My-fabulous-report-title--VmlldzoxOTc1Njk0` |



| 返り値 | |
| :--- | :--- |
| `path` で指定されたレポートを表す `BetaReport` オブジェクト |



| 例外 | |
| :--- | :--- |
| パスが無効な場合はwandb.Error |
### `project`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L701-L704)

```python
project(
 name, entity=None
)
```




### `projects`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L678-L699)

```python
projects(
 entity=None, per_page=200
)
```

指定されたエンティティのプロジェクトを取得します。

| 引数 | |
| :--- | :--- |
| `entity` | (str) 要求されたエンティティの名前。Noneの場合、`Api`に渡されたデフォルトのエンティティにフォールバックします。デフォルトのエンティティがない場合、`ValueError`を発生させます。|
| `per_page` | (int) クエリのページネーションに対するページサイズを設定します。Noneの場合、デフォルトのサイズを使用します。通常、これを変更する理由はありません。 |



| 戻り値 | |
| :--- | :--- |
| イテラブルな`Project`オブジェクトのコレクションである`Projects`オブジェクト。 |



### `queued_run`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L883-L904)

```python
queued_run(
 entity, project, queue_name, run_queue_item_id, container_job=(False),
 project_queue=None
)
```

パスに基づいて、1つのキューに入ったrunを返します。

entity/project/queue_id/run_queue_item_id の形式のパスを解析します。
### `reports`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L706-L735)

```python
reports(
 path="", name=None, per_page=50
)
```

指定したプロジェクトパスに対するレポートを取得します。

警告: このAPIはベータ版であり、今後のリリースで変更される可能性があります。

| 引数 |  |
| :--- | :--- |
| `path` | (str) レポートが存在するプロジェクトへのパスで、"entity/project"の形式であるべきです。 |
| `name` | (str) オプションで指定するレポートの名前。 |
| `per_page` | (int) クエリのページネーションにおけるページサイズを設定します。Noneを指定するとデフォルトサイズが使用されます。通常、この設定を変更する必要はありません。 |


| 戻り値 |  |
| :--- | :--- |
| `BetaReport`オブジェクトのイテラブルなコレクションである `Reports`オブジェクト。|
### `run`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L866-L881)

```python
run(
 path=""
)
```

エンティティ/プロジェクト/ランIDの形式でパスを解析して、単一のランを返します。

| 引数 | 説明 |
| :--- | :--- |
| `path` | (str) パスは`entity/project/run_id`の形式です。`api.entity`が設定されている場合、`project/run_id`の形式で、`api.project`が設定されている場合は、run_id だけで構いません。|



| 戻り値 | 説明 |
| :--- | :--- |
| `Run`オブジェクト。 |



### `runs`
[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L788-L864)

```python
runs(
 path: Optional[str] = None,
 filters: Optional[Dict[str, Any]] = None,
 order: str = "-created_at",
 per_page: int = 50,
 include_sweeps: bool = (True)
)
```

指定されたフィルタに一致するプロジェクトからのrunsのセットを返します。

`config.*`、`summary_metrics.*`、`tags`、`state`、`entity`、`createdAt`などでフィルタリングできます。

#### 例：

my_projectでconfig.experiment_nameが"foo"に設定されているrunsを検索
```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

my_projectでconfig.experiment_nameが"foo"または"bar"に設定されているrunsを検索
```
api.runs(
 path="my_entity/my_project",
 filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```
私のプロジェクトでconfig.experiment_nameが正規表現に一致するrunを見つける（アンカーはサポートされていません）
```
api.runs(
 path="my_entity/my_project",
 filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

私のプロジェクトでrun名が正規表現に一致するrunを見つける（アンカーはサポートされていません）
```
api.runs(
 path="my_entity/my_project",
 filters={"display_name": {"$regex": "^foo.*"}}
)
```

私のプロジェクトのrunを昇順の損失で並べ替える
```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```



| 引数 | |
| :--- | :--- |
| `path` | (str) プロジェクトへのパスで、"entity/project"の形式である必要があります |
| `filters` | (dict) MongoDBクエリ言語を使用して特定のrunをクエリします。config.key、summary_metrics.key、state、entity、createdAtなどのrunプロパティで絞り込むことができます。例：{"config.experiment_name": "foo"} は実験名が"foo"に設定されているrunを見つけます。詳細なクエリを作成するために、操作を組み合わせることができます。言語のリファレンスは https://docs.mongodb.com/manual/reference/operator/query で参照できます。 |
| `order` | (str) 順序は `created_at`、`heartbeat_at`、`config.*.value`、または `summary_metrics.*` にすることができます。順序に+を付けると昇順になります。順序に-を付けると降順になります（デフォルト）。デフォルトの順序は最新から最古のrun.created_atです。 |
| 戻り値 | |
| :--- | :--- |
| `Runs`オブジェクト。これは、`Run`オブジェクトのイテレータブルなコレクションです。|



### `sweep`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L906-L921)

```python
sweep(
 path=""
)
```

`entity/project/sweep_id`の形式でパスを解析して、スイープを返します。


| 引数 | |
| :--- | :--- |
| `path` | (str, optional) entity/project/sweep_idの形式で指定されたスイープへのパス。`api.entity`が設定されている場合は、project/sweep_idの形式で指定でき、`api.project`が設定されている場合は、sweep_idだけで指定できます。 |



| 戻り値 | |
| :--- | :--- |
| `Sweep`オブジェクト。 |
### `sync_tensorboard`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L493-L515)

```python
sync_tensorboard(
 root_dir, run_id=None, project=None, entity=None
)
```

tfeventファイルを含むローカルディレクトリをwandbに同期します。

### `team`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L749-L750)

```python
team(
 team
)
```
### `user`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L752-L772)

```python
user(
 ユーザー名_または_メール
)
```

ユーザー名またはメールアドレスからユーザーを返します。

注: この関数はローカル管理者専用です。自分自身のユーザーオブジェクトを取得しようとする場合は、`api.viewer`を使用してください。

| 引数 | |
| :--- | :--- |
| `ユーザー名_または_メール` | (str) ユーザーのユーザー名またはメールアドレス |



| 返り値 | |
| :--- | :--- |
| `User`オブジェクト または ユーザーが見つからない場合はNone |
### `users`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L774-L786)

```python
users(
 username_or_email
)
```

部分的なユーザー名またはメールアドレスのクエリからすべてのユーザーを返します。

注: この関数はローカル管理者のみが使用できます。自分のユーザーオブジェクトを取得しようとしている場合は、`api.viewer`を使用してください。

| 引数 | |
| :--- | :--- |
| `username_or_email` | (str) 検索したいユーザーの接頭辞または接尾辞 |



| 戻り値 | |
| :--- | :--- |
| `User`オブジェクトの配列 |
| クラス変数 | |

| :--- | :--- |

| `CREATE_PROJECT` | |

| `USERS_QUERY` | |

| `VIEWER_QUERY` | |