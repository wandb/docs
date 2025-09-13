---
title: API
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-api
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/api.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B に保存したデータをエクスポートまたは更新するには Public API を使用します。 

この API を使う前に、まずスクリプトからデータをログしてください。詳しくは [クイックスタート](https://docs.wandb.ai/quickstart) を参照してください。 

Public API の用途例: 
 - 実験が完了した後に、その実験のメタデータやメトリクスを更新する 
 - 結果をデータフレームとして取得し、Jupyter ノートブックで事後分析を行う 
 - 保存済みのモデル アーティファクトのうち、`ready-to-deploy` タグが付いたものを確認する 

Public API の使い方の詳細は [こちらのガイド](https://docs.wandb.com/guides/track/public-api-guide) を参照してください。 


## <kbd>class</kbd> `Api`
W&B サーバーにクエリを投げるために使用します。 



**例:**
 ```python
import wandb

wandb.Api()
``` 

### <kbd>method</kbd> `Api.__init__`

```python
__init__(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) → None
```

API を初期化します。 



**引数:**
 
 - `overrides`:  `base_url` を設定できます 
 - `using a W&B server other than `https`: //api.wandb.ai`. You can also set defaults for `entity`, `project`, and `run`. 
 - `timeout`:  API リクエストの HTTP タイムアウト（秒）。未指定の場合はデフォルトのタイムアウトが使われます。 
 - `api_key`:  認証に使用する APIキー。未指定の場合、現在の 環境または設定から APIキー が使用されます。 


---

### <kbd>property</kbd> Api.api_key

W&B の APIキー を返します。 

---

### <kbd>property</kbd> Api.client

クライアント オブジェクトを返します。 

---

### <kbd>property</kbd> Api.default_entity

デフォルトの W&B entity を返します。 

---

### <kbd>property</kbd> Api.user_agent

W&B Public ユーザーエージェントを返します。 

---

### <kbd>property</kbd> Api.viewer

viewer オブジェクトを返します。 



**発生しうる例外:**
 
 - `ValueError`:  W&B から viewer データを取得できない場合。 
 - `requests.RequestException`:  GraphQL リクエストの実行中にエラーが発生した場合。 



---

### <kbd>method</kbd> `Api.artifact`

```python
artifact(name: str, type: Optional[str] = None)
```

単一のアーティファクトを返します。 



**引数:**
 
 - `name`:  アーティファクトの名前。アーティファクトの名前はファイルパスに似ており、少なくとも アーティファクトがログされた project 名、アーティファクト名、アーティファクトの バージョンまたはエイリアス から構成されます。先頭に entity をスラッシュ区切りで付けることもできます。name に entity を指定しない場合は、Run または API の設定で指定された entity が使用されます。 
 - `type`:  取得するアーティファクトのタイプ。 



**戻り値:**
 `Artifact` オブジェクト。 



**発生しうる例外:**
 
 - `ValueError`:  アーティファクト名が指定されていない場合。 
 - `ValueError`:  アーティファクトタイプが指定されているが、取得したアーティファクトのタイプと一致しない場合。 



**例:**
 以下のコードスニペット中の "entity"、"project"、"artifact"、"version"、"alias" は、それぞれあなたの W&B entity、アーティファクトが存在する project 名、アーティファクト名、アーティファクトのバージョンのプレースホルダーです。 

```python
import wandb

# project、アーティファクト名、アーティファクトのエイリアスを指定
wandb.Api().artifact(name="project/artifact:alias")

# project、アーティファクト名、特定のアーティファクトバージョンを指定
wandb.Api().artifact(name="project/artifact:version")

# entity、project、アーティファクト名、アーティファクトのエイリアスを指定
wandb.Api().artifact(name="entity/project/artifact:alias")

# entity、project、アーティファクト名、特定のアーティファクトバージョンを指定
wandb.Api().artifact(name="entity/project/artifact:version")
``` 



**Note:**

> このメソッドは外部からの使用のみを意図しています。wandb リポジトリーのコード内で `api.artifact()` を呼び出さないでください。 

---

### <kbd>method</kbd> `Api.artifact_collection`

```python
artifact_collection(type_name: str, name: str) → public.ArtifactCollection
```

タイプで 1 つのアーティファクトコレクションを返します。 

返された `ArtifactCollection` オブジェクトを使うと、そのコレクション内の特定のアーティファクトに関する情報取得などができます。 



**引数:**
 
 - `type_name`:  取得するアーティファクトコレクションのタイプ。 
 - `name`:  アーティファクトコレクション名。先頭に アーティファクトをログした entity をスラッシュ区切りで付けることもできます。 



**戻り値:**
 `ArtifactCollection` オブジェクト。 



**例:**
 以下のコードスニペット中の "type"、"entity"、"project"、"artifact_name" は、コレクションタイプ、あなたの W&B entity、アーティファクトがある project 名、アーティファクト名のプレースホルダーです。 

```python
import wandb

collections = wandb.Api().artifact_collection(
    type_name="type", name="entity/project/artifact_name"
)

# コレクションの最初のアーティファクトを取得
artifact_example = collections.artifacts()[0]

# 指定したルートディレクトリーにアーティファクトの内容をダウンロード
artifact_example.download()
``` 

---

### <kbd>method</kbd> `Api.artifact_collection_exists`

```python
artifact_collection_exists(name: str, type: str) → bool
```

指定した project と entity 内にアーティファクトコレクションが存在するかどうか。 



**引数:**
 
 - `name`:  アーティファクトコレクション名。先頭に アーティファクトをログした entity をスラッシュ区切りで付けることもできます。entity または project が指定されていない場合、存在すれば override パラメータからコレクションを推測します。そうでなければ entity はユーザー設定から取得され、project は "uncategorized" がデフォルトになります。 
 - `type`:  アーティファクトコレクションのタイプ。 



**戻り値:**
 コレクションが存在すれば True、存在しなければ False。 



**例:**
 以下のコードスニペット中の "type" と "collection_name" は、それぞれアーティファクトコレクションのタイプとコレクション名を指します。 

```python
import wandb

wandb.Api.artifact_collection_exists(type="type", name="collection_name")
``` 

---

### <kbd>method</kbd> `Api.artifact_collections`

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: int = 50
) → public.ArtifactCollections
```

条件に合うアーティファクトコレクションの集合を返します。 



**引数:**
 
 - `project_name`:  フィルタ対象の project 名。 
 - `type_name`:  フィルタ対象のアーティファクトタイプ名。 
 - `per_page`:  クエリのページネーションにおけるページサイズを設定します。None の場合はデフォルトサイズ。通常これを変更する必要はありません。 



**戻り値:**
 反復可能な `ArtifactCollections` オブジェクト。 

---

### <kbd>method</kbd> `Api.artifact_exists`

```python
artifact_exists(name: str, type: Optional[str] = None) → bool
```

指定した project と entity 内に、アーティファクトのバージョンが存在するかどうか。 



**引数:**
 
 - `name`:  アーティファクトの名前。先頭にアーティファクトの entity と project を付けてください。末尾にコロンで区切ってアーティファクトのバージョンまたはエイリアスを付けます。entity または project が指定されていない場合、W&B は override パラメータがあればそれを使用します。なければ entity はユーザー設定から取得され、project は "Uncategorized" に設定されます。 
 - `type`:  アーティファクトのタイプ。 



**戻り値:**
 バージョンが存在すれば True、存在しなければ False。 



**例:**
 以下のコードスニペット中の "entity"、"project"、"artifact"、"version"、"alias" は、それぞれあなたの W&B entity、アーティファクトがある project 名、アーティファクト名、アーティファクトのバージョンのプレースホルダーです。 

```python
import wandb

wandb.Api().artifact_exists("entity/project/artifact:version")
wandb.Api().artifact_exists("entity/project/artifact:alias")
``` 

---

### <kbd>method</kbd> `Api.artifact_type`

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) → public.ArtifactType
```

一致する `ArtifactType` を返します。 



**引数:**
 
 - `type_name`:  取得するアーティファクトタイプ名。 
 - `project`:  指定した場合、フィルタ対象の project 名またはパス。 



**戻り値:**
 `ArtifactType` オブジェクト。 

---

### <kbd>method</kbd> `Api.artifact_types`

```python
artifact_types(project: Optional[str] = None) → public.ArtifactTypes
```

条件に合うアーティファクトタイプの集合を返します。 



**引数:**
 
 - `project`:  フィルタ対象の project 名またはパス。 



**戻り値:**
 反復可能な `ArtifactTypes` オブジェクト。 

---

### <kbd>method</kbd> `Api.artifact_versions`

```python
artifact_versions(type_name, name, per_page=50)
```

非推奨。代わりに `Api.artifacts(type_name, name)` メソッドを使用してください。 

---

### <kbd>method</kbd> `Api.artifacts`

```python
artifacts(
    type_name: str,
    name: str,
    per_page: int = 50,
    tags: Optional[List[str]] = None
) → public.Artifacts
```

`Artifacts` コレクションを返します。 



**引数:**
 type_name: 取得するアーティファクトのタイプ。 name: アーティファクトのコレクション名。先頭に アーティファクトをログした entity をスラッシュ区切りで付けることもできます。 per_page: クエリのページネーションにおけるページサイズを設定します。`None` の場合はデフォルトサイズ。通常これを変更する必要はありません。 tags: 指定したすべてのタグを持つアーティファクトのみ返します。 



**戻り値:**
  反復可能な `Artifacts` オブジェクト。 



**例:**
 以下のコードスニペット中の "type"、"entity"、"project"、"artifact_name" は、それぞれアーティファクトタイプ、W&B entity、アーティファクトがログされた project 名、アーティファクト名のプレースホルダーです。 

```python
import wandb

wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
``` 

---

### <kbd>method</kbd> `Api.automation`

```python
automation(name: str, entity: Optional[str] = None) → Automation
```

指定したパラメータに一致するただ 1 つの オートメーション を返します。 



**引数:**
 
 - `name`:  取得するオートメーションの名前。 
 - `entity`:  そのオートメーションを取得する対象の entity。 



**発生しうる例外:**
 
 - `ValueError`:  検索条件に一致する オートメーション が 0 件または複数件の場合。 



**例:**
 "my-automation" という既存のオートメーションを取得: 

```python
import wandb

api = wandb.Api()
automation = api.automation(name="my-automation")
``` 

entity "my-team" にある "other-automation" という既存のオートメーションを取得: 

```python
automation = api.automation(name="other-automation", entity="my-team")
``` 

---

### <kbd>method</kbd> `Api.automations`

```python
automations(
    entity: Optional[str] = None,
    name: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('Automation')]
```

指定したパラメータに一致するすべての オートメーション に対するイテレーターを返します。 

パラメータを指定しない場合、返されるイテレーターには ユーザー がアクセスできるすべての オートメーション が含まれます。 



**引数:**
 
 - `entity`:  オートメーションを取得する対象の entity。 
 - `name`:  取得するオートメーションの名前。 
 - `per_page`:  1 ページあたりに取得するオートメーション数。デフォルトは 50。通常これを変更する必要はありません。 



**戻り値:**
 オートメーションのリスト。 



**例:**
 entity "my-team" の既存のオートメーションをすべて取得: 

```python
import wandb

api = wandb.Api()
automations = api.automations(entity="my-team")
``` 

---

### <kbd>method</kbd> `Api.create_automation`

```python
create_automation(
    obj: 'NewAutomation',
    fetch_existing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

新しい オートメーション を作成します。 



**引数:**
  obj:  作成するオートメーション。  fetch_existing:  True の場合、競合するオートメーションがすでに存在すれば エラーを出さずに既存のオートメーションを取得することを試みます。  **kwargs:  作成前にオートメーションへ追加で設定する値。 指定した場合、すでに設定済みの値よりも優先されます: 
        - `name`: オートメーションの名前 
        - `description`: オートメーションの説明 
        - `enabled`: オートメーションを有効にするかどうか 
        - `scope`: オートメーションのスコープ 
        - `event`: オートメーションをトリガーするイベント 
        - `action`: オートメーションでトリガーされるアクション 



**戻り値:**
  保存された Automation。 



**例:**
 特定の project 内の run が、指定したしきい値を超えるメトリクをログしたときに Slack 通知を送る "my-automation" という新しいオートメーションを作成: 

```python
import wandb
from wandb.automations import OnRunMetric, RunEvent, SendNotification

api = wandb.Api()

project = api.project("my-project", entity="my-team")

# チームの 1 つ目の Slack インテグレーションを使用
slack_hook = next(api.slack_integrations(entity="my-team"))

event = OnRunMetric(
     scope=project,
     filter=RunEvent.metric("custom-metric") > 10,
)
action = SendNotification.from_integration(slack_hook)

automation = api.create_automation(
     event >> action,
     name="my-automation",
     description="Send a Slack message whenever 'custom-metric' exceeds 10.",
)
``` 

---

### <kbd>method</kbd> `Api.create_custom_chart`

```python
create_custom_chart(
    entity: str,
    name: str,
    display_name: str,
    spec_type: Literal['vega2'],
    access: Literal['private', 'public'],
    spec: Union[str, dict]
) → str
```

カスタムチャートのプリセットを作成し、その ID を返します。 



**引数:**
 
 - `entity`:  チャートの所有者（ユーザーまたは team） 
 - `name`:  チャートプリセットの一意な識別子 
 - `display_name`:  UI に表示される人間が読みやすい名前 
 - `spec_type`:  仕様のタイプ。Vega-Lite v2 の仕様には "vega2" を指定します。 
 - `access`:  チャートの アクセス レベル: 
        - "private": チャートの作成者の entity のみアクセス可能 
        - "public": 公開アクセス可能 
 - `spec`:  Vega/Vega-Lite の仕様（辞書または JSON 文字列） 



**戻り値:**
 作成されたチャートプリセットの ID（"entity/name" の形式） 



**発生しうる例外:**
 
 - `wandb.Error`:  チャートの作成に失敗した場合 
 - `UnsupportedError`:  サーバーがカスタムチャートをサポートしていない場合 



**例:**
 ```python
    import wandb

    api = wandb.Api()

    # シンプルな棒グラフ仕様を定義
    vega_spec = {
         "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
         "mark": "bar",
         "data": {"name": "wandb"},
         "encoding": {
             "x": {"field": "${field:x}", "type": "ordinal"},
             "y": {"field": "${field:y}", "type": "quantitative"},
         },
    }

    # カスタムチャートを作成
    chart_id = api.create_custom_chart(
         entity="my-team",
         name="my-bar-chart",
         display_name="My Custom Bar Chart",
         spec_type="vega2",
         access="private",
         spec=vega_spec,
    )

    # wandb.plot_table() と併用
    chart = wandb.plot_table(
         vega_spec_name=chart_id,
         data_table=my_table,
         fields={"x": "category", "y": "value"},
    )
    ``` 

---

### <kbd>method</kbd> `Api.create_project`

```python
create_project(name: str, entity: str) → None
```

新しい project を作成します。 



**引数:**
 
 - `name`:  新しい project の名前。 
 - `entity`:  新しい project の entity。 

---

### <kbd>method</kbd> `Api.create_registry`

```python
create_registry(
    name: str,
    visibility: Literal['organization', 'restricted'],
    organization: Optional[str] = None,
    description: Optional[str] = None,
    artifact_types: Optional[List[str]] = None
) → Registry
```

新しいレジストリを作成します。 



**引数:**
 
 - `name`:  レジストリ名。組織内で一意である必要があります。 
 - `visibility`:  レジストリの “公開範囲”。 
 - `organization`:  組織のメンバーであれば誰でもこのレジストリを閲覧できます。後で UI の設定からロールを編集できます。 
 - `restricted`:  UI から招待されたメンバーのみがこのレジストリに アクセス できます。公開共有は無効です。 
 - `organization`:  レジストリの所属組織。設定で organization が未設定の場合、entity が 1 つの organization にのみ属していればそこから取得されます。 
 - `description`:  レジストリの説明。 
 - `artifact_types`:  レジストリで受け付けるアーティファクトタイプ。タイプは 
 - `more than 128 characters and do not include characters `/` or ``: `. If not specified, all types are accepted. Allowed types added to the registry cannot be removed later. 



**戻り値:**
 レジストリ オブジェクト。 



**例:**
 ```python
import wandb

api = wandb.Api()
registry = api.create_registry(
    name="my-registry",
    visibility="restricted",
    organization="my-org",
    description="This is a test registry",
    artifact_types=["model"],
)
``` 

---

### <kbd>method</kbd> `Api.create_run`

```python
create_run(
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) → public.Run
```

新しい run を作成します。 



**引数:**
 
 - `run_id`:  run に割り当てる ID。未指定の場合、W&B がランダムな ID を作成します。 
 - `project`:  run をログする project。未指定の場合、"Uncategorized" という project にログされます。 
 - `entity`:  project の所有 entity。未指定の場合、デフォルトの entity にログされます。 



**戻り値:**
 新しく作成された `Run`。 

---

### <kbd>method</kbd> `Api.create_run_queue`

```python
create_run_queue(
    name: str,
    type: 'public.RunQueueResourceType',
    entity: Optional[str] = None,
    prioritization_mode: Optional[ForwardRef('public.RunQueuePrioritizationMode')] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) → public.RunQueue
```

W&B Launch に新しい run キューを作成します。 



**引数:**
 
 - `name`:  作成するキューの名前 
 - `type`:  キューで使用するリソースタイプ。"local-container"、"local-process"、"kubernetes"、"sagemaker"、"gcp-vertex" のいずれか。 
 - `entity`:  キューを作成する entity 名。`None` の場合は、設定済みまたはデフォルトの entity を使用します。 
 - `prioritization_mode`:  優先度付けのバージョン。"V0" か `None`。 
 - `config`:  キューで使用するデフォルトのリソース設定。テンプレート変数の指定にはハンドルバー（例: `{{var}}`）を使用します。 
 - `template_variables`:  config と併せて使うテンプレート変数スキーマの辞書。 



**戻り値:**
 新しく作成された `RunQueue`。 



**発生しうる例外:**
 パラメータが無効な場合は `ValueError`、wandb API エラーが発生した場合は `wandb.Error`。 

---

### <kbd>method</kbd> `Api.create_team`

```python
create_team(team: str, admin_username: Optional[str] = None) → public.Team
```

新しい team を作成します。 



**引数:**
 
 - `team`:  team の名前 
 - `admin_username`:  team の管理者ユーザー名。デフォルトは現在のユーザー。 



**戻り値:**
 `Team` オブジェクト。 

---

### <kbd>method</kbd> `Api.create_user`

```python
create_user(email: str, admin: Optional[bool] = False)
```

新しいユーザーを作成します。 



**引数:**
 
 - `email`:  ユーザーのメールアドレス。 
 - `admin`:  ユーザーをグローバルなインスタンス管理者に設定。 



**戻り値:**
 `User` オブジェクト。 

---

### <kbd>method</kbd> `Api.delete_automation`

```python
delete_automation(obj: Union[ForwardRef('Automation'), str]) → Literal[True]
```

オートメーションを削除します。 



**引数:**
 
 - `obj`:  削除するオートメーション、またはその ID。 



**戻り値:**
 オートメーションが正常に削除された場合は True。 

---

### <kbd>method</kbd> `Api.flush`

```python
flush()
```

ローカルキャッシュをフラッシュします。 

api オブジェクトは run のローカルキャッシュを保持します。そのため、スクリプトの実行中に run の状態が変わる可能性がある場合、最新の値を取得するには `api.flush()` でローカルキャッシュをクリアしてください。 

---

### <kbd>method</kbd> `Api.from_path`

```python
from_path(path: str)
```

パスから run、sweep、project、またはレポートを返します。 



**引数:**
 
 - `path`:  project、run、sweep、またはレポートへのパス 



**戻り値:**
 `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス。 



**発生しうる例外:**
 パスが無効、または対象オブジェクトが存在しない場合は `wandb.Error`。 



**例:**
 以下のコードスニペット中の "project"、"team"、"run_id"、"sweep_id"、"report_name" は、それぞれ project、team、run ID、sweep ID、特定のレポート名のプレースホルダーです。 

```python
import wandb

api = wandb.Api()

project = api.from_path("project")
team_project = api.from_path("team/project")
run = api.from_path("team/project/runs/run_id")
sweep = api.from_path("team/project/sweeps/sweep_id")
report = api.from_path("team/project/reports/report_name")
``` 

---

### <kbd>method</kbd> `Api.integrations`

```python
integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('Integration')]
```

指定した entity のすべてのインテグレーションに対するイテレーターを返します。 



**引数:**
 
 - `entity`:  インテグレーションを取得する対象の entity（例: team 名）。未指定の場合はユーザーのデフォルト entity が使用されます。 
 - `per_page`:  1 ページあたりに取得するインテグレーション数。デフォルトは 50。通常これを変更する必要はありません。 



**返すもの:**
 
 - `Iterator[SlackIntegration | WebhookIntegration]`:  サポートされているインテグレーションのイテレーター。 

---

### <kbd>method</kbd> `Api.job`

```python
job(name: Optional[str], path: Optional[str] = None) → public.Job
```

`Job` オブジェクトを返します。 



**引数:**
 
 - `name`:  ジョブの名前。 
 - `path`:  ジョブのアーティファクトをダウンロードするルートパス。 



**戻り値:**
 `Job` オブジェクト。 

---

### <kbd>method</kbd> `Api.list_jobs`

```python
list_jobs(entity: str, project: str) → List[Dict[str, Any]]
```

指定した entity と project に対して、存在すればジョブのリストを返します。 



**引数:**
 
 - `entity`:  一覧表示するジョブの entity。 
 - `project`:  一覧表示するジョブの project。 



**戻り値:**
 条件に一致するジョブのリスト。 

---

### <kbd>method</kbd> `Api.project`

```python
project(name: str, entity: Optional[str] = None) → public.Project
```

指定した名前（および entity 指定時はその entity）の `Project` を返します。 



**引数:**
 
 - `name`:  project 名。 
 - `entity`:  リクエスト対象の entity 名。None の場合は `Api` に渡されたデフォルトの entity を使用します。デフォルト entity がなければ `ValueError` を送出します。 



**戻り値:**
 `Project` オブジェクト。 

---

### <kbd>method</kbd> `Api.projects`

```python
projects(entity: Optional[str] = None, per_page: int = 200) → public.Projects
```

指定した entity の projects を取得します。 



**引数:**
 
 - `entity`:  リクエスト対象の entity 名。None の場合は `Api` に渡されたデフォルトの entity を使用します。デフォルト entity がなければ `ValueError` を送出します。 
 - `per_page`:  クエリのページネーションにおけるページサイズを設定します。`None` の場合はデフォルトサイズ。通常これを変更する必要はありません。 



**戻り値:**
 `Project` オブジェクトの反復可能なコレクションである `Projects` オブジェクト。 

---

### <kbd>method</kbd> `Api.queued_run`

```python
queued_run(
    entity: str,
    project: str,
    queue_name: str,
    run_queue_item_id: str,
    project_queue=None,
    priority=None
)
```

指定したパスに基づいて 1 つのキューイングされた run を返します。 

`entity/project/queue_id/run_queue_item_id` 形式のパスをパースします。 

---

### <kbd>method</kbd> `Api.registries`

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) → Registries
```

`Registry` オブジェクトの遅延イテレーターを返します。 

このイテレーターを使って、組織のレジストリ、コレクション、またはアーティファクトのバージョンを検索・フィルタできます。 



**引数:**
 
 - `organization`:  （str, 省略可）取得するレジストリの組織。未指定の場合はユーザー設定の organization が使用されます。 
 - `filter`:  （dict, 省略可）遅延レジストリ イテレーター内の各オブジェクトに適用する MongoDB 形式のフィルタ。 レジストリでフィルタ可能なフィールド: `name`, `description`, `created_at`, `updated_at`。 コレクションでフィルタ可能なフィールド: `name`, `tag`, `description`, `created_at`, `updated_at`。 バージョンでフィルタ可能なフィールド: `tag`, `alias`, `created_at`, `updated_at`, `metadata`。 



**戻り値:**
 `Registry` オブジェクトの遅延イテレーター。 



**例:**
 名前に "model" を含むすべてのレジストリを検索 

```python
import wandb

api = wandb.Api()  # entity が複数の org に属する場合は org を指定してください
api.registries(filter={"name": {"$regex": "model"}})
``` 

名前が "my_collection"、タグが "my_tag" のコレクションを含むレジストリを検索 

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
``` 

コレクション名に "my_collection" を含み、エイリアス "best" を持つバージョンを含むアーティファクトバージョンをすべて検索 

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
``` 

"model" を含み、タグが "prod" またはエイリアスが "best" のアーティファクトバージョンをすべて検索 

```python
api.registries(filter={"name": {"$regex": "model"}}).versions(
    filter={"$or": [{"tag": "prod"}, {"alias": "best"}]}
)
``` 

---

### <kbd>method</kbd> `Api.registry`

```python
registry(name: str, organization: Optional[str] = None) → Registry
```

レジストリ名を指定してレジストリを返します。 



**引数:**
 
 - `name`:  レジストリの名前。`wandb-registry-` プレフィックスは含みません。 
 - `organization`:  レジストリの所属組織。設定で organization が未設定の場合、entity が 1 つの organization にのみ属していればそこから取得されます。 



**戻り値:**
 レジストリ オブジェクト。 



**例:**
 レジストリの取得と更新 

```python
import wandb

api = wandb.Api()
registry = api.registry(name="my-registry", organization="my-org")
registry.description = "This is an updated description"
registry.save()
``` 

---

### <kbd>method</kbd> `Api.reports`

```python
reports(
    path: str = '',
    name: Optional[str] = None,
    per_page: int = 50
) → public.Reports
```

指定した project パスのレポートを取得します。 

注意: `wandb.Api.reports()` API はベータ版であり、今後のリリースで変更される可能性があります。 



**引数:**
 
 - `path`:  レポートが存在する project へのパス。先頭に、その project を作成した entity をスラッシュ区切りで付けます。 
 - `name`:  取得するレポート名。 
 - `per_page`:  クエリのページネーションにおけるページサイズを設定します。`None` の場合はデフォルトサイズ。通常これを変更する必要はありません。 



**戻り値:**
 `BetaReport` オブジェクトの反復可能なコレクションである `Reports` オブジェクト。 



**例:**
 ```python
import wandb

wandb.Api.reports("entity/project")
``` 

---

### <kbd>method</kbd> `Api.run`

```python
run(path='')
```

`entity/project/run_id` 形式のパスをパースして、単一の run を返します。 



**引数:**
 
 - `path`:  `entity/project/run_id` 形式の run へのパス。`api.entity` が設定されている場合は `project/run_id`、`api.project` が設定されている場合は run_id のみでも可。 



**戻り値:**
 `Run` オブジェクト。 

---

### <kbd>method</kbd> `Api.run_queue`

```python
run_queue(entity: str, name: str)
```

指定した entity の、名前付き `RunQueue` を返します。 

run キューの作成方法は `Api.create_run_queue` を参照してください。 

---

### <kbd>method</kbd> `Api.runs`

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = '+created_at',
    per_page: int = 50,
    include_sweeps: bool = True
)
```

`Run` オブジェクトを遅延的に反復処理する `Runs` オブジェクトを返します。 

フィルタできるフィールド: 
- `createdAt`: run が作成されたタイムスタンプ（ISO 8601 形式。例: "2023-01-01T12:00:00Z"） 
- `displayName`: run の人間が読める表示名（例: "eager-fox-1"） 
- `duration`: run の総実行時間（秒） 
- `group`: 関連する run をまとめるためのグループ名 
- `host`: run が実行されたホスト名 
- `jobType`: run のジョブタイプまたは目的 
- `name`: run の一意識別子（例: "a1b2cdef"） 
- `state`: run の現在の状態 
- `tags`: run に関連付けられたタグ 
- `username`: run を開始したユーザーのユーザー名 

run の config や summary metrics の項目でもフィルタできます。例: `config.experiment_name`、`summary_metrics.loss` など。 

より複雑なフィルタには MongoDB のクエリ演算子を使用できます。詳細: https://docs.mongodb.com/manual/reference/operator/query サポートされる演算子: 
- `$and` 
- `$or` 
- `$nor` 
- `$eq` 
- `$ne` 
- `$gt` 
- `$gte` 
- `$lt` 
- `$lte` 
- `$in` 
- `$nin` 
- `$exists` 
- `$regex` 







**引数:**
 
 - `path`:  （str）project へのパス。"entity/project" の形式 
 - `filters`:  （dict）MongoDB のクエリ言語で、特定の run を検索します。config.key、summary_metrics.key、state、entity、createdAt などの run プロパティでフィルタ可能です。 
 - `For example`:  `{"config.experiment_name": "foo"}` は、config の experiment_name が "foo" に設定された run を見つけます 
 - `order`:  （str）`created_at`、`heartbeat_at`、`config.*.value`、`summary_metrics.*` を指定可能。先頭に + を付けると昇順（デフォルト）、- を付けると降順。デフォルト順序は run.created_at の古い順から新しい順。 
 - `per_page`:  （int）クエリのページネーションにおけるページサイズ。 
 - `include_sweeps`:  （bool）結果に sweep の run を含めるかどうか。 



**戻り値:**
 `Run` オブジェクトの反復可能なコレクションである `Runs` オブジェクト。 



**例:**
 ```python
# config.experiment_name が "foo" に設定された run を project から検索
api.runs(path="my_entity/project", filters={"config.experiment_name": "foo"})
``` 

```python
# config.experiment_name が "foo" または "bar" の run を project から検索
api.runs(
    path="my_entity/project",
    filters={
         "$or": [
             {"config.experiment_name": "foo"},
             {"config.experiment_name": "bar"},
         ]
    },
)
``` 

```python
# config.experiment_name が正規表現にマッチする run を project から検索
# （アンカーは未サポート）
api.runs(
    path="my_entity/project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
``` 

```python
# run 名が正規表現にマッチする run を project から検索
# （アンカーは未サポート）
api.runs(
    path="my_entity/project", filters={"display_name": {"$regex": "^foo.*"}}
)
``` 

```python
# 損失の昇順でソートして run を検索
api.runs(path="my_entity/project", order="+summary_metrics.loss")
``` 

---

### <kbd>method</kbd> `Api.slack_integrations`

```python
slack_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('SlackIntegration')]
```

指定した entity の Slack インテグレーションのイテレーターを返します。 



**引数:**
 
 - `entity`:  インテグレーションを取得する対象の entity（例: team 名）。未指定の場合はユーザーのデフォルト entity が使用されます。 
 - `per_page`:  1 ページあたりに取得するインテグレーション数。デフォルトは 50。通常これを変更する必要はありません。 



**返すもの:**
 
 - `Iterator[SlackIntegration]`:  Slack インテグレーションのイテレーター。 



**例:**
 team "my-team" に登録されている Slack インテグレーションをすべて取得: 

```python
import wandb

api = wandb.Api()
slack_integrations = api.slack_integrations(entity="my-team")
``` 

チャンネル名が "team-alerts-" で始まるものだけに絞り込み: 

```python
slack_integrations = api.slack_integrations(entity="my-team")
team_alert_integrations = [
    ig
    for ig in slack_integrations
    if ig.channel_name.startswith("team-alerts-")
]
``` 

---

### <kbd>method</kbd> `Api.sweep`

```python
sweep(path='')
```

`entity/project/sweep_id` 形式のパスをパースして sweep を返します。 



**引数:**
 
 - `path`:  entity/project/sweep_id 形式の sweep へのパス。`api.entity` が設定されている場合は project/sweep_id、`api.project` が設定されている場合は sweep_id のみでも可。 



**戻り値:**
 `Sweep` オブジェクト。 

---

### <kbd>method</kbd> `Api.sync_tensorboard`

```python
sync_tensorboard(root_dir, run_id=None, project=None, entity=None)
```

tfevent ファイルを含むローカルディレクトリーを wandb と同期します。 

---

### <kbd>method</kbd> `Api.team`

```python
team(team: str) → public.Team
```

指定した名前に一致する `Team` を返します。 



**引数:**
 
 - `team`:  team の名前。 



**戻り値:**
 `Team` オブジェクト。 

---

### <kbd>method</kbd> `Api.update_automation`

```python
update_automation(
    obj: 'Automation',
    create_missing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

既存のオートメーションを更新します。 



**引数:**
 
 - `obj`:  更新するオートメーション。既存のオートメーションである必要があります。 create_missing (bool):  True の場合、オートメーションが存在しなければ作成します。 **kwargs:  更新前にオートメーションへ追加で設定する値。 指定した場合、すでに設定済みの値よりも優先されます: 
        - `name`: オートメーションの名前 
        - `description`: オートメーションの説明 
        - `enabled`: オートメーションを有効にするかどうか 
        - `scope`: オートメーションのスコープ 
        - `event`: オートメーションをトリガーするイベント 
        - `action`: オートメーションでトリガーされるアクション 



**戻り値:**
 更新されたオートメーション。 



**例:**
 既存のオートメーション（"my-automation"）を無効化し、説明を編集: 

```python
import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")
automation.enabled = False
automation.description = "Kept for reference, but no longer used."

updated_automation = api.update_automation(automation)
``` 

または 

```python
import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")

updated_automation = api.update_automation(
    automation,
    enabled=False,
    description="Kept for reference, but no longer used.",
)
``` 

---

### <kbd>method</kbd> `Api.upsert_run_queue`

```python
upsert_run_queue(
    name: str,
    resource_config: dict,
    resource_type: 'public.RunQueueResourceType',
    entity: Optional[str] = None,
    template_variables: Optional[dict] = None,
    external_links: Optional[dict] = None,
    prioritization_mode: Optional[ForwardRef('public.RunQueuePrioritizationMode')] = None
)
```

W&B Launch に run キューをアップサートします。 



**引数:**
 
 - `name`:  作成するキューの名前 
 - `entity`:  省略可。キューを作成する entity 名。`None` の場合は、設定済みまたはデフォルトの entity を使用します。 
 - `resource_config`:  省略可。キューで使用するデフォルトのリソース設定。テンプレート変数の指定にはハンドルバー（例: `{{var}}`）を使用します。 
 - `resource_type`:  キューで使用するリソースタイプ。"local-container"、"local-process"、"kubernetes"、"sagemaker"、"gcp-vertex" のいずれか。 
 - `template_variables`:  config と併せて使うテンプレート変数スキーマの辞書。 
 - `external_links`:  省略可。キューで使用する外部リンクの辞書。 
 - `prioritization_mode`:  省略可。優先度付けのバージョン。"V0" または None。 



**戻り値:**
 アップサートされた `RunQueue`。 



**発生しうる例外:**
 パラメータが無効な場合は ValueError、wandb API エラーが発生した場合は wandb.Error。 

---

### <kbd>method</kbd> `Api.user`

```python
user(username_or_email: str) → Optional[ForwardRef('public.User')]
```

ユーザー名またはメールアドレスからユーザーを返します。 

この関数はローカル管理者にのみ有効です。自分自身のユーザーオブジェクトを取得するには `api.viewer` を使用してください。 



**引数:**
 
 - `username_or_email`:  ユーザーのユーザー名またはメールアドレス。 



**戻り値:**
 `User` オブジェクト、またはユーザーが見つからない場合は None。 

---

### <kbd>method</kbd> `Api.users`

```python
users(username_or_email: str) → List[ForwardRef('public.User')]
```

部分一致のユーザー名またはメールアドレスのクエリから、すべてのユーザーを返します。 

この関数はローカル管理者にのみ有効です。自分自身のユーザーオブジェクトを取得するには `api.viewer` を使用してください。 



**引数:**
 
 - `username_or_email`:  目的のユーザーを見つけるためのプレフィックスまたはサフィックス。 



**戻り値:**
 `User` オブジェクトの配列。 

---

### <kbd>method</kbd> `Api.webhook_integrations`

```python
webhook_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('WebhookIntegration')]
```

指定した entity の Webhook インテグレーションのイテレーターを返します。 



**引数:**
 
 - `entity`:  インテグレーションを取得する対象の entity（例: team 名）。未指定の場合はユーザーのデフォルト entity が使用されます。 
 - `per_page`:  1 ページあたりに取得するインテグレーション数。デフォルトは 50。通常これを変更する必要はありません。 



**返すもの:**
 
 - `Iterator[WebhookIntegration]`:  Webhook インテグレーションのイテレーター。 



**例:**
 team "my-team" に登録されている Webhook インテグレーションをすべて取得: 

```python
import wandb

api = wandb.Api()
webhook_integrations = api.webhook_integrations(entity="my-team")
``` 

"https://my-fake-url.com" にリクエストを送る Webhook インテグレーションだけを抽出: 

```python
webhook_integrations = api.webhook_integrations(entity="my-team")
my_webhooks = [
    ig
    for ig in webhook_integrations
    if ig.url_endpoint.startswith("https://my-fake-url.com")
]
```