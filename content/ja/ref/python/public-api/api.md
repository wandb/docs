---
title: api
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-api
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/api.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API を使って、W&B に保存したデータをエクスポートまたは更新できます。

この API を使う前に、スクリプトからデータを記録しておく必要があります。詳細は [クイックスタート](https://docs.wandb.ai/quickstart) をご覧ください。

Public API の主な用途は以下の通りです。
 - 実験が完了した後にメタデータやメトリクスを更新する
 - 実験結果を dataframe でダウンロードして Jupyter ノートブックで分析する
 - `ready-to-deploy` タグが付いたモデルアーティファクトを確認する

Public API の詳細な使い方は[こちらのガイド](https://docs.wandb.com/guides/track/public-api-guide)をご確認ください。


## <kbd>class</kbd> `Api`
W&B サーバーへのクエリに使用します。



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
 
 - `overrides`:  `base_url` を指定して、デフォルト以外の W&B サーバー（例: `https://api.wandb.ai`）を使用したい場合にセットできます。また、`entity`、`project`、`run` のデフォルト値も指定できます。
 - `timeout`:  APIリクエストの HTTP タイムアウト（秒）。未指定の場合はデフォルト値が使われます。
 - `api_key`:  認証に使用する API キー。未指定の場合、現在の環境変数または設定から取得します。


---

### <kbd>property</kbd> Api.api_key

W&B API キーを返します。

---

### <kbd>property</kbd> Api.client

クライアントオブジェクトを返します。

---

### <kbd>property</kbd> Api.default_entity

デフォルトの W&B entity を返します。

---

### <kbd>property</kbd> Api.user_agent

W&B public user agent を返します。

---

### <kbd>property</kbd> Api.viewer

viewer オブジェクトを返します。



---

### <kbd>method</kbd> `Api.artifact`

```python
artifact(name: str, type: Optional[str] = None)
```

指定した artifact を 1 つ返します。



**引数:**
 
 - `name`:  アーティファクトの名前。artifact の名前は少なくとも、artifact が記録された project 名、artifact 名、artifact のバージョンまたはエイリアスで構成されるファイルパスに似ています。必要に応じて、artifact を記録した entity をプレフィックスとして `/` で区切って指定できます。entity を指定しない場合は Run または API 設定の entity が使用されます。
 - `type`:  取得したい artifact のタイプ。



**戻り値:**
 `Artifact` オブジェクト。



**例外:**
 
 - `ValueError`:  artifact 名が指定されていない場合
 - `ValueError`:  artifact タイプが指定されたが、取得した artifact のタイプと一致しない場合



**例:**
 以下のコードスニペット内の "entity", "project", "artifact", "version", "alias" はそれぞれ、ご自身の W&B entity、artifact が含まれるプロジェクト名、artifact 名、artifact のバージョンやエイリアスです。

```python
import wandb

# プロジェクト、artifact 名、artifact のエイリアスを指定
wandb.Api().artifact(name="project/artifact:alias")

# プロジェクト、artifact 名、特定の artifact バージョンを指定
wandb.Api().artifact(name="project/artifact:version")

# entity, プロジェクト, artifact 名, エイリアスを指定
wandb.Api().artifact(name="entity/project/artifact:alias")

# entity, プロジェクト, artifact 名, 特定の artifact バージョンを指定
wandb.Api().artifact(name="entity/project/artifact:version")
```

**注意:**

> このメソッドは外部利用向けです。wandb リポジトリのコード内で `api.artifact()` を呼び出さないでください。

---

### <kbd>method</kbd> `Api.artifact_collection`

```python
artifact_collection(type_name: str, name: str) → public.ArtifactCollection
```

指定したタイプの artifact collection を 1 つ返します。

返された `ArtifactCollection` オブジェクトから、コレクション内の特定の artifact などの情報を取得できます。



**引数:**
 
 - `type_name`:  取得したい artifact collection のタイプ
 - `name`:  artifact collection 名。entity をプリフィックスとして `/` で区切って追加可能



**戻り値:**
 `ArtifactCollection` オブジェクト



**例:**
 以下のコードスニペット内の "type", "entity", "project", "artifact_name" はそれぞれ collection type、W&B entity、artifact が含まれるプロジェクト名、そして artifact 名です。

```python
import wandb

collections = wandb.Api().artifact_collection(
    type_name="type", name="entity/project/artifact_name"
)

# コレクション内の最初の artifact を取得
artifact_example = collections.artifacts()[0]

# artifact の内容を指定したディレクトリにダウンロード
artifact_example.download()
```

---

### <kbd>method</kbd> `Api.artifact_collection_exists`

```python
artifact_collection_exists(name: str, type: str) → bool
```

指定した entity・project 内に artifact collection が存在するか確認します。



**引数:**
 
 - `name`:  artifact collection 名。entity をプリフィックスとして `/` で区切って追加可能。entity や project が未指定の場合、overrides のパラメータがあればそこから推測、なければユーザー設定から entity を、project には "uncategorized" を使用
 - `type`:  artifact collection のタイプ



**戻り値:**
 artifact collection が存在する場合は True、存在しない場合は False



**例:**
 以下のコードスニペット内の "type" および "collection_name" は artifact collection のタイプと名前です。

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

条件に合う artifact collection の一覧を返します。



**引数:**
 
 - `project_name`:  絞り込み対象のプロジェクト名
 - `type_name`:  絞り込み対象の artifact タイプ名
 - `per_page`:  ページネーション用ページサイズ。通常、変更する必要はありません。



**戻り値:**
 イテレータブルな `ArtifactCollections` オブジェクト

---

### <kbd>method</kbd> `Api.artifact_exists`

```python
artifact_exists(name: str, type: Optional[str] = None) → bool
```

指定プロジェクト・entity 内に artifact バージョンが存在するかどうかを判定します。



**引数:**
 
 - `name`:  artifact 名。entity と project をプレフィックスとして指定し、artifact のバージョンやエイリアスはコロン区切りで追加。entity や project が未指定の場合は override パラメータ、なければユーザー設定（プロジェクトは "Uncategorized"）が使われます。
 - `type`:  artifact のタイプ



**戻り値:**
 artifact バージョンが存在する場合は True、存在しない場合は False



**例:**
 以下のコードスニペット内の "entity", "project", "artifact", "version", "alias" はそれぞれ entity、プロジェクト名、artifact 名、artifact のバージョン、エイリアスです。

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

type 名とプロジェクト名（オプション）から対象の `ArtifactType` を返します。



**引数:**
 
 - `type_name`:  取得したい artifact_type の名前
 - `project`:  必要に応じ、名前またはパスでプロジェクトを指定



**戻り値:**
 `ArtifactType` オブジェクト

---

### <kbd>method</kbd> `Api.artifact_types`

```python
artifact_types(project: Optional[str] = None) → public.ArtifactTypes
```

条件に一致する artifact type 一覧を返します。



**引数:**
 
 - `project`:  名前またはパスでフィルターしたいプロジェクト



**戻り値:**
 イテレータブルな `ArtifactTypes` オブジェクト

---

### <kbd>method</kbd> `Api.artifact_versions`

```python
artifact_versions(type_name, name, per_page=50)
```

非推奨です。代わりに `Api.artifacts(type_name, name)` をご利用ください。

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
 type_name: 取得する artifacts のタイプ
 name: artifact コレクション名。entity をプレフィックスとして `/` 区切りで指定可能
 per_page: ページサイズ。None の場合はデフォルト値。普通は変更不要
 tags: 指定されたタグすべてを持つ artifact のみ返します



**戻り値:**
 イテレータブルな `Artifacts` オブジェクト



**例:**
 以下のコードスニペット内の "type", "entity", "project", "artifact_name" は artifact のタイプ、W&B エンティティ、artifact を記録したプロジェクト名、artifact 名です。

```python
import wandb

wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
```

---

### <kbd>method</kbd> `Api.automation`

```python
automation(name: str, entity: Optional[str] = None) → Automation
```

指定パラメータに一致する唯一の Automation を返します。



**引数:**
 
 - `name`:  取得したいオートメーション名
 - `entity`:  取得対象となる entity



**例外:**
 
 - `ValueError`:  検索条件に合致する Automation が 0 件または複数の場合



**例:**
 "my-automation" という既存オートメーションを取得

```python
import wandb

api = wandb.Api()
automation = api.automation(name="my-automation")
```

entity "my-team" から "other-automation" を取得

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

指定した条件に合ったすべての Automation をイテレータで返します。

パラメータを指定しない場合、ユーザーがアクセス可能なすべての Automation を取得します。



**引数:**
 
 - `entity`:  取得対象の entity
 - `name`:  オートメーション名
 - `per_page`:  1ページあたりの件数（デフォルト50。通常は変更不要）



**戻り値:**
 オートメーションのリスト



**例:**
 entity "my-team" のすべてのオートメーションを取得

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

新しいオートメーションを作成します。



**引数:**
  obj:  作成するオートメーション
  fetch_existing:  True の場合、既存の名前重複オートメーションがあれば取得（エラーにしない）
  **kwargs:  オートメーション作成前に割り当てる追加値（指定時、すでにセット済みの値より優先）
        - `name`: オートメーション名
        - `description`: オートメーションの説明
        - `enabled`: 有効かどうか
        - `scope`: 適用範囲
        - `event`: トリガーとなるイベント
        - `action`: 実行されるアクション



**戻り値:**
 作成した Automation



**例:**
 "my-automation" という名前の新規オートメーションを作成し、特定プロジェクト内でカスタムメトリクスが閾値を超えた時 Slack 通知を送る例

```python
import wandb
from wandb.automations import OnRunMetric, RunEvent, SendNotification

api = wandb.Api()

project = api.project("my-project", entity="my-team")

# チームの 1 つ目の Slack インテグレーションを利用
slack_hook = next(api.slack_integrations(entity="my-team"))

event = OnRunMetric(
     scope=project,
     filter=RunEvent.metric("custom-metric") > 10,
)
action = SendNotification.from_integration(slack_hook)

automation = api.create_automation(
     event >> action,
     name="my-automation",
     description="custom-metric が 10 を超えたら Slack に通知します。",
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

カスタムチャートのプリセットを作成し、その id を返します。



**引数:**
 
 - `entity`:  チャートの所有者となる entity（ユーザーやチーム）
 - `name`:  チャートプリセットの一意な identifier
 - `display_name`:  UI 上で表示するわかりやすい名前
 - `spec_type`:  スペックタイプ。"vega2"（Vega-Lite v2仕様）を指定
 - `access`:  チャートのアクセスレベル
        - "private": 作成 entity のみアクセス可能
        - "public": 誰でもアクセス可能
 - `spec`:  Vega/Vega-Lite 仕様（辞書または JSON 文字列）



**戻り値:**
 作成したチャートプリセットの ID（"entity/name" の形式）



**例外:**
 
 - `wandb.Error`:  チャート作成失敗時
 - `UnsupportedError`:  サーバーがカスタムチャート未対応の場合



**例:**
 ```python
    import wandb

    api = wandb.Api()

    # シンプルなバーグラフ仕様
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

    # wandb.plot_table() で利用
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

新しいプロジェクトを作成します。



**引数:**
 
 - `name`:  新規プロジェクト名
 - `entity`:  新規プロジェクトの entity

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

新しい registry を作成します。



**引数:**
 
 - `name`:  registry の名前（組織内で一意である必要があります）
 - `visibility`:  registry の公開範囲
 - `organization`:  組織内にいる人は誰でも閲覧可能。UI の設定画面からロールの編集も可。
 - `restricted`:  招待されたメンバーのみが UI からアクセス可能。公開共有は不可。
 - `organization`:  この registry の組織名。設定で指定しなければ、entity が1つの組織のみに属している場合は entity から取得
 - `description`:  registry の説明
 - `artifact_types`:  registry で受け入れ可能な artifact type。128 文字まで。「/」や「:」を含まないこと。指定しない場合は全タイプ受け入れ。追加した type は後から削除できません。



**戻り値:**
 registry オブジェクト



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
 
 - `run_id`:  この run に割り当てるID。未指定の場合はランダムIDを生成
 - `project`:  run を記録するプロジェクト名。未指定の場合は "Uncategorized" プロジェクトに記録
 - `entity`:  プロジェクト所有 entity。未指定の場合はデフォルト entity に記録



**戻り値:**
 作成した `Run`

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

W&B Launch で新しい run queue を作成します。



**引数:**
 
 - `name`:  作成するキュー名
 - `type`:  使用するリソースタイプ。"local-container"、"local-process"、"kubernetes"、"sagemaker"、"gcp-vertex" のいずれか
 - `entity`:  キューを作成する entity 名。None の場合は設定済みまたはデフォルト entity
 - `prioritization_mode`:  優先順位バージョン。"V0" または None
 - `config`:  デフォルトのリソースコンフィグ。テンプレート変数にはハンドルバー記法（例: `{{var}}`）を利用
 - `template_variables`:  config で使うテンプレート変数スキーマの辞書



**戻り値:**
 作成した `RunQueue`



**例外:**
 `ValueError`：無効なパラメータ時
 `wandb.Error`：wandb API エラー時

---

### <kbd>method</kbd> `Api.create_team`

```python
create_team(team: str, admin_username: Optional[str] = None) → public.Team
```

新しい team を作成します。



**引数:**
 
 - `team`:  チーム名
 - `admin_username`:  チーム管理者ユーザー名（省略時は現在のユーザー）



**戻り値:**
 `Team` オブジェクト

---

### <kbd>method</kbd> `Api.create_user`

```python
create_user(email: str, admin: Optional[bool] = False)
```

新しいユーザーを作成します。



**引数:**
 
 - `email`:  作成するユーザーのメールアドレス
 - `admin`:  グローバル管理者として作成する場合は True



**戻り値:**
 `User` オブジェクト

---

### <kbd>method</kbd> `Api.delete_automation`

```python
delete_automation(obj: Union[ForwardRef('Automation'), str]) → Literal[True]
```

オートメーションを削除します。



**引数:**
 
 - `obj`:  削除したいオートメーション、またはそのID



**戻り値:**
 削除に成功した場合は True

---

### <kbd>method</kbd> `Api.flush`

```python
flush()
```

ローカルキャッシュをフラッシュします。

api オブジェクトは run のローカルキャッシュを保持しています。スクリプトの実行中に run の状態が変わる可能性がある場合、`api.flush()` でキャッシュをクリアして最新値を取得してください。

---

### <kbd>method</kbd> `Api.from_path`

```python
from_path(path: str)
```

パスから run、sweep、project、report のいずれかを返します。



**引数:**
 
 - `path`:  project、run、sweep、report へのパス



**戻り値:**
 `Project`, `Run`, `Sweep` または `BetaReport` インスタンス



**例外:**
 `wandb.Error`：パスが不正、または該当オブジェクトが存在しない場合



**例:**
 以下のスニペット内の "project", "team", "run_id", "sweep_id", "report_name" は各自、該当プロジェクト名、チーム、run ID、sweep ID、特定 report 名です。

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

entity に紐づくインテグレーションをすべて取得できるイテレータを返します。



**引数:**
 
 - `entity`:  処理対象の entity（例: チーム名）。未指定の場合はユーザーのデフォルト entity
 - `per_page`:  1ページあたり取得するインテグレーション数（デフォルト50。通常は変更不要）



**戻り値:**
 
 - `Iterator[SlackIntegration | WebhookIntegration]`: 利用可能インテグレーションのイテレータ

---

### <kbd>method</kbd> `Api.job`

```python
job(name: Optional[str], path: Optional[str] = None) → public.Job
```

`Job` オブジェクトを返します。



**引数:**
 
 - `name`:  ジョブ名
 - `path`:  ジョブアーティファクトのダウンロードパス



**戻り値:**
 `Job` オブジェクト

---

### <kbd>method</kbd> `Api.list_jobs`

```python
list_jobs(entity: str, project: str) → List[Dict[str, Any]]
```

指定 entity・プロジェクト内のジョブ一覧（あれば）を返します。



**引数:**
 
 - `entity`:  対象とするジョブの entity
 - `project`:  対象とするジョブの project



**戻り値:**
 条件に当てはまるジョブのリスト

---

### <kbd>method</kbd> `Api.project`

```python
project(name: str, entity: Optional[str] = None) → public.Project
```

指定した名前（必要に応じて entity も）の `Project` を返します。



**引数:**
 
 - `name`:  プロジェクト名
 - `entity`:  entity 名。未指定の場合は `Api` で指定したデフォルト entity を利用。デフォルトなしの場合は `ValueError`



**戻り値:**
 `Project` オブジェクト

---

### <kbd>method</kbd> `Api.projects`

```python
projects(entity: Optional[str] = None, per_page: int = 200) → public.Projects
```

指定 entity のプロジェクト一覧を取得します。



**引数:**
 
 - `entity`:  entity 名。未指定の場合は `Api` で指定のデフォルト entity。デフォルト entity もなければ `ValueError`
 - `per_page`:  1ページあたりのプロジェクト件数（デフォルト200）



**戻り値:**
 イテレータブルな `Projects` オブジェクト

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

パスに基づいて1つの queued run を返します。

パスの形式は `entity/project/queue_id/run_queue_item_id` です。

---

### <kbd>method</kbd> `Api.registries`

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) → Registries
```

Registry のイテレータを返します。

このイテレータを利用して、組織の registry 横断で registry、コレクション、artifact バージョンを検索・絞り込み可能です。



**引数:**
 
 - `organization`:  （str・オプション）取得したい registry の組織名。未指定の場合、ユーザー設定内の組織を利用
 - `filter`:  （辞書・オプション）各オブジェクトに適用される MongoDB風フィルタ。collection 用フィールドは name、description、created_at、updated_at。バージョン用は tag、alias、created_at、updated_at、metadata など



**戻り値:**
 registry イテレータ



**例:**
 名前に "model" が含まれる registry 全件検索

```python
import wandb

api = wandb.Api()  # 複数 org 所属の場合 org を指定すること
api.registries(filter={"name": {"$regex": "model"}})
```

名前が "my_collection" かつタグ "my_tag" のコレクションを持つすべての registry 検索

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
```

コレクション名に "my_collection" が含まれ、バージョンのエイリアスが "best" の artifact バージョンをすべて表示

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
```

"model" を含み、タグ"prod"またはエイリアス"best"の artifact バージョン一覧

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

registry 名を基に registry を返します。



**引数:**
 
 - `name`:  registry の名前（`wandb-registry-` プレフィックスなし）
 - `organization`:  registry の組織名。設定なければ、entity が1つの組織のみに属する場合は entity から取得



**戻り値:**
 registry オブジェクト



**例:**
 registry を取得し、内容を更新

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

指定したプロジェクトパス内の Reports を取得します。

注: `wandb.Api.reports()` API はまだベータ版です。将来変更される可能性があります。



**引数:**
 
 - `path`:  レポートが存在するプロジェクトのパス。プロジェクトを作成した entity を `/` で区切って追加
 - `name`:  取得したいレポート名
 - `per_page`:  1ページあたりのページネーション件数（通常は変更不要）



**戻り値:**
 `Reports` オブジェクト（イテレータブルな `BetaReport` のコレクション）



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

`entity/project/run_id` 形式のパスから 1 つの run を返します。



**引数:**
 
 - `path`:  `entity/project/run_id` の形式によるパス。`api.entity` 指定済みなら `project/run_id`、`api.project` も指定済みなら run_id のみでも取得可能



**戻り値:**
 `Run` オブジェクト

---

### <kbd>method</kbd> `Api.run_queue`

```python
run_queue(entity: str, name: str)
```

指定 entity の指定名 `RunQueue` を返します。

キュー作成法は `Api.create_run_queue` を参照ください。

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

指定プロジェクトの run を、フィルター条件に基づき検索して返します。

フィルタ対象フィールド：
- `createdAt`: run の作成タイムスタンプ（ISO 8601 例: "2023-01-01T12:00:00Z"）
- `displayName`: run の表示名（例: "eager-fox-1"）
- `duration`: run の実行時間（秒）
- `group`: run グループ名
- `host`: 実行ホスト名
- `jobType`: run のジョブタイプ
- `name`: run の一意識別子（例: "a1b2cdef"）
- `state`: run のステータス
- `tags`: run のタグ
- `username`: run 開始ユーザー名

config や summary metrics 項目にもフィルタ可能（例：`config.experiment_name`、`summary_metrics.loss`）

より複雑なフィルタには MongoDB クエリオペレータが利用できます。詳細: https://docs.mongodb.com/manual/reference/operator/query サポートされる操作：
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
 
 - `path`:  （str）プロジェクトへのパス形式（"entity/project"）
 - `filters`:  （dict）MongoDB クエリ言語で run を検索（例：`{"config.experiment_name": "foo"}` なら experiment_name="foo" の run を取得）
 - `order`:  （str）`created_at`, `heartbeat_at`, `config.*.value`, `summary_metrics.*` のいずれかで並び替え。+ で昇順、- で降順（デフォルト）。デフォルトは run.created_at の古い→新しい順。
 - `per_page`:  （int）1ページあたりの件数
 - `include_sweeps`:  （bool）Sweep run を結果に含めるか



**戻り値:**
 `Runs` オブジェクト（`Run` のイテレータブル）



**例:**
 ```python
# config.experiment_name が "foo" の run を取得
api.runs(path="my_entity/project", filters={"config.experiment_name": "foo"})
```

```python
# config.experiment_name が "foo" または "bar" の run を取得
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
# config.experiment_name が指定の正規表現にマッチする run を取得（anchors未対応）
api.runs(
    path="my_entity/project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
```

```python
# run の名前が正規表現にマッチする run を取得（anchors未対応）
api.runs(
    path="my_entity/project", filters={"display_name": {"$regex": "^foo.*"}}
)
```

```python
# 損失値（loss）が昇順となるよう run を取得
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

指定 entity の Slack インテグレーションのイテレータを返します。



**引数:**
 
 - `entity`:  対象の entity（例: チーム名）。未指定時はユーザーのデフォルト entity
 - `per_page`:  1ページあたりの件数（デフォルト50。通常変更不要）



**戻り値:**
 
 - `Iterator[SlackIntegration]`：Slack インテグレーションのイテレータ



**例:**
 チーム "my-team" の登録済み Slack インテグレーション全取得

```python
import wandb

api = wandb.Api()
slack_integrations = api.slack_integrations(entity="my-team")
```

チャンネル名が "team-alerts-" で始まる Slack インテグレーションのみフィルタ

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

`entity/project/sweep_id` 形式のパスから sweep を取得します。



**引数:**
 
 - `path`:  entity/project/sweep_id 形式のパス。`api.entity` 指定時は project/sweep_id, `api.project` もあれば sweep_id のみでも可



**戻り値:**
 `Sweep` オブジェクト

---

### <kbd>method</kbd> `Api.sync_tensorboard`

```python
sync_tensorboard(root_dir, run_id=None, project=None, entity=None)
```

tfevent ファイルを含むローカルディレクトリを wandb に同期します。

---

### <kbd>method</kbd> `Api.team`

```python
team(team: str) → public.Team
```

指定した名前の `Team` を返します。



**引数:**
 
 - `team`:  チーム名



**戻り値:**
 `Team` オブジェクト

---

### <kbd>method</kbd> `Api.update_automation`

```python
update_automation(
    obj: 'Automation',
    create_missing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

既存オートメーションを更新します。



**引数:**
 
 - `obj`:  更新したいオートメーション（要既存）。create_missing (bool):  True かつ対象オートメーションがなければ新規作成。**kwargs:  更新時に割り当てる追加値（指定時、すでにセット済みの値より優先）
        - `name`: オートメーション名
        - `description`: オートメーション説明
        - `enabled`: 有効かどうか
        - `scope`: 適用範囲
        - `event`: トリガーとなるイベント
        - `action`: 実行アクション



**戻り値:**
 更新された automation



**例:**
 既存 "my-automation" を無効化し、説明も編集

```python
import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")
automation.enabled = False
automation.description = "参照用として保持。今は利用していません。"

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
    description="参照用として保持。今は利用していません。",
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

W&B Launch の run queue を upsert します。



**引数:**
 
 - `name`:  作成するキュー名
 - `entity`:  （オプション）作成対象 entity 名。None の場合は設定済みまたはデフォルト entity
 - `resource_config`:  （オプション）デフォルトのリソース設定。テンプレート変数にはハンドルバー記法（例: `{{var}}`）を利用
 - `resource_type`:  利用リソースタイプ。"local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" のいずれか
 - `template_variables`:  config で使うテンプレート変数スキーマの辞書
 - `external_links`:  キューで使う外部リンクの辞書（オプション）
 - `prioritization_mode`:  優先順位バージョン。"V0" または None



**戻り値:**
 upsert した `RunQueue`



**例外:**
 ValueError：無効なパラメータ
 wandb.Error：API エラー

---

### <kbd>method</kbd> `Api.user`

```python
user(username_or_email: str) → Optional[ForwardRef('public.User')]
```

ユーザー名またはメールアドレスからユーザーを返します。

この関数はローカル管理者のみ利用可能です。自身のユーザーを取得したい場合は `api.viewer` を利用してください。



**引数:**
 
 - `username_or_email`:  ユーザー名またはメールアドレス



**戻り値:**
 `User` オブジェクト（見つからない場合は None）

---

### <kbd>method</kbd> `Api.users`

```python
users(username_or_email: str) → List[ForwardRef('public.User')]
```

部分一致ユーザー名またはメールアドレスでユーザー一覧を返します。

この関数はローカル管理者のみ利用可能です。自身のユーザーを取得したい場合は `api.viewer` を利用してください。



**引数:**
 
 - `username_or_email`:  検索したいユーザーの部分文字列



**戻り値:**
 `User` オブジェクトの配列

---

### <kbd>method</kbd> `Api.webhook_integrations`

```python
webhook_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('WebhookIntegration')]
```

指定 entity の webhook インテグレーションのイテレータを返します。



**引数:**
 
 - `entity`:  対象の entity（例: チーム名）。未指定時はユーザーのデフォルト entity
 - `per_page`:  1ページあたりの件数（デフォルト50。通常変更不要）



**戻り値:**
 
 - `Iterator[WebhookIntegration]`：Webhook インテグレーションのイテレータ



**例:**
 チーム "my-team" の登録済み webhook インテグレーション取得

```python
import wandb

api = wandb.Api()
webhook_integrations = api.webhook_integrations(entity="my-team")
```

" https://my-fake-url.com" にリクエストを送信する webhook インテグレーションのみをフィルタ

```python
webhook_integrations = api.webhook_integrations(entity="my-team")
my_webhooks = [
    ig
    for ig in webhook_integrations
    if ig.url_endpoint.startswith("https://my-fake-url.com")
]
```
