---
title: api
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/api.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API を使用して、W&B に保存したデータのエクスポートや更新ができます。

この API を使用する前に、スクリプトからデータをログしておく必要があります。詳細は [クイックスタート](https://docs.wandb.ai/quickstart) をご参照ください。

Public API の主な用途例:
 - 実験完了後にメタデータやメトリクスを更新する
 - 結果をデータフレームとして取得し、Jupyter Notebook などで後処理・分析する
 - `ready-to-deploy` タグ付きの保存済みモデル Artifacts を確認する

Public API の詳しい使い方は [こちらのガイド](https://docs.wandb.com/guides/track/public-api-guide) をご覧ください。


## <kbd>class</kbd> `Api`
W&B サーバーへのクエリ用クラスです。



**使用例:**
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
 
 - `overrides`:  `base_url` の指定ができます（例: 
 - `https`: //api.wandb.ai 以外の W&B サーバーを利用する場合）。また、`entity`、`project`、`run` の初期値も設定可能です。
 - `timeout`:  API リクエストの HTTP タイムアウト値（秒）。指定しない場合はデフォルト値となります。
 - `api_key`:  認証用の API キー。指定しなければ、現在の環境変数や設定ファイルから API キーが利用されます。


---

### <kbd>property</kbd> Api.api_key

W&B の API キーを返します。

---

### <kbd>property</kbd> Api.client

クライアントオブジェクトを返します。

---

### <kbd>property</kbd> Api.default_entity

デフォルトの W&B エンティティを返します。

---

### <kbd>property</kbd> Api.user_agent

W&B パブリックユーザーエージェントを返します。

---

### <kbd>property</kbd> Api.viewer

ビューア オブジェクトを返します。



---

### <kbd>method</kbd> `Api.artifact`

```python
artifact(name: str, type: Optional[str] = None)
```

指定したアーティファクトを取得します。



**引数:**
 
 - `name`:  アーティファクト名。ファイルパスのような形式で、プロジェクト名/アーティファクト名:バージョンまたはエイリアス で構成されます。先頭に entity を付与する場合はスラッシュで区切ります。entity が省略された場合は Run または API 設定の entity が使われます。
 - `type`:  取得するアーティファクトの種類。



**戻り値:**
 `Artifact` オブジェクトを返します。



**例外:**
 
 - `ValueError`:  アーティファクト名が指定されていない場合
 - `ValueError`:  タイプ指定が実際のアーティファクトタイプと異なる場合



**使用例:**
 以下のコードスニペット中の "entity", "project", "artifact", "version", "alias" はそれぞれエンティティ・プロジェクト名・アーティファクト名・バージョン・エイリアスのプレースホルダです。

```python
import wandb

# プロジェクト名、アーティファクト名、エイリアスの指定
wandb.Api().artifact(name="project/artifact:alias")

# プロジェクト名、アーティファクト名、バージョンの指定
wandb.Api().artifact(name="project/artifact:version")

# entity、プロジェクト名、アーティファクト名、エイリアスの指定
wandb.Api().artifact(name="entity/project/artifact:alias")

# entity、プロジェクト名、アーティファクト名、バージョンの指定
wandb.Api().artifact(name="entity/project/artifact:version")
``` 



**注意:**

> このメソッドは外部からの利用専用です。wandb レポジトリ内部で `api.artifact()` を呼ばないでください。

---

### <kbd>method</kbd> `Api.artifact_collection`

```python
artifact_collection(type_name: str, name: str) → public.ArtifactCollection
```

指定したタイプのアーティファクトコレクションを取得します。

返された `ArtifactCollection` オブジェクトを使って、そのコレクション内の特定アーティファクトの情報取得等が行えます。



**引数:**
 
 - `type_name`:  取得したいアーティファクトコレクションの種類
 - `name`:  アーティファクトコレクション名。先頭に entity をスラッシュ区切りで追加可能。



**戻り値:**
 `ArtifactCollection` オブジェクトを返します。



**使用例:**
 「type」「entity」「project」「artifact_name」はコレクション種別・エンティティ・プロジェクト名・アーティファクト名のプレースホルダです。

```python
import wandb

collections = wandb.Api().artifact_collection(
    type_name="type", name="entity/project/artifact_name"
)

# コレクション内で最初のアーティファクトを取得
artifact_example = collections.artifacts()[0]

# アーティファクトの内容を指定したディレクトリにダウンロード
artifact_example.download()
``` 

---

### <kbd>method</kbd> `Api.artifact_collection_exists`

```python
artifact_collection_exists(name: str, type: str) → bool
```

指定したプロジェクトと entity 内にそのアーティファクトコレクションが存在するか確認します。



**引数:**
 
 - `name`:  アーティファクトコレクション名。entity をスラッシュ区切りで追加可能。entity または project が未指定の場合はオーバーライドパラメータやユーザー設定から推論されます。project はデフォルトで "uncategorized" となります。
 - `type`:  アーティファクトコレクションの種類。



**戻り値:**
 アーティファクトコレクションが存在する場合は True、存在しない場合は False を返します。



**使用例:**
「type」と「collection_name」はそれぞれ種類・コレクション名のプレースホルダです。

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

条件に一致する複数のアーティファクトコレクションを取得します。



**引数:**
 
 - `project_name`:  絞り込むプロジェクト名
 - `type_name`:  絞り込むアーティファクト種別名
 - `per_page`:  クエリページネーション時のページサイズ設定。通常変更する必要はありません。



**戻り値:**
 イテラブルな `ArtifactCollections` オブジェクト。

---

### <kbd>method</kbd> `Api.artifact_exists`

```python
artifact_exists(name: str, type: Optional[str] = None) → bool
```

指定したプロジェクトと entity に、アーティファクトのバージョンが存在するかどうかを判定します。



**引数:**
 
 - `name`:  アーティファクト名。entity, project をスラッシュ区切りで、末尾にコロン区切りでバージョンまたはエイリアスを追加。entity, project がなければオーバーライドパラメータやユーザー設定から取得し、project は "Uncategorized"になります。
 - `type`:  アーティファクトの種別。



**戻り値:**
 指定したアーティファクトバージョンが存在する場合 True、それ以外は False。



**使用例:**
「entity」「project」「artifact」「version」「alias」はそれぞれプレースホルダです。

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

指定した `ArtifactType` を返します。



**引数:**
 
 - `type_name`:  取得するアーティファクト種別名
 - `project`:  （任意）絞り込むプロジェクト名またはパス



**戻り値:**
 `ArtifactType` オブジェクト。

---

### <kbd>method</kbd> `Api.artifact_types`

```python
artifact_types(project: Optional[str] = None) → public.ArtifactTypes
```

条件に一致するアーティファクト種別のコレクションを返します。



**引数:**
 
 - `project`:  絞り込むプロジェクト名またはパス



**戻り値:**
 イテラブルな `ArtifactTypes` オブジェクト。

---

### <kbd>method</kbd> `Api.artifact_versions`

```python
artifact_versions(type_name, name, per_page=50)
```

非推奨です。代わりに `Api.artifacts(type_name, name)` メソッドを使用してください。

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
 type_name: 取得したいアーティファクトのタイプ name: コレクション名。先頭に entity をスラッシュ区切りで追加可能 per_page: クエリページネーション時のページサイズ。通常変更不要です。 tags: 指定タグ全てを持つアーティファクトのみ取得。



**戻り値:**
 イテラブルな `Artifacts` オブジェクト。



**使用例:**
「type」「entity」「project」「artifact_name」はそれぞれ種別・エンティティ・プロジェクト名・アーティファクト名のプレースホルダです。

```python
import wandb

wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
``` 

---

### <kbd>method</kbd> `Api.automation`

```python
automation(name: str, entity: Optional[str] = None) → Automation
```

パラメータに一致する唯一の Automation を返します。



**引数:**
 
 - `name`:  取得する Automation の名前
 - `entity`:  取得対象の entity



**例外:**
 
 - `ValueError`:  一致する Automation が 0 件または複数見つかった場合



**使用例:**
"my-automation" という名前の Automation を取得:

```python
import wandb

api = wandb.Api()
automation = api.automation(name="my-automation")
``` 

"other-automation" という名前・エンティティ"my-team" の Automation を取得:

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

パラメータにマッチする全ての Automation を取得するイテレータを返します。

パラメータを指定しない場合は、ユーザーがアクセスできる全 Automation を返します。



**引数:**
 
 - `entity`:  取得対象の entity
 - `name`:  取得対象の Automation 名
 - `per_page`: 1ページあたりの取得件数（デフォルト 50、通常変更不要）



**戻り値:**
 Automation のリスト



**使用例:**
エンティティ "my-team" の全 Automation を取得

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

新しい Automation を作成します。



**引数:**
  obj:  作成する Automation。  fetch_existing:  True の場合、重複 Automation がある場合は既存を取得しエラーになりません。  **kwargs:  Automation 作成前に上書き設定するフィールド。以下が指定可能:
        - `name`: Automation の名前。
        - `description`: Automation の説明。
        - `enabled`: Automation の有効化／無効化。
        - `scope`: Automation の作用範囲。
        - `event`: Automation をトリガーするイベント。
        - `action`: Automation で実行されるアクション。



**戻り値:**
  保存された Automation



**使用例:**
特定プロジェクトの run でメトリックがしきい値超過時に Slack 通知を送る Automation を作成:

```python
import wandb
from wandb.automations import OnRunMetric, RunEvent, SendNotification

api = wandb.Api()

project = api.project("my-project", entity="my-team")

# チームで最初に登録されている Slack フックを利用
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
 
 - `entity`:  チャートの所有者（ユーザーまたはチーム）
 - `name`:  チャートプリセットのユニーク名
 - `display_name`:  UI に表示される人が読みやすい名前
 - `spec_type`:  スペック種別。"vega2"（Vega-Lite v2）固定
 - `access`:  チャートの公開範囲:
        - "private": チャートの作成 entity のみアクセス可
        - "public": 誰でもアクセス可能
 - `spec`:  Vega または Vega-Lite 仕様の dict または JSON 文字列



**戻り値:**
 作成されたチャートプリセットの ID（"entity/name" 形式）



**例外:**
 
 - `wandb.Error`:  作成失敗時
 - `UnsupportedError`:  サーバー側カスタムチャート未対応時



**使用例:**
 ```python
    import wandb

    api = wandb.Api()

    # シンプルな棒グラフのチャート仕様を定義
    vega_spec = {
         "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
         "mark": "bar",
         "data": {"name": "wandb"},
         "encoding": {
             "x": {"field": "${field:x}", "type": "ordinal"},
             "y": {"field": "${field:y}", "type": "quantitative"},
         },
    }

    # カスタムチャート作成
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
 
 - `name`:  新しいプロジェクト名
 - `entity`:  作成先 entity

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
 
 - `name`:  レジストリ名。組織内でユニークである必要があります
 - `visibility`:  レジストリの公開範囲
 - `organization`:  組織内の誰でもこのレジストリを閲覧可能です。ロールの編集は UI の設定から変更可能です
 - `restricted`:  招待したメンバーのみ UI 経由でこのレジストリへのアクセスが可能。外部公開は無効です
 - `organization`:  レジストリの所属組織。設定で未指定の場合、entity が 1 つの組織だけに所属していればそこから取得されます
 - `description`:  レジストリ説明文
 - `artifact_types`:  レジストリで許容するアーティファクト種別。128 文字以内、「/」「:」は不可。指定しなければ全種許容。許可した種別は後から削除不可



**戻り値:**
 レジストリオブジェクト



**使用例:**
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
 
 - `run_id`:  割り当てる run の ID。指定しなければ W&B がランダムで生成します
 - `project`:  ログ先のプロジェクト。未指定時は "Uncategorized" プロジェクトにログします
 - `entity`:  プロジェクト所有 entity。未指定の場合はデフォルト entity になります



**戻り値:**
 新しく作成された `Run`

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

W&B Launch に新しいランキューを作成します。



**引数:**
 
 - `name`:  作成するキュー名
 - `type`:  利用リソース種別（"local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" のいずれか）
 - `entity`:  作成先 entity。`None` は設定やデフォルト entity を利用
 - `prioritization_mode`:  優先度バージョン（"V0" または `None`）
 - `config`:  キュー既定リソース構成。テンプレート変数は `{{var}}` 形式で指定
 - `template_variables`:  config 用テンプレート変数スキーマの辞書



**戻り値:**
 新しく作成した `RunQueue`



**例外:**
 パラメータ無効時 `ValueError`、W&B API エラー時 `wandb.Error`

---

### <kbd>method</kbd> `Api.create_team`

```python
create_team(team: str, admin_username: Optional[str] = None) → public.Team
```

新しいチームを作成します。



**引数:**
 
 - `team`:  チーム名
 - `admin_username`:  チーム管理者ユーザー名。デフォルトは現在のユーザー



**戻り値:**
 `Team` オブジェクト

---

### <kbd>method</kbd> `Api.create_user`

```python
create_user(email: str, admin: Optional[bool] = False)
```

新しいユーザーを作成します。



**引数:**
 
 - `email`:  ユーザーのメールアドレス
 - `admin`:  インスタンス管理者権限を付与する場合 True



**戻り値:**
 `User` オブジェクト

---

### <kbd>method</kbd> `Api.delete_automation`

```python
delete_automation(obj: Union[ForwardRef('Automation'), str]) → Literal[True]
```

Automation を削除します。



**引数:**
 
 - `obj`:  削除対象 Automation またはその ID



**戻り値:**
 削除成功時 True

---

### <kbd>method</kbd> `Api.flush`

```python
flush()
```

ローカルキャッシュをクリアします。

api オブジェクトは run のローカルキャッシュを保持しているため、スクリプト実行中に run の状態が変化する可能性がある場合は `api.flush()` でキャッシュをクリアして最新値を取得してください。

---

### <kbd>method</kbd> `Api.from_path`

```python
from_path(path: str)
```

パスから run、sweep、project、report を返します。



**引数:**
 
 - `path`:  project、run、sweep、report のパス



**戻り値:**
 `Project`、`Run`、`Sweep`、または `BetaReport` インスタンス



**例外:**
 パスが不正または対象が存在しない場合 `wandb.Error`



**使用例:**
「project」「team」「run_id」「sweep_id」「report_name」はそれぞれプロジェクト・チーム・run ID・sweep ID・レポート名のプレースホルダです。

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

指定した entity の全 integration イテレータを返します。



**引数:**
 
 - `entity`:  取得対象 entity（例: チーム名）。未指定ならユーザーのデフォルト entity
 - `per_page`: 1ページあたりの件数（デフォルト 50、通常変更不要）



**Yields:**
 
 - `Iterator[SlackIntegration | WebhookIntegration]`:  対応する integration のイテレータ

---

### <kbd>method</kbd> `Api.job`

```python
job(name: Optional[str], path: Optional[str] = None) → public.Job
```

`Job` オブジェクトを取得します。



**引数:**
 
 - `name`:  ジョブ名
 - `path`:  ジョブアーティファクトのダウンロード先パス



**戻り値:**
 `Job` オブジェクト

---

### <kbd>method</kbd> `Api.list_jobs`

```python
list_jobs(entity: str, project: str) → List[Dict[str, Any]]
```

指定した entity・プロジェクトのジョブ一覧（存在すれば）を返します。



**引数:**
 
 - `entity`:  検索対象の entity
 - `project`:  検索対象のプロジェクト



**戻り値:**
 条件に合うジョブリスト

---

### <kbd>method</kbd> `Api.project`

```python
project(name: str, entity: Optional[str] = None) → public.Project
```

指定した名前（および entity）の `Project` を返します。



**引数:**
 
 - `name`:  プロジェクト名
 - `entity`:  リクエスト先 entity 名。None の場合は `Api` で指定されたデフォルト entity。デフォルト entity なしの場合は `ValueError` となります



**戻り値:**
 `Project` オブジェクト

---

### <kbd>method</kbd> `Api.projects`

```python
projects(entity: Optional[str] = None, per_page: int = 200) → public.Projects
```

指定した entity のプロジェクト一覧を取得します。



**引数:**
 
 - `entity`:  リクエスト先 entity 名。None の場合は `Api` で指定されたデフォルト entity。デフォルト entity なしの場合は `ValueError`
 - `per_page`:  クエリページネーション時のページサイズ。通常変更不要



**戻り値:**
 `Project`オブジェクトのイテラブルなコレクション

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

パスから1件の queued run を返します。

`entity/project/queue_id/run_queue_item_id` 形式のパスを解析します。

---

### <kbd>method</kbd> `Api.registries`

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) → Registries
```

レジストリエイタレータを返します。

このイテレータを使い、組織内の registry・collection・artifact バージョン検索や絞り込みが可能です。



**引数:**
 
 - `organization`:  （str・任意）取得対象レジストリの所属組織。未指定時はユーザー設定の所属組織を利用
 - `filter`:  （dict・任意）各オブジェクトへの MongoDB 形式のフィルタ。collection では `name`, `description`, `created_at`, `updated_at` がフィルタ対象、version では `tag`, `alias`, `created_at`, `updated_at`, `metadata` が利用できます



**戻り値:**
 レジストリエイタレータ



**使用例:**
「model」を含む名前の全 registry を検索

```python
import wandb

api = wandb.Api()  # 複数組織に entity が属する場合 org 指定
api.registries(filter={"name": {"$regex": "model"}})
``` 

名前が "my_collection" かつタグが "my_tag" の全 registry 内 collection を検索

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
``` 

コレクション名に "my_collection" を含み、バージョンの alias が "best" の全 artifact バージョンを検索

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
``` 

"model" を含み、"prod" タグまたは "best" エイリアスのバージョンを持つ artifact バージョンを検索

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

レジストリ名から該当レジストリを取得します。



**引数:**
 
 - `name`:  レジストリ名（`wandb-registry-` プレフィックス無し）
 - `organization`:  レジストリの所属組織。未指定時、設定で組織が定義されていなければ、entity が 1 組織のみ所属の場合その組織を利用



**戻り値:**
 レジストリオブジェクト



**使用例:**
レジストリ取得と更新

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

指定プロジェクトパスの reports を取得します。

注意: `wandb.Api.reports()` API はベータ版であり、今後のリリースで変更される可能性があります。



**引数:**
 
 - `path`:  レポート対象プロジェクトのパス。作成 entity をスラッシュ区切りで先頭付与
 - `name`:  取得したいレポート名
 - `per_page`:  クエリページネーション時のページサイズ。通常変更不要



**戻り値:**
 イテラブルな `BetaReport` オブジェクトのコレクション (`Reports`) を返します。



**使用例:**
 ```python
import wandb

wandb.Api.reports("entity/project")
``` 

---

### <kbd>method</kbd> `Api.run`

```python
run(path='')
```

`entity/project/run_id` 形式のパスから run を1件返します。



**引数:**
 
 - `path`:  `entity/project/run_id` 形式で渡す。`api.entity` 設定時は `project/run_id`、`api.project` 設定時は `run_id` のみも可



**戻り値:**
 `Run` オブジェクト

---

### <kbd>method</kbd> `Api.run_queue`

```python
run_queue(entity: str, name: str)
```

指定 entity の `RunQueue` を返します。

run キューの作成については `Api.create_run_queue` を参照してください。

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

指定プロジェクトから、フィルター条件に合う run を取得します。

フィルタ可能な項目例:
- `createdAt`: run の作成日時（ISO 8601 形式, 例 "2023-01-01T12:00:00Z"）
- `displayName`: run の表示名（例 "eager-fox-1"）
- `duration`: ランタイム総秒数
- `group`: グループ名（一連の run をまとめる際に利用）
- `host`: run 実行ホスト名
- `jobType`: run のジョブ種別や目的
- `name`: run のユニーク識別子（例 "a1b2cdef"）
- `state`: run の現在の状態
- `tags`: run に紐づくタグ
- `username`: run 発行ユーザーのユーザー名

また、run 設定（config）や summary メトリクスでもフィルタ可能です（例: `config.experiment_name`, `summary_metrics.loss` など）。

複雑な条件でのフィルタには MongoDB クエリオペレータが使えます。詳しくは https://docs.mongodb.com/manual/reference/operator/query をご確認ください。サポートされている演算子:
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
 
 - `path`:  （str）プロジェクトへのパス（例: "entity/project"）
 - `filters`:  （dict）MongoDBフォーマットで run をフィルタ。config.key, summary_metrics.key, state, entity, createdAt などで指定可能
 - `For example`:  `{"config.experiment_name": "foo"}` の場合 experiment_name="foo" の run を検索
 - `order`:  （str）`created_at`・`heartbeat_at`・`config.*.value`・`summary_metrics.*` でソート。`+` 先頭で昇順、`-` 先頭で降順（デフォルト）。デフォルトは run.created_at 昇順。
 - `per_page`:  （int）ページサイズ指定
 - `include_sweeps`:  （bool）sweep run も含めるかどうか



**戻り値:**
 `Run` オブジェクトのイテラブルなコレクション（`Runs`）。



**使用例:**
 ```python
# プロジェクト内で、config.experiment_name="foo" の run を検索
api.runs(path="my_entity/project", filters={"config.experiment_name": "foo"})
``` 

```python
# config.experiment_name が "foo" または "bar" の run を検索
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
# config.experiment_name が "b" で始まる run を正規表現で検索
api.runs(
    path="my_entity/project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
``` 

```python
# run 名（display_name）が "foo" で始まる run を正規表現で検索
api.runs(
    path="my_entity/project", filters={"display_name": {"$regex": "^foo.*"}}
)
``` 

```python
# 損失値 loss の昇順で run をソート
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

指定 entity の Slack integration のイテレータを返します。



**引数:**
 
 - `entity`:  取得対象 entity（例: チーム名）。未指定ならデフォルト entity
 - `per_page`: 1ページあたりの取得数。通常変更不要（デフォルト: 50）



**Yields:**
 
 - `Iterator[SlackIntegration]`:  Slack integration のイテレータ



**使用例:**
チーム "my-team" に登録された全 Slack integration を取得

```python
import wandb

api = wandb.Api()
slack_integrations = api.slack_integrations(entity="my-team")
``` 

チャンネル名が "team-alerts-" で始まる Slack integration のみ取得

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

`entity/project/sweep_id` 形式のパスから sweep を1件取得します。



**引数:**
 
 - `path`:  entity/project/sweep_id 形式で渡す。`api.entity` 設定時は project/sweep_id、`api.project` 設定時は sweep_id のみも可



**戻り値:**
 `Sweep` オブジェクト

---

### <kbd>method</kbd> `Api.sync_tensorboard`

```python
sync_tensorboard(root_dir, run_id=None, project=None, entity=None)
```

tfevent ファイルを含むローカルディレクトリを wandb と同期します。

---

### <kbd>method</kbd> `Api.team`

```python
team(team: str) → public.Team
```

指定名の `Team` を返します。



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

既存の Automation を更新します。



**引数:**
 
 - `obj`:  更新対象 Automation（既存である必要あり） create_missing (bool): True で Automation 無存在時に作成 **kwargs: 更新前に指定フィールドを上書き
        - `name`: Automation 名
        - `description`: 説明
        - `enabled`: 有効・無効
        - `scope`: 作用範囲
        - `event`: トリガーイベント
        - `action`: トリガーアクション



**戻り値:**
 更新済みの Automation



**使用例:**
既存 Automation ("my-automation") の無効化・説明変更

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

W&B Launch で run キューの upsert（更新または新規作成）を行います。



**引数:**
 
 - `name`:  作成するキュー名
 - `entity`:  （任意）作成先 entity。None の場合は設定やデフォルト entity 利用
 - `resource_config`:  キュー用デフォルトリソース構成。テンプレート変数は `{{var}}` 指定
 - `resource_type`:  利用するリソースのタイプ（"local-container", "local-process", "kubernetes", "sagemaker", "gcp-vertex" から選択）
 - `template_variables`:  config で利用するテンプレート変数スキーマ
 - `external_links`:  （任意）キューに利用する外部リンク情報
 - `prioritization_mode`:  優先度バージョン（"V0" または None）



**戻り値:**
 upsert された `RunQueue`



**例外:**
 パラメータのいずれかが無効な場合は ValueError、W&B API エラー時は wandb.Error

---

### <kbd>method</kbd> `Api.user`

```python
user(username_or_email: str) → Optional[ForwardRef('public.User')]
```

username またはメールアドレスからユーザーを取得します。

この関数はローカル管理者のみ利用可能です。自身のユーザーオブジェクトを取得するには `api.viewer` を利用してください。



**引数:**
 
 - `username_or_email`:  検索対象ユーザー名またはメールアドレス



**戻り値:**
 `User` オブジェクトまたは該当者なしなら None

---

### <kbd>method</kbd> `Api.users`

```python
users(username_or_email: str) → List[ForwardRef('public.User')]
```

部分一致するユーザー名やメールアドレスで全ユーザーを取得します。

この関数はローカル管理者のみ利用可能です。自身のユーザーオブジェクトを取得するには `api.viewer` を利用してください。



**引数:**
 
 - `username_or_email`:  検索したいユーザーのプレフィックスまたはサフィックス



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

指定した entity の webhook integration のイテレータを返します。



**引数:**
 
 - `entity`:  取得対象 entity（例: チーム名）。未指定時はユーザーのデフォルト entity
 - `per_page`:  1ページあたりの件数（デフォルト 50, 通常変更不要）



**Yields:**
 
 - `Iterator[WebhookIntegration]`:  webhook integration のイテレータ



**使用例:**
チーム "my-team" の全 webhook integration を取得

```python
import wandb

api = wandb.Api()
webhook_integrations = api.webhook_integrations(entity="my-team")
``` 

URL が "https://my-fake-url.com" で始まる webhook integration だけを抽出

```python
webhook_integrations = api.webhook_integrations(entity="my-team")
my_webhooks = [
    ig
    for ig in webhook_integrations
    if ig.url_endpoint.startswith("https://my-fake-url.com")
]
```