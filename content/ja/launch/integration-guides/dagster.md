---
title: Dagster
description: W&B を Dagster と統合する方法のガイド。
menu:
  launch:
    identifier: ja-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster と W&B を使って MLOps パイプラインをオーケストレーションし、ML アセットを管理しましょう。W&B とのインテグレーションにより、Dagster 内で以下が簡単に行えます。

* [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の作成と利用
* [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) での Registered Models の利用と作成
* [W&B Launch]({{< relref path="/launch/" lang="ja" >}}) を通じた専用計算リソースでのトレーニングジョブ実行
* ops や assets 内で [wandb]({{< relref path="/ref/python/" lang="ja" >}}) クライアントの利用

W&B Dagster インテグレーションは、W&B 専用の Dagster リソースと IO Manager を提供します。

* `wandb_resource`: W&B API への認証と通信に使う Dagster リソース
* `wandb_artifacts_io_manager`: W&B Artifacts の利用に使う Dagster IO Manager

以下のガイドでは、Dagster で W&B を使う際の準備、ops/assets での W&B Artifacts の作成・利用方法、W&B Launch の利用やベストプラクティスについて説明します。

## 始める前に
Dagster で W&B を使うには、以下のリソースが必要です。

1. **W&B API Key**
2. **W&B entity（ユーザーまたは team）**：entity とは、W&B Runs や Artifacts を送信するユーザーネームや team 名のことです。事前に W&B App UI でアカウントまたは team entity を作成しましょう。entity を指定しない場合、run はデフォルト entity（通常は自分のユーザーネーム）に送信されます。デフォルト entity は、**Project Defaults** 内の設定から変更できます。
3. **W&B project**：[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を格納するプロジェクト名。

W&B entity は、W&B App で該当ユーザーまたは team のプロフィールページから確認できます。既存のプロジェクトを使うか、新規作成することも可能です。新しいプロジェクトは W&B App のホームやプロフィールページで作成できます。プロジェクトが存在しない場合、初回利用時に自動的に作成されます。API キーの取得方法は以下の通りです。

### API キーの取得方法
1. [W&B にログイン](https://wandb.ai/login)。※W&B Server を使っている場合は、管理者にインスタンスのホスト名を確認してください。
2. [認証ページ](https://wandb.ai/authorize) またはユーザー/チーム設定で API キーを取得します。プロダクション環境では、キーの所有者として [サービスアカウント]({{< relref path="/support/kb-articles/service_account_useful.md" lang="ja" >}}) の利用を推奨します。
3. 環境変数として API キーを設定します：`export WANDB_API_KEY=YOUR_KEY`

以下のサンプルで、Dagster コード内のどこで API キーを指定するか示しています。`wandb_config` のネストされた辞書内に entity と project 名を明記してください。複数の ops/assets で異なる W&B プロジェクトを使いたい場合は、それぞれに異なる `wandb_config` を渡せます。渡せるキーの詳細は下記 Configuration セクションをご参照ください。


{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
例：`@job` 用の設定例
```python
# この内容を config.yaml に追加
# または Dagit の Launchpad や JobDefinition.execute_in_process でセットも可能
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # W&B entity に置き換えてください
     project: my_project # W&B project に置き換えてください


@job(
   resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
       "io_manager": wandb_artifacts_io_manager,
   }
)
def simple_job_example():
   my_op()
```
{{% /tab %}}
{{% tab "Config for @repository using assets" %}}

例：assets を使った `@repository` 用の設定例

```python
from dagster_wandb import wandb_artifacts_io_manager, wandb_resource
from dagster import (
   load_assets_from_package_module,
   make_values_resource,
   repository,
   with_resources,
)

from . import assets

@repository
def my_repository():
   return [
       *with_resources(
           load_assets_from_package_module(assets),
           resource_defs={
               "wandb_config": make_values_resource(
                   entity=str,
                   project=str,
               ),
               "wandb_resource": wandb_resource.configured(
                   {"api_key": {"env": "WANDB_API_KEY"}}
               ),
               "wandb_artifacts_manager": wandb_artifacts_io_manager.configured(
                   {"cache_duration_in_minutes": 60} # ファイルは 1 時間のみキャッシュ
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # W&B entity に置き換えてください
                       "project": "my_project", # W&B project に置き換えてください
                   }
               }
           },
       ),
   ]
```
この例では、`@job` の例と異なり IO Manager のキャッシュ時間を設定しています。
{{% /tab %}}
{{< /tabpane >}}


### 設定（Configuration）
以下は、インテグレーションにより提供される W&B 専用 Dagster resource・IO Manager の設定オプションです。

* `wandb_resource`: W&B API との通信に使う Dagster [resource](https://docs.dagster.io/concepts/resources)。API キーで自動認証。プロパティ：
    * `api_key`：（str, 必須）W&B API との通信に必要な API キー
    * `host`：（str, 任意）利用したい API ホストサーバー。W&B Server 利用時のみ必須。デフォルトはパブリッククラウド `https://api.wandb.ai`
* `wandb_artifacts_io_manager`: W&B Artifacts の利用に使う Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ：
    * `base_dir`：（int, 任意）ローカルストレージ・キャッシュに使うベースディレクトリー。デフォルトは `DAGSTER_HOME` ディレクトリー
    * `cache_duration_in_minutes`：（int, 任意）ローカルに保持する W&B Artifacts・W&B Run ログの保存時間。該当時間アクセスされなかったファイル・ディレクトリはキャッシュから削除。キャッシュ削除は IO Manager の実行終了時。0 にするとキャッシュ無効化。デフォルトは 30 日間
    * `run_id`：（str, 任意）run を再開するためのユニーク ID。プロジェクト内で一意である必要、run を削除した ID は再利用不可。短い説明名は name フィールドに、比較用にハイパーパラメーター管理は config へ。ID には `/\#?%:..` 含められません。Dagster で実験管理時 Run ID の設定必須。デフォルトは Dagster の Run ID 例：`7e4df022-1bf2-44b5-a383-bb852df4077e`
    * `run_name`：（str, 任意）UI 上で識別しやすい短い表示名。デフォルトは `dagster-run-[Dagster Run ID の最初の8文字]` 例：`dagster-run-7e4df022`
    * `run_tags`：（list[str], 任意）run のタグ一覧。タグで runs を整理したり、`baseline` や `production` など一時的なラベル付与にも便利。UI で追加・削除やフィルターも簡単。インテグレーションで利用された W&B Run には `dagster_wandb` タグが付きます。

## W&B Artifacts の利用

W&B Artifact との連携は、Dagster IO Manager に依存します。

[IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットまたは op の出力を保存し、下流のアセット・ops への入力としてロードする責任を持つユーザー提供オブジェクトです。例えば、IO Manager はファイルシステム上のファイルからオブジェクトを保存・読み込みできます。

W&B Artifacts 用の専用 IO Manager により、どの Dagster の `@op` や `@asset` でも W&B Artifacts の作成・利用が可能です。たとえば、Python リストを含む type dataset の W&B Artifact を生成するシンプルな `@asset` は下記の通りです。

```python
@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
    return [1, 2, 3] # この内容が Artifact として保存されます
```

`@op`, `@asset`, `@multi_asset` にはメタデータ設定を注釈して Artifacts を保存可能です。同様に、外部で作成された W&B Artifacts も消費できます。

## W&B Artifacts の書き込み
続ける前に、W&B Artifacts の基本を理解しておくことをおすすめします。[Artifacts ガイド]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) もご参照ください。

Python 関数の return オブジェクトが、そのまま W&B Artifact として保存されます。W&B でサポートされているオブジェクト：

* Python オブジェクト（int, dict, list など）
* W&B オブジェクト（Table, Image, Graph など）
* W&B Artifact オブジェクト

以下では、Dagster のアセット（`@asset`）で W&B Artifacts を書き込む例を紹介します。


{{< tabpane text=true >}}
{{% tab "Python オブジェクト" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズ可能なデータは、インテグレーションにより pickled されて Artifact 保存されます。Dagster 内で Artifact を読む際にアンピックル（デシリアライズ）されます（詳しくは [アーティファクトの読み込み]({{< relref path="#read-wb-artifacts" lang="ja" >}}) 参照）。

```python
@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
    return [1, 2, 3]
```

W&B は複数種類の Pickle 系ライブラリ ([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートします。[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などの高度なシリアライズ形式も利用できます。詳細は [Serialization]({{< relref path="#serialization-configuration" lang="ja" >}}) セクションをご覧ください。
{{% /tab %}}
{{% tab "W&B オブジェクト" %}}
[Table]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) や [Image]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) などの W&B オブジェクトも、インテグレーションで作成された Artifact に追加可能です。下記は Table を保存する例です。

```python
import wandb

@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset_in_table():
    return wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
```

{{% /tab %}}
{{% tab "W&B Artifact" %}}

より複雑なユースケースでは Artifact オブジェクトを自分で作成することもできます。その際もインテグレーションのメタデータ拡張などの便利な機能が利用できます。

```python
import wandb

MY_ASSET = "my_asset"

@asset(
    name=MY_ASSET,
    io_manager_key="wandb_artifacts_manager",
)
def create_artifact():
   artifact = wandb.Artifact(MY_ASSET, "dataset")
   table = wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
   artifact.add(table, "my_table")
   return artifact
```
{{% /tab %}}
{{< /tabpane >}}


### 設定
`wandb_artifact_configuration` という設定辞書を、`@op`, `@asset`, `@multi_asset` にメタデータとして設定できます。この設定は IO Manager が W&B Artifacts を読み書きする際に必要です。

`@op` では [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) の metadata 引数に、
`@asset` では asset の metadata 引数に、
`@multi_asset` では各出力ごとの [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) の metadata 引数に設定してください。

各計算における設定例は以下。

{{< tabpane text=true >}}
{{% tab "Example for @op" %}}
`@op` の例：
```python 
@op(
   out=Out(
       metadata={
           "wandb_artifact_configuration": {
               "name": "my_artifact",
               "type": "dataset",
           }
       }
   )
)
def create_dataset():
   return [1, 2, 3]
```
{{% /tab %}}
{{% tab "Example for @asset" %}}
`@asset` の例：
```python
@asset(
   name="my_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
   return [1, 2, 3]
```

@asset はすでに名前を持っているため、設定で name を渡す必要はありません。インテグレーション側で Artifact 名とアセット名は同じになります。

{{% /tab %}}
{{% tab "Example for @multi_asset" %}}

`@multi_asset` の例：

```python
@multi_asset(
   name="create_datasets",
   outs={
       "first_table": AssetOut(
           metadata={
               "wandb_artifact_configuration": {
                   "type": "training_dataset",
               }
           },
           io_manager_key="wandb_artifacts_manager",
       ),
       "second_table": AssetOut(
           metadata={
               "wandb_artifact_configuration": {
                   "type": "validation_dataset",
               }
           },
           io_manager_key="wandb_artifacts_manager",
       ),
   },
   group_name="my_multi_asset_group",
)
def create_datasets():
   first_table = wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
   second_table = wandb.Table(columns=["d", "e"], data=[[4, 5]])

   return first_table, second_table
```
{{% /tab %}}
{{< /tabpane >}}



対応プロパティ：
* `name`：（str）この Artifact の UI 上での識別名、および use_artifact 呼び出し時に参照する名称。プロジェクト内で一意である必要あり。@op では必須。
* `type`：（str）Artifact のタイプ。整理・区別用。dataset, model など。任意の文字列利用可。出力がすでに Artifact でない場合に必要。
* `description`：（str）Artifact の説明テキスト。UI で markdown レンダリングされるため、テーブルやリンクなどにも利用可。
* `aliases`：（list[str]）Artifact に適用したいエイリアス一覧。latest タグは自動追加されます（明記有無問わず）。モデルやデータセットのバージョン管理にも有効。
* [`add_dirs`]({{< relref path="/ref/python/sdk/classes/artifact#add_dir" lang="ja" >}})：（list[dict[str, Any]]）Artifact に含めたいローカルディレクトリーごとの設定。
* [`add_files`]({{< relref path="/ref/python/sdk/classes/artifact#add_file" lang="ja" >}})：（list[dict[str, Any]]）Artifact に含めたいローカルファイルごとの設定。
* [`add_references`]({{< relref path="/ref/python/sdk/classes/artifact#add_reference" lang="ja" >}})：（list[dict[str, Any]]）Artifact に含めたい外部参照ごとの設定。
* `serialization_module`：（dict）利用するシリアライズモジュールの設定。詳しくは Serialization セクション参照。
    * `name`：（str）シリアライズモジュールの名前。利用可能：`pickle`, `dill`, `cloudpickle`, `joblib`。導入済みである必要あり。
    * `parameters`：（dict[str, Any]）シリアライズ関数への追加引数。各モジュールの dump メソッドと同じパラメータを使用可。例：`{"compress": 3, "protocol": 4}`

応用例：

```python
@asset(
   name="my_advanced_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
           "description": "My *Markdown* description",
           "aliases": ["my_first_alias", "my_second_alias"],
           "add_dirs": [
               {
                   "name": "My directory",
                   "local_path": "path/to/directory",
               }
           ],
           "add_files": [
               {
                   "name": "validation_dataset",
                   "local_path": "path/to/data.json",
               },
               {
                   "is_tmp": True,
                   "local_path": "path/to/temp",
               },
           ],
           "add_references": [
               {
                   "uri": "https://picsum.photos/200/300",
                   "name": "外部 HTTP 画像参照",
               },
               {
                   "uri": "s3://my-bucket/datasets/mnist",
                   "name": "外部 S3 参照",
               },
           ],
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_advanced_artifact():
   return [1, 2, 3]
```



アセットはインテグレーション両側で有用なメタデータが付与されて materialize されます。
* W&B 側：ソースインテグレーション名・バージョン、利用 Python バージョン、ピックルプロトコルバージョン など
* Dagster 側：
    * Dagster Run ID
    * W&B Run：ID, name, path, URL
    * W&B Artifact：ID, name, type, version, size, URL
    * W&B Entity
    * W&B Project

次の画像は、W&B から取得したメタデータが Dagster アセットに追加された様子です。これはインテグレーションがなければ取得できません。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

下記画像は、W&B Artifact 側の設定がメタデータでどのように拡張されたかを示します。これは再現性・保守性に役立つ情報です。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}


{{% alert %}}
mypy などの静的型チェッカーをご利用の場合、下記のように型定義オブジェクトを import してください。

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### パーティション（partitions）の利用

インテグレーションは [Dagster の partition](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) にもネイティブ対応しています。

`DailyPartitionsDefinition` でパーティション分割された例は以下の通りです。

```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2023-01-01", end_date="2023-02-01"),
    name="my_daily_partitioned_asset",
    compute_kind="wandb",
    metadata={
        "wandb_artifact_configuration": {
            "type": "dataset",
        }
    },
)
def create_my_daily_partitioned_asset(context):
    partition_key = context.asset_partition_key_for_output()
    context.log.info(f"Creating partitioned asset for {partition_key}")
    return random.randint(0, 100)
```
このコードは、パーティションごとに 1 つの W&B Artifact を生成します。アセット名＋パーティションキー（例：`my_daily_partitioned_asset.2023-01-01` など）で Artifact パネル（UI）から確認できます。複数次元のパーティションはドット区切り（例：`my_asset.car.blue`）で表示されます。

{{% alert color="secondary" %}}
1 run で複数パーティションの materialize はできません。アセットごとに複数 run を実行して materialize してください。Dagit でのアセット materialize 時にも適用されます。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 応用例
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [Simple partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [Multi-partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [Advanced partitioned usage](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)


## W&B Artifacts の読み込み
W&B Artifacts の読み込みも書きこみと同様に設定します。`wandb_artifact_configuration` 辞書を `@op` や `@asset` の input 側に設定するのが唯一の違いです。

`@op` では [In](https://docs.dagster.io/_apidocs/ops#dagster.In) の metadata で明示的に Artifact 名を渡します。

`@asset` では [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) の In metadata で親アセットと同じ名前にするため Artifact 名は不要です。

外部で作成された Artifact に依存したい場合は [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) をご利用ください。常に最新版のアセットを読み込みます。

以下は様々な ops で Artifact を読む例です。

{{< tabpane text=true >}}
{{% tab "From an @op" %}}
`@op` からの Artifact 読み込み
```python
@op(
   ins={
       "artifact": In(
           metadata={
               "wandb_artifact_configuration": {
                   "name": "my_artifact",
               }
           }
       )
   },
   io_manager_key="wandb_artifacts_manager"
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```
{{% /tab %}}
{{% tab "Created by another @asset" %}}
別の `@asset` で作成された Artifact の読み込み
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # input 引数名を変更しない場合 'key' を省略可能
           key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

{{% /tab %}}
{{% tab "Artifact created outside Dagster" %}}

Dagster 外部で作成された Artifact の読み込み：

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B Artifact の名前
   description="Artifact created outside Dagster",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```
{{% /tab %}}
{{< /tabpane >}}


### 設定
この設定は、IO Manager がどのように入力を収集し、関数の input として渡すかを指定します。以下の read パターンに対応しています。

1. Artifact 内のネーム付きオブジェクトを取得したい場合は get を指定：

```python
@asset(
   ins={
       "table": AssetIn(
           key="my_artifact_with_table",
           metadata={
               "wandb_artifact_configuration": {
                   "get": "my_table",
               }
           },
           input_manager_key="wandb_artifacts_manager",
       )
   }
)
def get_table(context, table):
   context.log.info(table.get_column("a"))
```


2. Artifact 内のダウンロードファイルのローカルパスを取得したい場合は get_path：

```python
@asset(
   ins={
       "path": AssetIn(
           key="my_artifact_with_file",
           metadata={
               "wandb_artifact_configuration": {
                   "get_path": "name_of_file",
               }
           },
           input_manager_key="wandb_artifacts_manager",
       )
   }
)
def get_path(context, path):
   context.log.info(path)
```

3. Artifact オブジェクト全体（ローカルへ内容ダウンロード済）を取得する場合：
```python
@asset(
   ins={
       "artifact": AssetIn(
           key="my_artifact",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def get_artifact(context, artifact):
   context.log.info(artifact.name)
```


対応プロパティ：
* `get`：（str）Artifact 内の相対名で指定された W&B オブジェクトを取得
* `get_path`：（str）Artifact 内のファイルのローカルパスを取得

### シリアライズ設定（Serialization configuration）
デフォルトでは標準 [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使いますが、一部オブジェクトは非対応です（特に yield 付き関数などはエラーになります）。

さらに、[dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib) といった Pickle 系モジュールや、用途次第で [ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) なども利用可能です（シリアライズ済み文字列などで対応）。用途に応じて最適な方法を文献等で調査してください。

### Pickle 系のシリアライズモジュール

{{% alert color="secondary" %}}
ピックルにはセキュリティリスクがあります。安全性が必要な場合は W&B オブジェクトのみご利用を推奨します。データ署名やハッシュ管理など、より高度な用途はご相談ください。
{{% /alert %}}

`wandb_artifact_configuration` の `serialization_module` 辞書で利用モジュールを選択可能です。Dagster の実行環境に該当モジュールを導入してください。

インテグレーションは Artifact 読み込み時にモジュールを自動判別します。

現時点で対応しているのは、`pickle`, `dill`, `cloudpickle`, `joblib` です。

下記は joblib でシリアライズした “モデル” を作成し、それを推論に使う例です。

```python
@asset(
    name="my_joblib_serialized_model",
    compute_kind="Python",
    metadata={
        "wandb_artifact_configuration": {
            "type": "model",
            "serialization_module": {
                "name": "joblib"
            },
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_model_serialized_with_joblib():
    # 実際の ML モデルではないですが、pickle では不可な例です
    return lambda x, y: x + y

@asset(
    name="inference_result_from_joblib_serialized_model",
    compute_kind="Python",
    ins={
        "my_joblib_serialized_model": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        )
    },
    metadata={
        "wandb_artifact_configuration": {
            "type": "results",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def use_model_serialized_with_joblib(
    context: OpExecutionContext, my_joblib_serialized_model
):
    inference_result = my_joblib_serialized_model(1, 2)
    context.log.info(inference_result)  # 3 を出力
    return inference_result
```

### 高度なシリアライズ形式（ONNX, PMML）
ONNX や PMML のような交換ファイルフォーマットは一般的です。これらにインテグレーションも対応していますが、Pickle 系より少し追加作業が必要です。

利用方法は大きく2つ：
1. 選択したフォーマットでモデルを変換し、文字列として返すだけ。インテグレーションがそのままピックルします。受け取り側で復元可能です。
2. シリアライズしたファイルをローカルに保存し、add_file 設定でカスタム Artifact を作成。

Scikit-learn モデルを ONNX でシリアライズする例：

```python
import numpy
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dagster import AssetIn, AssetOut, asset, multi_asset

@multi_asset(
    compute_kind="Python",
    outs={
        "my_onnx_model": AssetOut(
            metadata={
                "wandb_artifact_configuration": {
                    "type": "model",
                }
            },
            io_manager_key="wandb_artifacts_manager",
        ),
        "my_test_set": AssetOut(
            metadata={
                "wandb_artifact_configuration": {
                    "type": "test_set",
                }
            },
            io_manager_key="wandb_artifacts_manager",
        ),
    },
    group_name="onnx_example",
)
def create_onnx_model():
    # 元ネタ: https://onnx.ai/sklearn-onnx/

    # モデルを学習
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 形式に変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # Artifact 書き込み (model + test_set)
    return onx.SerializeToString(), {"X_test": X_test, "y_test": y_test}

@asset(
    name="experiment_results",
    compute_kind="Python",
    ins={
        "my_onnx_model": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        ),
        "my_test_set": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        ),
    },
    group_name="onnx_example",
)
def use_onnx_model(context, my_onnx_model, my_test_set):
    # 元ネタ: https://onnx.ai/sklearn-onnx/

    # ONNX Runtime で予測
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### パーティションの利用

Dagster の [partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) にネイティブ対応。

特定のパーティションのみ・複数・全パーティションいずれも選択して読み込み可。

全パーティションは dict で渡され、キー・値はそれぞれパーティションキーと Artifact 内容です。

{{< tabpane text=true >}}
{{% tab "全パーティションを読む" %}}
上流 `@asset` の全パーティションを dict 形式で受け取ります。キー、値はパーティションキーと Artifact 内容です。
```python
@asset(
    compute_kind="wandb",
    ins={"my_daily_partitioned_asset": AssetIn()},
    output_required=False,
)
def read_all_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
{{% /tab %}}
{{% tab "特定パーティションを読む" %}}
`AssetIn` の `partition_mapping` 設定で、特定パーティションだけ選択可能。ここでは `TimeWindowPartitionMapping` の例です。
```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2023-01-01", end_date="2023-02-01"),
    compute_kind="wandb",
    ins={
        "my_daily_partitioned_asset": AssetIn(
            partition_mapping=TimeWindowPartitionMapping(start_offset=-1)
        )
    },
    output_required=False,
)
def read_specific_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
{{% /tab %}}
{{< /tabpane >}}

`metadata` オブジェクトで、W&B がプロジェクト内の各 Artifact パーティションをどう扱うか設定できます。

`metadata` オブジェクトには `wandb_artifact_configuration` キー、その中にさらに `partitions` オブジェクトがあります。

`partitions` オブジェクトは各パーティション名と設定内容のマッピングです。ここには `get`, `version`, `alias` などが指定できます。

**設定キー**

1. `get`:
W&B オブジェクト（Table, Image など）の名前を指定
2. `version`:
特定 Artifact バージョン取得時に使用
3. `alias`:
エイリアスで Artifact を取得する際に使用

**ワイルドカード設定**

ワイルドカード `"*"` は明示設定されていない全パーティションのデフォルト設定となります。

例：

```python
"*": {
    "get": "default_table_name",
},
```
この場合、個別設定のないパーティションは `default_table_name` のテーブルからデータを取得します。

**特定パーティションの個別設定**

該当パーティションのキーで個別上書き可能です。

例：

```python
"yellow": {
    "get": "custom_table_name",
},
```

「yellow」パーティションは `custom_table_name` からデータを取得します（ワイルドカード設定より優先）。

**バージョン/エイリアス指定**

バージョン・エイリアス活用には `version`, `alias` キーで指定します。

バージョン例：

```python
"orange": {
    "version": "v0",
},
```

この設定で `orange` パーティションの `v0` バージョンを取得。

エイリアス例：

```python
"blue": {
    "alias": "special_alias",
},
```

この場合、「blue」エイリアスの Artifact パーティションから `default_table_name` テーブルのデータを取得します。

### 応用例
より高度な利用例は以下を参照してください。
* [高度な asset 利用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py) 
* [パーティション化ジョブの例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルを Model Registry にリンクする例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)


## W&B Launch の利用

{{% alert color="secondary" %}}
ベータ開発中プロダクト
Launch に興味のある方は、W&B Launch のカスタマーパイロットプログラム参加についてアカウントチームまでお問い合わせください。
パイロットには AWS EKS または SageMaker のご利用環境が必要です。将来的には他プラットフォームも対応予定です。
{{% /alert %}}

続ける前に、W&B Launch の基本を理解しておくことをおすすめします。[Launch ガイド]({{< relref path="/launch/" lang="ja" >}}) をご覧ください。

Dagster インテグレーションでできること：
* Dagster インスタンスで 1つ以上の Launch agent を実行
* ローカルで Launch ジョブを実行
* オンプレまたはクラウドでの Launch ジョブの実行

### Launch agent
インテグレーションは `run_launch_agent` という import 可能な `@op` を提供します。これは Launch Agent を開始し、手動で停止されるまで長時間稼働プロセスとして動きます。

Agent は launch キューをポーリングし、ジョブを実行（または外部サービスへディスパッチ）します。

詳しくは [Launch ページ]({{< relref path="/launch/" lang="ja" >}}) をご覧ください。

Launchpad 内で全プロパティの説明確認も可能です。

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

シンプルな例：
```python
# この内容を config.yaml に追加
# または Dagit の Launchpad や JobDefinition.execute_in_process で設定可能
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # W&B entity に置き換えてください
     project: my_project # W&B project に置き換えてください
ops:
 run_launch_agent:
   config:
     max_jobs: -1
     queues: 
       - my_dagster_queue

from dagster_wandb.launch.ops import run_launch_agent
from dagster_wandb.resources import wandb_resource

from dagster import job, make_values_resource

@job(
   resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
   },
)
def run_launch_agent_example():
   run_launch_agent()
```

### Launch ジョブ
`run_launch_job` という import 可能な `@op` も用意されています。これで Launch ジョブを実行します。

Launch ジョブは、実行するためにキューに割り当てられます。キューは新規作成またはデフォルトでもOK。該当キューにアクティブな agent がいることを確認してください。Dagster インスタンス内で agent を動作させるか、Kubernetes でデプロイ型 agent をご検討ください。

詳しくは [Launch ページ]({{< relref path="/launch/" lang="ja" >}}) をご覧ください。

Launchpad 内でも全プロパティの説明を確認可能です。

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}


シンプルな例：
```python
# この内容を config.yaml に追加
# または Dagit の Launchpad や JobDefinition.execute_in_process でも設定可能
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # W&B entity に置き換えてください
     project: my_project # W&B project に置き換えてください
ops:
 my_launched_job:
   config:
     entry_point:
       - python
       - train.py
     queue: my_dagster_queue
     uri: https://github.com/wandb/example-dagster-integration-with-launch


from dagster_wandb.launch.ops import run_launch_job
from dagster_wandb.resources import wandb_resource

from dagster import job, make_values_resource


@job(resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
   },
)
def run_launch_job_example():
   run_launch_job.alias("my_launched_job")() # job 名をエイリアスでリネーム
```

## ベストプラクティス

1. Artifacts の読み書きには IO Manager を利用しましょう。  
[`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact#download" lang="ja" >}}) や [`Run.log_artifact()`]({{< relref path="/ref/python/sdk/classes/run#log_artifact" lang="ja" >}}) を直接使わないでください。これらはインテグレーション側で適切に処理されます。保存したいデータを戻り値として返せば、管理やリネージ（系譜）も良好に保てます。

2. 複雑なケース以外では自分で Artifact オブジェクトを作らずに済ませましょう。
Python オブジェクトや W&B オブジェクトを ops/assets で返せば十分です。複雑な場合は Artifact オブジェクトを直接渡しても良いですが、その際もインテグレーションがソースやバージョンなどのメタデータを自動で付与します。

3. Artifacts にファイル、ディレクトリ、外部参照を追加する場合は metadata（`wandb_artifact_configuration`）を活用しましょう。  
Amazon S3, GCS, HTTP などのリファレンス追加は [Artifact の設定セクション]({{< relref path="#configuration-1" lang="ja" >}}) の応用例を参照。

4. Artifact を生成する場合は @asset の利用を推奨します。
Artifact は asset です。Dagster が asset を管理する場合も @op より @asset が推奨されます。これにより Dagit の Asset Catalog で管理性・可視性も向上します。

5. Dagster 外部で作成された Artifact を利用するには SourceAsset を使いましょう。
これによりインテグレーションの力を活かせます。SourceAsset を使わない場合、インテグレーションで作成された Artifact のみ利用できます。

6. 大規模モデルなど専用計算リソースでトレーニングする場合は W&B Launch を活用しましょう。
小規模なモデルは Dagster クラスター内で直接動かせますが、大規模モデルは Launch を活用すると負荷分散や適切な計算リソース確保ができます。

7. Dagster で実験管理時、W&B Run ID を Dagster Run ID に設定しましょう。
[Run の Resumable 設定]({{< relref path="/guides/models/track/runs/resuming.md" lang="ja" >}}) に加えて、W&B Run ID を Dagster Run ID または任意文字列に設定することを推奨します。これで W&B メトリクスおよび W&B Artifacts が同じ Run に紐付き、管理も一元化できます。

W&B Run ID を Dagster Run ID にセット：
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または任意の ID を IO Manager 設定に渡す方法も可能：
```python
wandb.init(
    id="my_resumable_run_id",
    resume="allow",
    ...
)

@job(
   resource_defs={
       "io_manager": wandb_artifacts_io_manager.configured(
           {"wandb_run_id": "my_resumable_run_id"}
       ),
   }
)
```

8. 特に大きな W&B Artifacts の場合は get や get_path で必要なデータだけ取得しましょう。
デフォルトでは Artifact 全体をダウンロードしますが、必要なファイルやオブジェクトだけの取得で効率・速度ともに向上します。

9. Python オブジェクト利用時は用途に応じて最適なピックルモジュールを選定しましょう。
デフォルトは [pickle](https://docs.python.org/3/library/pickle.html) ですが、一部オブジェクト（yield 付き関数等）は非対応。W&B は他にも [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib) などをサポートしています。

ONNX や PMML などの高度なシリアライズ形式も、用途によっては（シリアライズ済み文字列や Artifact の直接作成で）活用可能です。最適な選択肢はユースケースごとにご検討ください。

