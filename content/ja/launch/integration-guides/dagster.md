---
title: Dagster
description: W&B を Dagster と統合するためのガイド。
menu:
  launch:
    identifier: dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster と W&B（W&B）を利用して MLOps パイプラインのオーケストレーションや ML アセットの管理を行うことができます。W&B とのインテグレーションにより、Dagster 内で次のことが簡単に行えます：

* [W&B Artifact]({{< relref "/guides/core/artifacts/" >}}) の作成と利用
* [W&B Registry]({{< relref "/guides/core/registry/" >}}) の Registered Models の使用・作成
* [W&B Launch]({{< relref "/launch/" >}}) を使った専用コンピュートでのトレーニングジョブ実行
* ops や assets で [wandb]({{< relref "/ref/python/" >}}) クライアントの利用

W&B Dagster インテグレーションは、W&B 専用の Dagster リソースと IO Manager を提供します：

* `wandb_resource`: W&B API への認証と通信を行うための Dagster リソース
* `wandb_artifacts_io_manager`: W&B Artifacts を取り扱う Dagster IO Manager

このガイドでは、Dagster で W&B を使うための前提条件の準備、ops や assets での W&B Artifacts の作成・利用方法、W&B Launch の利用手順、おすすめのベストプラクティスについて説明します。

## はじめに
W&B で Dagster を利用するには、以下のリソースが必要です：
1. **W&B API Key**
2. **W&B entity（ユーザーまたはチーム）**：Entity は、W&B Runs や Artifacts を送るためのユーザー名やチーム名です。run を記録する前に、W&B App UI でアカウントまたはチーム entity を作成しておきましょう。entity を指定しなかった場合、run はデフォルト entity（通常は自身のユーザ名）に送信されます。デフォルト entity は **Project Defaults** の設定から変更可能です。
3. **W&B project**：[W&B Runs]({{< relref "/guides/models/track/runs/" >}}) が保存されるプロジェクト名

W&B entity は、W&B App のユーザーまたはチームのプロフィールページで確認できます。既存の W&B project を利用することも、新しく作成することも可能です。新規プロジェクトは W&B App のホームページやユーザー／チームのプロフィールページから作成できます。プロジェクトが存在しない場合、初回利用時に自動で作成されます。API キーの取得方法は以下の通りです。

### API キーの取得方法
1. [W&B にログイン](https://wandb.ai/login) します。注：W&B Server を使う場合は、管理者にインスタンスのホスト名を確認してください。
2. [認証ページ](https://wandb.ai/authorize) またはユーザー／チームの設定から API キーを取得します。プロダクション環境では、そのキーを所有する [サービスアカウント]({{< relref "/support/kb-articles/service_account_useful.md" >}}) の利用を推奨します。
3. API キーを環境変数として設定します。例：`export WANDB_API_KEY=YOUR_KEY`

以降の例では、Dagster のコード内で API キーを指定する箇所を示しています。エンティティやプロジェクト名は、`wandb_config` のネストされた辞書内で指定してください。異なる ops/assets で異なる `wandb_config` を使えば、別の W&B Project を利用することもできます。設定可能なキーの詳細は下記 Configuration セクションをご覧ください。

{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
例：`@job` 用の設定
```python
# これを config.yaml に追加します
# または Dagit の Launchpad や JobDefinition.execute_in_process で設定も可能です
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

例：assets を使った `@repository` の設定

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
                   {"cache_duration_in_minutes": 60} # ファイルは1時間だけキャッシュ
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # W&B entity に置換
                       "project": "my_project", # W&B project に置換
                   }
               }
           },
       ),
   ]
```
この例では IO Manager のキャッシュ時間を設定している点が、`@job` の例と異なります。
{{% /tab %}}
{{< /tabpane >}}

### 設定
このセクションで紹介する設定オプションは、インテグレーションによる W&B 専用 Dagster リソースと IO Manager の設定方法です。

* `wandb_resource`: W&B API との通信を行う Dagster [resource](https://docs.dagster.io/concepts/resources)。指定した API キーで自動的に認証を行います。プロパティ：
    * `api_key`: (必須, str) W&B API 用の API キー
    * `host`: (任意, str) 利用したい API ホストサーバー。W&B Server 使用時のみ必須。デフォルトは Public Cloud の `https://api.wandb.ai`
* `wandb_artifacts_io_manager`: W&B Artifacts を取り扱う Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ：
    * `base_dir`: (任意, int) ローカル保存およびキャッシュに使用するベースディレクトリ。Artifacts や Run ログが書き込まれ／読み込まれます。デフォルトは `DAGSTER_HOME`
    * `cache_duration_in_minutes`: (任意, int) Artifacts や Run ログをローカル保持する期間。指定時間開かれていないファイル／ディレクトリはキャッシュから削除。キャッシュ削除は IO Manager 実行後に行われます。完全にキャッシュを無効にする場合は 0 に設定可能。キャッシュは同一マシン上でジョブがアーティファクトを使いまわす場合に高速化に寄与します（デフォルト：30日）。
    * `run_id`: (任意, str) 再開用の unique な Run ID。プロジェクト内で一意で、run を削除した場合は再利用不可。短い説明名には name フィールドか設定を利用。ID には `/\#?%:..` 等の記号は不可。Dagster で実験管理をする場合は Run ID の指定が必要です。デフォルトは Dagster Run ID 例：`7e4df022-1bf2-44b5-a383-bb852df4077e`
    * `run_name`: (任意, str) UI で判別しやすい短い表示名。デフォルトは `dagster-run-[Dagster Run ID の先頭8文字]` 例：`dagster-run-7e4df022`
    * `run_tags`: (任意, list[str]) run に付けるタグのリスト。タグは run の整理や一時的な "baseline" や "production" などのラベルに便利。UI での付け外しや特定タグでのフィルターも簡単。利用する run には自動で `dagster_wandb` タグが追加されます。

## W&B Artifacts の利用

W&B Artifact とのインテグレーションは Dagster IO Manager を利用します。

[IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) は資産（asset）や op の出力を保存し、下流の asset や op への入力としてそれをロードする責任を持つユーザー定義オブジェクトです。たとえば IO Manager はファイルシステム上のファイルからオブジェクトを保存・ロードする場合があります。

このインテグレーションでは W&B Artifacts 用の IO Manager を提供しており、Dagster の `@op` や `@asset` で W&B Artifacts をネイティブに作成・利用できます。下記は Python リストを含むデータセット型 W&B Artifact を生成する `@asset` の例です。

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
    return [1, 2, 3] # このデータが Artifact として保存されます
```

`@op`, `@asset`, `@multi_asset` には metadata 設定を使って Artifact の書き込みができます。外部で作成された W&B Artifacts も同様に消費可能です。

## W&B Artifacts の書き込み
続けて進む前に、W&B Artifacts の使い方を理解しておくことをおすすめします。[Artifacts ガイド]({{< relref "/guides/core/artifacts/" >}}) をご参照ください。

Python 関数からオブジェクトを返すことで W&B Artifact を作成できます。W&B でサポートされるオブジェクトは以下です：
* Python オブジェクト（int, dict, list…）
* W&B オブジェクト（Table, Image, Graph…）
* W&B Artifact オブジェクト

以下に Dagster asset（`@asset`）で W&B Artifacts を書き込む例を示します。

{{< tabpane text=true >}}
{{% tab "Python objects" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズ可能なものであれば、インテグレーションによる Artifact にピクル化されて追加されます。Dagster 内で Artifact を読み込む際にアンピクルされます（詳細は [Read artifacts]({{< relref "#read-wb-artifacts" >}}) 参照）。

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

W&B は複数の Pickle 系シリアライズモジュール（[pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)）に対応しています。より高度なシリアライズ（[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) など）も利用可能です。詳細は[シリアライズ設定]({{< relref "#serialization-configuration" >}}) セクションを参照してください。
{{% /tab %}}
{{% tab "W&B Object" %}}
[Table]({{< relref "/ref/python/sdk/data-types/table.md" >}}) や [Image]({{< relref "/ref/python/sdk/data-types/image.md" >}}) など、どんな W&B オブジェクトでも Artifact に追加できます。この例では Table を Artifact に追加しています。

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

より複雑な用途では、自分で Artifact オブジェクトを作成する場合もあります。この場合も両側でのメタデータ付与など便利な機能は利用できます。

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
`@op`, `@asset`, `@multi_asset` それぞれに `wandb_artifact_configuration` という辞書型の設定が可能です。この辞書はデコレータの metadata 引数として渡します。この設定は W&B Artifacts の IO Manager の読み書きを制御する際に必須です。

`@op` では、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) の metadata 引数に設定します。
`@asset` では、asset の metadata 引数に設定します。
`@multi_asset` では、各出力の [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) の metadata 引数に指定します。

以下のコード例では、`@op`、`@asset`、`@multi_asset` での設定方法を示します：

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

@asset はすでに名前を持つため、設定で name を指定する必要はありません。インテグレーションが Artifact 名として asset 名を設定します。

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
* `name`: (str) このアーティファクトの識別名。UI や use_artifact 呼び出しで参照する際に利用。プロジェクト内で一意である必要があります。`@op` では必須
* `type`: (str) アーティファクトの型（分類のために利用）。一般的に "dataset" や "model" などを利用しますが、任意の文字列（英数字、アンダースコア、ハイフン、ドット）も可。出力がすでに Artifact でない場合必須
* `description`: (str) アーティファクトの説明。UI では Markdown 表示
* `aliases`: (list[str]) アーティファクトに付与したいエイリアス。自動的に `latest` タグも付与されます。バージョン管理などに便利です
* [`add_dirs`]({{< relref "/ref/python/sdk/classes/artifact#add_dir" >}}): (list[dict[str, Any]]) Artifact に含めたいローカルディレクトリの設定
* [`add_files`]({{< relref "/ref/python/sdk/classes/artifact#add_file" >}}): (list[dict[str, Any]]) Artifact に含めたいローカルファイルの設定
* [`add_references`]({{< relref "/ref/python/sdk/classes/artifact#add_reference" >}}): (list[dict[str, Any]]) Artifact に含めたい外部参照（Amazon S3 等）の設定
* `serialization_module`: (dict) シリアライズモジュールの設定。詳細はシリアライズ設定セクション参照
    * `name`: (str) 使用するモジュール名。`pickle`, `dill`, `cloudpickle`, `joblib` のいずれか
    * `parameters`: (dict[str, Any]) シリアライズ関数への追加引数。例：`{"compress": 3, "protocol": 4}`

やや高度な例：

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
                   "name": "External HTTP reference to an image",
               },
               {
                   "uri": "s3://my-bucket/datasets/mnist",
                   "name": "External S3 reference",
               },
           ],
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_advanced_artifact():
   return [1, 2, 3]
```

この asset は、インテグレーションの両側で便利なメタデータが付与されて materialize されます：
* W&B 側: ソース統合名とバージョン、Python バージョン、pickle プロトコルバージョンなど
* Dagster 側:
    * Dagster Run ID
    * W&B Run: ID, 名前, パス, URL
    * W&B Artifact: ID, 名前, 型, バージョン, サイズ, URL
    * W&B Entity
    * W&B Project

次の画像は、W&B 側のメタデータが Dagster asset に追加された例です。インテグレーションがなければ得られない情報です。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

続く画像は、設定内容が W&B Artifact 側でどのように有益なメタデータで強化されたかを示します。再現性や保守性の向上に寄与します。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
mypy などの静的型チェッカーを使う場合、次のように型定義オブジェクトをインポートしてください。

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### パーティションの利用

このインテグレーションは [Dagster のパーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)をネイティブサポートしています。

DailyPartitionsDefinition を利用したパーティション例：
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
このコードは、各パーティションごとに 1 つの W&B Artifact を生成します。Artifacts は asset 名＋パーティションキーで UI のアーティファクトパネルに表示されます（例：`my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02` など）。複数の次元でパーティションするとドット区切りで表示されます（例： `my_asset.car.blue`）。

{{% alert color="secondary" %}}
このインテグレーションでは 1 run で複数パーティションの materialize はできません。asset を materialize する場合は複数回 run を実行してください。Dagit で実行することが可能です。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 応用例
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [シンプルなパーティション asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [多次元パーティション asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [応用パーティション例](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts の読み込み
W&B Artifacts の読み込み方法も書き込みとほぼ同様です。`wandb_artifact_configuration` 辞書を `@op` または `@asset` に設定します。唯一の違いは入出力先が異なり、読み込みの場合は入力側に設定します。

`@op` では、[In](https://docs.dagster.io/_apidocs/ops#dagster.In) の metadata に設定します。Artifact の名前指定が必要です。

`@asset` では、入力 metadata に [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In metadata を指定します。この場合は従属する asset の名前を使うので Artifact 名を渡さないでください。

インテグレーション外で作られた Artifact に依存したい場合は [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を利用します。常にその asset の最新バージョンを読み込みます。

以下の例では、さまざまなパターンで Artifact を読み込む方法を示します。

{{< tabpane text=true >}}
{{% tab "From an @op" %}}
`@op` から Artifact を読む例
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
別の `@asset` が作成した Artifact の読み込み
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 引数名変更しない場合 'key' を省略可能
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

Dagster 外で作成された Artifact の読み込み：

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
この設定は、どの IO Manager がどの入力に何を提供するかを指定するものです。次の読み込みパターンをサポートしています。

1. アーティファクトから名前付きオブジェクトを取得する
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

2. ダウンロードしたファイルのローカルパスを取得する
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

3. Artifact オブジェクト全体（ローカルでダウンロード済み）を取得する:
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

対応プロパティ
* `get`: (str) アーティファクト内の W&B オブジェクト名で特定オブジェクトを取得
* `get_path`: (str) アーティファクト内ファイルのローカルパスを取得

### シリアライズ設定
デフォルトでは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使いますが、一部のオブジェクト（yield を含む関数など）は pickle できません。

W&B では他にも Pickle 系モジュール（[dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)）に対応しています。また、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などの高度な形式も、シリアライズ文字列を返すか Artifact を直接生成することで利用できます。目的やユースケースに適したものを選択してください。

### Pickle 系シリアライズモジュール

{{% alert color="secondary" %}}
Pickle にはセキュリティ上のリスクがあります。セキュリティ要件がある場合は W&B オブジェクトのみを利用してください。データの署名やハッシュキー管理などご検討ください。複雑なユースケースはサポートまでご相談ください。
{{% /alert %}}

`wandb_artifact_configuration` 内の辞書 `serialization_module` で使用モジュールを設定します。Dagster を稼働させるマシンにそのモジュールをインストールしておきましょう。

Artifact 読み込み時は正しいシリアライズモジュールが自動判別されます。

対応モジュール：`pickle`, `dill`, `cloudpickle`, `joblib`

以下は "model" を joblib でシリアライズし、その後推論に利用する簡易例です。

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
    # 本物の ML モデルではありませんが pickle では不可能な例
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
    context.log.info(inference_result)  # 出力結果: 3
    return inference_result
```

### 高度なシリアライズ形式（ONNX, PMML）
ONNX や PMML などの交換用ファイル形式もよく使われます。インテグレーションでこれらもサポートしていますが、Pickle 系より少し手順が増えます。

利用方法は2通りあります。
1. モデルを対象フォーマットに変換し、その文字列表現を普通の Python オブジェクトとして返します。インテグレーションはその文字列を pickle します。再構築時はその文字列からモデルを復元します。
2. シリアライズ済みモデルをローカルファイルに保存し、そのファイルを add_file 設定でカスタム Artifact として追加します。

下記は Scikit-learn モデルを ONNX でシリアライズする例です。

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
    # https://onnx.ai/sklearn-onnx/ を参考

    # モデル学習
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX フォーマットに変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクト（モデル＋テストセット）保存
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
    # https://onnx.ai/sklearn-onnx/ を参考

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

このインテグレーションは [Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)をネイティブサポートしています。

1つ、複数、または全パーティションを選択的に読み込むことができます。

全パーティションは辞書として渡され、キーにパーティションキー、値に対応する Artifact コンテンツが対応します。

{{< tabpane text=true >}}
{{% tab "Read all partitions" %}}
上流 `@asset` の全パーティションを読み込む例。各パーティションキーと内容が辞書で渡されます。
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
{{% tab "Read specific partitions" %}}
`AssetIn` の `partition_mapping` 設定で特定パーティションを指定可。この例では `TimeWindowPartitionMapping` を利用。
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

設定オブジェクト `metadata` で W&B がプロジェクト内の各アーティファクトパーティションをどう扱うかを指定可能です。

`metadata` の中に `wandb_artifact_configuration` キーがあり、その中で `partitions` オブジェクトが使えます。

`partitions` オブジェクトは、各パーティション名とその設定をマッピングします。各設定では `get`, `version`, `alias` 等が使えます。

**設定用キー**

1. `get`:
`get` キーは、データ取得に使う W&B オブジェクト名を指定します。

2. `version`:
`version` キーはアーティファクトの特定バージョン取得時に利用します。

3. `alias`:
`alias` キーでエイリアスによるアーティファクト取得ができます。

**ワイルドカード設定**

`"*"` は、その他未指定パーティションすべてに適用されるデフォルト設定です。

例：

```python
"*": {
    "get": "default_table_name",
},
```
明示されていないすべてのパーティションで `default_table_name` からデータを取得する設定です。

**特定パーティションの設定**

ワイルドカード設定を上書きしたい場合は個別にキー指定で設定します。

例：

```python
"yellow": {
    "get": "custom_table_name",
},
```
`yellow` パーティションのみ、`custom_table_name` からデータを取得します。

**バージョン管理やエイリアス指定**

特定バージョンやエイリアスも設定できます。

バージョン例：

```python
"orange": {
    "version": "v0",
},
```
`orange` パーティションのバージョン `v0` からデータを取得。

エイリアス例：

```python
"blue": {
    "alias": "special_alias",
},
```
`blue` パーティションの `special_alias` エイリアスで Artifact（`default_table_name` テーブル）を取得。

### 応用例
応用的な使い方を知りたい場合は以下のコード例をご参照ください:
* [asset 用応用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [Partitioned job の例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルを Model Registry にリンク](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch の利用

{{% alert color="secondary" %}}
ベータ版製品で現在開発中
Launch のご利用にご興味がある場合は、担当アカウントチームまでご相談いただき W&B Launch のカスタマーパイロットプログラムへの参加についてお話しください。
パイロット参加には AWS EKS または SageMaker の利用が条件です。順次サポートプラットフォームは拡大予定です。
{{% /alert %}}

進む前に [Launch ガイド]({{< relref "/launch/" >}}) を読むことをおすすめします。

Dagster インテグレーションは次の作業に役立ちます：
* Dagster インスタンスでの複数 Launch エージェントの運用
* Dagster インスタンス内でのローカル Launch ジョブの実行
* オンプレミスまたはクラウドでのリモート Launch ジョブ実行

### Launch エージェント
このインテグレーションでは `run_launch_agent` という import 可能な `@op` を提供しています。Launch エージェントを起動し、手動停止までプロセスが継続されます。

エージェントは Launch のジョブキューを監視し、ジョブを順次実行（または外部サービスにディスパッチ）するプロセスです。

詳細は [Launch ページ]({{< relref "/launch/" >}}) 参照。

Launchpad では各プロパティの説明も確認できます。

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

簡単な例
```python
# これを config.yaml に追加
# または Dagit の Launchpad や JobDefinition.execute_in_process で設定
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # W&B entity に置換
     project: my_project # W&B project に置換
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
`run_launch_job` という import 可能な `@op` で Launch ジョブを実行します。

Launch ジョブはキューに割り当てられます。デフォルトキューを使うか、キューを作成してください。必ずそのキューに対応するアクティブなエージェントを用意すること。エージェントは Dagster インスタンス内、または Kubernetes などのデプロイ可能エージェントでも可。

詳細は [Launch ページ]({{< relref "/launch/" >}}) 参照。

Launchpad にはすべてのプロパティ説明があります。

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}

簡単な例
```python
# これを config.yaml に追加
# または Dagit の Launchpad や JobDefinition.execute_in_process で設定
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # W&B entity に置換
     project: my_project # W&B project に置換
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
   run_launch_job.alias("my_launched_job")() # ジョブ名をエイリアスでリネーム
```

## ベストプラクティス

1. Artifacts の読み書きは IO Manager を使いましょう。
[`Artifact.download()`]({{< relref "/ref/python/sdk/classes/artifact#download" >}}) や [`Run.log_artifact()`]({{< relref "/ref/python/sdk/classes/run#log_artifact" >}}) を直接呼ぶのは避けてください。これらのメソッドはインテグレーションが管理するので、保存したいデータを返すだけでOKです。この方法だと Artifacts のリネージが明確になります。

2. 複雑な用途を除き、自前で Artifact オブジェクトを生成しない
Python オブジェクトや W&B オブジェクトを ops/assets から返してください。インテグレーション側で自動的に Artifact 化します。より複雑な要件がある場合だけ Artifact オブジェクトを自作し、その際はメタデータ（統合名・バージョンやピクルバージョン等）が enrich されます。

3. ファイル・ディレクトリ・外部参照は metadata 設定で Artifacts に追加
ファイルやディレクトリ、外部（S3、GCS、HTTPなど）参照は `wandb_artifact_configuration` オブジェクトで簡単に追加できます。詳しくは [Artifact 設定例]({{< relref "#configuration-1" >}}) の advanced example 参照。

4. Artifact を出力する場合は @op よりも @asset を推奨
Artifact は asset です。Dagster で asset として管理することで、Dagit の Asset Catalog での可視性が大きく向上します。

5. Dagster 外部で作られた Artifact 利用には SourceAsset を使う
これにより統合のメリットを活かし、外部 Artifact も読み込めます。統合で作成した Artifact でなければ使えないという制限を回避できます。

6. 大規模モデルのトレーニングには W&B Launch で専用環境を使う
小規模モデルは Dagster クラスタ内でトレーニングでき、GPU ノードのある Kubernetes クラスタ上の Dagster でも運用できます。大規模モデルには W&B Launch で分散トレーニングするのが推奨です。これでインスタンスへの過負荷を防ぎ、最適な計算リソースを使えます。

7. Dagster で実験管理する際、W&B Run ID を Dagster Run ID に設定する
[Run の再開機能]({{< relref "/guides/models/track/runs/resuming.md" >}}) を有効にし、W&B Run ID を Dagster Run ID か任意文字列にするのが推奨です。これにより W&B の metrics や Artifacts が同じ W&B Run にまとめて保存でき、再現性も向上します。

W&B Run ID を Dagster Run ID に設定する例：
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または独自の W&B Run ID を IO Manager 設定で渡す場合：
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

8. 大きな W&B Artifact では get や get_path で必要なデータだけを収集
デフォルトでは Artifact 全体がダウンロードされます。巨大な Artifact 利用時は必要なファイルやオブジェクトだけを取得すると高速化でき、リソースも節約できます。

9. Python オブジェクトの場合、ユースケースに応じて pickling モジュールを選択
W&B インテグレーションのデフォルトは [pickle](https://docs.python.org/3/library/pickle.html) ですが、yield を含む関数など一部は pickle できません。他の Pickle 系モジュール（[dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)）の利用も可能です。

さらに、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) など高度なシリアライズも、シリアライズした文字列の返却または Artifact 作成で実現できます。どの方法が適しているかはユースケースで異なるため、関連文献等もご参考ください。