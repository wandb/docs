---
title: Dagster
description: W&B を Dagster と統合する方法のガイド.
menu:
  launch:
    identifier: ja-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster と W&B (W&B) を使用して、MLOps パイプラインをオーケストレーションし、ML 資産を管理します。W&B との統合により、Dagster 内で以下のことが簡単にできます：

* [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の使用と作成。
* [W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) での Registered Models の使用と作成。
* [W&B Launch]({{< relref path="/launch/" lang="ja" >}}) を使用して専用のコンピュートでトレーニングジョブを実行します。
* ops とアセットで [wandb]({{< relref path="/ref/python/" lang="ja" >}}) クライアントを使用。

W&B Dagster の統合は、W&B 専用の Dagster リソースと IO マネージャーを提供します。

* `wandb_resource`: W&B API に認証して通信するために使用される Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するために使用される Dagster IO マネージャー。

次のガイドでは、Dagster で W&B を使用するための事前条件を満たす方法、ops とアセットで W&B Artifacts を作成して使用する方法、W&B Launch を使用する方法、および推奨されるベストプラクティスを示しています。

## 始める前に
Weights and Biases 内で Dagster を使用するために次のリソースが必要です：
1. **W&B API Key**。
2. **W&B エンティティ (ユーザーまたはチーム)**: エンティティとは、W&B Runs と Artifacts を送信するユーザー名またはチーム名です。run を記録する前に、W&B アプリ UI でアカウントまたはチームエンティティを作成することを確認してください。エンティティを指定しない場合、run は通常ユーザー名であるデフォルトのエンティティに送信されます。デフォルトのエンティティは、**Project Defaults** の設定で変更できます。
3. **W&B プロジェクト**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) が保存されているプロジェクトの名前。

W&B アプリのユーザーまたはチームのプロファイル ページを確認することで、W&B エンティティを見つけることができます。既存の W&B プロジェクトを使用したり、新しいプロジェクトを作成することができます。新しいプロジェクトは、W&B アプリのホームページまたはユーザー/チームのプロファイルページで作成できます。プロジェクトが存在しない場合は、最初に使用したときに自動的に作成されます。以下の手順では、API キーを取得する方法を示しています：

### API キーの取得方法
1. [W&B にログインする](https://wandb.ai/login)。注: W&B Server を使用している場合は、管理者にインスタンスホスト名を確認してください。
2. [承認ページ](https://wandb.ai/authorize) に移動するか、ユーザー/チーム設定で API キーを取得します。プロダクション環境では、そのキーを所有するために [サービス アカウント]({{< relref path="../../support/service_account_useful.md" lang="ja" >}}) を使用することをお勧めします。
3. 環境変数を設定し、その API キーを `WANDB_API_KEY=YOUR_KEY` としてエクスポートします。

以下の例では、Dagster コード内で API キーを指定する場所を示しています。エンティティとプロジェクト名を `wandb_config` 内のネストされた辞書に指定してください。別の W&B Project を使用する場合は、異なる `wandb_config` 値を別々の ops/アセットに渡すことができます。渡すことができる可能性のあるキーに関する詳細については、以下の設定のセクションを参照してください。

{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
Example: configuration for `@job`
```python
# add this to your config.yaml
# alternatively you can set the config in Dagit's Launchpad or JobDefinition.execute_in_process
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # replace this with your W&B entity
     project: my_project # replace this with your W&B project


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

Example: configuration for `@repository` using assets

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
                   {"cache_duration_in_minutes": 60} # only cache files for one hour
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # replace this with your W&B entity
                       "project": "my_project", # replace this with your W&B project
                   }
               }
           },
       ),
   ]
```
Note that we are configuring the IO Manager cache duration in this example contrary to the example for `@job`.
{{% /tab %}}
{{< /tabpane >}}

### Configuration
続く設定オプションは、統合によって提供される W&B 専用の Dagster リソースと IO マネージャーの設定として使用されます。

* `wandb_resource`: Dagster [resource](https://docs.dagster.io/concepts/resources) は W&B API と通信するために使用されます。提供された API キーを使用して自動的に認証されます。プロパティ:
    * `api_key`: (str, 必須): W&B API と通信するために必要な W&B API キー。
    * `host`: (str, オプション): 使用する API のホストサーバー。W&B Server を使用する場合にのみ必要です。デフォルトでは Public Cloud ホスト、`https://api.wandb.ai` を使用します。
* `wandb_artifacts_io_manager`: Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) は、W&B Artifacts を消費するために使用されます。プロパティ:
    * `base_dir`: (int, オプション) ローカルストレージとキャッシュに使用される基本ディレクトリ。W&B Artifacts と W&B Run のログは、そのディレクトリから読み書きされます。デフォルトでは `DAGSTER_HOME` ディレクトリを使用しています。
    * `cache_duration_in_minutes`: (int, オプション) ローカルストレージに W&B Artifacts と W&B Run logs が保持される時間を定義します。その期間にアクセスされなかったファイルとディレクトリのみがキャッシュから削除されます。キャッシュの削除は IO マネージャーの実行の最後に行われます。キャッシュを完全にオフにする場合は、0 に設定します。同じマシンで実行されているジョブ間でアーティファクトを再利用する場合、キャッシュは速度を向上させます。デフォルトでは 30 日間に設定されています。
    * `run_id`: (str, オプション): 本 run のユニーク ID。プロジェクト内で一意でなければならず、run を削除する場合は ID を再利用できません。短く記述的な名前には name フィールドを使用し、run 間で比較するハイパーパラメーターの保存には config を使用します。ID には以下の特殊文字が含まれていてはなりません: `/\#?%:..` 実験管理を Dagster 内で行う際、IO マネージャーが run を再開できるようにするために Run ID を設定する必要があります。デフォルトでは Dagster Run ID に設定されています。例：`7e4df022-1bf2-44b5-a383-bb852df4077e`。
    * `run_name`: (str, オプション) このrun の短いディスプレイ名で、UI でこのrun を識別するのに役立ちます。デフォルトでは、次の形式の文字列です: `dagster-run-[Dagster Run ID の最初の 8 文字]`。例：`dagster-run-7e4df022`。
    * `run_tags`: (list[str], オプション): UI でこのrun のタグ一覧に反映させる文字列のリスト。タグはrun をまとめて整理したり、`baseline` や `production` などの一時的なラベルを適用するのに便利です。UI でタグを追加および削除したり、特定のタグを持つrun のみに絞り込むことが簡単です。統合に使用された W&B Run には `dagster_wandb` タグが付けられます。

## W&B Artifacts の使用
W&B Artifact との統合は、Dagster IO Manager に依存します。

[IO Managers](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットや op の出力を保存し、それを下流のアセットや ops への入力としてロードする役割を持つユーザー提供のオブジェクトです。例えば、IO マネージャーはファイルシステム上のファイルからオブジェクトを保存したりロードしたりすることができます。

統合は W&B Artifacts 用の IO マネージャーを提供します。これにより、どの Dagster `@op` または `@asset` もネイティブに W&B Artifacts を作成および消費できます。ここでは、Python リストを含むデータセット型の W&B Artifact を生成する `@asset` の簡単な例を示します。

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
    return [1, 2, 3] # これはアーティファクトに保存されます
```

`@op`, `@asset` および `@multi_asset` にメタデータ設定を注釈し、アーティファクトを書き込むことができます。同様に、Dagster 外部で作成された W&B Artifacts も消費できます。

## W&B Artifacts 書き込み
続ける前に、W&B Artifacts の使用方法についてよく理解していることをお勧めします。[Artifact のガイド]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を読むことを検討してください。

Python 関数からオブジェクトを返すことで、W&B Artifact に書き込みます。W&B でサポートされているオブジェクトは、次のとおりです：
* Python オブジェクト (int, dict, list…)
* W&B オブジェクト (Table, Image, Graph…)
* W&B Artifact オブジェクト

続く例では、Dagster アセット (`@asset`) を使用して W&B Artifacts を書き込む方法を示しています。

{{< tabpane text=true >}}
{{% tab "Python objects" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズできるものは、統合で作成されたアーティファクトにピクルされ追加されます。そのアーティファクトを Dagster 内で読み込むときに内容が取り出されます（詳細については [アーティファクトの読み込み]({{< relref path="#read-wb-artifacts" lang="ja" >}}) を参照してください）。

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

W&B は複数の Pickle ベースのシリアライゼーションモジュール ([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。さらに、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) といったより高度なシリアライゼーションを使用することもできます。詳細については [Serialization セクション]({{< relref path="#serialization-configuration" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{% tab "W&B Object" %}}
ネイティブの W&B オブジェクト（例: [Table]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}), [Image]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}}), or [Graph]({{< relref path="/ref/python/data-types/graph.md" lang="ja" >}})）はいずれも統合によって作成されたアーティファクトに追加されます。ここでは Table を使用した例を示します。

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

複雑なユースケースの場合、自分のアーティファクトオブジェクトを作成する必要があるかもしれません。統合は統合の両側のメタデータに注釈を付けるような有用な追加機能を提供します。

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

### Configuration
`wandb_artifact_configuration` と呼ばれる設定辞書は `@op`, `@asset`, および `@multi_asset` に設定できます。この辞書はメタデータとしてデコレーター引数で渡される必要があります。この設定は、W&B Artifacts の IO マネージャーの読み取りと書き込みを制御するために必要です。

`@op` の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を通じて出力メタデータに配置されています。
`@asset` の場合、明確に入力されているメタデータ引数に配置されています。
`@multi_asset` の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数を通じて各出力メタデータに配置されています。

続いてのコード例では、`@op`, `@asset` および `@multi_asset` 計算の辞書を設定する方法を示しています。

{{< tabpane text=true >}}
{{% tab "Example for @op" %}}
Example for `@op`:
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
Example for `@asset`:
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


`@asset` はすでに名前を持っているため、設定によって名前を渡す必要はありません。統合はアセット名としてアーティファクト名を設定します。

{{% /tab %}}
{{% tab "Example for @multi_asset" %}}

Example for `@multi_asset`:

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

サポートされているプロパティ:
* `name`：(str) このアーティファクトの人間が読める名前で、UI でこのアーティファクトを特定したり、`use_artifact` 呼び出しで参照するためのものです。名前には、文字、数字、アンダースコア、ハイフン、ドットが含まれている可能性があります。名前はプロジェクト内で一意でなければなりません。`@op` に必須です。
* `type`：(str) アーティファクトの種類で、アーティファクトを整理し、区別するために使用されます。一般的なタイプには dataset または model がありますが、文字、数字、アンダースコア、ハイフン、ドットを含む任意の文字列を使用できます。出力がすでにアーティファクトでない場合に必須です。
* `description`：(str) アーティファクトの説明を提供するフリーテキスト。説明が UI でマークダウンとしてレンダリングされるため、表、リンクなどを配置するのに適した場所です。
* `aliases`：(list[str]) アーティファクトに適用する 1 つ以上のエイリアスを含む配列。このリストには、設定されていない場合でも「latest」タグが追加されます。モデルとデータセットのバージョニングを管理するための効果的な方法です。
* [`add_dirs`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}): (list[dict[str, Any]])：アーティファクトに含めるローカル ディレクトリごとの構成を含む配列。SDK の同名のメソッドと同じ引数をサポートします。
* [`add_files`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}): (list[dict[str, Any]])：アーティファクトに含めるローカル ファイルごとの構成を含む配列。SDK の同名のメソッドと同じ引数をサポートします。
* [`add_references`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}): (list[dict[str, Any]])：アーティファクトに含める外部参照ごとの構成を含む配列。SDK の同名のメソッドと同じ引数をサポートします。
* `serialization_module`: (dict) 使用するシリアライゼーションモジュールの構成。詳細については「シリアライゼーション」セクションを参照してください。
    * `name`: (str) シリアライゼーションモジュールの名前。受け入れられる値: `pickle`、`dill`、`cloudpickle`、`joblib`。モジュールはローカルで利用可能でなければなりません。
    * `parameters`: (dict[str, Any]) シリアライゼーション関数に渡されるオプションの引数。そのモジュールのダンプメソッドと同じパラメーターを受け入れます。例: `{"compress": 3, "protocol": 4}`.

高度な例:

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

アセットは統合の両側に有用なメタデータが付けられてマテリアライズされます:
* W&B サイド: ソース統合名とバージョン、使用された Python バージョン、ピクルプロトコルバージョンなど。
* Dagster サイド: 
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、タイプ、バージョン、サイズ、URL
    * W&B エンティティ
    * W&B プロジェクト

以下の画像は、W&B から追加されたメタデータを Dagster アセットに示しています。この情報は統合なしでは利用できません。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

次の画像は、提供された構成がどのように W&B Artifact で有用なメタデータで補完されたかを示しています。この情報は再現性とメンテナンスに役立ちます。統合なしでは利用できません。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
静的タイプチェッカー（たとえば mypy）を使用している場合は、次のようにして構成タイプ定義オブジェクトをインポートします：

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### パーティションを使用する

この統合は [Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をネイティブにサポートしています。

以下は `DailyPartitionsDefinition` を使用しているパーティション化されたアセットの例です。
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
このコードは各パーティションに対して 1 つの W&B アーティファクトを生成します。アセット名の下にあるアーティファクトパネル（UI）でアーティファクトを表示し、パーティションキーが追加されています。たとえば、`my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、または `my_daily_partitioned_asset.2023-01-03` などです。複数の次元にわたってパーティション化されたアセットの場合、各次元はドットで区切られた形式で示されます。たとえば、`my_asset.car.blue` のようになります。

{{% alert color="secondary" %}}
この統合は、1 つのラン内で複数のパーティションを具現化することを許可しません。資産を作成するには、複数のランを実行する必要があります。これは、資産を具現化する際に Dagit で実行できます。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 高度な使用法
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [Simple partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [Multi-partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [Advanced partitioned usage](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts の読み込み
W&B Artifacts の読み込みは、それらを書き込む場合と似ています。`wandb_artifact_configuration` と呼ばれる構成辞書を `@op` または `@asset` に設定できます。唯一区別は、設定を出力ではなく入力に設定する必要があることです。

`@op` の場合、[In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数を通じて入力メタデータに配置されています。アーティファクトの名前を明示的に渡す必要があります。

`@asset` の場合、[AssetIn](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) の入力メタデータ引数を通じて配置されています。親アセットの名前がそれにマッチするはずなので、アーティファクト名を渡す必要はありません。

統合外で作成されたアーティファクトに依存したい場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。これは、そのアセットの最新バージョンを常に読み込みます。

以下の例は、さまざまな ops からアーティファクトを読む方法を示しています。

{{< tabpane text=true >}}
{{% tab "From an @op" %}}
Reading an artifact from an `@op`
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
Reading an artifact created by another `@asset`
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # if you don't want to rename the input argument you can remove 'key'
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

Reading an Artifact created outside Dagster:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # the name of the W&B Artifact
   description="Artifact created outside Dagster",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```
{{% /tab %}}
{{< /tabpane >}}

### Configuration
続く設定は、IO マネージャーが収集し、装飾された関数への入力として提供する内容を指示するために使用されます。次の読み取りパターンがサポートされています。

1. アーティファクト内の名前付きオブジェクトを取得するには、get を使用します。

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


2. アーティファクト内に含まれるダウンロードしたファイルのローカルパスを取得するには、get_path を使用します。

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

3. 全体のアーティファクトオブジェクトを取得する (コンテンツをローカルにダウンロードした状態で):

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

サポートされているプロパティ
* `get`: (str) アーティファクト相対名に配置されている W&B オブジェクトを取得します。
* `get_path`: (str) アーティファクト相対名に位置するファイルへのパスを取得します。

### シリアライゼーション構成
デフォルトでは、統合は標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、いくつかのオブジェクトはこれに対応していません。例えば、yield を含む関数は、ピクルを試みるとエラーを引き起こします。

私たちは、より多くの Pickle ベースのシリアライゼーションモジュール ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。また、シリアライズされた文字列を返す、またはウィジェットを作成することで、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などのより高度なシリアライゼーションを使用することもできます。適切な選択肢は使用ケースによって異なりますので、このテーマに関する文献を参照してください。	

### Pickle ベースのシリアライゼーションモジュール

{{% alert color="secondary" %}}
ピクルには知られているセキュリティの脆弱性が存在しています。セキュリティが懸念される場合は、W&B オブジェクトのみを使用してください。データに署名し、ハッシュキーを自身のシステムに保存することをお勧めします。より複雑なユースケースに関して、何かご不明点がございましたらお気兼ねなくお問い合わせください。喜んでサポートいたします。
{{% /alert %}}

ピクルされたモジュールの名称は、`wandb_artifact_configuration` 内の `serialization_module` 辞書を通じて設定することができます。Dagster を実行しているマシン上でモジュールが利用可能であることを確認してください。

統合は、アーティファクトを読むときにどのシリアライゼーションモジュールを使用するかを自動的に把握します。

現在サポートされているモジュールは `pickle`、`dill`、`cloudpickle`、および `joblib` です。

以下は joblib で直列化された「モデル」を作成し、その後推論に使用する簡略化された例です。

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
    # これは実際の ML モデルではありませんが、pickle モジュールでは不可能なことです
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
    context.log.info(inference_result)  # 印刷: 3
    return inference_result
```

### 高度なシリアライゼーション形式 (ONNX, PMML)
ONNX や PMML のような交換ファイル形式を使用することが一般的です。この統合はこれらのフォーマットをサポートしていますが、Pickle ベースのシリアライゼーションよりも少し手間がかかります。

これらのフォーマットを使用するには、2 つの異なる方法があります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常の Python オブジェクトとして返します。統合によってその文字列がピクルされます。その文字列を使用してモデルを再構築できます。
2. シリアライズされたモデルを含むローカルファイルを新しく作成し、そのファイルを add_file 設定を使用してカスタムアーティファクトとして作成します。

以下は Scikit-learn モデルを ONNX を使用してシリアライズした例です。

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
    # https://onnx.ai/sklearn-onnx/ から影響を受けた例

    # モデルをトレーニングします。
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX フォーマットに変換します
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクトを書き込みます (model + test_set)
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
    # https://onnx.ai/sklearn-onnx/ から影響を受けた例

    # ONNX Runtime で予測を計算します
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### パーティションを使用する

統合は [Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をネイティブにサポートしています。

特定の 1 つの、複数の、またはすべてのパーティションを選択して読み取りを行うことができます。

すべてのパーティションは辞書形式で提供され、キーと値はそれぞれパーティションキーとアーティファクトコンテンツを表します。

{{< tabpane text=true >}}
{{% tab "すべてのパーティションを読み取る" %}}
上流の `@asset` のすべてのパーティションが辞書として読み取られます。この辞書では、キーと値はそれぞれパーティションキーとアーティファクトコンテンツに対応します。
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
{{% tab "特定のパーティションを読み取る" %}}
`AssetIn` の `partition_mapping` 設定を使用して特定のパーティションを選択することができます。この例では、`TimeWindowPartitionMapping` を使用しています。
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

設定オブジェクト `metadata` は、プロジェクト内の異なるアーティファクトパーティションに対する Weights & Biases (wandb) のやり取りを管理するために使用されます。

オブジェクト `metadata` には、キー `wandb_artifact_configuration` があります。このオブジェクトにはさらにネストされたオブジェクト `partitions` があります。

オブジェクト `partitions` は、各パーティションの名前をその構成にマッピングします。各パーティションの構成では、データを取得するための設定を指定できます。これらの構成には、`get`、`version`、および `alias` という異なるキーが含まれている可能性があります。

**構成キー**

1. `get`:
`get` キーは、データを取得する W&B オブジェクト (Table, Image...) の名前を指定します。
2. `version`:
`version` キーは、特定のバージョンのアーティファクトを取得したい場合に使用されます。
3. `alias`:
`alias` キーは、エイリアスによってアーティファクトを取得するために使用されます。

**ワイルドカード構成**

ワイルドカード `"*"` は、構成されていないすべてのパーティションを表します。これは、`partitions` オブジェクトに明示的に記載されていないパーティションに対するデフォルト構成を提供します。

例えば、

```python
"*": {
    "get": "default_table_name",
},
```
この構成は、明示的に構成されていないすべてのパーティションの場合、`default_table_name` というテーブルからデータを取得することを意味します。

**特定パーティションの構成**

特定のパーティションに対して構成を明示的に示すことで、ワイルドカード構成をオーバーライドできます。

例えば、

```python
"yellow": {
    "get": "custom_table_name",
},
```

この構成は、`yellow` という名前のパーティションの場合、`custom_table_name` というテーブルからデータが取得されるということを意味し、ワイルドカード構成をオーバーライドします。

**バージョニングとエイリアス化**

バージョニングとエイリアス化のために、`version` と `alias` キーを構成で指定することができます。

バージョンの場合、

```python
"orange": {
    "version": "v0",
},
```

この構成は、`orange` アーティファクトパーティションのバージョン `v0` からデータを取得することを意味します。

エイリアスの場合、

```python
"blue": {
    "alias": "special_alias",
},
```

この構成は、エイリアス `special_alias`（設定で `blue` として示されている）を持つアーティファクトパーティションの `default_table_name` というテーブルからデータを取得することを意味します。

### 高度な使用
統合の高度な使用法を確認するには、以下の完全なコードサンプルを参照してください。
* [アセットの高度な使用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py) 
* [Partitioned job example](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルをモデルレジストリにリンクする](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch を使用する

{{% alert color="secondary" %}}
開発中のベータ製品
Launch に興味がありますか？ W&B Launch のカスタマーパイロットプログラムに参加するためにアカウントチームに連絡してください。
パイロット顧客様は、ベータプログラムに参加する資格を得るために AWS EKS または SageMaker を使用する必要があります。最終的には、他のプラットフォームをサポートする予定です。
{{% /alert %}}

続ける前に、W&B Launch の使用方法についてよく理解していることをお勧めします。Launch のガイドを読むことを考慮してください: /guides/launch。

Dagster の統合と助けにより：
* Dagsterインスタンスでの Launch エージェントの実行が可能。
* Dagster インスタンス内でのローカル Launch ジョブの実行。
* オンプレミスまたはクラウド上での リモート Launch ジョブの実行が可能。

### Launch エージェント
統合は `run_launch_agent` というインポート可能な `@op` を提供します。これは Launch エージェントを起動し、手動で停止するまで長時間実行プロセスとして実行されます。

エージェントは launch キューをポーリングし、ジョブを実行または外部サービスにディスパッチするプロセスです。

設定に関する詳細は、[リファレンスドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad でもすべてのプロパティについての有用な説明を閲覧することができます。

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

シンプルな例
```python
# add this to your config.yaml
# alternatively you can set the config in Dagit's Launchpad or JobDefinition.execute_in_process
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # replace this with your W&B entity
     project: my_project # replace this with your W&B project
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
統合は `run_launch_job` というインポート可能な `@op` を提供します。これは Launch ジョブを実行します。

Launch ジョブはキューに割り当てられて実行されます。新しいキューを作成するか、デフォルトのものを使用することができます。そのキューを監視している有効なエージェントがあることを確認してください。Dagster インスタンス内でエージェントを実行することもできますが、Kubernetes 内で展開可能なエージェントを検討することもできます。

設定に関する詳細は、[リファレンスドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad でもすべてのプロパティについての有用な説明を閲覧することができます。

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}

シンプルな例
```python
# add this to your config.yaml
# alternatively you can set the config in Dagit's Launchpad or JobDefinition.execute_in_process
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # replace this with your W&B entity
     project: my_project # replace this with your W&B project
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
   run_launch_job.alias("my_launched_job")() # ジョブをエイリアスで名前変更します
```

## ベストプラクティス

1. IO マネージャーを使用してアーティファクトを読み書きする。
[`Artifact.download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) や [`Run.log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) を直接使用する必要はありません。これらのメソッドは統合によって処理されます。アーティファクトに保存したいデータを単に返して、残りは統合にお任せください。これにより、W&B でのアーティファクトのラインエージが改善されます。

2. 複雑なユースケースのためにのみアーティファクトオブジェクトを自分で作成する。
Python オブジェクトと W&B オブジェクトは、ops/アセットから返されるべきです。統合は、アーティファクトのバンドルを処理します。
複雑なユースケースの場合、Dagster ジョブの中でアーティファクトを直接構築できます。統合のためにアーティファクトオブジェクトを渡し、メタデータの注釈を追加することをお勧めします。ソース統合名とバージョン、使用された Python バージョン、ピクルプロトコルバージョンなど。

3. ファイル、ディレクトリ、および外部参照をメタデータを通じてアーティファクトに追加する。
ファイル、ディレクトリ、または外部参照 (Amazon S3、GCS、HTTP…) を追加するには、統合の `wandb_artifact_configuration` オブジェクトを使用してください。[アーティファクト構成セクション]({{< relref path="#configuration-1" lang="ja" >}}) の高度な例を参照して詳細情報を確認してください。

4. アーティファクトが生成される場合は、@op の代わりに @asset を使用する。
アーティファクトは資産です。Dagster がその資産を管理する場合は、資産（アセット）を使用することをお勧めします。これにより、Dagit アセットカタログ内での可観測性が向上します。

5. Dagster 外部で作成されたアーティファクトを消費するために SourceAsset を使用する。
これにより、外部で作成されたアーティファクトを読み込む際に統合の利点を活用することができます。そうしない場合、統合によって作成されたアーティファクトのみ使用できます。

6. 大規模なモデルについては、専用のコンピュートでのトレーニングをオーケストレーションするために W&B Launch を使用する。
小規模なモデルを Dagster クラスター内でトレーニングすることができ、GPU ノードを持つ Kubernetes クラスターで Dagster を実行することができます。大規模なモデルトレーニングには W&B Launch を使用することを推奨します。これにより、インスタンスが過負荷にならず、より適切なコンピューティングリソースへのアクセスが提供されます。

7. Dagster 内で実験管理する際は、W&B Run ID を Dagster Run ID に設定する。
[Run を再実行可能にする]({{< relref path="/guides/models/track/runs/resuming.md" lang="ja" >}}) と、W&B Run ID を Dagster Run ID または任意の文字列に設定することをお勧めします。この勧告に従うことで、Dagster 内でモデルをトレーニングする際に、W&B メトリクスと W&B Artifacts が同じ W&B Run に保存されることが保証されます。

W&B Run ID を Dagster Run ID に設定します。
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

あるいは、自分の W&B Run ID を選択し、それを IO マネージャー構成に渡します。
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

8. 大規模な W&B Artifacts の場合、get や get_path で必要なデータのみを収集する。
デフォルトでは、統合はアーティファクト全体をダウンロードします。非常に大きなアーティファクトを使用する場合、必要な特定のファイルまたはオブジェクトのみを収集したいかもしれません。これにより、速度とリソースの利用が向上します。

9. Python オブジェクトの場合、ピクルモジュールをユースケースに適応させる。
デフォルトでは、W&B 統合は標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用します。しかし、いくつかのオブジェクトはこれに適合しません。例えば、yield を含む関数は、ピクルを試みるとエラーを引き起こします。W&B は他の Pickle ベースのシリアライゼーションモジュール ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。

また、シリアライズされた文字列を返したり、直接ウィジェットを作成することで、[ONNX](https://onnx.ai/)  や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のようなより高度なシリアライゼーションを使用することもできます。適切な選択肢はユースケースによって異なりますので、このテーマに関する利用可能な文献を参照してください。