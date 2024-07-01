---
description: W&B を Dagster と統合する方法に関するガイド。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dagster

Dagster と W&B を使用して MLOps パイプラインをオーケストレーションし、ML 資産を維持します。W&B とのインテグレーションにより、Dagster 内で以下が容易になります:

* [W&B Artifacts](../artifacts/intro.md) を使用および作成する。
* [W&B Model Registry](../model_registry/intro.md) で Registered Models を使用および作成する。
* [W&B Launch](../launch/intro.md) を使用して、専用のコンピュートでトレーニングジョブを実行する。
* ops および assets で [wandb](../../ref/python/README.md) クライアントを使用する。

W&B Dagster インテグレーションには、W&B 固有の Dagster リソースと IO Manager が提供されます:

* `wandb_resource`: W&B API に認証して通信するための Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster IO Manager。

以下のガイドでは、Dagster で W&B を使用するための前提条件を満たす方法、ops と asset で W&B Artifacts を作成および使用する方法、W&B Launch を使用する方法、および推奨されるベストプラクティスについて説明します。

## 始める前に
Weights & Biases 内で Dagster を使用するには、以下のリソースが必要です:
1. **W&B API キー**。
2. **W&B エンティティ (ユーザーまたはチーム)**: エンティティは、W&B Runs と Artifacts を送信するユーザー名またはチーム名です。run をログに記録する前に、W&B アプリ UI でアカウントまたはチームエンティティを作成してください。エンティティを指定しない場合、run はデフォルトのエンティティ (通常はユーザー名) に送信されます。**Project Defaults** の設定でデフォルトのエンティティを変更できます。
3. **W&B プロジェクト**: [W&B Runs](../runs/intro.md) が保存されるプロジェクトの名前。

W&B アプリ内のユーザーまたはチームのプロフィールページを確認して、W&B エンティティを見つけてください。既存の W&B プロジェクトを使用するか、新しいプロジェクトを作成できます。新しいプロジェクトは、W&B アプリホームページまたはユーザー/チームプロフィールページで作成できます。プロジェクトが存在しない場合は、初めて使用する際に自動的に作成されます。以下の手順では、API キーの取得方法を示します:

### API キーの取得方法
1. [W&B にログイン](https://wandb.ai/login)します。注意: W&B サーバーを使用している場合は、管理者にインスタンスホスト名を尋ねてください。
2. [authorize ページ](https://wandb.ai/authorize) またはユーザー/チーム設定で API キーを取得します。プロダクション環境では、そのキーを所有するために [サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) を使用することをお勧めします。
3. 環境変数にそのAPIキーを設定します: `WANDB_API_KEY=YOUR_KEY`。


以下の例では、Dagster コードで API キーを指定する場所を説明します。`wandb_config` 内の辞書でエンティティおよびプロジェクト名を指定してください。異なる W&B プロジェクトを使用したい場合は、異なる `wandb_config` 値を異なる ops/assets に渡すことができます。渡すことができるキーの詳細については、以下の設定セクションを参照してください。


<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

Example: configuration for `@job`
```python
# これを config.yaml に追加します。
# 代わりに、Dagit's Launchpad または JobDefinition.execute_in_process で設定することもできます。
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これを W&B エンティティに置き換えます
     project: my_project # これを W&B プロジェクトに置き換えます


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

  </TabItem>
  <TabItem value="repository">


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
                   {"cache_duration_in_minutes": 60} # 一時間だけキャッシュ
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # これを W&B エンティティに置き換えます
                       "project": "my_project", # これを W&B プロジェクトに置き換えます
                   }
               }
           },
       ),
   ]
```
この例では、`@job` の例とは対照的に IO Manager キャッシュ期間が設定されている点に注意してください。

  </TabItem>
</Tabs>


### 設定
以下の設定オプションは、インテグレーションが提供する W&B 固有の Dagster リソースおよび IO マネージャの設定として使用されます。

* `wandb_resource`: W&B API と通信するために使用される Dagster [resource](https://docs.dagster.io/concepts/resources)。提供された API キーを使って自動認証します。プロパティ:
    * `api_key`: (str, 必須): W&B API と通信するために必要な W&B API キー。
    * `host`: (str, オプション): 使用する API ホストサーバー。W&B サーバーを使用している場合のみ必要です。デフォルトは Public Cloud ホスト [https://api.wandb.ai](https://api.wandb.ai)。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: (int, オプション) ローカルストレージとキャッシュのために使用されるベースディレクトリ。W&B Artifacts と W&B Run のログはそのディレクトリに書き込まれ、そこから読み取られます。デフォルトでは `DAGSTER_HOME` ディレクトリを使用します。
    * `cache_duration_in_minutes`: (int, オプション) W&B Artifacts と W&B Run ログがローカルストレージに保持される時間を定義します。その時間内に開かれなかったファイルやディレクトリのみがキャッシュから削除されます。キャッシュのパージは、IO Manager の実行終了時に行われます。キャッシュを完全に無効にしたい場合は 0 に設定できます。同じマシンでジョブを実行する間に Artifact が再利用される場合、キャッシュは速度を向上させます。デフォルトは 30 日です。
    * `run_id`: (str, オプション): この run の一意のID、再開に使用されます。プロジェクト内で一意である必要があり、run を削除した場合は ID を再利用できません。名前フィールドを使用して短い説明名を付けたり、設定でハイパーパラメーターを保存して run 間で比較できます。ID には `/\#?%:..` の特殊文字を含めることはできません。Dagster 内で実験管理を行う際に Run ID を設定する必要があり、IO Manager が run を再開できるようにします。デフォルトは Dagster Run ID です e.g `7e4df022-1bf2-44b5-a383-bb852df4077e`。
    * `run_name`: (str, オプション) この run の短い表示名、UI で run を識別する方法。デフォルトは次の形式の文字列に設定されています: dagster-run-[8 文字の Dagster Run ID] e.g. `dagster-run-7e4df022`。
    * `run_tags`: (list[str], オプション): この run の UI でタグ一覧を埋める文字列のリスト。タグは run の整理や一時的なラベル（例: "baseline" や "production"）の適用に便利です。UIではタグの追加や削除が簡単にでき、特定のタグ付きrun のみを表示できます。インテグレーションで使用される W&B Run には `dagster_wandb` タグが付けられます。

## W&B Artifacts を使用する

W&B Artifact とのインテグレーションは、Dagster IO Manager に依存しています。

[IO Managers](https://docs.dagster.io/concepts/io-management/io-managers) は、資産やオプの出力を保存し、それを下流の資産やオプに入力として読み込む役割を持つユーザー提供オブジェクトです。たとえば、IO Manager はファイルシステム上のファイルからオブジェクトを保存およびロードします。

インテグレーションでは、W&B Artifacts 用の IO Manager が提供されます。これにより、任意の Dagster `@op` または `@asset` が W&B Artifacts をネイティブに作成および消費できます。以下は、Python リストを含むデータセット型の W&B Artifact を生成する `@asset` のシンプルな例です。

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
    return [1, 2, 3] # これは Artifact に保存されます
```

`@op`、`@asset` および `@multi_asset` にメタデータ設定を注釈して Artifact を書き込むことができます。同様に、Dagster 以外で作成された W&B Artifacts を消費することもできます。

## W&B Artifacts の書き込み
続行する前に、W&B Artifacts の使用方法についてよく理解していることをお勧めします。[Artifacts に関するガイド](../artifacts/intro.md) を読むことを検討してください。

W&B Artifact を書き込むには、Python 関数からオブジェクトを返します。次のオブジェクトが W&B によってサポートされています:
* Python オブジェクト (int, dict, list…)
* W&B オブジェクト (Table, Image, Graph…)
* W&B Artifact オブジェクト

以下の例では、Dagster 資産 (`@asset`) を使用して W&B Artifacts を書き込む方法を示します:


<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python objects', value: 'python_objects'},
    {label: 'W&B object', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズできるものはすべてピクルされ、インテグレーションによって作成された Artifact に追加されます。内容は Dagster 内で Artifact を読むときにアンピクルされます（詳細については [artifacts の読む](#read-wb-artifacts) を参照）。 

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


W&B は複数のピクルベースのシリアル化モジュールをサポートしています ([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))。 [ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のようなより高度なシリアル化も使用できます。詳細については、[シリアル化](#serialization-configuration) セクションを参照してください。

  </TabItem>
  <TabItem value="wb_object">

ネイティブの W&B オブジェクト（例: [Table](../../ref/python/data-types/table.md), [Image](../../ref/python/data-types/image.md), [Graph](../../ref/python/data-types/graph.md)）は、インテグレーションによって作成された Artifact に追加されます。以下は、テーブルを使用する例です。

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

  </TabItem>
  <TabItem value="wb_artifact">

複雑なユースケースには、独自の Artifact オブジェクトを構築する必要があるかもしれません。インテグレーションは、インテグレーションの両側でメタデータを強化するなど、有用な追加機能を提供します。

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

  </TabItem>
</Tabs>


### 設定
`wandb_artifact_configuration` と呼ばれる設定辞書が `@op`、`@asset` および `@multi_asset` に設定できます。この辞書はデコレータ引数としてメタデータで渡さなければなりません。この設定によって、IO Manager が W&B Artifacts の読み書きを制御できます。

`@op` の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) のメタデータ引数として出力メタデータの中にあります。 
`@asset` の場合、資産のメタデータ引数の中にあります。 
`@multi_asset` の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) のメタデータ引数の中に各出力メタデータにあります。

以下のコード例では、`@op`、`@asset`、`@multi_asset` 計算に辞書を設定する方法を示しています:

<Tabs
  defaultValue="op"
  values={[
    {label: 'Example for @op', value: 'op'},
    {label: 'Example for @asset', value: 'asset'},
    {label: 'Example for @multi_asset', value: 'multi_asset'},
  ]}>
  <TabItem value="op">

`@op` の例:
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

  </TabItem>
  <TabItem value="asset">

`@asset` の例:
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

名前を設定する必要はありません。`@asset` にはすでに名前があります。インテグレーションは Artifact 名を資産名として設定します。

  </TabItem>
  <TabItem value="multi_asset">

`@multi_asset` の例:

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

  </TabItem>
</Tabs>



サポートされるプロパティ:
* `name`: (str) このアーティファクトの人間が読める名前で、UIでこのアーティファクトを識別する方法や `use_artifact` 呼び出しで参照する方法です。名前には、文字、数字、アンダースコア、ハイフン、およびドットを含めることができます。名前はプロジェクト全体で一意である必要があります。`@op` では必須です。
* `type`: (str) アーティファクトのタイプ。これはアーティファクトを整理し、区別するために使用されます。一般的なタイプにはデータセットやモデルがありますが、文字、数字、アンダースコア、ハイフン、およびドットを含む任意の文字列を使用できます。出力がすでにアーティファクトでない場合に必須です。
* `description`: (str) アーティファクトの説明を提供する自由なテキスト。説明は UI で Markdown 形式でレンダリングされるため、テーブル、リンクなどを配置するのに適しています。
* `aliases`: (list[str]) アーティファクトに適用する1つ以上のエイリアスを含む配列。このインテグレーションは、設定の有無にかかわらず "latest" タグをそのリストに追加します。モデルやデータセットのバージョン管理に効果的です。
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): アーティファクトに含める各ローカルディレクトリの設定を含む配列。SDKの同名メソッドと同じ引数をサポートします。
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): アーティファクトに含める各ローカルファイルの設定を含む配列。SDKの同名メソッドと同じ引数をサポートします。
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): アーティファクトに含める各外部参照の設定を含む配列。SDKの同名メソッドと同じ引数をサポートします。
* `serialization_module`: (dict) 使用するシリアル化モジュールの設定。詳細についてはシリアル化セクションを参照してください。
    * `name`: (str) シリアル化モジュールの名前。受け入れられる値は `pickle`、`dill`、`cloudpickle`、`joblib` です。そのモジュールはローカルに利用可能でなければなりません。
    * `parameters`: (dict[str, Any]) シリアル化関数に渡されるオプションの引数。そのモジュールの dump メソッドと同じ引数を受け入れます。例: `{"compress": 3, "protocol": 4}`。

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
                   "name": "外部 HTTP 参照による画像",
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

この資産にはインテグレーションの両側に有用なメタデータがマテリアライズされました:
* W&B 側: ソースインテグレーションの名前とバージョン、使用された Python バージョン、ピクルプロトコルのバージョンなど。
* Dagster 側: 
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、タイプ、バージョン、サイズ、URL
    * W&B Entity
    * W&B Project

次の画像は、インテグレーションから追加された Dagster 資産の W&B メタデータを示しています。この情報がなければインテグレーションは利用できません。

![](/images/integrations/dagster_wb_metadata.png)

次の画像は、提供された設定が W&B Artifact 上で有用なメタデータにより強化された様子を示しています。この情報は再現性および保守に役立ちます。インテグレーションなしでは利用できません。

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)


:::info
mypy のような静的型チェッカーを使用する場合、設定型定義オブジェクトを以下のようにインポートします:

```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

### パーティションの使用

インテグレーションはネイティブに [Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をサポートしています。

以下は `DailyPartitionsDefinition` を使用したパーティションの例です。
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
    context.log.info(f"{partition_key} のパーティション資産を作成中")
    return random.randint(0, 100)
```
このコードは各パーティションに対して一つの W&B Artifact を生成します。それらはアーティファクトパネル（UI）で資産名の下に見つけることができ、パーティションキーが追加されます。例: `my_daily_partitioned_asset.2023-01-01`, `my_daily_partitioned_asset.2023-01-02`, `my_daily_partitioned_asset.2023-01-03` など。複数の次元にわたってパーティションされた資産は、各次元をドットで分けられます。例: `my_asset.car.blue`。

:::caution
インテグレーションは一つの run で複数のパーティションをマテリアライズすることを許可しません。複数の run を実行して、資産をマテリアライズする必要があります。これは Dagit で資産をマテリアライズする際に実行できます。

![](/images/integrations/dagster_multiple_runs.png)
:::

#### 高度な使用法
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [シンプルなパーティション資産](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [マルチパーティション資産](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [高度なパーティション使用例](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)


## W&B Artifacts の読み込み
W&B Artifacts を読む方法は書き込む方法と似ています。`wandb_artifact_configuration` と呼ばれる設定辞書が `@op` または `@asset` に設定できます。唯一の違いは、設定を出力ではなく入力に設定することです。

`@op` の場合、入力メタデータとして [In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数の中にあります。 
`@asset` の場合、入力メタデータとして [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) 入力メタデータ引数の中にあります。アーティファクト名を入力する必要はありません、親資産の名前が一致する必要があります。

インテグレーション外で作成されたアーティファクトに依存関係を持ちたい場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。常にその資産の最新バージョンを読み取ります。

次の例では、さまざまな ops からアーティファクトを読み取る方法を示します。

<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'op'},
    {label: 'Created by another @asset', value: 'asset'},
    {label: 'Artifact created outside Dagster', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

`@op` からアーティファクトを読む
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

  </TabItem>
  <TabItem value="asset">

別の `@asset` によって作成されたアーティファクトを読む
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 入力引数名を変更したくない場合は 'key' を削除してください
           key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

  </TabItem>
  <TabItem value="outside_dagster">

Dagster 外で作成されたアーティファクトを読む:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B Artifact の名前
   description="Dagster 外で作成されたアーティファクト",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>


### 設定
以下の設定は、IO Manager が収集して装飾された関数の入力として提供するものを示すために使用されます。以下の読み取りパターンがサポートされています。

1. アーティファクト内の名前付きオブジェクトを取得するには get を使用します:

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


2. アーティファクト内に含まれるダウンロードファイルのローカルパスを取得するには get_path を使用します:

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

3. アーティファクト全体のオブジェクトを取得するには (その内容がローカルにダウンロードされています):
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


サポートされるプロパティ
* `get`: (str) アーティファクトの相対名に位置する W&B オブジェクトを取得します。
* `get_path`: (str) アーティファクトの相対名に位置するファイルのパスを取得します。

### シリアル化設定
デフォルトでは、インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、一部のオブジェクトはこれと互換性がありません。例えば、yield を持つ関数はピクルしようとするとエラーが発生します。 

私たちは他のピクルベースのシリアル化モジュール ([dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)) をサポートしています。より高度なシリアル化（[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) など）も使用できます。文字列を直にシリアル化して返すか、アーティファクトを直接作成してください。使用に応じた適切な選択を行い、このトピックの文献を参照してください。

### ピクルベースのシリアル化モジュール

:::caution
ピクルは安全でないことが知られています。セキュリティ懸念がある場合は、W&B オブジェクトのみを使用してください。データに署名し、自身のシステムでハッシュキーを保存することをお勧めします。複雑なユースケースについては、ぜひお問い合わせください。喜んでお手伝いいたします。
:::

シリアル化は、`wandb_artifact_configuration` の `serialization_module` 辞書で設定できます。Dagster を実行するマシンでモジュールが利用可能であることを確認してください。

インテグレーションはアーティファクトを読む際にどのシリアル化モジュールを使用するかを自動的に判断します。

現在サポートされているモジュールには pickle、dill、cloudpickle、および joblib があります。

以下は、joblib でシリアル化された "モデル" を作成し、それを推論に使用するシンプルな例です。

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
    # これは実際の ML モデルではありませんが、ピクル モジュールでは不可能です
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
    context.log.info(inference_result)  # 出力: 3
    return inference_result
```

### 高度なシリアル化フォーマット (ONNX, PMML)
ONNX や PMML などの交換ファイル形式を使用するのは一般的です。インテグレーションはこれらの形式をサポートしていますが、ピクルベースのシリアル化よりも少し作業が必要です。

これらの方法を使用して、これらの形式を利用することができます。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常の Python オブジェクトとして返します。インテグレーションはその文字列をピクルします。その文字列を使用してモデルを再構築できます。
2. モデルをシリアル化した新しいローカルファイルを作成し、add_file 設定を使用してカスタムアーティファクトを作成します。

以下はScikit-learnモデルを ONNX を使用してシリアル化する例です。

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
    # https://onnx.ai/sklearn-onnx/ にインスパイアされました
    # モデルをトレーニングします。
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 形式に変換します
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクトを書き込みます (モデル + テストセット)
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
    # https://onnx.ai/sklearn-onnx/ にインスパイアされました
    # ONNX ランタイムで予測を計算します
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### パーティションの使用

インテグレーションはネイティブに [Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をサポートしています。

1つ、複数、またはすべてのパーティションを選択的に読み取ることができます。

すべてのパーティションは辞書で提供され、そのキーと値はそれぞれパーティションキーとアーティファクト内容を表します。

<Tabs
  defaultValue="all"
  values={[
    {label: 'Read all partitions', value: 'all'},
    {label: 'Read specific partitions', value: 'specific'},
  ]}>
  <TabItem value="all">

アップストリーム `@asset` のすべてのパーティションを読み込み、それらは辞書として提供されます。この辞書では、キーと値がそれぞれパーティションクキーとアーティファクトの内容に対応しています。
```python
@asset(
    compute_kind="wandb",
    ins={"my_daily_partitioned_asset": AssetIn()},
    output_required=False,
)
def read_all_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"パーティション={partition}, 内容={content}")
```
  </TabItem>
  <TabItem value="specific">

`AssetIn` の `partition_mapping` 設定を使用して、特定のパーティションを選択できます。この場合、`TimeWindowPartitionMapping` を使用します。
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
        context.log.info(f"パーティション={partition}, 内容={content}")
```
  </TabItem>
</Tabs>

設定オブジェクト `metadata` は、プロジェクト内の異なるアーティファクトパーティションと Weights & Biases (wandb) がどのように相互作用するかを設定するために使用されます。

オブジェクト `metadata` は `wandb_artifact_configuration` というキーを含み、さらにその中に `partitions` オブジェクトを含んでいます。

`partitions` オブジェクトは各パーティションの名前をその設定にマッピングします。各パーティションの設定では、そこからデータを取得する方法を指定できます。これらの設定には `get`、`version`、`alias` というキーが含まれる場合があります。

**設定キー**

1. `get`:
`get` キーは、データを取得する W&B オブジェクト（Table, Image など）の名前を指定します。
2. `version`:
`version` キーは、特定のバージョンのアーティファクトを取得する場合に使用します。
3. `alias`:
`alias` キーは、エイリアスでアーティファクトを取得する場合に使用します。

**ワイルドカード設定**

ワイルドカード `"*"` は、全ての未設定のパーティションを表します。これは、明示的に言及されていないパーティションに対してデフォルトの設定を提供します。

例えば、

```python
"*": {
    "get": "default_table_name",
},
```
この設定は、明示的に設定されていないすべてのパーティションに対して、テーブル `default_table_name` からデータを取得することを意味します。

**特定パーティション設定**

各パーティションの特定の設定を提供することで、ワイルドカード設定を上書きできます。

例えば、

```python
"yellow": {
    "get": "custom_table_name",
},
```

この設定は、パーティション `yellow` に対してカスタムテーブル `custom_table_name` からデータを取得することを意味します。ワイルドカード設定を上書きします。

**バージョン管理とエイリアシング**

バージョン管理とエイリアシングの目的で、設定の中で特定の `version` と `alias` キーを提供できます。

バージョンについて、

```python
"orange": {
    "version": "v0",
},
```

この設定により、`orange` アーティファクトパーティションのバージョン `v0` からデータを取得します。

エイリアスについて、

```python
"blue": {
    "alias": "special_alias",
},
```

この設定により、エイリアス `special_alias` を持つアーティファクトパーティションのテーブル `default_table_name` からデータが取得されます。

### 高度な使用法
インテグレーションの高度な使用法を確認するには、次の完全なコード例を参照してください:
* [資産の高度な使用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py) 
* [パーティションジョブの例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルをモデルレジストリにリンクする](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)


## W&B Launch を使用する

:::caution
アクティブ開発中のベータ製品
Launch に興味がありますか？ W&B Launch のカスタマーパイロットプログラムへの参加については、アカウントチームにお問い合わせください。
パイロット顧客は、ベータプログラムに参加するために AWS EKS または SageMaker を使用する必要があります。最終的には追加のプラットフォームをサポートする予定です。
:::

続行する前に、W&B Launch の使用方法についてよく理解していることをお勧めします。ガイドのLaunchを読むことを検討してください: https://docs.wandb.ai/guides/launch。

Dagster インテグレーションは次のことに役立ちます:
* Dagster インスタンス内で1つまたは複数の Launch エージェントを実行する。
* Dagster インスタンス内でローカル Launch ジョブを実行する。
* オンプレミスまたはクラウドでリモート Launch ジョブを実行する。

### Launch エージェント
インテグレーションは `run_launch_agent` という輸入可能な `@op` を提供します。これは Launch Agent を開始し、手動で停止するまで長時間実行するプロセスとして実行されます。

エージェントは Launch キューをポーリングし、ジョブを実行または外部サービスにディスパッチして実行します。

[リファレンスドキュメント](../launch/intro.md) を参照して設定を確認してください

Launchpad でもすべてのプロパティの有用な説明を見ることができます。

![](/images/integrations/dagster_launch_agents.png)

シンプルな例
```python
# これを config.yaml に追加します。
# 代わりに、Dagit's Launchpad または JobDefinition.execute_in_process で設定することもできます。
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
    entity: my_entity # これを W&B エンティティに置き換えます
    project: my_project # これを W&B プロジェクトに置き換えます
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
インテグレーションは、`run_launch_job` という輸入可能な `@op` を提供します。これは Launch ジョブを実行します。

Launch ジョブはキューに割り当てられて実行されます。キューを作成するか、デフォルトのキューを使用することができます。そのキューを待ち受けるアクティブなエージェントがいることを確認してください。エージェントを Dagster インスタンス内で実行できますが、Kubernetes にデプロイ可能なエージェントの使用も検討できます。

[リファレンスドキュメント](../launch/intro.md) を参照して設定を確認してください。

Launchpad でもすべてのプロパティの有用な説明を見ることができます。

![](/images/integrations/dagster_launch_jobs.png)


シンプルな例
```python
# これを config.yaml に追加します。
# 代わりに、Dagit's Launchpad または JobDefinition.execute_in_process で設定することもできます。
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
    entity: my_entity # これを W&B エンティティに置き換えます
    project: my_project # これを W&B プロジェクトに置き換えます
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
  run_launch_job.alias("my_launched_job")() # エイリアスでジョブの名前を変更します
```

## ベストプラクティス

1. IO Manager を使用して Artifacts を読み書きします。 
[`Artifact.download()`](../../ref/python/artifact.md#download) や [`Run.log_artifact()`](../../ref/python/run.md#log_artifact) を直接使用する必要はありません。これらのメソッドはインテグレーションによって処理されます。保存したいデータを戻すだけで、インテグレーションが残りを処理します。これにより、W&B でアーティファクトのリネージが向上します。

2. 複雑なユースケース以外では自分でArtifactオブジェクトを作成しないでください。
PythonオブジェクトやW&Bオブジェクトをops/assetsから返すべきです。インテグレーションがアーティファクトをバンドル処理します。
複雑なユースケースでは、Dagsterジョブ内で直接アーティファクトを作成できます。インテグレーションにアーティファクトオブジェクトを渡して、インテグレーション名やバージョン、使用されたPythonバージョン、ピクルプロトコルバージョンなどのメタデータを充実させることをお勧めします。

3. メタデータを通じてファイル、ディレクトリ、外部参照をアーティファクトに追加します。
インテグレーションの `wandb_artifact_configuration` オブジェクトを使用して、任意のファイル、ディレクトリ、または外部参照（Amazon S3、GCS、HTTP…）を追加できます。詳細については、[アーティファクト設定セクション](#configuration-1) の高度な例を参照してください。

4. アーティファクトが生成される場合、`@op` の代わりに `@asset` を使用する。
アーティファクトは資産です。Dagsterがその資産を管理する場合、資産を使用することをお勧めします。これにより、Dagit Asset Catalogでの監視性が向上します。

5. Dagster外で作成されたアーティファクトを消費するには SourceAsset を使用します。
これにより、インテグレーションの利点を享受して外部で作成されたアーティファクトを読み取ることができます。そうでなければ、インテグレーションによって作成されたアーティファクトのみを使用できます。

6. W&B Launch を使用して大規模モデルのトレーニングを専用のコンピュートでオーケストレーションする。
小規模なモデルはDagsterクラスター内でトレーニングできます。また、GPUノードを持つDagsterクラスターをKubernetesで実行できます。大規模なモデルのトレーニングにはW&B Launchを使用することをお勧めします。これにより、インスタンスの過負荷を防ぎ、適切なコンピュートを提供できます。

7. Dagster内で実験管理する際、W&B Run ID を Dagster Run ID に設定してください。
[Runを再開可能](../runs/resuming.md)にし、W&B Run ID を Dagster Run ID または任意の文字列に設定することをお勧めします。この推奨事項に従うことで、Dagster内でモデルをトレーニングする際にW&BのメトリクスとW&Bのアーティファクトが同じW&B Runに保存されることを保証します。

W&B Run ID を Dagster Run ID に設定するか:
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または、任意の W&B Run ID を選択し、IO Manager 設定に渡します:
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

8. 大きな W&B Artifacts に対して特定のデータのみを get または get_path で収集する。
デフォルトでは、インテグレーションはアーティファクト全体をダウンロードします。非常に大きなアーティファクトを使用している場合、特定のファイルやオブジェクトのみを収集することを検討する価値があります。これにより、速度とリソースの利用が向上します。

9. Python オブジェクトの場合、ピクルモジュールをユースケースに適応させる。
デフォルトでは、W&B インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用します。しかし、一部のオブジェクトはこれに対応していません。たとえば、yield を持つ関数をピクルしようとするとエラーが発生します。W&B は他のピクルベースのシリアル化モジュール ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) にも対応しています。

