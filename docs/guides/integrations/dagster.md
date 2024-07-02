---
description: W&B を Dagster と統合するためのガイド。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dagster

DagsterとW&B (Weights & Biases) を使用して、MLOpsパイプラインを管理し、ML資産を維持します。W&Bとのインテグレーションにより、Dagster内で簡単に以下のことができます:

* [W&B Artifacts](../artifacts/intro.md) の使用および作成。
* [W&B Model Registry](../model_registry/intro.md) でのRegistered Modelsの使用および作成。
* [W&B Launch](../launch/intro.md) を使用して専用のコンピュータでトレーニングジョブを実行。
* opsと資産で [wandb](../../ref/python/README.md) クライアントを使用。

W&BとDagsterのインテグレーションは、W&B-specificなDagsterリソースとIO Managerを提供します:

* `wandb_resource`: W&B APIに認証し、通信するために使用されるDagsterリソース。
* `wandb_artifacts_io_manager`: W&B Artifactsを消費するためのDagster IO Manager。

次のガイドでは、DagsterでW&Bを使用するための前提条件を満たす方法、opsと資産でW&B Artifactsを作成および使用する方法、W&B Launchを使用する方法、推奨されるベストプラクティスについて説明します。

## 始める前に
Weights & BiasesでDagsterを使用するには、次のリソースが必要です:
1. **W&B API Key**.
2. **W&B entity (user or team)**: エンティティは、W&B RunsおよびArtifactsを送信するユーザー名またはチーム名です。runをログに記録する前に、W&BのアプリケーションUIでアカウントまたはチームエンティティを必ず作成してください。エンティティを指定しなかった場合、runは通常、ユーザー名であるデフォルトのエンティティに送信されます。 **Project Defaults** の設定でデフォルトエンティティを変更できます。
3. **W&B project**: [W&B Runs](../runs/intro.md) が保存されるプロジェクトの名前。

W&Bエンティティは、W&B Appのユーザーまたはチームのプロフィールページで確認できます。既存のW&Bプロジェクトを使用することも、新しいプロジェクトを作成することもできます。新しいプロジェクトはW&B Appのホームページまたはユーザー/チームのプロフィールページで作成できます。プロジェクトが存在しない場合、初回使用時に自動的に作成されます。以下の手順では、APIキーの取得方法を示しています。

### APIキーの取得方法
1. [W&Bにログイン](https://wandb.ai/login) します。注: W&B Serverを使用している場合、管理者にインスタンスホスト名を問い合わせてください。
2. [認可ページ](https://wandb.ai/authorize) またはユーザー/チームの設定でAPIキーを取得します。プロダクション環境では、 [サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) を所有しているキーを使用することをお勧めします。
3. そのAPIキーを環境変数に設定します。 `export WANDB_API_KEY=YOUR_KEY`。

以下の例では、Dagsterコード内でAPIキーを指定する方法を示します。 `wandb_config` ネストされた辞書内にエンティティとプロジェクト名を指定してください。異なるW&B Projectを使用する場合、異なる `wandb_config`値を異なるops/assetsに渡すことができます。渡すことができる可能なキーについては、以下の設定セクションを参照してください。

<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

Example: `@job`の設定
```python
# config.yamlにこれを追加
# あるいはDagit's LaunchpadまたはJobDefinition.execute_in_processで設定できます
# リファレンス: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをW&Bエンティティに置き換える
     project: my_project # これをW&Bプロジェクト名に置き換える


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

Example: 資産を使用する`@repository`の設定

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
                   {"cache_duration_in_minutes": 60} # ファイルを1時間のみキャッシュ
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # これをW&Bエンティティに置き換える
                       "project": "my_project", # これをW&Bプロジェクト名に置き換える
                   }
               }
           },
       ),
   ]
```
この例では、`@job`の例とは異なり、IO Managerのキャッシュ期間を設定しています。

  </TabItem>
</Tabs>

### 設定
以下の設定オプションは、インテグレーションが提供するW&B-specificなDagsterリソースおよびIO Managerの設定として使用されます。

* `wandb_resource`: W&B APIと通信するために使用されるDagster [リソース](https://docs.dagster.io/concepts/resources)。提供されたAPIキーを使用して自動認証します。プロパティ:
    * `api_key`: (str, 必須): W&B APIと通信するために必要なW&B APIキー。
    * `host`: (str, 任意): 使用するAPIホストサーバー。W&B Serverを使用する場合のみ必要です。デフォルトはPublic Cloudホスト: [https://api.wandb.ai](https://api.wandb.ai)
* `wandb_artifacts_io_manager`: W&B Artifactsを消費するためのDagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: (int, 任意) ローカルストレージとキャッシングに使用されるベースディレクトリ。W&B ArtifactsおよびW&B Runログはそのディレクトリから読み書きされます。デフォルトは`DAGSTER_HOME`ディレクトリです。
    * `cache_duration_in_minutes`: (int, 任意) W&B ArtifactsおよびW&B Runログがローカルストレージに保持される時間を定義します。その時間内に開かれなかったファイルとディレクトリのみがキャッシュから削除されます。キャッシュのパージはIOマネージャーの実行終了時に行われます。キャッシュを完全に無効にしたい場合は0に設定してください。キャッシュは、同じマシンで実行されるジョブ間でアーティファクトが再利用される場合に速度を向上させます。デフォルトは30日です。
    * `run_id`: (str, 任意): このランの一意のIDで、再開に使用されます。プロジェクト内で一意でなければならず、ランを削除するとIDを再利用できません。名前フィールドには短い説明名を使用するか、run間でハイパーパラメータを比較するためにconfigを使用してください。IDには次の特殊文字を含めることはできません: `/\#?%:..`。実験追跡をDagster内で行う場合、IOマネージャーを使用してランを再開するためにRun IDを設定する必要があります。デフォルトではDagster Run IDに設定されています。例: `7e4df022-1bf2-44b5-a383-bb852df4077e`。
    * `run_name`: (str, 任意) このランの短い表示名です。UIでこのランを識別するために使用します。デフォルトでは、次の形式の文字列に設定されています dagster-run-[最初のDagster Run IDの8文字] 例: `dagster-run-7e4df022`。
    * `run_tags`: (list[str], 任意): このランのUIに表示されるタグのリストです。タグはランをまとめたり、一時的なラベルを適用するために便利です。ランをフィルタリングする際に特定のタグを持つランだけに絞り込むことができます。インテグレーションで使用されるすべてのW&B Runには、`dagster_wandb`タグが付きます。

## W&B Artifactsの使用

W&B ArtifactとのインテグレーションはDagster IO Managerに依存しています。

[IO Managers](https://docs.dagster.io/concepts/io-management/io-managers) は、資産やopの出力を保存し、下流の資産やopに入力として読み込む責任を負うユーザー提供のオブジェクトです。例えば、IO Managerはファイルシステム上のファイルからオブジェクトを保存および読み込みすることができます。

インテグレーションはW&B Artifacts用のIO Managerを提供します。これにより、任意のDagster `@op`または`@asset`がネイティブにW&B Artifactsを作成および消費できます。以下はPythonリストを含むデータセットタイプのW&B Artifactを生成する`@asset`の簡単な例です。

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
    return [1, 2, 3] # これはArtifactに保存されます
```

`@op`、`@asset`、および`@multi_asset`をメタデータ設定で注釈し、Artifactsを書き込むことができます。同様に、Dagster外で作成されたW&B Artifactsを消費することもできます。

## W&B Artifactsの書き込み
続ける前に、W&B Artifactsの使用方法を十分に理解していることを推奨します。[Artifactsに関するガイド](../artifacts/intro.md)を読むことを検討してください。

Python関数からオブジェクトを返すことで、W&B Artifactを書き込むことができます。W&Bがサポートするオブジェクトは以下の通りです:
* Pythonオブジェクト (int, dict, list…)
* W&Bオブジェクト (Table, Image, Graph…)
* W&B Artifactオブジェクト

以下の例では、Dagsterの資産 (`@asset`) を使用してW&B Artifactsを書き込む方法を示します。

<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python objects', value: 'python_objects'},
    {label: 'W&B object', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[ピクル](https://docs.python.org/3/library/pickle.html) モジュールでシリアル化できるものは何でもピクルされ、インテグレーションによって作成されたアーティファクトに追加されます。内容はDagster内でそのアーティファクトを読むときにunpickleされます（詳細については [Read artifacts](#read-wb-artifacts) を参照してください）。

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

W&Bは複数のピクルベースのシリアル化モジュール（[pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートしています。さらに高度なシリアル化として [ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) も使用できます。詳細については [Serialization](#serialization-configuration) セクションを参照してください。

  </TabItem>
  <TabItem value="wb_object">

ネイティブのW&Bオブジェクト（例: [Table](../../ref/python/data-types/table.md)、[Image](../../ref/python/data-types/image.md)、[Graph](../../ref/python/data-types/graph.md)）はインテグレーションによって作成されたアーティファクトに追加されます。以下はTableを使用する例です。

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

複雑なユースケースでは、独自のアーティファクトオブジェクトを構築する必要がある場合があります。この場合でも、インテグレーションはメタデータを両側に追加する便利な機能を提供します。

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
`@op`、`@asset`および`@multi_asset`には、`wandb_artifact_configuration`と呼ばれる設定辞書をメタデータとしてデコレータ引数に渡すことができます。この設定はW&B ArtifactsのIO Managerの読み書きを制御するためのものです。

`@op`の場合、これは [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を通じて出力メタデータにあります。 
`@asset`の場合、資産のメタデータ引数にあります。
`@multi_asset`の場合、各出力メタデータ ([AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数) にあります。

以下のコード例では、`@op`、`@asset`および`@multi_asset`の計算に辞書を設定する方法を示しています。

<Tabs
  defaultValue="op"
  values={[
    {label: 'Example for @op', value: 'op'},
    {label: 'Example for @asset', value: 'asset'},
    {label: 'Example for @multi_asset', value: 'multi_asset'},
  ]}>
  <TabItem value="op">

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

  </TabItem>
  <TabItem value="asset">

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

名前は設定を経由して渡す必要はありません。@assetには既に名前があります。インテグレーションはアーティファクト名を資産名として設定します。

  </TabItem>
  <TabItem value="multi_asset">

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

  </TabItem>
</Tabs>

サポートされているプロパティ:
* `name`: (str) このアーティファクトの人間が読める名前。UIでこのアーティファクトを識別したり、use_artifactコールで参照するために使用されます。名前には文字、数字、アンダースコア、ハイフン、ドットを含むことができます。プロジェクト内で一意でなければなりません。`@op`では必須です。
* `type`: (str) アーティファクトのタイプ。アーティファクトを整理し、区別するために使用されます。一般的なタイプにはdatasetやmodelがありますが、任意の文字列を使用できます。出力がすでにアーティファクトでない場合、必須です。
* `description`: (str) アーティファクトの説明を提供するフリーテキスト。説明はUIでマークダウンとしてレンダリングされるため、表やリンクなどを配置するのに適しています。
* `aliases`: (list[str]) アーティファクトに適用したい一つ以上のエイリアスを含む配列。インテグレーションはセットされているかどうかにかかわらず、「latest」タグもそのリストに追加します。これは、モデルやデータセットのバージョン管理に効果的な方法です。
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): アーティファクトに含める各ローカルディレクトリの設定を含む配列。SDKの同名のメソッドと同じ引数をサポートします。
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): アーティファクトに含める各ローカルファイルの設定を含む配列。SDKの同名のメソッドと同じ引数をサポートします。
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): アーティファクトに含める各外部参照の設定を含む配列。SDKの同名のメソッドと同じ引数をサポートします。
* `serialization_module`: (dict) 使用するシリアル化モジュールの設定。詳細についてはシリアル化セクションを参照してください。
    * `name`: (str) シリアル化モジュールの名前。受け入れ可能な値: `pickle`, `dill`, `cloudpickle`, `joblib`。モジュールはローカルで利用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアル化関数に渡される任意の引数。モジュールのdumpメソッドの引数と同じ引数を受け入れます。例: `{"compress": 3, "protocol": 4}`。

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

資産は両側の統合に有用なメタデータで具体化されます:
* W&B側: ソースインテグレーションの名前とバージョン、使用されたPythonバージョン、ピクルプロトコルバージョンなど。
* Dagster側:
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、タイプ、バージョン、サイズ、URL
    * W&B Entity
    * W&B Project

次の画像は、統合によりDagster資産に追加されたW&Bのメタデータを示しています。この情報は統合なしでは利用できません。

![](/images/integrations/dagster_wb_metadata.png)

次の画像は、提供された設定がW&B Artifactの有用なメタデータで豊かにされたことを示しています。この情報は再現性とメンテナンスに役立ちます。それは統合なしでは利用できません。

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)

:::info
mypyのような静的型チェックツールを使用する場合、設定タイプ定義オブジェクトを次のようにインポートしてください:

```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

### パーティションの使用

インテグレーションはネイティブで [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をサポートします。

以下は `DailyPartitionsDefinition` を使用してパーティションされている例です。
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
    context.log.info(f"{partition_key}のためのパーティション化された資産を作成しています")
    return random.randint(0, 100)
```
このコードは各パーティションごとに1つのW&B Artifactを生成します。それらは資産名の下でアーティファクトパネル（UI）に表示され、パーティションキーで区切られます。例えば`my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、`my_daily_partitioned_asset.2023-01-03`など。複数次元でパーティション化された資産は、各次元がドットで区切られます。例: `my_asset.car.blue`。

:::caution
インテグレーションは1つのrunで複数のパーティションを具体化させることを許可していません。資産を具体化させるために複数のrunを実行する必要があります。Dagitで資産を具体化する際に実行することができます。

![](/images/integrations/dagster_multiple_runs.png)
:::

#### 高度な使用法
- [パーティショーンドジョブ](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [シンプルなパーティショーンド資産](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [複数パーティショーンド資産](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [高度なパーティション使用](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts を読む
W&B Artifactsを読むのは書くのと似ています。設定辞書 `wandb_artifact_configuration` を`@op`または`@asset`に設定できます。違いは、出力の代わりに入力に設定することです。

`@op`の場合、入力メタデータを通して [In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数に設定されます。アーティファクトの名前を明示的に渡す必要があります。

`@asset`の場合、入力メタデータを通して [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) Inメタデータ引数に設定されます。親資産の名前と一致すべきなので、アーティファクト名を渡す必要はありません。

統合外で作成されたアーティファクトに依存関係を持たせたい場合 [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。常にその資産の最新バージョンを読み込みます。

以下の例では、さまざまなopsからアーティファクトを読む方法を示します。

<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'op'},
    {label: 'Created by another @asset', value: 'asset'},
    {label: 'Artifact created outside Dagster', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

`@op`からアーティファクトを読む
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

他の`@asset`によって作成されたアーティファクトを読む
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 入力引数名を変更したくない場合、'key'を削除できます
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

Dagster外で作成されたアーティファクトを読む:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B アーティファクトの名前
   description="Dagster外で作成されたアーティファクト",
   io_manager_key="wandb_artifacts_manager",
)

@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>

### 設定
以下の設定は、IO マネージャーが収集して、装飾関数への入力として提供する内容を示しています。以下の読み取りパターンがサポートされています。

1. アーティファクト内の名前付きオブジェクトを取得するには、get を使用します:

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

2. アーティファクト内のダウンロードされたファイルのローカルパスを取得するには、get_path を使用します:

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

3. アーティファクトオブジェクト全体を取得する (内容はローカルにダウンロードされます):
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
* `get`: (str) 指定された相対名のアーティファクトにあるW&Bオブジェクトを取得します。
* `get_path`: (str) 指定された相対名のアーティファクトにあるファイルへのパスを取得します。

### シリアル化設定
デフォルトでは、インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、一部のオブジェクトはそれと互換性がありません。たとえば、yield を含む関数をピクルしようとするとエラーが発生します。

ピクルベースのシリアル化モジュール（[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートしています。さらに [ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のような高度なシリアル化もサポートしています。それらを使用するには、シリアル化された文字列を返すか、直接アーティファクトを作成する必要があります。使用ケースに応じて適切なものを選んでください。この主題に関する利用可能な文献を参照してください。

### ピクルベースのシリアル化モジュール

:::caution
ピクルは既知のセキュリティリスクを伴います。セキュリティが懸念される場合は、W&Bオブジェクトのみを使用してください。データに署名し、ハッシュキーを自身のシステムに保存することをお勧めします。複雑なユースケースについては、お気軽にお問い合わせください。お手伝いさせていただきます。
:::

シリアル化を設定するには、`wandb_artifact_configuration`の`serialization_module`辞書を使用して設定します。実行するマシンでモジュールが利用可能であることを確認してください。

インテグレーションは、そのアーティファクトを読む時に適切なシリアル化モジュールを自動的に認識します。

現在サポートされているモジュールは、pickle、dill、cloudpickle、joblibです。

以下は、joblibでシリアル化された「モデル」を作成し、それを推論に使用する簡単な例です。

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
    # これは本当のMLモデルではありませんが、ピクルモジュールでは不可能です
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

### 高度なシリアル化フォーマット (ONNX、PMML)
ONNXやPMMLのような交換ファイル形式を使用することは一般的です。インテグレーションはそれらの形式をサポートしていますが、ピクルベースのシリアル化とは異なる方法が必要です。

2つの異なる方法があります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常のPythonオブジェクトのように返します。インテグレーションがその文字列をピクルします。その文字列を使用してモデルを再構築できます。
2. シリアル化されたモデルを含む新しいローカルファイルを作成し、そのファイルを使用してカスタムArtifactを構築します。

以下は、Scikit-learnモデルを使用してONNXを使用してシリアル化する例です。

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
    # https://onnx.ai/sklearn-onnx/ からインスパイアされた例

    # モデルのトレーニング
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX形式に変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクトを書き込み（モデル＋テストセット）
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
    # https://onnx.ai/sklearn-onnx/ からインスパイアされた例

    # ONNXランタイムでの予測を計算
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

インテグレーションはネイティブで [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をサポートします。

特定のパーティションまたはすべてのパーティションを選択的に読み込むことができます。

すべてのパーティションは辞書で提供され、キーと値がパーティションキーとアーティファクトの内容を表します。

<Tabs
  defaultValue="all"
  values={[
    {label: 'Read all partitions', value: 'all'},
    {label: 'Read specific partitions', value: 'specific'},
  ]}>
  <TabItem value="all">

上流の `@asset` のすべてのパーティションを読み込み、それらは辞書で与えられます。この辞書では、キーと値がパーティションキーとアーティファクトの内容に対応します。
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
  </TabItem>
  <TabItem value="specific">

`AssetIn` の `partition_mapping` 設定を使用すると、特定のパーティションを選択できます。この場合、 `TimeWindowPartitionMapping` を使用します。
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
  </TabItem>
</Tabs>

設定オブジェクト `metadata` は、Weights & Biases (wandb) がプロジェクト内の異なるアーティファクトパーティションとどのように対話するかを設定します。

オブジェクト `metadata` には `wandb_artifact_configuration` という名前のキーがあり、その中にはさらにネストされたオブジェクト `partitions` があります。

この `partitions` オブジェクトは、各パーティションの名前をその設定にマッピングします。各パーティションの設定は、それを取得する方法を指定できます。これらの設定には `get`、`version`、`alias` などのキーが含まれることがあります。

**設定キー**

1. `get`:
`get`キーは、データを取得するW&Bオブジェクト（Table, Imageなど）名を指定します。
2. `version`:
`version`キーは、アーティファクトの特定のバージョンを取得したい場合に使用します。
3. `alias`:
`alias`キーは、エイリアスでアーティファクトを取得するのに使用します。

**ワイルドカード設定**

ワイルドカード `"*"` は、設定されていないすべてのパーティションを意味します。これは、明示的に `partitions` オブジェクトに記載されていないパーティションのためのデフォルト設定を提供します。

例えば、

```python
"*": {
    "get": "default_table_name",
},
```
この設定は、明示的に設定されていないすべてのパーティションについて、データが `default_table_name` という名前のテーブルから取得されることを意味します。

**特定のパーティション設定**

特定のパーティションのための設定をそのキーを使用して提供することで、ワイルドカード設定を上書きできます。

例えば、

```python
"yellow": {
    "get": "custom_table_name",
},
```
この設定は、`yellow`という名前のパーティションについて、データが `custom_table_name` という名前のテーブルから取得されることを意味しています。ワイルドカード設定を上書きすることになります。

**バージョン管理およびエイリアス**

バージョン管理およびエイリアスのために、`version`および`alias`キーを設定内で指定できます。

例えば、
```python
"orange": {
    "version": "v0",
},
```
この設定は、`orange`アーティファクトパーティションのバージョン`v0`からデータを取得します。

エイリアスの場合、
```python
"blue": {
    "alias": "special_alias",
},
```
この設定は、エイリアス`special_alias`（設定内で`blue`と呼ばれる）を持つアーティファクトパーティションの`default_table_name`テーブルからデータを取得します。

### 高度な使用法
インテグレーションの高度な使用方法を見るには、以下の完全なコード例を参照してください:
* [資産の高度な使用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [パーティショーンドジョブ例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルをモデルレジストリにリンクする例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launchの使用

:::caution
アクティブな開発中のベータプロダクト
Launchに興味がありますか？W&B Launchのカスタマーパイロットプログラムに参加するために、アカウントチームにご連絡ください。
パイロット顧客はAWS EKSまたはSageMakerを使用する必要があります。最終的には、追加のプラットフォームをサポートする予定です。
:::

続ける前に、W&B Launchの使用方法を十分に理解することをお勧めします。[Launcherのガイド](https://docs.wandb.ai/guides/launch)を読んでください。

Dagsterのインテグレーションでは以下のことができます:
* Dagsterインスタンス内でLaunchエージェントを実行する。
* Dagsterインスタンス内でローカルLaunchジョブを実行する。
* オンプレミスまたはクラウドでのリモートLaunchジョブを実行する。

### Launchエージェント
インテグレーションは、`run_launch_agent`というインポート可能な`@op`を提供します。Launchエージェントを起動し、手動で停止されるまで長時間実行プロセスとして実行します。

エージェントはLaunchキューをポーリングし、ジョブを実行する（または外部サービスにディスパッチする）プロセスです。

設定については [リファレンス文書](../launch/intro.md) を参照してください。

Launchpadでプロパティの有用な説明を表示することもできます。

![](/images/integrations/dagster_launch_agents.png)

簡単な例
```python
# config.yamlにこれを追加
# あるいはDagit's LaunchpadまたはJobDefinition.execute_in_processで設定できます
# リファレンス: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをW&Bエンティティ名に置き換える
     project: my_project # これをW&Bプロジェクト名に置き換える
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

### Launchジョブ
インテグレーションは、`run_launch_job`というインポート可能な`@op`を提供します。これはLaunchジョブを実行します。

Launchジョブはキューに割り当てられ実行されます。キューを作成するか、デフォルトのキューを使用できます。キューを監視するアクティブなエージェントが必要です。エージェントをDagsterインスタンス内で実行することもできますが、Kubernetesで展開可能なエージェントの使用も検討できます。

設定については [リファレンス文書](../launch/intro.md) を参照してください。

Launchpadでプロパティの有用な説明を表示することもできます。

![](/images/integrations/dagster_launch_jobs.png)

簡単な例
```python
# config.yamlにこれを追加
# あるいはDagit's LaunchpadまたはJobDefinition.execute_in_processで設定できます
# リファレンス: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをW&Bエンティティ名に置き換える
     project: my_project # これをW&Bプロジェクト名に置き換える
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
   run_launch_job.alias("my_launched_job")()  # ジョブにエイリアスを付けて名前を変更します
```

## ベストプラクティス

1. IO マネージャーを使用してアーティファクトを読み書きする。
[`Artifact.download()`](../../ref/python/artifact.md#download) または [`Run.log_artifact()`](../../ref/python/run.md#log_artifact) を直接使用する必要はありません。これらのメソッドは統合で処理されます。保存したいデータを返すだけで、統合に任せます。これにより、W&Bのアーティファクトの系統が改善されます。

2. 複雑なユースケースではない限り、アーティファクトオブジェクトを直接構築しない。
PythonオブジェクトおよびW&Bオブジェクトは、ops/assetsから返されるべきです。統合がアーティファクトのパッケージングを処理します。
複雑なユースケースでは、Dagsterジョブ内で直接アーティファクトを構築できます。ソースインテグレーションの名前とバージョン、使用されたPythonバージョン、ピクルプロトコルバージョンなどのメタデータをインテグレーションで拡充することをお勧めします。

3. メタデータを通じてファイル、ディレクトリおよび外部参照をアーティファクトに追加する。
統合の `wandb_artifact_configuration` オブジェクトを使用して、任意のファイル、ディレクトリまたは外部参照（Amazon S3、GCS、HTTPなど）を追加します。詳細については [Artifact 設定セクション](#configuration-1) の高度な例を参照してください。

4. アーティファクトが生成される場合は、@asset を使用する。
アーティファクトは資産です。Dagsterで資産を管理する場合は、資産を使用することをお勧めします。これにより、Dagit資産カタログでの可視性が向上します。

5. SourceAsset を使用して、Dagster外で作成されたアーティファクトを消費する。
これにより、統合の利点を利用して外部で作成されたアーティファクトを読み込むことができます。それ以外の場合、統合で作成されたアーティファクトのみを使用できます。

6. 大規模なモデルには、専用のコンピュータでのトレーニングを調整するためにW&B Launchを使用する。
小規模なモデルはDaksterクラスター内でトレーニングできます。DagsterをGPUノードを持つKubernetesクラスターで実行することもできます。大規模なモデルのトレーニングにはW&B Launchを使用することをお勧めします。これにより、インスタンスの過負荷を防ぎ、より適切なコンピュータにアクセスできます。

7. Dagster 内での実験追跡時には、W&B Run ID を Dagster Run ID と同じ値に設定する。
[Runが再開可能](../runs/resuming.md) であることを確認し、W&B Run ID を Dakster Run ID または任意の文字列に設定することをお勧めします。この推奨事項に従うことで、Dagster内でモデルをトレーニングする際に、W&B のメトリクスと W&B のアーティファクトが同じ W&B Run に保存されます。

W&B Run ID を Dagster Run ID に設定します。
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または、独自のW&B Run IDを選択して、IOマネージャー設定に渡します。
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

8. 大規模なW&Bアーティファクトの場合、必要なデータのみをgetまたはget_pathで収集する。
デフォルトでは、インテグレーションはアーティファクト全体をダウンロードします。非常に大きなアーティファクトを使用している場合、必要な特定のファイルやオブジェクトのみを収集することを検討すると良いでしょう。これにより、速度とリソースの利用効率が向上します。

9. Pythonオブジェクトの場合、ピクルモジュールをユースケースに合わせて調整する。
デフォルトでは、W&Bインテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用します。しかし、一部のオブジェクトはそれと互換性がありません。例えば、yieldを含む関数をピクルしようとするとエラーが発生します。W&Bは他のピクルベースのシリアル化モジュール（[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートしています。

