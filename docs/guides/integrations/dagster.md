---
description: W&B を Dagster と統合する方法に関するガイド。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Dagster

Dagster と W&B (W&B) を使用して MLOps パイプラインをオーケストレーションし、ML アセットを管理します。W&B とのインテグレーションにより、Dagster 内で以下のことが簡単に行えます：

* [W&B Artifacts](../artifacts/intro.md) の使用と作成。
* [W&B Model Registry](../model_registry/intro.md) での Registered Models の使用と作成。
* [W&B Launch](../launch/intro.md) を使用して専用のコンピューティングでトレーニングジョブを実行。
* ops およびアセットで [wandb](../../ref/python/README.md) クライアントを使用。

W&B Dagster インテグレーションは、W&B 固有の Dagster リソースと IO マネージャーを提供します：

* `wandb_resource`: W&B API への認証と通信に使用される Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts の消費に使用される Dagster IO マネージャー。

以下のガイドは、Dagster で W&B を使用するための前提条件を満たす方法、ops およびアセットで W&B Artifacts を作成および使用する方法、W&B Launch の使用方法、推奨されるベストプラクティスについて説明します。

## 始める前に
Weights and Biases 内で Dagster を使用するには、以下のリソースが必要です：
1. **W&B API Key**。
2. **W&B entity (user or team)**: entity は、W&B Runs と Artifacts を送信するユーザー名またはチーム名です。runs をログに記録する前に、W&B App UI でアカウントまたはチームエンティティを作成してください。エンティティを指定しない場合、run は通常はユーザー名であるデフォルトのエンティティに送信されます。デフォルトのエンティティは **Project Defaults** の設定で変更できます。
3. **W&B project**: [W&B Runs](../runs/intro.md) が格納されるプロジェクトの名前。

W&B entity は、W&B アプリのそのユーザーまたはチームのプロフィールページを確認して見つけることができます。既存の W&B プロジェクトを使用することも、新しいプロジェクトを作成することもできます。新しいプロジェクトは、W&B アプリのホームページまたはユーザー/チームのプロフィールページで作成できます。プロジェクトが存在しない場合、最初に使用するときに自動的に作成されます。以下の手順は、APIキーの取得方法を示しています：

### APIキーの取得方法
1. [W&B にログイン](https://wandb.ai/login)。注：W&B サーバーを使用している場合は、インスタンスのホスト名を管理者に問い合わせてください。
2. [authorize ページ](https://wandb.ai/authorize) またはユーザー/チーム設定に移動し、APIキーを収集します。プロダクション環境では、そのキーを所有する [サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) を使用することをお勧めします。
3. その APIキーの環境変数を設定します。`export WANDB_API_KEY=YOUR_KEY`。

以下の例は、Dagster コード内で APIキーを指定する場所を示しています。`wandb_config` ネストされた辞書内にエンティティとプロジェクト名を必ず指定してください。異なる W&B Project を使用したい場合は、異なる `wandb_config` 値を異なる ops/アセット に渡すことができます。渡すことができるキーの詳細については、以下の設定セクションをご覧ください。

<Tabs defaultValue="job" values={[ {label: 'configuration for @job', value: 'job'}, {label: 'configuration for @repository using assets', value: 'repository'}, ]}>
  <TabItem value="job">

Example: configuration for `@job`
```python
# これを config.yaml に追加します
# または、Dagit's Launchpad または JobDefinition.execute_in_process で設定を行うこともできます
# 参考： https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
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
                   {"cache_duration_in_minutes": 60} # ファイルを1時間のみキャッシュ
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
この例では、`@job` の例とは異なり、IOマネージャーのキャッシュ期間を設定しています。

  </TabItem>
</Tabs>

### 設定
以下の設定オプションは、インテグレーションによって提供される W&B 固有の Dagster リソースと IO マネージャーの設定として使用されます。

* `wandb_resource`: W&B API と通信するための Dagster [リソース](https://docs.dagster.io/concepts/resources)。提供された APIキーを使用して自動的に認証します。プロパティ：
    * `api_key`: (str, 必須): W&B API と通信するために必要な W&B APIキー。
    * `host`: (str, 任意): 使用したい API ホストサーバー。W&B Server を使用している場合にのみ必要です。デフォルトはパブリッククラウドホスト：[https://api.wandb.ai](https://api.wandb.ai)
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster [IOマネージャー](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ：
    * `base_dir`: (int, 任意) ローカルストレージとキャッシュのために使用されるベースディレクトリ。W&B Artifacts と W&B Run のログはそのディレクトリに書き込まれ、読み取られます。デフォルトでは `DAGSTER_HOME` ディレクトリが使用されます。
    * `cache_duration_in_minutes`: (int, 任意) W&B Artifacts と W&B Run のログをローカルストレージに保持する時間を定義します。その時間開かれなかったファイルとディレクトリのみがキャッシュから削除されます。キャッシュのパージは IO マネージャーの実行終了時に行われます。キャッシュを完全に無効にしたい場合は 0 に設定できます。キャッシュは、同じマシンで実行されるジョブ間で Artifact が再利用される場合に速度を向上させます。デフォルトは 30 日です。
    * `run_id`: (str, 任意): この run のための一意の ID であり、再開に使用されます。それはプロジェクト内で一意である必要があり、run を削除した場合、その ID を再利用することはできません。短い説明的な名前には name フィールドを使用するか、run 間で比較するハイパーパラメータを保存するために config を使用します。ID には次の特殊文字を含めることはできません: `/\#?%:..` 実験管理内で run を再開できるようにするために、Dagster 内で run_id を設定する必要があります。デフォルトでは、Dagster run ID 例: `7e4df022-1bf2-44b5-a383-bb852df4077e` に設定されています。
    * `run_name`: (str, optional) この run の短い表示名で、UI でこの run を識別する方法です。デフォルトでは、次の形式の文字列が設定されています: dagster-run-[Dagster Run ID の最初の 8 文字] 例: `dagster-run-7e4df022`.
    * `run_tags`: (list[str], optional): UI でこの run のタグのリストを生成する文字列のリスト。タグは run を一緒に整理したり、「ベースライン」や「プロダクション」などの一時的なラベルを適用したりするのに便利です。UI でタグを追加および削除するのは簡単ですし、特定のタグを持つ run のみに絞り込むことも簡単です。インテグレーションによって使用される任意の W&B Run には `dagster_wandb` タグが付けられます。

## W&B Artifacts の使用

W&B Artifacts とのインテグレーションは、Dagster IO マネージャーに依存しています。

[IO マネージャー](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットまたは op の出力を保存し、それを下流のアセットまたは op に入力として読み込む責任を持つユーザー提供のオブジェクトです。例えば、IO マネージャーはファイルシステム上のファイルからオブジェクトを保存および読み込むかもしれません。

インテグレーションは W&B Artifacts 用の IO マネージャーを提供します。これにより、任意の Dagster `@op` または `@asset` がネイティブに W&B Artifacts を作成および消費できます。以下は、Python リストを含むデータセットタイプの W&B Artifact を生成する `@asset` の簡単な例です。

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

`@op`、`@asset`、および `@multi_asset` をメタデータ設定でアノテートして、Artifacts を書き込むことができます。同様に、Dagster 外で作成された W&B Artifacts も消費することができます。

## W&B Artifacts の作成
続行する前に、W&B Artifacts の使用方法について十分に理解しておくことをお勧めします。[Artifacts ガイド](../artifacts/intro.md) を読むことを検討してください。

Python 関数からオブジェクトを返して、W&B Artifact を作成します。以下のオブジェクトが W&B によってサポートされています：
* Pythonオブジェクト (int, dict, list…)
* W&Bオブジェクト (Table, Image, Graph…)
* W&B Artifact オブジェクト

以下の例は、Dagster アセット（`@asset`）で W&B Artifacts を作成する方法を示しています：

<Tabs defaultValue="python_objects" values={[ {label: 'Python objects', value: 'python_objects'}, {label: 'W&B object', value: 'wb_object'}, {label: 'W&B Artifacts', value: 'wb_artifact'}, ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズ可能なものはすべてピクルされ、インテグレーションによって作成された Artifact に追加されます。Dagster 内でその Artifact を読むときに内容がアンピクルされます（詳細については [アーティファクトを読む](#read-wb-artifacts) を参照）。

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

W&B は複数のピクルベースのシリアライズモジュール（[pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートしています。より高度なシリアライズを使用することもできます。詳細については、[Serialization](#serialization-configuration) セクションを参照してください。

  </TabItem>
  <TabItem value="wb_object">

ネイティブの W&B オブジェクト [Table](../../ref/python/data-types/table.md)、[Image](../../ref/python/data-types/image.md)、[Graph](../../ref/python/data-types/graph.md) などがインテグレーションによって作成された Artifact に追加されます。ここでは Table を使用した例を示します。

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

複雑なユースケースでは、自分自身で Artifact オブジェクトを構築する必要があるかもしれません。それでもインテグレーションは、両側のメタデータを増強するなどの有用な追加機能を提供します。

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

### Configuration
`@op`、`@asset`、および `@multi_asset` に設定できる設定辞書 `wandb_artifact_configuration` があります。この辞書はデコレーターの引数としてメタデータで渡す必要があります。この設定は、W&B Artifacts の IO マネージャーの読み書きを制御するために必要です。

`@op` の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を通じて出力メタデータに配置されます。
`@asset` の場合、アセットのメタデータ引数に配置されます。
`@multi_asset` の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数を通じて各出力メタデータに配置されます。

以下のコード例は、`@op`、`@asset`、`@multi_asset` 計算に辞書を設定する方法を示しています。

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

設定では名前を渡す必要はありません。`@asset` には既に名前が付いているためです。インテグレーションはアセット名をArtifactの名前として設定します。

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

サポートされているプロパティ:
* `name`: (str) このArtifactの人間が読める名前。この名前でUI上でArtifactを識別したり、`use_artifact` 呼び出しで参照したりできます。名前には、アルファベット、数字、アンダースコア、ハイフン、およびドットを含めることができます。プロジェクト全体で一意である必要があります。`@op` に必須。
* `type`: (str) Artifactのタイプ。これはArtifactsを組織化および区別するために使用されます。一般的なタイプにはデータセットやモデルがありますが、文字、数字、アンダースコア、ハイフン、およびドットを含む任意の文字列を使用できます。出力が既にArtifactでない場合に必須。
* `description`: (str) Artifactの説明を含む自由なテキスト。説明はUI上でmarkdownレンダリングされるため、テーブルやリンクなどを記載するのに適しています。
* `aliases`: (list[str]) Artifactに適用したい1つ以上のエイリアスを含む配列。インテグレーションは「latest」タグもそのリストに追加します（設定されているかどうかにかかわらず）。これはモデルやデータセットのバージョン管理を効果的に行う方法です。
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): Artifactに含める各ローカルディレクトリの設定を含む配列。SDK内の同名メソッドと同じ引数をサポートします。
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): Artifactに含める各ローカルファイルの設定を含む配列。SDK内の同名メソッドと同じ引数をサポートします。
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): Artifactに含める各外部参照の設定を含む配列。SDK内の同名メソッドと同じ引数をサポートします。
* `serialization_module`: (dict) 使用するシリアル化モジュールの設定。詳細についてはシリアル化セクションを参照してください。
    * `name`: (str) シリアル化モジュールの名前。受け入れられる値: `pickle`, `dill`, `cloudpickle`, `joblib`。モジュールはローカルに利用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアル化関数に渡されるオプションの引数。モジュールのdumpメソッドと同じ引数を受け入れます。例: `{"compress": 3, "protocol": 4}`。

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

アセットは、インテグレーションの両側で有用なメタデータで具現化されます:
* W&B側: ソースインテグレーションの名前とバージョン、使用されたPythonバージョン、pickleプロトコルバージョンなど。
* Dagster側:
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、タイプ、バージョン、サイズ、URL
    * W&B Entity
    * W&B Project

以下の画像は、W&B側からDagsterアセットに追加されたメタデータを示しています。これはインテグレーションなしでは利用できない情報です。

![](/images/integrations/dagster_wb_metadata.png)

以下の画像は、提供された設定がW&B Artifactで有用なメタデータで豊かにされた方法を示しています。この情報は再現性とメンテナンスに役立ちます。これはインテグレーションなしでは利用できません。

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)

:::info
mypyのような静的型チェッカーを使用している場合、設定の型定義オブジェクトを次のようにインポートします:

```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

### パーティションの使用

インテグレーションは[Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)をネイティブにサポートしています。

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
    context.log.info(f"Creating partitioned asset for {partition_key}")
    return random.randint(0, 100)
```
このコードは各パーティションごとに1つのW&B Artifactを生成します。これらはアセット名の下でArtifactパネル（UI）に表示され、パーティションキーが付加されます。例えば、`my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、`my_daily_partitioned_asset.2023-01-03`などです。複数次元でパーティション化されたアセットは各次元がドットで区切られます。例: `my_asset.car.blue`

:::caution
インテグレーションは1つのrun内で複数のパーティションを具現化することを許可していません。アセットの具現化には複数のrunを実行する必要があります。これはDagitでアセットを具現化する際に実行できます。

![](/images/integrations/dagster_multiple_runs.png)
:::

#### 高度な使用法
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [Simple partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [Multi-partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [Advanced partitioned usage](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifactsを読む
W&B Artifactsの読み込みは書き込みと似ています。設定辞書 `wandb_artifact_configuration` を `@op` または `@asset` に設定できます。唯一の違いは、この設定を出力ではなく入力に設定する必要がある点です。

`@op` の場合、[In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数を通じて入力メタデータに配置されます。Artifactの名前を明示的に渡す必要があります。

`@asset` の場合、[Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In メタデータ引数を通じて入力メタデータに配置されます。親アセットの名前と一致するはずなので、Artifactの名前を渡すべきではありません。

インテグレーションの外部で作成されたArtifactに依存関係を持ちたい場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。これは常にそのアセットの最新バージョンを読み取ります。

以下の例は、配慮されたさまざまな方法でArtifactを読む方法を示しています。

<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'op'},
    {label: 'Created by another @asset', value: 'asset'},
    {label: 'Artifact created outside Dagster', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

`@op` からArtifactを読む
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

他の`@asset`によって作成されたArtifactを読む
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 入力引数の名前を変更したくない場合、 'key' を削除できます
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

Dagsterの外部で作成されたArtifactを読む

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B Artifactの名前
   description="Artifact created outside Dagster",
   io_manager_key="wandb_artifacts_manager",
)

@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>

### 設定
以下の設定は、IOマネージャーが収集し、装飾された関数に入力として提供すべき内容を示しています。以下の読み取りパターンがサポートされています。

1. Artifactに含まれる名前付きオブジェクトを取得するには、`get`を使用します:
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

2. Artifactに含まれるダウンロードされたファイルのローカルパスを取得するには、`get_path` を使用します:
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

3. Artifactオブジェクト全体を取得するには（コンテンツはローカルにダウンロードされます）:
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
* `get`: (str) artifact相対名にあるW&Bオブジェクトを取得します。
* `get_path`: (str) artifact相対名にあるファイルのパスを取得します。

### シリアル化の設定
デフォルトでは、インテグレーションは標準の[pickle](https://docs.python.org/3/library/pickle.html)モジュールを使用しますが、一部のオブジェクトはこれに対応していません。例えば、yieldを持つ関数はそれをピクルするとエラーを引き起こします。

よりPickleベースのシリアル化モジュール([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) がサポートされています。また、[ONNX](https://onnx.ai/)や[PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)のようなより高度なシリアル化もサポートしています。返されるシリアル化された文字列や直接Artifactを作成することができます。適切な選択はユースケースによります。この主題の利用可能な文献を参照してください。

### Pickle ベースのシリアライズモジュール

:::caution
Pickling は安全でないことが知られています。セキュリティが懸念される場合は、W&B オブジェクトのみを使用してください。データに署名し、自身のシステムでハッシュキーを保存することをお勧めします。複雑なユースケースについては、遠慮なくお問い合わせください。喜んでお手伝いします。
:::

シリアライズに使用するモジュールは、`wandb_artifact_configuration` の辞書内の `serialization_module` で設定できます。このモジュールがDagsterを実行しているマシンで利用可能であることを確認してください。

このインテグレーションは、読まれる Artifact に応じて自動的にシリアライズモジュールを識別します。

現在サポートされているモジュールは pickle、dill、cloudpickle、および joblib です。

ここでは、joblib を使用してシリアライズした「モデル」を作成し、それを推論に使用する簡単な例を紹介します。

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
    # これは本物の ML モデルではありませんが、pickle モジュールでは不可能です
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

### 高度なシリアライズ形式 (ONNX, PMML)
交換ファイル形式として ONNX や PMML を使用することは一般的です。このインテグレーションはこれらの形式をサポートしていますが、Pickle ベースのシリアライズよりも少し手間がかかります。

これらの形式を使用するには、2つの異なる方法があります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常の Python オブジェクトとして返します。インテグレーションはその文字列をピクルすることができます。その文字列を使用してモデルを再構築します。
2. シリアライズされたモデルを含む新しいローカルファイルを作成し、そのファイルを使用してカスタム Artifact を作成します。

ここでは、Scikit-learn モデルを使用して ONNX を使用してシリアライズする例を示します。

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
    # 基本的には https://onnx.ai/sklearn-onnx/ のインスピレーションを受けています

    # モデルをトレーニングします。
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 形式に変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクトを書き出します（モデル + テストセット）
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
    # 基本的には https://onnx.ai/sklearn-onnx/ のインスピレーションを受けています

    # ONNX ランタイムを使用して予測を計算
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

このインテグレーションは[Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をネイティブにサポートしています。

アセットの一部、一つまたはすべてのパーティションを選択的に読み取ることができます。

全てのパーティションは辞書で提供されており、キーと値はそれぞれパーティションキーと Artifact コンテンツを表します。

<Tabs
  defaultValue="all"
  values={[
    {label: 'すべてのパーティションを読む', value: 'all'},
    {label: '特定のパーティションを読む', value: 'specific'},
  ]}>
  <TabItem value="all">

アップストリームの `@asset` のすべてのパーティションを読み取ります。それらは辞書形式で提供され、キーと値はそれぞれパーティションキーと Artifact コンテンツに対応します。
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

`AssetIn` の `partition_mapping` 設定を使用して特定のパーティションを選択できます。この例では `TimeWindowPartitionMapping` を使用しています。
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

Weights & Biases (wandb) がプロジェクト内の異なるアーティファクトパーティションとどのようにやり取りするかを設定するために、 `metadata` オブジェクトが使用されます。

オブジェクト `metadata` は、 `wandb_artifact_configuration` というキーを含んでおり、その中にはさらに `partitions` というネストされたオブジェクトが含まれています。

`partitions` オブジェクトは、それぞれのパーティションの名前を設定にマップします。各パーティションの設定には、データの取得方法を指定できます。これらの設定には、特定のパーティションの要件に応じて `get`、`version`、および `alias` といったキーが含まれます。

**設定キー**

1. `get`:
`get` キーは、データを取得するW&Bオブジェクト（テーブル、画像など）の名前を指定します。
2. `version`:
`version` キーは、特定のバージョンのArtifactを取得したいときに使用されます。
3. `alias`:
`alias` キーは、エイリアスでArtifactを取得できるようにします。

**ワイルドカード設定**

ワイルドカード `"*"` は、すべての未設定のパーティションを表します。これは、 `partitions` オブジェクトで明示的に言及されていないパーティションに対するデフォルト設定を提供します。

例、

```python
"*": {
    "get": "default_table_name",
},
```
この設定は、明示的に設定されていないすべてのパーティションに対して、`default_table_name` というテーブルからデータを取得することを意味します。

**特定のパーティション設定**

特定のパーティションに対してそのキーを使用して特定の設定を提供することで、ワイルドカード設定をオーバーライドできます。

例、

```python
"yellow": {
    "get": "custom_table_name",
},
```

この設定は、`yellow` という名前のパーティションに対して、`custom_table_name` というテーブルからデータを取得し、ワイルドカード設定をオーバーライドすることを意味します。

**バージョン管理とエイリアス**

バージョン管理とエイリアスの目的のために、設定内に特定の `version` および `alias` キーを提供することができます。

バージョンの場合、

```python
"orange": {
    "version": "v0",
},
```

この設定は、`orange` アーティファクトパーティションのバージョン `v0` からデータを取得します。

エイリアスの場合、

```python
"blue": {
    "alias": "special_alias",
},
```

この設定は、エイリアス `special_alias` を持つアーティファクトパーティションのテーブル `default_table_name` からデータを取得します（設定では `blue` として参照されています）。

### 高度な使用方法
インテグレーションの高度な使用方法については、以下の完全なコード例を参照してください。
* [アセットの高度な使用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [パーティショニングされたジョブの例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルを Model Registry にリンクする](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch の使用

:::caution
アクティブ開発中のベータ製品
Launch に興味がありますか？Launch のカスタマーパイロットプログラムに参加するために、アカウントチームに連絡してください。
パイロットのお客様はベータプログラムに参加するために AWS EKS または SageMaker を利用する必要があります。最終的には追加のプラットフォームをサポートする予定です。
:::

続行する前に、W&B Launch の使用方法についてよく理解しておくことをお勧めします。Launch のガイドを読むことを検討してください: https://docs.wandb.ai/guides/launch.

Dagster インテグレーションは以下のことを支援します:
* Dagster インスタンスで1つまたは複数の Launch エージェントを実行。
* Dagster インスタンス内でローカルの Launch ジョブを実行。
* オンプレミスまたはクラウドでリモート Launch ジョブを実行。

### Launch エージェント
このインテグレーションは `run_launch_agent` というインポート可能な `@op` を提供します。これは Launch Agent を起動し、手動で停止されるまでの長時間のプロセスとして実行します。

エージェントは Launch キューをポーリングし、ジョブを実行するプロセス（または外部サービスにディスパッチして実行）します。

設定については[リファレンスドキュメント](../launch/intro.md) を参照してください。

Launchpad でプロパティの有用な説明も表示できます。

![](/images/integrations/dagster_launch_agents.png)

簡単な例
```python
# config.yamlにこれを追加
# または、Dagit's Launchpad または JobDefinition.execute_in_process で設定を行うことができます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これを自身の W&B エンティティに置き換えてください
     project: my_project # これを自身の W&B プロジェクトに置き換えてください
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
このインテグレーションは `run_launch_job` というインポート可能な `@op` を提供し、Launch ジョブを実行します。

Launch ジョブはキューに割り当てられ、実行されます。キューを作成するか、デフォルトのものを使用することができます。そのキューをリスニングしているアクティブなエージェントがいることを確認してください。Dagster インスタンス内でエージェントを実行することもできますが、Kubernetes 内でデプロイ可能なエージェントを使用することも検討してください。

設定については[リファレンスドキュメント](../launch/intro.md) を参照してください。

Launchpad でプロパティの有用な説明も表示できます。

![](/images/integrations/dagster_launch_jobs.png)

簡単な例
```python
# config.yamlにこれを追加
# または、Dagit's Launchpad または JobDefinition.execute_in_process で設定を行うことができます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これを自身の W&B エンティティに置き換えてください
     project: my_project # これを自身の W&B プロジェクトに置き換えてください
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
   run_launch_job.alias("my_launched_job")() # ジョブ名をエイリアスで変更
```

## ベストプラクティス

1. IO Managerを使用してArtifactsの読み書きを行う。
[`Artifact.download()`](../../ref/python/artifact.md#download)や[`Run.log_artifact()`](../../ref/python/run.md#log_artifact)を直接使用する必要はありません。これらのメソッドは統合によって処理されます。保存したいデータをArtifactとして返すだけで、統合が他の作業を行います。これにより、W&B内でのArtifactのリネージがより良くなります。

2. 複雑なユースケースにはArtifactオブジェクトを自分で構築する。
PythonオブジェクトとW&Bオブジェクトは、オペレーションやアセットから返されるべきです。統合はArtifactのバンドルを処理します。複雑なユースケースでは、Dagsterジョブ内でArtifactを直接構築できます。Artifactオブジェクトを統合に渡すことで、ソース統合名やバージョン、使用されたPythonバージョン、pickleプロトコルバージョンなどのメタデータの充実を図ることをお勧めします。

3. メタデータを 통해ファイル、ディレクトリー、外部参照をArtifactsに追加する。
統合`wandb_artifact_configuration`オブジェクトを使用して、ファイル、ディレクトリー、外部参照（Amazon S3、GCS、HTTP…）を追加します。詳細については[Artifactの設定セクション](#configuration-1)の高度な例を参照してください。

4. Artifactが生成される場合は、@opではなく@assetを使用する。
Artifactsはアセットです。Dagsterがそのアセットを管理する場合には、アセットを使用することをお勧めします。これにより、Dagit Asset Catalogでの可観測性が向上します。

5. Dagster外で作成されたArtifactを使用するにはSourceAssetを使用する。
これにより、外部で作成されたArtifactsを読み取るための統合の利便性を活用できます。そうでなければ、統合によって作成されたArtifactsのみを使用できます。

6. 大規模なモデルのトレーニングを専用のコンピュートでオーケストレーションするには、W&B Launchを使用する。
小規模なモデルをDagsterクラスター内でトレーニングすることができますし、GPUノードを備えたKubernetesクラスター内でDagsterを実行することも可能です。大規模なモデルトレーニングにはW&B Launchの使用をお勧めします。これにより、インスタンスの過負荷を防ぎ、より適切なコンピュートのアアクセスを提供します。

7. Dagsterで実験管理を行う際は、W&B Run IDをDagster Run IDの値に設定する。
[Runを再開可能](../runs/resuming.md) にし、W&B Run IDをDagster Run IDまたは任意の文字列に設定することをお勧めします。この推奨に従うことで、Dagster内でモデルをトレーニングする際に、W&BのメトリクスとW&B Artifactsが同じW&B Runに保存されることが保証されます。

W&B Run IDをDagster Run IDに設定する場合:
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または、独自のW&B Run IDを選び、それをIO Manager設定に渡す場合:
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

8. 大規模なW&B Artifactsの場合、必要なデータだけをgetまたはget_pathで収集する。
標準では、統合はArtifact全体をダウンロードします。非常に大きなArtifactsを使用している場合、必要な特定のファイルやオブジェクトだけを収集することを検討してください。これにより、速度とリソースの利用効率が向上します。

9. Pythonオブジェクトについては、ユースケースに合わせてピックルモジュールを適応させる。
標準では、W&B統合は標準の[pickle](https://docs.python.org/3/library/pickle.html)モジュールを使用します。しかし、すべてのオブジェクトがこのモジュールと互換性があるわけではありません。例えば、`yield`を持つ関数はピックルしようとするとエラーを発生させます。W&Bは他のPickleベースのシリアライゼーションモジュール（[dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)）をサポートしています。

また、シリアライズされた文字列を返すか、Artifactを直接作成することで、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のようなより高度なシリアライズを使用することもできます。適切な選択はユースケースに依存しますので、このテーマに関する文献を参照してください。