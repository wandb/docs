---
description: W&B を Dagster と統合する方法に関するガイド。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Dagster

Dagster と W&B (W&B) を使用して MLOps パイプラインをオーケストレーションし、ML アセットを管理します。W&B とのインテグレーションにより、Dagster 内で以下が容易になります:

* [W&B Artifacts](../artifacts/intro.md) の使用と作成。
* [W&B Model Registry](../model_registry/intro.md) での登録済みモデルの使用と作成。
* [W&B Launch](../launch/intro.md) を使用して専用のコンピューティングリソースでトレーニングジョブを実行。
* ops とアセットで [wandb](../../ref/python/README.md) クライアントを使用。

W&B Dagster インテグレーションは、W&B 専用の Dagster リソースと IO マネージャーを提供します:

* `wandb_resource`: W&B API への認証と通信に使用される Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するために使用される Dagster IO マネージャー。

次のガイドでは、Dagster で W&B を使用するための前提条件の満たし方、ops とアセットで W&B Artifacts を作成および使用する方法、W&B Launch の使用方法、および推奨されるベストプラクティスについて説明します。

## 始める前に
Weights and Biases 内で Dagster を使用するために必要なリソースは次のとおりです:
1. **W&B API Key**。
2. **W&B entity（user または team）**: entity とは、W&B Runs や Artifacts を送信するユーザー名またはチーム名です。run を記録する前に、W&B アプリ UI でアカウントまたはチームエンティティを作成してください。entity を指定しない場合、run は通常ユーザー名であるデフォルトエンティティに送信されます。設定の **Project Defaults** でデフォルトエンティティを変更できます。
3. **W&B project**: [W&B Runs](../runs/intro.md) が保存されるプロジェクトの名前。

W&B エンティティは、W&B アプリのそのユーザーまたはチームのプロファイルページで確認できます。既存の W&B プロジェクトを使用するか、新しいプロジェクトを作成できます。新しいプロジェクトは、W&B アプリのホームページまたはユーザー/チームプロファイルページで作成できます。プロジェクトが存在しない場合、最初に使用したときに自動的に作成されます。次の指示に従って API キーを取得してください:

### APIキーの取得方法
1. [W&B にログイン](https://wandb.ai/login)してください。注意: W&B Server を使用している場合は、管理者からインスタンスホスト名を問い合わせてください。
2. [認証ページ](https://wandb.ai/authorize) またはユーザー/チーム設定で API キーを収集します。プロダクション環境では、そのキーを所有するために [サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) を使用することをお勧めします。
3. 環境変数としてその API キーを設定します。`export WANDB_API_KEY=YOUR_KEY`。

次の例では、Dagster のコード内で API キーを指定する場所を示しています。`wandb_config` のネストされた辞書内でエンティティとプロジェクト名を指定するようにしてください。異なる W&B Project を使用するために異なる `wandb_config` 値を異なる ops/アセットに渡すことができます。渡すことができるキーの詳細については、以下の設定セクションを参照してください。

<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

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
この例では、`@job` の例とは対照的に、IO マネージャーのキャッシュ期間を設定しています。

  </TabItem>
</Tabs>


### 設定
以下の設定オプションは、インテグレーションによって提供される W&B 専用の Dagster リソースおよび IO マネージャーに対する設定として使用されます。

* `wandb_resource`: W&B API と通信するために使用される Dagster [resource](https://docs.dagster.io/concepts/resources)。提供された API キーを使用して自動的に認証します。プロパティ:
    * `api_key`: (str, 必須): W&B API と通信するために必要な W&B API キー。
    * `host`: (str, オプション): 使用したい API ホストサーバー。W&B Server を使用している場合にのみ必要です。デフォルトでは、パブリッククラウドホスト: [https://api.wandb.ai](https://api.wandb.ai) を使用します。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するために使用する Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: (int, オプション) ローカルストレージとキャッシュに使用される基礎ディレクトリ。W&B Artifacts と W&B Run のログはそのディレクトリから書き込みおよび読み取りされます。デフォルトでは、`DAGSTER_HOME` ディレクトリを使用します。
    * `cache_duration_in_minutes`: (int, オプション) W&B Artifacts と W&B Run のログがローカルストレージに保持される時間を定義します。その時間内に開かれなかったファイルとディレクトリはキャッシュから削除されます。キャッシュのパージは IO マネージャーの実行終了時に発生します。完全にキャッシングを無効にしたい場合は0に設定できます。同じマシン上でジョブが再利用される場合、キャッシングは速度を改善します。デフォルトでは30日間。
    * `run_id`: (str, オプション): この run の一意の ID。再開に使用されます。プロジェクト内で一意でなければならず、run を削除するとその ID は再利用できません。短い説明的な名前として名前フィールドを使用するか、ハイパーパラメータを保存して run 間で比較するために設定を使用します。ID には以下の特殊文字を含めることはできません: `/\#?%:..`。Dagster 内で experiment tracking を行う場合、IO マネージャーが run を再開できるように Run ID を設定する必要があります。デフォルトでは Dagster Run ID になります。例: `7e4df022-1bf2-44b5-a383-bb852df4077e`。
    * `run_name`: (str, オプション) この run の短い表示名。この run を UI で識別する方法です。デフォルトでは、次の形式の文字列に設定されます: dagster-run-[Dagster Run ID の最初の8文字] 例: `dagster-run-7e4df022`。
    * `run_tags`: (list[str], オプション): この run のタグリストを埋めるための文字列リスト。タグは run を一緒に整理したり、「baseline」や「production」などの一時的なラベルを適用したりするのに便利です。UI でタグを簡単に追加および削除したり、特定のタグを持つ run のみにフィルタリングしたりできます。インテグレーションが使用するすべての W&B Run には `dagster_wandb` タグが付けられます。

## W&B Artifacts の使用

W&B Artifact とのインテグレーションは、Dagster IO マネージャーに依存しています。

[IO マネージャー](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットまたは op の出力を保存し、それを下流のアセットまたは ops に入力として読み込む責任を持つユーザー提供のオブジェクトです。たとえば、IO マネージャーはファイルシステム上のファイルからオブジェクトを保存および読み込むことがあります。

インテグレーションは W&B Artifacts 用の IO マネージャーを提供します。これにより、任意の Dagster `@op` または `@asset` が W&B Artifacts をネイティブに作成および消費できるようになります。以下は、データセットタイプの W&B Artifact を Python リストに含めて出力する `@asset` の簡単な例です。

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

`@op`、`@asset`、および `@multi_asset` にメタデータ設定を注釈付けして、Artifacts を記録します。同様に、Dagster 外で作成された場合でも W&B Artifacts を消費することもできます。

## W&B Artifacts の記録
続行する前に、W&B Artifacts の使用方法を十分に理解していることをお勧めします。[Artifacts に関するガイド](../artifacts/intro.md) を読むことを検討してください。

Python 関数からオブジェクトを返して W&B Artifact を記録します。以下のオブジェクトが W&B によってサポートされています:
* Python オブジェクト (int, dict, list…)
* W&B オブジェクト (Table, Image, Graph…)
* W&B Artifact オブジェクト

以下の例は、Dagster アセット (`@asset`) を使用して W&B Artifacts を記録する方法を示しています。

<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python objects', value: 'python_objects'},
    {label: 'W&B object', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズできるものは何でも pickled され、インテグレーションによって作成された Artifact に追加されます。詳細については [Read artifacts](#read-wb-artifacts) セクションを参照してください。

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

W&B は複数の Pickle ベースのシリアライゼーションモジュール（[pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートしています。より高度なシリアライゼーション（[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)）も使用できます。詳細については、[Serialization](#serialization-configuration) セクションを参照してください。

  </TabItem>
  <TabItem value="wb_object">

ネイティブ W&B オブジェクト (例: [Table](../../ref/python/data-types/table.md)、[Image](../../ref/python/data-types/image.md)、[Graph](../../ref/python/data-types/graph.md)) はすべてインテグレーションによって作成された Artifact に追加されます。以下は Table を使用した例です。

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

複雑なユースケースの場合、独自の Artifact オブジェクトを構築する必要があります。インテグレーションは、両側のメタデータを拡張する便利な機能を提供し続けます。

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
wandb_artifact_configuration と呼ばれる設定辞書を `@op`、`@asset` および `@multi_asset` に設定できます。この辞書はメタデータとしてデコレータ引数に渡す必要があります。これは W&B Artifacts の IO マネージャーの読み取りおよび書き込みを制御するために必要です。

`@op` には、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を通じて出力メタデータに設定されます。
`@asset` には、アセットのメタデータ引数に設定されます。
`@multi_asset` には、それぞれの出力メタデータに [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数を通じて設定されます。

以下のコード例は、辞書を `@op`、`@asset` および `@multi_asset` 計算に設定する方法を示しています。

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

名前を設定する必要はありません。`@asset` にはすでに名前があるためです。インテグレーションは Artifact の名前をアセット名として設定します。

  </TabItem>
  <TabItem value="multi_asset">

Example for `@multi_asset`