---
description: Guide on how to integrate W&B with Dagster.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dagster

DagsterとWeights & Biases（W&B）を使ってMLOpsの開発フローを組織化し、MLアセットを維持します。DagsterでのW&Bとの統合により、次のことが簡単に行えます:

* [W&Bアーティファクト](../artifacts/intro.md)の使用と作成。
* [W&Bモデルレジストリ](../models/intro.md)内のRegistered Modelsの使用と作成。
* [W&B Launch](../launch/intro.md)を使用して専用の計算リソースでトレーニングジョブを実行。
* opsとアセットで[wandb](../../ref/python/README.md)クライアントを使用。

W&BのDagster統合では、W&B固有のDagsterリソースとIOマネージャーが提供されます:

* `wandb_resource`: W&B APIと認証、通信を行うためのDagsterリソース。
* `wandb_artifacts_io_manager`: W&Bアーティファクトを使用するためのDagster IOマネージャー。

次のガイドでは、DagsterでW&Bを使用するための前提条件の満たし方、opsとアセットでW&Bアーティファクトを作成・使用する方法、W&B Launchの使い方、および推奨されるベストプラクティスについて説明しています。

## はじめに
Weights and Biases内でDagsterを使用するには、次のリソースが必要です:
1. **W&B APIキー**。
2. **W&Bエンティティ（ユーザーまたはチーム）**: エンティティは、W&BのRunsやアーティファクトを送信するユーザー名またはチーム名です。ログの送信前にW&BアプリUIでアカウントまたはチームのエンティティを作成してください。エンティティが指定されていない場合、Runはデフォルトのエンティティ（通常はあなたのユーザー名）に送信されます。**プロジェクトのデフォルト**の設定でデフォルトのエンティティを変更できます。
3. **W&Bプロジェクト**: [W&B Runs](../runs/intro.md)が格納されるプロジェクトの名前。

W&Bアプリでユーザーまたはチームのプロファイルページを確認して、W&Bエンティティを見つけることができます。既存のW&Bプロジェクトを使用するか、新しいプロジェクトを作成することができます。新しいプロジェクトは、W&Bアプリのホームページまたはユーザー/チームのプロファイルページで作成できます。プロジェクトが存在しない場合、最初に使用すると自動的に作成されます。次の手順では、APIキーを取得する方法を示しています。
### APIキーの取得方法
1. [W&Bにログイン](https://wandb.ai/login)します。注：W&B Serverを使用している場合は、インスタンスのホスト名を管理者に尋ねてください。
2. [認証ページ](https://wandb.ai/authorize)に移動してAPIキーを収集するか、ユーザー/チームの設定で見つけます。プロダクション環境では、そのキーを所有する[サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)を使用することをお勧めします。
3. そのAPIキーの環境変数を設定します。`WANDB_API_KEY=YOUR_KEY`をエクスポートします。


以下の例では、Dagsterのコード内でAPIキーを指定する場所を示しています。`wandb_config`ネストされた辞書の中でエンティティとプロジェクト名を指定してください。異なるW&Bプロジェクトを使用したい場合は、異なる`wandb_config`値を異なるops/assetsに渡すことができます。渡すことができるキーの詳細については、以下の構成セクションを参照してください。


<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

例：`@job`の構成
```python
# これをconfig.yamlに追加します
# あるいは、DagitのLaunchpadまたはJobDefinition.execute_in_processで設定を行うこともできます
# 参照：https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをあなたのW&Bエンティティに置き換えてください
     project: my_project # これをあなたのW&Bプロジェクトに置き換えてください


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

例: `@repository`でアセットを使った設定

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
                   {"cache_duration_in_minutes": 60} # 1時間だけファイルをキャッシュする
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # ここにあなたのW&Bエンティティを入れてください
                       "project": "my_project", # ここにあなただのW&Bプロジェクトを入れてください
                   }
               }
           },
       ),
   ]
```

この例では、`@job`の例とは異なり、IOマネージャーのキャッシュ期間を設定しています。
</TabItem>
</Tabs>

### 設定
以下の設定オプションは、この統合によって提供されるW&B特有のDagsterリソースとIOマネージャに設定として使用されます。

* `wandb_resource`: W&B APIと通信するために使用されるDagster [resource](https://docs.dagster.io/concepts/resources) 。提供されたAPIキーを使用して自動的に認証します。プロパティ:
    * `api_key`: （str, 必須）：W&B APIと通信するために必要なW&B APIキー。
    * `host`: （str, 任意）：使用したいAPIホストサーバー。W&Bサーバーを使用している場合のみ必要です。デフォルトはパブリッククラウドホスト： [https://api.wandb.ai](https://api.wandb.ai) です。
* `wandb_artifacts_io_manager`: W&Bアーティファクトを消費するためのDagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: （int, 任意）ローカルストレージとキャッシュに使用されるベースディレクトリー。 W&BアーティファクトとW&B Runログは、このディレクトリーから書き込みと読み込みが行われます。デフォルトでは、`DAGSTER_HOME`ディレクトリーを使用しています。
    * `cache_duration_in_minutes`: （int, 任意）ローカルストレージにW&BアーティファクトとW&B Runログを保持する時間を定義します。この時間が経過しても開かれなかったファイルやディレクトリーのみがキャッシュから削除されます。キャッシュの消去はIOマネージャの実行が終了した後に行われます。キャッシュを完全に無効にする場合は、0に設定できます。キャッシュは、同じマシン上で複数のジョブ間でアーティファクトが再利用される際の速度向上に役立ちます。デフォルトでは30日間に設定されています。
    * `run_id`: (str, 任意): このrunの一意のIDで、再開時に使用されます。プロジェクト内で一意でなければならず、runを削除するとIDを再利用することはできません。短い説明名のためのnameフィールドを使用するか、ランとハイパーパラメーターを比較するためのconfigを使用します。IDには次の特殊文字を含めることができません： `/\#?%:..`  Dagster内で実験トラッキングを行う際には、Run IDを設定して、IOマネージャがrunを再開できるようにする必要があります。デフォルトではDagster Run IDに設定されています。例：`7e4df022-1bf2-44b5-a383-bb852df4077e`
    * `run_name`: （str, 任意）このrunの短い表示名で、UI上でこのrunを識別する方法です。デフォルトでは、以下のフォーマットの文字列に設定されています。`dagster-run-[Dagster Run IDの最初の8文字]` 例：`dagster-run-7e4df022`
    * `run_tags`: （list[str], 任意）：UI上のこのrunのタグリストに表示される文字列のリスト。タグは、run同士をまとめたり、一時的なラベル（`baseline`や`production`など）を適用するのに便利です。UIでタグを追加/削除したり、特定のタグのあるrunのみをフィルタリングすることが簡単です。統合によって使用されるすべてのW&B Runには`dagster_wandb`タグが付けられます。

## W&Bアーティファクトを使用する

W&Bアーティファクトとの統合は、Dagster IOマネージャに依存しています。

[IO Managers](https://docs.dagster.io/concepts/io-management/io-managers)は、ユーザーが提供するオブジェクトであり、アセットやopの出力を格納し、それを下流のアセットやopに入力として読み込む責任があります。例えば、IOマネージャは、ファイルシステム上のファイルからオブジェクトを格納したり、読み込んだりすることができます。

この統合では、W&BアーティファクトのためのIOマネージャが提供されています。これにより、任意のDagster `@op`や`@asset`でW&Bアーティファクトをネイティブに作成および消費することができます。以下は、Pythonリストを含むデータセットタイプのW&Bアーティファクトを生成する`@asset`の簡単な例です。

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
    return [1, 2, 3] # これはアーティファクトに格納されます
```
`@op`、`@asset`、`@multi_asset` をメタデータ設定で注釈付けして、アーティファクトを書き込むことができます。同様に、Dagsterの外で作成されたW&Bアーティファクトも消費できます。

## W&Bアーティファクトを書き込む
続ける前に、W&Bアーティファクトの使い方をしっかり理解することをお勧めします。[アーティファクトのガイド](../artifacts/intro.md)を読んでみてください。

Python関数からオブジェクトを返すことで、W&Bアーティファクトを書き込むことができます。W&Bでは以下のオブジェクトがサポートされています。
* Pythonオブジェクト（int、dict、list...）
* W&Bオブジェクト（Table、Image、Graph...）
* W&Bアーティファクトオブジェクト

以下の例では、Dagsterアセット（`@asset`）でW&Bアーティファクトを書き込む方法を示しています。

<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Pythonオブジェクト', value: 'python_objects'},
    {label: 'W&Bオブジェクト', value: 'wb_object'},
    {label: 'W&Bアーティファクト', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html)モジュールでシリアライズできるものは、すべてピクル化され、インテグレーションによって作成されたアーティファクトに追加されます。その内容は、Dagster内でそのアーティファクトを読むときにアンピクル化されます（詳細は[アーティファクトの読み取り](#read-wb-artifacts)を参照してください）。

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
W&Bは、複数のPickleベースのシリアル化モジュール([pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib))をサポートしています。また、[ONNX](https://onnx.ai/)や[PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)のようなより高度なシリアル化も利用できます。詳細は[シリアル化](#serialization-configuration)セクションを参照してください。

  </TabItem>
  <TabItem value="wb_object">

W&Bのネイティブオブジェクト（例: [Table](../../ref/python/data-types/table.md)、[Image](../../ref/python/data-types/image.md)、[Graph](../../ref/python/data-types/graph.md)）は、このインテグレーションによって作成されたアーティファクトに追加されます。以下に、Tableを使用した例を示します。

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

複雑なユースケースでは、独自のArtifactオブジェクトを構築する必要があるかもしれません。このインテグレーションは、両方のインテグレーションのメタデータを拡張する便利な追加機能も提供しています。

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
`@op`、`@asset`、`@multi_asset`に、wandb_artifact_configurationという名前の設定辞書を設定することができます。この辞書は、メタデータとしてデコレータ引数に渡す必要があります。この設定は、W&BアーティファクトのIOマネージャーの読み込みと書き込みを制御するために必要です。

`@op`の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out)メタデータ引数を介した出力メタデータに位置しています。
`@asset`の場合、アセットのメタデータ引数に位置しています。
`@multi_asset`の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut)メタデータ引数を介した各出力メタデータに位置しています。

以下のコード例では、`@op`、`@asset`、`@multi_asset`計算に辞書を設定する方法を示しています：

<Tabs
  defaultValue="op"
  values={[
    {label: '@opの例', value: 'op'},
    {label: '@assetの例', value: 'asset'},
    {label: '@multi_assetの例', value: 'multi_asset'},
  ]}>
  <TabItem value="op">

`@op`の例:
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

`@asset`の例:
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
設定で名前を渡す必要はありません。なぜなら、@アーティファクトはすでに名前を持っているからです。インテグレーションはアーティファクト名をアセット名に設定します。

  </TabItem>
  <TabItem value="multi_asset">

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

  </TabItem>
</Tabs>



サポートされるプロパティ:
* `name`: (str) このアーティファクトの人間が読める名前。UIでこのアーティファクトを識別したり、`use_artifact`呼び出しで参照するための方法です。名前には文字、数字、アンダースコア、ハイフン、ドットが含まれていることができます。プロジェクト全体で一意である必要があります。 `@op`に必要。
* `type`: (str) アーティファクトのタイプで、アーティファクトを整理し、区別するために使用されます。一般的なタイプには、データセットやモデルなどがありますが、文字、数字、アンダースコア、ハイフン、ドットを含む任意の文字列を使用できます。出力がすでにアーティファクトでない場合に必要。
* `description`: (str) アーティファクトの説明を提供するフリーテキスト。この説明はUIでマークダウンとして表示されるため、表やリンクなどを表示するのに適しています。
* `aliases`: (list[str]) アーティファクトに適用したいエイリアスを1つまたは複数含む配列。統合では、「latest」タグもリストに追加されます（設定されていなくても）。これは、モデルやデータセットのバージョン管理を行う効果的な方法です。
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): アーティファクトに含めるローカルディレクトリの構成を含む配列。SDKの同名メソッドと同じ引数をサポートします。
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): アーティファクトに含めるローカルファイルの構成を含む配列。SDKの同名メソッドと同じ引数をサポートします。
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): アーティファクトに含める外部参照の各構成を含む配列。SDKの同名メソッドと同じ引数をサポートします。
* `serialization_module`: (dict) 使用されるシリアル化モジュールの構成。詳細については、シリアル化セクションを参照してください。
    * `name`: (str) シリアル化モジュールの名前。受け入れられる値：`pickle`、`dill`、`cloudpickle`、`joblib`。ローカルで利用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアル化機能に渡されるオプション引数。そのモジュールのdumpメソッドと同じパラメータを受け入れます。例：`{"compress": 3, "protocol": 4}`。

高度な例：
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

このアセットは、両方の統合部分で有用なメタデータとともに具体化されます。
* W&B側：ソース統合名とバージョン、使用されたPythonバージョン、ピクルプロトコルバージョンなど。
* Dagster側：
    * Dagster Run ID
    * W&B Run：ID、名前、パス、URL
    * W&Bアーティファクト：ID、名前、タイプ、バージョン、サイズ、URL
    * W&Bエンティティ
    * W&Bプロジェクト

以下の画像は、Dagsterアセットに追加されたW&Bからのメタデータを示しています。この情報は、統合がなければ利用できません。

![](/images/integrations/dagster_wb_metadata.png)

次の画像は、提供された設定がW&Bアーティファクトで役立つメタデータを使用してどのように強化されたかを示しています。この情報は、再現性とメンテナンスに役立ちます。統合がなければ利用できません。

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)


:::info
mypyのような静的型チェッカーを使用している場合は、次のようにして設定型定義オブジェクトをインポートしてください。
```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

## W&Bアーティファクトの読み込み
W&Bアーティファクトの読み込みは、書き込むのと同様です。`wandb_artifact_configuration`という設定ディクショナリを`@op`か`@asset`に設定できます。唯一の違いは、出力ではなく入力に設定を行う必要がある点です。

`@op`の場合、入力メタデータを通じて[In](https://docs.dagster.io/_apidocs/ops#dagster.In)メタデータ引数である入力メタデータに位置しています。アーティファクトの名前を明示的に渡す必要があります。

`@asset`の場合、入力メタデータを通じて[Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn)Inメタデータ引数に位置しています。親アセットの名前がそれに一致するはずなので、アーティファクト名を渡すべきではありません。

統合の外部で作成されたアーティファクトに依存する場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset)を使用する必要があります。それは常にそのアセットの最新のバージョンを読み込みます。

以下の例は、さまざまなopからアーティファクトを読み込む方法を示しています。

<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'op'},
    {label: 'Created by another @asset', value: 'asset'},
    {label: 'Artifact created outside Dagster', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

`@op`からアーティファクトを読み込む
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

他の `@asset` によって作成されたアーティファクトの読み込み
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 入力引数の名前を変更したくない場合は、'asset_key' を削除できます
           asset_key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

  </TabItem>
  <TabItem value="outside_dagster">

Dagsterの外部で作成されたアーティファクトの読み込み：

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&Bアーティファクトの名前
   description="Dagsterの外部で作成されたアーティファクト",
   io_manager_key="wandb_artifacts_manager",
)
@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>


### 設定
上記の設定は、IOマネージャが収集してデコレートされた関数に提供するべき入力を示すために使用されます。次のリードパターンがサポートされています。

1. アーティファクト内に含まれる名前付きオブジェクトを取得するには、getを使用します。

```python
@asset(
   ins={
       "table": AssetIn(
           asset_key="my_artifact_with_table",
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
2. Artifact内のダウンロード済みファイルのローカルパスを取得するには、get_pathを使用します：
```python
@asset(
   ins={
       "path": AssetIn(
           asset_key="my_artifact_with_file",
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

3. コンテンツがローカルにダウンロードされたArtifactオブジェクト全体を取得するには：
```python
@asset(
   ins={
       "artifact": AssetIn(
           asset_key="my_artifact",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def get_artifact(context, artifact):
   context.log.info(artifact.name)
```



サポートされているプロパティ

* `get`: (str) アーティファクトの相対名にあるW&Bオブジェクトを取得します。
* `get_path`: (str) アーティファクトの相対名にあるファイルへのパスを取得します。

### シリアライズ設定
デフォルトでは、このインテグレーションでは標準の[pickle](https://docs.python.org/3/library/pickle.html)モジュールを使用しますが、一部のオブジェクトはそれと互換性がありません。例えば、yieldを持つ関数は、pickleしようとするとエラーが発生します。

ありがたいことに、より多くのPickleベースのシリアライゼーションモジュール([dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib))をサポートしています。また、シリアル化された文字列を返すか、Artifactを直接作成することで、[ONNX](https://onnx.ai/)や[PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)のような高度なシリアル化も使用できます。適切な選択肢は、ユースケースによって異なりますので、この件に関する利用可能な文献を参照してください。


### Pickleベースのシリアリゼーションモジュール

:::warning
ピクリングは、安全性が低いことが知られています。 セキュリティが懸念される場合は、W＆Bオブジェクトのみを使用してください。 データに署名し、ハッシュキーを独自のシステムに保存することをお勧めします。 より複雑な使用例については、お気軽にお問い合わせください。喜んでお手伝いいたします。
:::

`wandb_artifact_configuration`の`serialization_module`ディクショナリを通じて、使用されるシリアライズが設定できます。Dagsterを実行しているマシンでモジュールが利用可能であることを確認してください。

インテグレーションは、そのアーティファクトを読むときにどのシリアル化モジュールを使用するかを自動的に認識します。

現在サポートされているモジュールは、pickle、dill、cloudpickle、およびjoblibです。

以下は、joblibでシリアル化された「モデル」を作成し、それを推論に使用する簡略化された例です。

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
    # これは実際のMLモデルではありませんが、pickleモジュールでは不可能です
    return lambda x, y: x + y
```
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
    context.log.info(inference_result)  # 出力：3
    return inference_result
```

### 高度なシリアル化形式（ONNX、PMML）
ONNXやPMMLのような共通ファイル形式を使用することが一般的です。このインテグレーションもこれらの形式をサポートしていますが、Pickleベースのシリアル化よりも少し手間がかかります。

それらのフォーマットを使用するには、2つの異なる方法があります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常のPythonオブジェクトであるかのように返します。インテグレーションはその文字列をピクルします。その文字列を使用してモデルを再構築できます。
2. シリアル化されたモデルを含む新しいローカルファイルを作成し、add_file設定でそのファイルを使用してカスタムアーティファクトを構築します。
以下は、Scikit-learnモデルがONNXを使用してシリアル化される例です。

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
    # https://onnx.ai/sklearn-onnx/ からのインスピレーション
```
# モデルをトレーニングする
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX形式に変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクト（モデル + テストセット）を書き込む
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
    # https://onnx.ai/sklearn-onnx/ から着想を得た
# ONNXランタイムで予測を計算する
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

<!-- ### 高度な使用法
インテグレーションの高度な使用法については、以下の完全なコード例を参照してください:
* アセットの高度な使用例: TODO: 
* パーティション化されたジョブの例: TODO: リンク
* モデルレジストリへのモデルのリンク: TODO: リンク -->

## W&B Launchの使用法

:::warning
アクティブ開発中のベータ製品
Launchに興味がありますか？W&B Launchの顧客パイロットプログラムに参加するために、アカウントチームに連絡してください。
パイロット顧客は、AWS EKSまたはSageMakerを使用する必要があります。最終的には、他のプラットフォームもサポートする予定です。
:::

続ける前に、W&B Launchの使い方について十分に理解しておくことをお勧めします。Launchに関するガイドを読むことを検討してください: https://docs.wandb.ai/guides/launch.

Dagsterとの統合は以下のように役立ちます:
* Dagsterインスタンスで1つまたは複数のLaunchエージェントを実行する。
* Dagsterインスタンス内でローカルLaunchジョブを実行する。
* オンプレミスまたはクラウドでのリモートLaunchジョブ。
### エージェントの起動
この統合により、 `run_launch_agent` という名前のインポート可能な `@op` が提供されます。 Launch Agentを起動し、手動で停止するまで長時間実行を続けるプロセスとして実行します。

エージェントは、ローンチキューをポーリングし、ジョブを順番に実行するプロセスです（または外部サービスにジョブをディスパッチして実行させます）。

<!-- 設定に関する参考ドキュメントを参照してください: TODO: リンク -->

Launchpad内ですべてのプロパティの便利な説明を表示することもできます。

![](/images/integrations/dagster_launch_agents.png)

シンプルな例
```python
# これを config.yaml に追加
# あるいは、DagitのLaunchpadやJobDefinition.execute_in_processで設定することもできます
# 参考：https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをあなたのW&Bエンティティに置き換えます
     project: my_project # これをあなたのW&Bプロジェクトに置き換えます
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

### ジョブの起動
この統合では、`run_launch_job`というインポート可能な`@op`が提供されています。Launchジョブを実行します。

Launchジョブは、実行されるためにキューに割り当てられます。キューを作成するか、デフォルトのキューを使用できます。そのキューをリッスンしているアクティブなエージェントがあることを確認してください。Dagsterインスタンス内でエージェントを実行することもできますが、Kubernetesでデプロイ可能なエージェントを使用することも検討してください。

<!-- 設定に関しては、参照ドキュメントを参照してください: TODO: link -->

Launchpadのすべてのプロパティについても、役立つ説明を見ることができます。

![](/images/integrations/dagster_launch_jobs.png)


簡単な例
```python

# config.yamlにこれを追加してください
# 代わりにDagitのLaunchpadやJobDefinition.execute_in_processで設定を行うこともできます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # ここをあなたのW&Bエンティティに置き換えてください
     project: my_project # ここをあなたのW&Bプロジェクトに置き換えてください
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

1. IOマネージャを使ってアーティファクトを読み書きする。
[`Artifact.download()`](../../ref/python/artifact.md#download) や [`Run.log_artifact()`](../../ref/python/run.md#log_artifact) を直接使用する必要はありません。これらのメソッドはインテグレーションによって処理されます。単にアーティファクトに保存したいデータを返すだけで、インテグレーションが残りの処理を行います。これにより、W&B内でのアーティファクトの履歴が向上します。

2. 複雑なユースケースでのみ、自分でアーティファクトオブジェクトを作成する。
PythonオブジェクトとW&Bオブジェクトは、あなたのops/assetsから返されるべきです。インテグレーションがアーティファクトのバンドルを処理します。
複雑なユースケースでは、Dagsterのジョブ内で直接アーティファクトを作成することができます。メタデータの豊富さを提供するために、ソースのインテグレーション名、バージョン、使用されるPythonバージョン、pickeプロトコルバージョンなどの情報が含まれるアーティファクトオブジェクトをインテグレーションに渡すことをお勧めします。

3. メタデータを通じてファイル、ディレクトリ、外部参照をアーティファクトに追加する。
インテグレーションの `wandb_artifact_configuration` オブジェクトを使って、任意のファイル、ディレクトリ、外部参照（Amazon S3、GCS、HTTPなど）を追加してください。詳しくは、[アーティファクト設定セクション](#configuration-1)の高度な例を参照してください。

4. アーティファクトが生成される際に、@opの代わりに@assetを使用する。
アーティファクトはアセットです。Dagsterがそのアセットを維持する場合は、アセットを使用することをお勧めします。これにより、Dagitアセットカタログでの観測性が向上します。

5. SourceAssetを使用して、Dagsterの外部で作成されたアーティファクトを使用する。
これにより、外部で作成されたアーティファクトを読み込むためのインテグレーションを活用できます。それ以外の場合は、インテグレーションによって作成されたアーティファクトのみを使用することができます。

6. 大きなモデルのためにはW&B Launchを使用して、専用のコンピュータでのトレーニングを管理する。
小さなモデルはDagsterクラスター内でトレーニングすることができ、DagsterをGPUノードを持つKubernetesクラスターで実行することができます。大規模なモデルトレーニングには、W&B Launchの使用をお勧めします。これにより、インスタンスへの過負荷を防ぎ、より適切なコンピューティングが利用可能になります。

7. Dagster内での実験トラッキング時に、W&B Run IDをDagster Run IDの値に設定する。
[Runを再開可能](../runs/resuming.md)にし、W&B Run IDをDagster Run IDまたは任意の文字列に設定することをお勧めします。この勧告に従うことで、Dagster内でモデルをトレーニングする際に、W&BのメトリクスとW&Bアーティファクトが同じW&B Runに格納されることが確実になります。

　Dagster Run IDをW&B Run IDに設定します。
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```
自分のW&B Run IDを選択し、IO Managerの設定に渡してください。

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

8. 大規模なW&Bアーティファクトの場合、必要なデータのみをgetまたはget_pathで収集してください。

デフォルトでは、統合はアーティファクト全体をダウンロードします。非常に大きなアーティファクトを使用している場合は、必要な特定のファイルやオブジェクトのみを収集することをお勧めします。これにより、速度とリソースの利用効率が向上します。

9. Pythonオブジェクトには、用途に応じてpicklingモジュールを適応させてください。

デフォルトでは、W&B統合は標準の[pickle](https://docs.python.org/3/library/pickle.html)モジュールを使用します。ただし、一部のオブジェクトはそれと互換性がありません。例えば、yieldが含まれる関数は、pickle化しようとするとエラーが発生します。W&Bは、他のPickleベースのシリアライゼーションモジュール([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))にも対応しています。

また、[ONNX](https://onnx.ai/)や[PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)のような、より高度なシリアライゼーションを使用することもできます。シリアライズされた文字列を返すか、Artifactを直接作成します。適切な選択は、ユースケースによって異なります。この件に関する利用可能な文献を参照してください。