---
title: Dagster
description: W&B を Dagster に統合する方法についての ガイド 。
menu:
  launch:
    identifier: ja-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster と W&B (Weights & Biases) を使用して、MLOps パイプラインを編成し、ML アセットを維持します。W&B との統合により、Dagster 内で次のことが容易になります。

* [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の使用と作成。
* [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) で Registered Models の使用と作成。
* [W&B Launch]({{< relref path="/launch/" lang="ja" >}}) を使用した、専用コンピューティングでのトレーニングジョブの実行。
* ops およびアセットでの [wandb]({{< relref path="/ref/python/" lang="ja" >}}) クライアントの使用。

W&B Dagster インテグレーションは、W&B 固有の Dagster リソースと IO Manager を提供します。

* `wandb_resource`: W&B API に対して認証および通信するために使用される Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts の消費に使用される Dagster IO Manager。

次のガイドでは、Dagster で W&B を使用するための前提条件を満たす方法、ops およびアセットで W&B Artifacts を作成および使用する方法、W&B Launch の使用方法、推奨されるベストプラクティスについて説明します。

## 始める前に
Weights & Biases 内で Dagster を使用するには、次のリソースが必要です。
1. **W&B APIキー**。
2. **W&B entity (ユーザーまたは Team)**: エンティティとは、W&B Runs と Artifacts を送信するユーザー名または Team 名です。run を記録する前に、W&B App UI でアカウントまたは Team エンティティを作成してください。エンティティを指定しない場合、run はデフォルトのエンティティ (通常はユーザー名) に送信されます。**Project Defaults** の設定でデフォルトのエンティティを変更します。
3. **W&B project**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) が保存される Project の名前。

W&B App でユーザーまたは Team のプロフィールページを確認して、W&B エンティティを見つけてください。既存の W&B Project を使用することも、新しい Project を作成することもできます。新しい Project は、W&B App のホームページまたはユーザー/Team のプロフィールページで作成できます。Project が存在しない場合、最初に使用するときに自動的に作成されます。以下の手順では、APIキーを取得する方法について説明します。

### APIキーを取得する方法
1. [W&B にログイン](https://wandb.ai/login) します。注: W&B Server を使用している場合は、インスタンスのホスト名について管理者に問い合わせてください。
2. [認証ページ](https://wandb.ai/authorize) またはユーザー/Team の設定に移動して、APIキーを収集します。本番環境では、[サービスアカウント]({{< relref path="/support/kb-articles/service_account_useful.md" lang="ja" >}}) を使用してそのキーを所有することをお勧めします。
3. その APIキーの環境変数を設定します。`export WANDB_API_KEY=YOUR_KEY`.

以下の例では、Dagster コードで APIキーを指定する場所を示しています。`wandb_config` ネストされた辞書内でエンティティと Project 名を必ず指定してください。異なる W&B Project を使用する場合は、異なる `wandb_config` 値を異なる ops/アセットに渡すことができます。渡すことができるキーの詳細については、以下の構成セクションを参照してください。

{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
例: `@job` の構成
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

例: アセットを使用した `@repository` の構成

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
`@job` の例とは対照的に、この例では IO Manager のキャッシュ期間を構成していることに注意してください。
{{% /tab %}}
{{< /tabpane >}}

### 構成
以下の構成オプションは、インテグレーションによって提供される W&B 固有の Dagster リソースおよび IO Manager の設定として使用されます。

* `wandb_resource`: W&B API との通信に使用される Dagster [リソース](https://docs.dagster.io/concepts/resources)。提供された APIキーを使用して自動的に認証します。プロパティ:
    * `api_key`: (str, 必須): W&B API との通信に必要な W&B APIキー。
    * `host`: (str, オプション): 使用する API ホストサーバー。W&B Server を使用している場合にのみ必要です。デフォルトはパブリッククラウドホスト `https://api.wandb.ai` です。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: (int, オプション) ローカルストレージとキャッシュに使用されるベースディレクトリー。W&B Artifacts と W&B Run のログは、そのディレクトリーから書き込まれ、読み取られます。デフォルトでは、`DAGSTER_HOME` ディレクトリーを使用しています。
    * `cache_duration_in_minutes`: (int, オプション) W&B Artifacts と W&B Run ログをローカルストレージに保持する時間を定義します。その時間の間、開かれなかったファイルとディレクトリーのみがキャッシュから削除されます。キャッシュのパージは、IO Manager の実行の最後に発生します。キャッシュを完全にオフにする場合は、0 に設定できます。キャッシュを使用すると、同じマシンで実行されているジョブ間で Artifact が再利用される場合に速度が向上します。デフォルトは 30 日です。
    * `run_id`: (str, オプション): 再開に使用されるこの Run の一意の ID。Project 内で一意である必要があり、Run を削除した場合は ID を再利用できません。UI での識別に役立つ短い説明名には name フィールドを使用し、Run 全体でハイパーパラメーターを比較するために保存するには config を使用します。ID には、次の特殊文字を含めることはできません: `/\#?%:..` IO Manager が Run を再開できるようにするには、Dagster 内で実験管理を行うときに Run ID を設定する必要があります。デフォルトでは、Dagster Run ID (例: `7e4df022-1bf2-44b5-a383-bb852df4077e`) に設定されています。
    * `run_name`: (str, オプション) UI でこの Run を識別するのに役立つ、この Run の短い表示名。デフォルトでは、`dagster-run-[Dagster Run ID の最初の 8 文字]` という形式の文字列です。たとえば、`dagster-run-7e4df022` です。
    * `run_tags`: (list[str], オプション): UI でこの Run のタグのリストを設定する文字列のリスト。タグは、Run をまとめて編成したり、`baseline` や `production` などの一時的なラベルを適用したりするのに役立ちます。UI でタグを簡単に追加および削除したり、特定のタグを持つ Run のみに絞り込んだりできます。インテグレーションで使用される W&B Run には、`dagster_wandb` タグが付けられます。

## W&B Artifacts を使用する

W&B Artifact との統合は、Dagster IO Manager に依存しています。

[IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットまたは op の出力を保存し、ダウンストリームのアセットまたは op の入力としてロードする役割を担う、ユーザーが提供するオブジェクトです。たとえば、IO Manager はファイルシステムのファイルからオブジェクトを保存およびロードできます。

この統合は、W&B Artifacts の IO Manager を提供します。これにより、Dagster の `@op` または `@asset` は、W&B Artifacts をネイティブに作成および消費できます。以下は、Python リストを含むデータセット型の W&B Artifact を生成する `@asset` の簡単な例です。

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
    return [1, 2, 3] # this will be stored in an Artifact
```

Artifact を書き込むには、`@op`、`@asset`、および `@multi_asset` にメタデータ構成をアノテーションできます。同様に、Dagster の外部で作成された場合でも、W&B Artifacts を消費することもできます。

## W&B Artifacts を書き込む
続行する前に、W&B Artifacts の使用方法を十分に理解しておくことをお勧めします。[Artifacts に関するガイド]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

W&B Artifact を書き込むには、Python 関数からオブジェクトを返します。W&B では、次のオブジェクトがサポートされています。
* Python オブジェクト (int、dict、list...)
* W&B オブジェクト (Table, Image, Graph...)
* W&B Artifact オブジェクト

以下の例では、Dagster アセット (`@asset`) で W&B Artifacts を書き込む方法を示しています。

{{< tabpane text=true >}}
{{% tab "Python objects" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアル化できるものはすべて、インテグレーションによって作成された Artifact に pickle 化されて追加されます。コンテンツは、Dagster 内でその Artifact を読み取るときに unpickle 化されます ([アーティファクトの読み取り]({{< relref path="#read-wb-artifacts" lang="ja" >}}) を参照して詳細を確認してください)。

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

W&B は、複数の Pickle ベースのシリアル化モジュール ([pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)) をサポートしています。[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などのより高度なシリアル化を使用することもできます。詳細については、[シリアル化]({{< relref path="#serialization-configuration" lang="ja" >}}) セクションを参照してください。
{{% /tab %}}
{{% tab "W&B Object" %}}
ネイティブの W&B オブジェクト (例: [Table]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}})、[Image]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}})、または [Graph]({{< relref path="/ref/python/data-types/graph.md" lang="ja" >}})) は、インテグレーションによって作成された Artifact に追加されます。以下は、Table を使用した例です。

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

複雑なユースケースでは、独自の Artifact オブジェクトを構築する必要がある場合があります。インテグレーションは、インテグレーションの両側でメタデータを拡張するなど、役立つ追加機能も提供します。

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

### 構成
`wandb_artifact_configuration` と呼ばれる構成ディクショナリは、`@op`、`@asset`、および `@multi_asset` で設定できます。このディクショナリは、メタデータとしてデコレーター引数で渡す必要があります。この構成は、W&B Artifacts の IO Manager の読み取りと書き込みを制御するために必要です。

`@op` の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を介して出力メタデータに配置されます。
`@asset` の場合、アセットのメタデータ引数に配置されます。
`@multi_asset` の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数を介して各出力メタデータに配置されます。

以下のコード例では、`@op`、`@asset`、および `@multi_asset` 計算でディクショナリを構成する方法を示しています。

{{< tabpane text=true >}}
{{% tab "Example for @op" %}}
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
{{% /tab %}}
{{% tab "Example for @asset" %}}
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

@asset にはすでに名前があるため、構成を介して名前を渡す必要はありません。インテグレーションは、Artifact 名をアセット名として設定します。

{{% /tab %}}
{{% tab "Example for @multi_asset" %}}

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
{{% /tab %}}
{{< /tabpane >}}

サポートされているプロパティ:
* `name`: (str) この Artifact の人間が判読できる名前。これは、UI でこの Artifact を識別したり、use_artifact 呼び出しで参照したりする方法です。名前には、文字、数字、アンダースコア、ハイフン、およびドットを含めることができます。名前は、Project 全体で一意である必要があります。`@op` に必須。
* `type`: (str) Artifact のタイプ。Artifact を編成および区別するために使用されます。一般的なタイプにはデータセットまたはモデルがありますが、文字、数字、アンダースコア、ハイフン、およびドットを含む任意の文字列を使用できます。出力が Artifact でない場合は必須。
* `description`: (str) Artifact の説明を提供する自由形式のテキスト。説明は UI で Markdown レンダリングされるため、テーブルやリンクなどを配置するのに適しています。
* `aliases`: (list[str]) Artifact に適用する 1 つ以上のエイリアスを含む配列。インテグレーションは、設定されているかどうかに関係なく、「latest」タグもそのリストに追加します。これは、モデルとデータセットのバージョン管理を管理するための効果的な方法です。
* [`add_dirs`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}): (list[dict[str, Any]]): Artifact に含めるローカルディレクトリーごとの構成を含む配列。SDK の同名メソッドと同じ引数をサポートしています。
* [`add_files`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}): (list[dict[str, Any]]): Artifact に含めるローカルファイルごとの構成を含む配列。SDK の同名メソッドと同じ引数をサポートしています。
* [`add_references`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}): (list[dict[str, Any]]): Artifact に含める外部参照ごとの構成を含む配列。SDK の同名メソッドと同じ引数をサポートしています。
* `serialization_module`: (dict) 使用するシリアル化モジュールの構成。詳細については、シリアル化セクションを参照してください。
    * `name`: (str) シリアル化モジュールの名前。受け入れられる値: `pickle`、`dill`、`cloudpickle`、`joblib`。モジュールはローカルで利用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアル化関数に渡されるオプションの引数。そのモジュールのダンプメソッドと同じパラメーターを受け入れます。たとえば、`{"compress": 3, "protocol": 4}` です。

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

アセットは、インテグレーションの両側で役立つメタデータを使用して具体化されます。
* W&B 側: ソースインテグレーションの名前とバージョン、使用された Python バージョン、pickle プロトコルバージョンなど。
* Dagster 側:
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、タイプ、バージョン、サイズ、URL
    * W&B Entity
    * W&B Project

以下の画像は、Dagster アセットに追加された W&B からのメタデータを示しています。この情報は、インテグレーションなしでは利用できません。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

次の画像は、提供された構成が W&B Artifact 上の役立つメタデータでどのようにエンリッチされたかを示しています。この情報は、再現性とメンテナンスに役立ちます。この情報は、インテグレーションなしでは利用できません。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
mypy のような静的型チェッカーを使用する場合は、次のものを使用して構成型定義オブジェクトをインポートします。

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### パーティションの使用

このインテグレーションは、[Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をネイティブにサポートしています。

以下は、`DailyPartitionsDefinition` を使用してパーティション化された例です。
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
このコードは、パーティションごとに 1 つの W&B Artifact を生成します。パーティションキーが追加されたアセット名の下の Artifact パネル (UI) で Artifact を表示します。たとえば、`my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、または `my_daily_partitioned_asset.2023-01-03` です。複数のディメンションにわたってパーティション化されたアセットは、ドット区切りの形式で各ディメンションを示します。たとえば、`my_asset.car.blue` です。

{{% alert color="secondary" %}}
このインテグレーションでは、1 つの Run 内で複数のパーティションを具体化することはできません。アセットを具体化するには、複数の Run を実行する必要があります。これは、アセットを具体化するときに Dagit で実行できます。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 高度な使用法
- [パーティション化されたジョブ](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [単純なパーティション化されたアセット](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [マルチパーティション化されたアセット](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [高度なパーティション化された使用法](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts を読み取る
W&B Artifacts の読み取りは、書き込みに似ています。`wandb_artifact_configuration` と呼ばれる構成ディクショナリは、`@op` または `@asset` で設定できます。唯一の違いは、出力ではなく入力で構成を設定する必要があることです。

`@op` の場合、[In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数を介して入力メタデータに配置されます。Artifact の名前を明示的に渡す必要があります。

`@asset` の場合、[Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In メタデータ引数を介して入力メタデータに配置されます。親アセットの名前と一致する必要があるため、Artifact 名を渡さないでください。

インテグレーションの外部で作成された Artifact に依存関係を持たせたい場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。常にそのアセットの最新バージョンを読み取ります。

次の例では、さまざまな ops から Artifact を読み取る方法を示しています。

{{< tabpane text=true >}}
{{% tab "From an @op" %}}
`@op` からの Artifact の読み取り
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
別の `@asset` によって作成された Artifact の読み取り
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

Dagster の外部で作成された Artifact の読み取り:

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

### 構成
以下の構成は、IO Manager が収集し、装飾された関数への入力として提供する内容を示すために使用されます。次の読み取りパターンがサポートされています。

1. Artifact 内に含まれる名前付きオブジェクトを取得するには、get を使用します。

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

2. Artifact 内に含まれるダウンロードされたファイルのローカルパスを取得するには、get_path を使用します。

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

3. Artifact オブジェクト全体を取得するには (コンテンツをローカルにダウンロードした場合):
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
* `get`: (str) Artifact の相対名にある W&B オブジェクトを取得します。
* `get_path`: (str) Artifact の相対名にあるファイルのパスを取得します。

### シリアル化構成
デフォルトでは、インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、一部のオブジェクトはそれと互換性がありません。たとえば、yield を使用する関数は、pickle 化しようとするとエラーが発生します。

より多くの Pickle ベースのシリアル化モジュール ([dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)) をサポートしています。[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などのより高度なシリアル化を使用することもできます。シリアル化された文字列を返すか、Artifact を直接作成することで、それらを使用できます。適切な選択はユースケースによって異なります。この件に関する利用可能な文献を参照してください。

### Pickle ベースのシリアル化モジュール

{{% alert color="secondary" %}}
Pickle 化は安全でないことが知られています。セキュリティが懸念される場合は、W&B オブジェクトのみを使用してください。データを署名し、ハッシュキーを独自のシステムに保存することをお勧めします。より複雑なユースケースについては、お気軽にお問い合わせください。喜んでお手伝いさせていただきます。
{{% /alert %}}

`wandb_artifact_configuration` の `serialization_module` ディクショナリを使用して、使用するシリアル化を構成できます。モジュールが Dagster を実行しているマシンで使用できることを確認してください。

インテグレーションは、その Artifact を読み取るときに使用するシリアル化モジュールを自動的に認識します。

現在サポートされているモジュールは、`pickle`、`dill`、`cloudpickle`、および `joblib` です。

以下は、joblib でシリアル化された「モデル」を作成し、それを推論に使用する簡単な例です。

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
    # This is not a real ML model but this would not be possible with the pickle module
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
    context.log.info(inference_result)  # Prints: 3
    return inference_result
```

### 高度なシリアル化形式 (ONNX、PMML)
ONNX や PMML のような交換ファイル形式を使用するのが一般的です。インテグレーションはこれらの形式をサポートしていますが、Pickle ベースのシリアル化よりも少し手間がかかります。

これらの形式を使用するには、2 つの異なる方法があります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常の Python オブジェクトであるかのように返します。インテグレーションはその文字列を pickle 化します。次に、その文字列を使用してモデルを再構築できます。
2. シリアル化されたモデルを使用して新しいローカルファイルを作成し、add_file 構成を使用してそのファイルでカスタム Artifact を構築します。

以下は、ONNX を使用してシリアル化される Scikit-learn モデルの例です。

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
    # Inspired from https://onnx.ai/sklearn-onnx/

    # Train a model.
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # Write artifacts (model + test_set)
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
    # Inspired from https://onnx.ai/sklearn-onnx/

    # Compute the prediction with ONNX Runtime
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

このインテグレーションは、[Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をネイティブにサポートしています。

アセットの 1 つ、複数、またはすべてのパーティションを選択的に読み取ることができます。

すべてのパーティションは辞書で提供され、キーと値はそれぞれパーティションキーと Artifact コンテンツを表します。

{{< tabpane text=true >}}
{{% tab "Read all partitions" %}}
アップストリーム `@asset` のすべてのパーティションを読み取ります。これらは辞書として提供されます。この辞書では、キーと値はそれぞれパーティションキーと Artifact コンテンツに対応しています。
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
`AssetIn` の `partition_mapping` 構成を使用すると、特定のパーティションを選択できます。この場合、`TimeWindowPartitionMapping` を採用しています。
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

構成オブジェクト `metadata` は、Weights & Biases (wandb) が Project 内のさまざまな Artifact パーティションとどのように対話するかを構成するために使用されます。

オブジェクト `metadata` には、`wandb_artifact_configuration` という名前のキーが含まれており、さらにネストされたオブジェクト `partitions` が含まれています。

`partitions` オブジェクトは、各パーティションの名前をその構成にマップします。各パーティションの構成では、そこからデータを取得する方法を指定できます。これらの構成には、各パーティションの要件に応じて、`get`、`version`、および `alias` という異なるキーを含めることができます。

**構成キー**

1. `get`:
`get` キーは、データをフェッチする W&B オブジェクト (Table, Image...) の名前を指定します。
2. `version`:
`version` キーは、Artifact の特定のバージョンをフェッチする場合に使用されます。
3. `alias`:
`alias` キーを使用すると、エイリアスで Artifact を取得できます。

**ワイルドカード構成**

ワイルドカード `"*"` は、構成されていないすべてのパーティションを表します。これにより、`partitions` オブジェクトで明示的に言及されていないパーティションにデフォルト構成が提供されます。

たとえば、

```python
"*": {
    "get": "default_table_name",
},
```
この構成は、明示的に構成されていないすべてのパーティションについて、データが `default_table_name` という名前のテーブルからフェッチされることを意味します。

**特定のパーティション構成**

キーを使用して特定の構成を提供することで、特定のパーティションのワイルドカード構成をオーバーライドできます。

たとえば、

```python
"yellow": {
    "get": "custom_table_name",
},
```

この構成は、`yellow` という名前のパーティションの場合、データが `custom_table_name` という名前のテーブルからフェッチされ、ワイルドカード構成がオーバーライドされることを意味します。

**バージョン管理とエイリアス**

バージョン管理とエイリアスの目的で、構成に特定の `version` および `alias` キーを指定できます。

バージョンについては、

```python
"orange": {
    "version": "v0",
},
```

この構成は、`orange` Artifact パーティションのバージョン `v0` からデータをフェッチします。

エイリアスについては、

```python
"blue": {
    "alias": "special_alias",
},
```

この構成は、エイリアス `special_alias` (構成では `blue` と呼ばれる) を持つ Artifact パーティションのテーブル `default_table_name` からデータをフェッチします。

### 高度な使用法
インテグレーションの高度な使用法を表示するには、次の完全なコード例を参照してください。
* [アセットの高度な使用法の例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [パーティション化されたジョブの例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルをモデルレジストリにリンクする](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch の使用

{{% alert color="secondary" %}}
アクティブな開発中のベータ版製品
Launch に興味がありますか？W&B Launch のカスタマーパイロットプログラムへの参加について話し合うには、アカウントチームにお問い合わせください。
パイロットのお客様は、ベータプログラムの資格を得るために AWS EKS または SageMaker を使用する必要があります。最終的には、追加のプラットフォームをサポートする予定です。
{{% /alert %}}

続行する前に、W&B Launch の使用方法を十分に理解しておくことをお勧めします。Launch に関するガイド: /guides/launch を参照してください。

Dagster インテグレーションは、以下に役立ちます。
* Dagster インスタンスで 1 つまたは複数の Launch エージェントを実行します。
* Dagster インスタンス内でローカル Launch ジョブを実行します。
* オンプレミスまたはクラウドでリモート Launch ジョブを実行します。

### Launch エージェント
このインテグレーションは、`run_launch_agent` と呼ばれるインポート可能な `@op` を提供します。これは、Launch エージェントを起動し、手動で停止するまで長時間実行されるプロセスとして実行します。

エージェントは、Launch キューをポーリングし、ジョブを実行 (または外部サービスにディスパッチして実行) するプロセスです。

構成については、[リファレンスドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

また、Launchpad ですべてのプロパティの役立つ説明を表示することもできます。

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

簡単な例
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
このインテグレーションは、`run_launch_job` と呼ばれるインポート可能な `@op` を提供します。Launch ジョブを実行します。

Launch ジョブは、実行されるためにキューに割り当てられます。キューを作成するか、デフォルトのキューを使用できます。そのキューをリッスンするアクティブなエージェントがあることを確認してください。Dagster インスタンス内でエージェントを実行できますが、Kubernetes でデプロイ可能なエージェントを使用することも検討できます。

構成については、[リファレンスドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

また、Launchpad ですべてのプロパティの役立つ説明を表示することもできます。

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}

簡単な例
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
   run_launch_job.alias("my_launched_job")() # we rename the job with an alias
```

## ベストプラクティス

1. IO Manager を使用して Artifacts を読み書きします。
[`Artifact.download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) または [`Run.log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) を直接使用する必要はありません。これらのメソッドは、インテグレーションによって処理されます。Artifact に保存するデータを返すだけで、残りはインテグレーションに任せます。これにより、W&B での Artifact のリネージが向上します。

2. 複雑なユースケースでのみ Artifact オブジェクトを自分で構築します。
Python オブジェクトと W&B オブジェクトは、ops/アセットから返される必要があります。インテグレーションは、Artifact のバンドルを処理します。
複雑なユースケースでは、Dagster ジョブで Artifact を直接構築できます。ソースインテグレーションの名前とバージョン、使用された Python バージョン、pickle プロトコルバージョンなどのメタデータエンリッチメントのために、Artifact オブジェクトをインテグレーションに渡すことをお勧めします。

3. メタデータを介してファイル、ディレクトリー、および外部参照を Artifacts に追加します。
インテグレーション `wandb_artifact_configuration` オブジェクトを使用して、ファイル、ディレクトリー、または外部参照 (Amazon S3、GCS、HTTP...) を追加します。詳細については、[Artifact 構成セクション]({{< relref path="#configuration-1" lang="ja" >}}) の高度な例を参照してください。

4. Artifact が生成される場合は、@op の代わりに @asset を使用します。
Artifacts はアセットです。Dagster がそのアセットを維持する場合は、アセットを使用することをお勧めします。これにより、Dagit Asset Catalog での可観測性が向上します。

5. SourceAsset を使用して、Dagster の外部で作成された Artifact を消費します。
これにより、インテグレーションを利用して外部で作成された Artifact を読み取ることができます。それ以外の場合は、インテグレーションによって作成された Artifact のみを使用できます。

6. W&B Launch を使用して、大規模モデルの専用コンピューティングでのトレーニングを編成します。
Dagster クラスター内で小さなモデルをトレーニングでき、GPU ノードを備えた Kubernetes クラスターで Dagster を実行できます。大規模なモデルトレーニングには、W&B Launch を使用することをお勧めします。これにより、インスタンスの過負荷を防ぎ、より適切なコンピューティングへのアクセスを提供できます。

7. Dagster 内で実験管理を行う場合は、W&B Run ID を Dagster Run ID の値に設定します。
[Run を再開可能]({{< relref path="/guides/models/track/runs/resuming.md" lang="ja" >}}) にし、W&B Run ID を Dagster Run ID または選択した文字列に設定することをお勧めします。この推奨事項に従うと、Dagster 内でモデルをトレーニングするときに、W&B メトリクスと W&B Artifacts が同じ W&B Run に保存されるようになります。

W&B Run ID を Dagster Run ID に設定します。
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または、独自の W&B Run ID を選択して、IO Manager 構成に渡します。
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

8. 大規模な W&B Artifacts の場合は、get または get_path で必要なデータのみを収集します。
デフォルトでは、インテグレーションは Artifact 全体をダウンロードします。非常に大きな Artifact を使用している場合は、必要な特定のファイルまたはオブジェクトのみを収集する必要があります。これにより、速度とリソースの使用率が向上します。

9. Python オブジェクトの場合は、ユースケースに合わせて pickle モジュールを調整します。
デフォルトでは、W&B インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用します。ただし、一部のオブジェクトはそれと互換性がありません。たとえば、yield を使用する関数は、pickle 化しようとするとエラーが発生します。W&B は、他の Pickle ベースのシリアル化モジュール ([dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)) をサポートしています。

シリアル化された文字列を返すか、Artifact を直接作成することで、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などのより高度なシリアル化を使用することもできます。適切な選択はユースケースによって異なります。この件に関する利用可能な文献を参照してください。
```