---
title: Dagster
description: W&B と Dagster を統合する方法についての ガイド 。
menu:
  launch:
    identifier: ja-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster と W&B (Weights & Biases) を使用して、MLOps パイプラインを調整し、ML アセットを管理します。W&B との連携により、Dagster 内で次のことが容易になります。

* [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の使用と作成。
* [W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) で Registered Models の使用と作成。
* [W&B Launch]({{< relref path="/launch/" lang="ja" >}}) を使用した、専用コンピュートでのトレーニングジョブの実行。
* ops およびアセットでの [wandb]({{< relref path="/ref/python/" lang="ja" >}}) クライアントの使用。

W&B Dagster インテグレーションは、W&B 固有の Dagster リソースと IO Manager を提供します。

* `wandb_resource`: W&B API への認証と通信に使用される Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts の消費に使用される Dagster IO Manager。

以下のガイドでは、Dagster で W&B を使用するための前提条件を満たす方法、ops およびアセットで W&B Artifacts を作成および使用する方法、W&B Launch の使用方法、推奨されるベストプラクティスについて説明します。

## 始める前に
Weights & Biases 内で Dagster を使用するには、次のリソースが必要です。
1. **W&B API キー**。
2. **W&B エンティティ (ユーザーまたは Team)**: エンティティは、W&B Runs と Artifacts を送信するユーザー名または Team 名です。run をログに記録する前に、W&B App UI でアカウントまたは Team エンティティを作成してください。エンティティを指定しない場合、run はデフォルトのエンティティ (通常はユーザー名) に送信されます。[**Project Defaults**] の下にある設定でデフォルトのエンティティを変更します。
3. **W&B プロジェクト**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) が保存されるプロジェクトの名前。

W&B App でユーザーまたは Team のプロファイルページを確認して、W&B エンティティを見つけます。既存の W&B プロジェクトを使用するか、新しいプロジェクトを作成できます。新しいプロジェクトは、W&B App ホームページまたはユーザー/Team プロファイルページで作成できます。プロジェクトが存在しない場合は、最初にプロジェクトを使用するときに自動的に作成されます。以下の手順では、API キーを取得する方法を示します。

### API キーの取得方法
1. [W&B にログイン](https://wandb.ai/login) します。注: W&B Server を使用している場合は、インスタンスのホスト名を管理者に問い合わせてください。
2. [認証ページ](https://wandb.ai/authorize) に移動するか、ユーザー/Team 設定で API キーを収集します。本番環境では、[サービスアカウント]({{< relref path="../../support/service_account_useful.md" lang="ja" >}}) を使用してそのキーを所有することをお勧めします。
3. その API キーの環境変数を設定します。`export WANDB_API_KEY=YOUR_KEY`。

以下の例では、Dagster コードで API キーを指定する場所を示します。`wandb_config` ネストされた辞書内でエンティティとプロジェクト名を指定してください。別の W&B Project を使用する場合は、異なる `wandb_config` 値を異なる ops/アセットに渡すことができます。渡すことができるキーの詳細については、以下の構成セクションを参照してください。

{{< tabpane text=true >}}
{{% tab "@job の構成" %}}
@job の構成例
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
{{% tab "アセットを使用した @repository の構成" %}}

アセットを使用した `@repository` の構成例

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

* `wandb_resource`: W&B API との通信に使用される Dagster [リソース](https://docs.dagster.io/concepts/resources)。提供された API キーを使用して自動的に認証されます。プロパティ:
    * `api_key`: (str, 必須): W&B API との通信に必要な W&B API キー。
    * `host`: (str, オプション): 使用する API ホストサーバー。W&B Server を使用している場合にのみ必要です。デフォルトはパブリッククラウドホスト `https://api.wandb.ai` です。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費する Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: (int, オプション) ローカルストレージとキャッシュに使用されるベースディレクトリー。W&B Artifacts と W&B Run ログは、そのディレクトリーから書き込まれ、読み取られます。デフォルトでは、`DAGSTER_HOME` ディレクトリーを使用しています。
    * `cache_duration_in_minutes`: (int, オプション) W&B Artifacts と W&B Run ログをローカルストレージに保持する時間量を定義します。その時間開かれなかったファイルとディレクトリーのみがキャッシュから削除されます。キャッシュのパージは、IO Manager の実行の最後に発生します。キャッシュを完全にオフにする場合は、0 に設定できます。キャッシュは、同じマシンで実行されているジョブ間で Artifact が再利用される場合に速度を向上させます。デフォルトは 30 日です。
    * `run_id`: (str, オプション): 再開に使用されるこの Run の一意の ID。これはプロジェクト内で一意である必要があり、Run を削除した場合は ID を再利用できません。UI でこの Run を識別するのに役立つ短い記述名には name フィールドを使用し、Run 間で比較するためにハイパーパラメーターを保存するには config を使用します。ID には、次の特殊文字を含めることはできません: `/\#?%:..` IO Manager が Run を再開できるようにするには、Dagster 内で実験管理を行うときに Run ID を設定する必要があります。デフォルトでは、Dagster Run ID (例: `7e4df022-1bf2-44b5-a383-bb852df4077e`) に設定されています。
    * `run_name`: (str, オプション) UI でこの Run を識別するのに役立つこの Run の短い表示名。デフォルトでは、`dagster-run-[Dagster Run ID の最初の 8 文字]` という形式の文字列です。たとえば、`dagster-run-7e4df022` のようになります。
    * `run_tags`: (list[str], オプション): UI でこの Run のタグリストを設定する文字列のリスト。タグは、Run をまとめて整理したり、`ベースライン` や `本番` などの一時的なラベルを適用したりするのに役立ちます。UI でタグを簡単に追加および削除したり、特定のタグを持つ Run のみに絞り込んだりできます。インテグレーションで使用される W&B Run には、`dagster_wandb` タグが付けられます。

## W&B Artifacts の使用

W&B Artifact とのインテグレーションは、Dagster IO Manager に依存しています。

[IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットまたは op の出力を保存し、ダウンストリームのアセットまたは ops の入力としてロードする役割を担うユーザー提供のオブジェクトです。たとえば、IO Manager は、ファイルシステムのファイルからオブジェクトを保存およびロードする場合があります。

このインテグレーションは、W&B Artifacts 用の IO Manager を提供します。これにより、任意の Dagster `@op` または `@asset` が W&B Artifacts をネイティブに作成および消費できるようになります。以下は、Python リストを含むデータセット型の W&B Artifact を生成する `@asset` の簡単な例です。

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

Artifact を書き込むには、メタデータ構成で `@op`、`@asset`、および `@multi_asset` にアノテーションを付けることができます。同様に、Dagster の外部で作成された場合でも、W&B Artifacts を消費できます。

## W&B Artifacts の書き込み
続行する前に、W&B Artifacts の使用方法を十分に理解しておくことをお勧めします。[Artifacts のガイド]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

W&B Artifact を書き込むには、Python 関数からオブジェクトを返します。W&B では、次のオブジェクトがサポートされています。
* Python オブジェクト (int、dict、list...)
* W&B オブジェクト (Table, Image, Graph...)
* W&B Artifact オブジェクト

以下の例では、Dagster アセット (`@asset`) で W&B Artifacts を書き込む方法を示します。

{{< tabpane text=true >}}
{{% tab "Python オブジェクト" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアル化できるものはすべてピクルス化され、インテグレーションによって作成された Artifact に追加されます。コンテンツは、Dagster 内でその Artifact を読み取るときにアンピクルス化されます (詳細については、[Artifact の読み取り]({{< relref path="#read-wb-artifacts" lang="ja" >}}) を参照してください)。

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

W&B は、複数の Pickle ベースのシリアル化モジュール ([pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)) をサポートしています。より高度なシリアル化 (例: [ONNX](https://onnx.ai/) または [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)) を使用することもできます。詳細については、[シリアル化]({{< relref path="#serialization-configuration" lang="ja" >}}) セクションを参照してください。
{{% /tab %}}
{{% tab "W&B オブジェクト" %}}
任意のネイティブ W&B オブジェクト (例: [Table]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}})、[Image]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}})、または [Graph]({{< relref path="/ref/python/data-types/graph.md" lang="ja" >}})) が、インテグレーションによって作成された Artifact に追加されます。以下は、Table を使用する例です。

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

複雑なユースケースでは、独自の Artifact オブジェクトを構築する必要がある場合があります。このインテグレーションは、インテグレーションの両側でメタデータを拡張するなど、便利な追加機能も提供します。

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
`wandb_artifact_configuration` という構成ディクショナリは、`@op`、`@asset`、および `@multi_asset` で設定できます。このディクショナリは、メタデータとしてデコレーター引数で渡す必要があります。この構成は、W&B Artifacts の IO Manager の読み取りと書き込みを制御するために必要です。

`@op` の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を介して出力メタデータに配置されます。
`@asset` の場合、アセットのメタデータ引数に配置されます。
`@multi_asset` の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数を介して各出力メタデータに配置されます。

以下のコード例は、`@op`、`@asset`、および `@multi_asset` コンピューティングでディクショナリを構成する方法を示しています。

{{< tabpane text=true >}}
{{% tab "@op の例" %}}
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
{{% tab "@asset の例" %}}
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

@asset には既に名前があるため、構成を介して名前を渡す必要はありません。インテグレーションは、Artifact 名をアセット名として設定します。

{{% /tab %}}
{{% tab "@multi_asset の例" %}}

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
* `name`: (str) この Artifact の人間が読める名前。これは、UI でこの Artifact を識別したり、use_artifact 呼び出しで参照したりする方法です。名前には、文字、数字、アンダースコア、ハイフン、およびドットを含めることができます。名前は、プロジェクト全体で一意である必要があります。`@op` に必須。
* `type`: (str) Artifact の型。Artifact を整理および区別するために使用されます。一般的な型にはデータセットやモデルが含まれますが、文字、数字、アンダースコア、ハイフン、およびドットを含む任意の文字列を使用できます。出力が Artifact でない場合は必須です。
* `description`: (str) Artifact の説明を提供する自由形式のテキスト。説明は UI でマークダウンでレンダリングされるため、テーブルやリンクなどを配置するのに適した場所です。
* `aliases`: (list[str]) Artifact に適用する 1 つ以上のエイリアスを含む配列。インテグレーションは、設定されているかどうかに関係なく、「latest」タグもそのリストに追加します。これは、モデルとデータセットのバージョン管理を管理するための効果的な方法です。
* [`add_dirs`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}): (list[dict[str, Any]]): Artifact に含めるローカルディレクトリーごとの構成を含む配列。これは、SDK の同名のメソッドと同じ引数をサポートします。
* [`add_files`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}): (list[dict[str, Any]]): Artifact に含めるローカルファイルごとの構成を含む配列。これは、SDK の同名のメソッドと同じ引数をサポートします。
* [`add_references`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}): (list[dict[str, Any]]): Artifact に含める外部参照ごとの構成を含む配列。これは、SDK の同名のメソッドと同じ引数をサポートします。
* `serialization_module`: (dict) 使用するシリアル化モジュールの構成。詳細については、シリアル化セクションを参照してください。
    * `name`: (str) シリアル化モジュールの名前。使用可能な値: `pickle`、`dill`、`cloudpickle`、`joblib`。モジュールはローカルで利用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアル化関数に渡されるオプションの引数。これは、そのモジュールのダンプメソッドと同じパラメーターを受け入れます。たとえば、`{"compress": 3, "protocol": 4}` のようになります。

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
* W&B 側: ソースインテグレーションの名前とバージョン、使用された python バージョン、pickle プロトコルバージョンなど。
* Dagster 側:
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、型、バージョン、サイズ、URL
    * W&B エンティティ
    * W&B プロジェクト

以下の画像は、Dagster アセットに追加された W&B のメタデータを示しています。この情報は、インテグレーションなしでは利用できません。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

以下の画像は、提供された構成が W&B Artifact の役立つメタデータでどのように強化されたかを示しています。この情報は、再現性とメンテナンスに役立ちます。これは、インテグレーションなしでは利用できません。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
mypy などの静的型チェッカーを使用する場合は、次のものを使用して構成型定義オブジェクトをインポートします。

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
このコードは、パーティションごとに 1 つの W&B Artifact を生成します。パーティションキーが追加されたアセット名の下にある Artifact パネル (UI) で Artifact を表示します。たとえば、`my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、または`my_daily_partitioned_asset.2023-01-03` のようになります。複数のディメンションにわたってパーティション化されたアセットは、ドット区切りの形式で各ディメンションを表示します。たとえば、`my_asset.car.blue` のようになります。

{{% alert color="secondary" %}}
このインテグレーションでは、1 回の Run で複数のパーティションを具体化することはできません。アセットを具体化するには、複数の Run を実行する必要があります。これは、アセットを具体化するときに Dagit で実行できます。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 高度な使用法
- [パーティション化されたジョブ](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [シンプルなパーティション化されたアセット](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [マルチパーティション化されたアセット](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [高度なパーティション化された使用法](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts の読み取り
W&B Artifacts の読み取りは、書き込みに似ています。`wandb_artifact_configuration` という構成ディクショナリは、`@op` または `@asset` で設定できます。唯一の違いは、出力ではなく入力で構成を設定する必要があることです。

`@op` の場合、[In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数を介して入力メタデータに配置されます。Artifact の名前を明示的に渡す必要があります。

`@asset` の場合、[Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In メタデータ引数を介して入力メタデータに配置されます。親アセットの名前と一致する必要があるため、Artifact 名を渡さないでください。

インテグレーションの外部で作成された Artifact への依存関係が必要な場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。これは、常にそのアセットの最新バージョンを読み取ります。

以下の例は、さまざまな ops から Artifact を読み取る方法を示しています。

{{< tabpane text=true >}}
{{% tab "@op から" %}}
`@op` から Artifact を読み取る
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
{{% tab "別の @asset によって作成された" %}}
別の `@asset` によって作成された Artifact を読み取る
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
{{% tab "Dagster の外部で作成された Artifact" %}}

Dagster の外部で作成された Artifact を読み取る:

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
以下の構成は、IO Manager が収集し、装飾された関数の入力として提供するものを指定するために使用されます。次の読み取りパターンがサポートされています。

1. Artifact に含まれる名前付きオブジェクトを取得するには、get を使用します。

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

2. Artifact に含まれるダウンロードされたファイルのローカルパスを取得するには、get_path を使用します。

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

3. Artifact オブジェクト全体を取得するには (コンテンツはローカルにダウンロードされます)。
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

### シリアル化の構成
デフォルトでは、インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、一部のオブジェクトは互換性がありません。たとえば、yield を含む関数をピクルス化しようとすると、エラーが発生します。

より多くの Pickle ベースのシリアル化モジュール ([dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)) をサポートしています。[ONNX](https://onnx.ai/) または [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などのより高度なシリアル化を使用することもできます。シリアル化された文字列を返すか、Artifact を直接作成します。適切な選択はユースケースによって異なります。この件に関する利用可能な文献を参照してください。

### Pickle ベースのシリアル化モジュール

{{% alert color="secondary" %}}
ピクルス化は安全でないことが知られています。セキュリティが懸念される場合は、W&B オブジェクトのみを使用してください。データを署名し、ハッシュキーを独自のシステムに保存することをお勧めします。より複雑なユースケースについては、お気軽にお問い合わせください。喜んでお手伝いさせていただきます。
{{% /alert %}}

`wandb_artifact_configuration` の `serialization_module` ディクショナリを使用して、使用されるシリアル化を構成できます。モジュールが Dagster を実行しているマシンで利用可能であることを確認してください。

インテグレーションは、その Artifact を読み取るときにどのシリアル化モジュールを使用するかを自動的に認識します。

現在サポートされているモジュールは、`pickle`、`dill`、`cloudpickle`、および `joblib` です。

以下は、joblib でシリアル化された「モデル」を作成し、それを推論に使用する簡略化された例です。

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
ONNX や PMML などのインターチェンジファイル形式を使用するのが一般的です。インテグレーションはこれらの形式をサポートしていますが、Pickle ベースのシリアル化よりも少し手間がかかります。

これらの形式を使用するには、2 つの異なる方法があります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常の Python オブジェクトであるかのように返します。インテグレーションはその文字列をピクルス化します。その後、その文字列を使用してモデルを再構築できます。
2. シリアル化されたモデルで新しいローカルファイルを作成し、add_file 構成を使用してそのファイルでカスタム Artifact を構築します。

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

すべてのパーティションはディクショナリで提供され、キーと値はそれぞれパーティションキーと Artifact コンテンツを表します。

{{< tabpane text=true >}}
{{% tab "すべてのパーティションを読み取る" %}}
アップストリーム `@asset` のすべてのパーティションを読み取ります。これらはディクショナリとして指定されます。このディクショナリでは、キーと値はそれぞれパーティションキーと Artifact コンテンツに対応します。
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
`AssetIn` の `partition_mapping` 構成を使用すると、特定のパーティションを選択できます。この場合、`TimeWindowPartitionMapping` を使用しています。
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

構成オブジェクト `metadata` は、Weights & Biases (wandb) がプロジェクトのさまざまな Artifact パーティションとどのように対話するかを構成するために使用されます。

オブジェクト `metadata` には、`wandb_artifact_configuration` という名前のキーが含まれており、これにはさらにネストされたオブジェクト `partitions` が含まれています。

`partitions` オブジェクトは、各パーティションの名前をその構成にマップします。各パーティションの構成では、そこからデータを取得する方法を指定できます。これらの構成には、各パーティションの要件に応じて、`get`、`version`、および `alias` という異なるキーを含めることができます。

**構成キー**

1. `get`:
`get` キーは、データを取得する W&B Object (Table, Image...) の名前を指定します。
2. `version`:
Artifact の特定のバージョンを取得する場合は、`version` キーを使用します。
3. `alias`:
`alias` キーを使用すると、エイリアスで Artifact を取得できます。

**ワイルドカード構成**

ワイルドカード `"*"` は、構成されていないすべてのパーティションを表します。これにより、`partitions` オブジェクトで明示的に言及されていないパーティションのデフォルト構成が提供されます。

例:

```python
"*": {
    "get": "default_table_name",
},
```
この構成は、明示的に構成されていないすべてのパーティションについて、データが `default_table_name` という名前のテーブルから取得されることを意味します。

**特定のパーティション構成**

キーを使用して特定の構成を提供することにより、特定のパーティションのワイルドカード構成をオーバーライドできます。

例:

```python
"yellow": {
    "get": "custom_table_name",
},
```

この構成は、`yellow` という名前のパーティションの場合、データが `custom_table_name` という名前のテーブルから取得され、ワイルドカード構成をオーバーライドすることを意味します。

**バージョン管理とエイリアス**

バージョン管理とエイリアスの目的で、構成で特定の `version` および `alias` キーを提供できます。

バージョンの場合:

```python
"orange": {
    "version": "v0",
},
```

この構成は、`orange` Artifact パーティションのバージョン `v0` からデータを取得します。

エイリアスの場合:

```python
"blue": {
    "alias": "special_alias",
},
```

この構成は、エイリアス `special_alias` を持つ Artifact パーティションのテーブル `default_table_name` からデータを取得します (構成では `blue` と呼ばれます)。

### 高度な使用法
インテグレーションの高度な使用法を表示するには、以下の完全なコード例を参照してください。
* [アセットの高度な使用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [パーティション化されたジョブの例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルを Model Registry にリンクする](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch の使用

{{% alert color="secondary" %}}
アクティブな開発中のベータ版製品
Launch に興味がありますか? アカウントチームに連絡して、W&B Launch のカスタマーパイロットプログラムへの参加について相談してください。
パイロットカスタマーは、ベータプログラムの対象となるには、AWS EKS または SageMaker を使用する必要があります。最終的には、追加のプラットフォームをサポートする予定です。
{{% /alert %}}

続行する前に、W&B Launch の使用方法を十分に理解しておくことをお勧めします。Launch のガイド ( /guides/launch) を参照してください。

Dagster インテグレーションは、以下に役立ちます。
* Dagster インスタンスで 1 つまたは複数の Launch エージェントを実行します。
* Dagster インスタンス内でローカル Launch ジョブを実行します。
* オンプレミスまたはクラウドでリモート Launch ジョブを実行します。

### Launch エージェント
インテグレーションは、`run_launch_agent` というインポート可能な `@op` を提供します。これは Launch Agent を起動し、手動で停止するまで長期実行プロセスとして実行します。

エージェントは、Launch キューをポーリングし、ジョブを実行 (または外部サービスにディスパッチして実行) するプロセスです。

構成については、[参照ドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad でのすべてのプロパティの役立つ説明を表示することもできます。

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
インテグレーションは、`run_launch_job` というインポート可能な `@op` を提供します。これは Launch ジョブを実行します。

Launch ジョブは、実行されるためにキューに割り当てられます。キューを作成するか、デフォルトのキューを使用できます。そのキューをリッスンしているアクティブなエージェントがいることを確認してください。Dagster インスタンス内でエージェントを実行できますが、Kubernetes でデプロイ可能なエージェントを使用することも検討できます。

構成については、[参照ドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad でのすべてのプロパティの役立つ説明を表示することもできます。

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
[`Artifact.download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) または [`Run.log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) を直接使用する必要はありません。これらのメソッドは、インテグレーションによって処理されます。Artifact に保存するデータを返すだけで、残りはインテグレーションが行います。これにより、W&B での Artifact の系統が向上します。

2. 複雑なユースケースでのみ、Artifact オブジェクトを自分で構築します。
Python オブジェクトと W&B オブジェクトは、ops/アセットから返す必要があります。インテグレーションは Artifact のバンドルを処理します。
複雑なユースケースでは、Dagster ジョブで Artifact を直接構築できます。ソースインテグレーションの名前とバージョン、使用された python バージョン、pickle プロトコルバージョンなど、メタデータエンリッチメントのために Artifact オブジェクトをインテグレーションに渡すことをお勧めします。

3. メタデータを介してファイル、ディレクトリー、および外部参照を Artifacts に追加します。
インテグレーション `wandb_artifact_configuration` オブジェクトを使用して、ファイル、ディレクトリー、または外部参照 (Amazon S3、GCS、HTTP…) を追加します。詳細については、[Artifact 構成セクション]({{< relref path="#configuration-1" lang="ja" >}}) の高度な例を参照してください。

4. Artifact が生成される場合は、@op の代わりに @asset を使用します。
Artifacts はアセットです。Dagster がそのアセットを維持する場合は、アセットを使用することをお勧めします。これにより、Dagit アセットカタログでの可観測性が向上します。

5. Dagster の外部で作成された Artifact を消費するには、SourceAsset を使用します。
これにより、インテグレーションを利用して外部で作成された Artifacts を読み取ることができます。それ以外の場合は、インテグレーションによって作成された Artifacts のみを使用できます。

6. 大規模モデルのトレーニングを専用コンピュートで調整するには、W&B Launch を使用します。
Dagster クラスター内で小規模モデルをトレーニングしたり、GPU ノードを備えた Kubernetes クラスターで Dagster を実行したりできます。大規模モデルのトレーニングには W&B Launch を使用することをお勧めします。これにより、インスタンスの過負荷を防ぎ、より適切なコンピュートへのアクセスを提供できます。

7. Dagster 内で実験管理を行う場合は、W&B Run ID を Dagster Run ID の値に設定します。
[Run を再開可能]({{< relref path="/guides/models/track/runs/resuming.md" lang="ja" >}}) にし、W&B Run ID を Dagster Run ID または任意の文字列に設定することを推奨します。この推奨事項に従うことで、Dagster 内でモデルをトレーニングするときに、W&B メトリクスと W&B Artifacts が同じ W&B Run に保存されるようになります。

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
デフォルトでは、インテグレーションは Artifact 全体をダウンロードします。非常に大きな Artifact を使用している場合は、必要な特定のファイルまたはオブジェクトのみを収集することをお勧めします。これにより、速度とリソース使用率が向上します。

9. Python オブジェクトの場合は、ユースケースに合わせてピクルスモジュールを調整します。
デフォルトでは、W&B インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用します。ただし、一部のオブジェクトは互換性がありません。たとえば、yield を含む関数をピクルス化しようとすると、エラーが発生します。W&B は、他の Pickle ベースのシリアル化モジュール ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。

シリアル化された文字列を返すか、Artifact を直接作成することにより、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) などのより高度なシリアル化を使用することもできます。適切な選択はユースケースによって異なります。この件に関する利用可能な文献を参照してください。
```