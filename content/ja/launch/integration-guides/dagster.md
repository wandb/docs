---
title: Dagster
description: W&B と Dagster を統合するためのガイド。
menu:
  launch:
    identifier: ja-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster と W&B を使って MLOps のパイプラインをオーケストレーションし、ML アセットを維持しましょう。W&B との連携により、Dagster 内で次のことが簡単に行えます。

* [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の作成と利用
* [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) で Registered Models の利用と作成
* [W&B Launch]({{< relref path="/launch/" lang="ja" >}}) を使って専用コンピュート上でトレーニングジョブを実行
* ops や assets で [wandb]({{< relref path="/ref/python/" lang="ja" >}}) クライアントを利用

W&B の Dagster 連携は、W&B 専用の Dagster resource と IO Manager を提供します。

* `wandb_resource`: W&B API と認証・通信するための Dagster リソース
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster IO Manager

このガイドでは、Dagster で W&B を使うための前提条件の満たし方、ops や assets で W&B Artifacts を作成・利用する方法、W&B Launch の使い方、および推奨ベストプラクティスを説明します。

## 始める前に
Dagster を W&B で使うには、以下が必要です。
1. **W&B API Key**
2. **W&B entity（user または team）**: entity は、W&B の Runs や Artifacts を送る宛先となるユーザー名またはチーム名です。run をログする前に、W&B App の UI でアカウントまたは team の entity を作成してください。entity を指定しない場合、run はデフォルトの entity（通常はあなたのユーザー名）に送信されます。デフォルト entity は設定の **Project Defaults** で変更できます。
3. **W&B project**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) が保存されるプロジェクト名

W&B の entity は、W&B App でそのユーザーまたはチームのプロフィールページから確認できます。既存の W&B project を使っても、新しく作成してもかまいません。新規プロジェクトは W&B App のホームページやユーザー／チームのプロフィールページから作成できます。プロジェクトが存在しない場合、初回利用時に自動的に作成されます。以下では API key の取得方法を説明します。

### API key の取得方法
1. [W&B にログイン](https://wandb.ai/login) します。注: W&B Server を利用している場合は、インスタンスのホスト名を管理者に確認してください。
2. [authorize ページ](https://wandb.ai/authorize) かユーザー／チームの設定から API key を取得します。プロダクション環境では、そのキーの所有者として [service account]({{< relref path="/support/kb-articles/service_account_useful.md" lang="ja" >}}) の使用を推奨します。
3. 取得した API key を環境変数に設定します。例: export `WANDB_API_KEY=YOUR_KEY`.

以下の例では、Dagster のコード内で API key をどこに指定するかを示します。`wandb_config` のネストされた辞書の中で、entity と project 名を必ず指定してください。異なる W&B Project を使いたい場合は、ops／assets ごとに異なる `wandb_config` の値を渡せます。渡せるキーの詳細は、この後の Configuration セクションを参照してください。


{{< tabpane text=true >}}
{{% tab "@job の設定" %}}
例: `@job` の設定
```python
# config.yaml にこれを追加してください
# あるいは Dagit の Launchpad または JobDefinition.execute_in_process で設定できます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # ここをあなたの W&B entity に置き換えてください
     project: my_project # ここをあなたの W&B project に置き換えてください


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
{{% tab "assets を使う @repository の設定" %}}

例: assets を使う `@repository` の設定

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
                   {"cache_duration_in_minutes": 60} # ファイルを 1 時間のみキャッシュ
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # ここをあなたの W&B entity に置き換えてください
                       "project": "my_project", # ここをあなたの W&B project に置き換えてください
                   }
               }
           },
       ),
   ]
```
この例では、`@job` の例と異なり、IO Manager のキャッシュ保持期間を設定しています。
{{% /tab %}}
{{< /tabpane >}}


### Configuration
以下の設定項目は、この連携が提供する W&B 専用の Dagster resource と IO Manager の設定として利用されます。

* `wandb_resource`: W&B API と通信するための Dagster の [resource](https://docs.dagster.io/concepts/resources)。指定された API key を使って自動で認証します。プロパティ:
    * `api_key`: (str, 必須) W&B API と通信するために必要な W&B API key
    * `host`: (str, 任意) 使用したい API ホストサーバー。W&B Server を使う場合のみ必要です。デフォルトは Public Cloud のホスト `https://api.wandb.ai`。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster の [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ:
    * `base_dir`: (int, 任意) ローカルストレージとキャッシュに使うベースディレクトリー。W&B Artifacts と W&B Run ログはそのディレクトリーに書き込み／読み込みされます。デフォルトは `DAGSTER_HOME` ディレクトリー。
    * `cache_duration_in_minutes`: (int, 任意) W&B Artifacts と W&B Run ログをローカルに保持する時間。指定時間開かれていないファイルやディレクトリーのみがキャッシュから削除されます。キャッシュの削除は IO Manager の実行終了時に行われます。キャッシュを完全に無効にしたい場合は 0 に設定できます。同一マシン上でジョブ間で Artifact を再利用する場合にキャッシュは高速化に役立ちます。デフォルトは 30 日。
    * `run_id`: (str, 任意) この run の一意な ID。再開に使用します。プロジェクト内で一意である必要があり、run を削除した場合はその ID を再利用できません。短い説明には name フィールドを、run 間で比較するハイパーパラメーターの保存には config を使ってください。ID には次の特殊文字を含められません: `/\#?%:..` Dagster 内で実験管理を行う場合、IO Manager が run を再開できるように Run ID を設定する必要があります。デフォルトは Dagster Run ID（例: `7e4df022-1bf2-44b5-a383-bb852df4077e`）。
    * `run_name`: (str, 任意) UI 上で run を識別しやすくする短い表示名。デフォルトは `dagster-run-[Dagster Run ID の先頭 8 文字]` という形式の文字列（例: `dagster-run-7e4df022`）。
    * `run_tags`: (list[str], 任意) UI 上のこの run に付与されるタグの一覧。タグは run をまとめて整理したり、`baseline` や `production` のような一時的なラベルを付けるのに便利です。UI で簡単にタグの追加／削除やフィルタリングができます。連携で使われる W&B Run には `dagster_wandb` タグが付きます。

## W&B Artifacts を使う

W&B Artifact との連携は、Dagster の IO Manager に依存します。

[IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) は、asset や op の出力を保存し、下流の assets や ops の入力として読み込む役割を持つ、ユーザー提供のオブジェクトです。例えば、IO Manager はファイルシステム上のファイルからオブジェクトを保存／読み込みできます。

この連携は W&B Artifacts 用の IO Manager を提供します。これにより、任意の Dagster `@op` や `@asset` がネイティブに W&B Artifacts を作成・消費できます。以下は Python の list を含む dataset タイプの W&B Artifact を生成する `@asset` のシンプルな例です。

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
    return [1, 2, 3] # この戻り値は Artifact に保存されます
```

`@op`、`@asset`、`@multi_asset` にメタデータ設定を付けて Artifact を書き込めます。同様に、Dagster の外部で作られた W&B Artifacts も消費できます。

## W&B Artifacts の書き込み
先に、W&B Artifacts の使い方を理解しておくことをおすすめします。[Artifacts のガイド]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

Python 関数からオブジェクトを返すと、W&B Artifact に書き込まれます。W&B がサポートするオブジェクトは以下です。
* Python オブジェクト（int、dict、list など）
* W&B オブジェクト（Table、Image、Graph など）
* W&B Artifact オブジェクト

以下の例では、Dagster の assets（`@asset`）で W&B Artifacts を書き込む方法を示します。


{{< tabpane text=true >}}
{{% tab "Python オブジェクト" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズできるものはすべて pickle 化され、連携が作成する Artifact に追加されます。Dagster 内でその Artifact を読むときに内容は unpickle されます（詳細は [Read artifacts]({{< relref path="#read-wb-artifacts" lang="ja" >}}) を参照）。

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

W&B は複数の Pickle 系シリアライズモジュール（[pickle](https://docs.python.org/3/library/pickle.html)、[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートします。[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のような高度なシリアライズも利用できます。詳細は [Serialization]({{< relref path="#serialization-configuration" lang="ja" >}}) セクションを参照してください。
{{% /tab %}}
{{% tab "W&B オブジェクト" %}}
[Table]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) や [Image]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) などの任意の W&B オブジェクトは、連携が作成する Artifact に追加されます。次の例では Table を Artifact に追加します。

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

複雑なユースケースでは、自分で Artifact オブジェクトを構築する必要があるかもしれません。この連携は、その場合でも両者のメタデータを補強するといった有用な機能を提供します。

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
`wandb_artifact_configuration` という設定用の辞書を `@op`、`@asset`、`@multi_asset` に指定できます。この辞書はデコレーター引数の metadata として渡す必要があります。これは IO Manager が W&B Artifacts を読み書きする挙動を制御するために必要です。

`@op` の場合は [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) の metadata 引数で出力メタデータ内に設定します。
`@asset` の場合は asset の metadata 引数に設定します。
`@multi_asset` の場合は各出力について [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) の metadata 引数で設定します。

以下のコード例は、`@op`、`@asset`、`@multi_asset` の計算に辞書を設定する方法を示します。

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

この場合、@asset 自体に名前があるため、設定で name を渡す必要はありません。Artifact の名前は asset 名として連携が設定します。
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



サポートされるプロパティ:
* `name`: (str) Artifact の人間可読な名前。UI での識別や use_artifact 呼び出しで参照するために使います。名前には英数字、アンダースコア、ハイフン、ドットが使えます。プロジェクト内で一意である必要があります。`@op` では必須。
* `type`: (str) Artifact のタイプ。アーティファクトの整理や識別に使用します。一般的なタイプには dataset や model がありますが、英数字、アンダースコア、ハイフン、ドットを含む任意の文字列を使用できます。出力がすでに Artifact でない場合に必須。
* `description`: (str) Artifact の説明文。UI では Markdown としてレンダリングされるため、表やリンクなどに適しています。
* `aliases`: (list[str]) Artifact に適用したいエイリアスの配列。この連携は、設定の有無にかかわらず “latest” タグも追加します。これは models や datasets のバージョン管理に有効です。
* [`add_dirs`]({{< relref path="/ref/python/sdk/classes/artifact#add_dir" lang="ja" >}}): (list[dict[str, Any]]) Artifact に含める各ローカルディレクトリーの設定を持つ配列
* [`add_files`]({{< relref path="/ref/python/sdk/classes/artifact#add_file" lang="ja" >}}): (list[dict[str, Any]]) Artifact に含める各ローカルファイルの設定を持つ配列
* [`add_references`]({{< relref path="/ref/python/sdk/classes/artifact#add_reference" lang="ja" >}}): (list[dict[str, Any]]) Artifact に含める各外部参照の設定を持つ配列
* `serialization_module`: (dict) 使用するシリアライズモジュールの設定。詳細は Serialization セクションを参照してください。
    * `name`: (str) シリアライズモジュール名。許可される値: `pickle`、`dill`、`cloudpickle`、`joblib`。モジュールはローカルで利用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアライズ関数に渡す任意の引数。そのモジュールの dump メソッドと同じパラメーターを受け付けます（例: `{"compress": 3, "protocol": 4}`）。

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



この asset は、連携の両側にとって有用なメタデータ付きで materialize されます。
* W&B 側: ソース連携の名前とバージョン、使用した Python のバージョン、pickle プロトコルのバージョンなど
* Dagster 側:
    * Dagster Run ID
    * W&B Run: ID、name、path、URL
    * W&B Artifact: ID、name、type、version、size、URL
    * W&B Entity
    * W&B Project

以下の画像は、W&B から Dagster の asset に追加されたメタデータを示します。これは連携なしでは利用できません。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

次の画像は、提供した設定が W&B Artifact 上でどのように有用なメタデータで拡張されたかを示しています。これは再現性やメンテナンスに役立ちます。連携がなければ利用できません。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}


{{% alert %}}
mypy のような静的型チェッカーを使う場合は、次を使って設定の型定義オブジェクトをインポートしてください。

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### partitions の利用

この連携はネイティブに [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をサポートします。

以下は `DailyPartitionsDefinition` を使ったパーティションの例です。
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
このコードは、各パーティションごとに 1 つの W&B Artifact を生成します。Artifact は asset 名の後ろにパーティションキーを付けた名前で UI の Artifact パネルに表示されます（例: `my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、`my_daily_partitioned_asset.2023-01-03`）。複数次元でパーティションされた assets は、各次元をドット区切りで表示します（例: `my_asset.car.blue`）。

{{% alert color="secondary" %}}
この連携では、1 回の run で複数のパーティションを materialize することはできません。asset を materialize するには複数の run を実行する必要があります。これは Dagit で assets を materialize する際に実行できます。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 高度な使用法
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [Simple partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [Multi-partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [Advanced partitioned usage](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)


## W&B Artifacts の読み取り
W&B Artifacts の読み取りは、書き込みとよく似ています。`wandb_artifact_configuration` という設定用の辞書を `@op` または `@asset` に設定できます。唯一の違いは、出力ではなく入力側に設定する必要がある点です。

`@op` の場合は [In](https://docs.dagster.io/_apidocs/ops#dagster.In) の metadata 引数で入力メタデータ内に設定します。Artifact の名前を明示的に渡す必要があります。

`@asset` の場合は [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In の metadata 引数で入力メタデータ内に設定します。親 asset の名前が一致する必要があるため、Artifact 名は渡すべきではありません。

連携の外部で作成された Artifact に依存したい場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。常にその asset の latest バージョンを読み取ります。

以下の例は、さまざまな ops から Artifact を読む方法を示します。

{{< tabpane text=true >}}
{{% tab "@op から" %}}
`@op` から artifact を読み取る
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
{{% tab "別の @asset が作成したものから" %}}
別の `@asset` が作成した artifact を読み取る
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 入力引数名を変更したくない場合は 'key' を削除できます
           key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

{{% /tab %}}
{{% tab "Dagster の外で作られた Artifact" %}}

Dagster の外部で作成された Artifact を読む:

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


### Configuration
以下の設定は、IO Manager がデコレートされた関数の入力として何を収集・提供すべきかを示すために使用します。サポートされる読み取りパターンは次のとおりです。

1. Artifact に含まれる名前付きオブジェクトを取得するには get を使用します。

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

2. Artifact に含まれるダウンロード済みファイルのローカルパスを取得するには get_path を使用します。

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

3. Artifact 全体のオブジェクト（内容はローカルにダウンロード済み）を取得するには:
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
* `get`: (str) Artifact の相対名にある W&B オブジェクトを取得します。
* `get_path`: (str) Artifact の相対名にあるファイルのパスを取得します。

### Serialization の設定
デフォルトでは、この連携は標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、一部のオブジェクトは互換性がありません。例えば、yield を含む関数は pickle 化しようとするとエラーになります。

より多くの Pickle 系シリアライズモジュール（[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）をサポートしています。[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のような高度なシリアライズも、シリアライズ済みの文字列を返すか、直接 Artifact を作成することで利用できます。適切な選択はユースケースに依存します。関連文献を参照してください。	

### Pickle 系シリアライズモジュール

{{% alert color="secondary" %}}
Pickle は安全ではないことが知られています。セキュリティが懸念される場合は、W&B オブジェクトのみを使用してください。データに署名し、ハッシュキーを自社システムに保存することを推奨します。より複雑なユースケースについては、お気軽にご相談ください。喜んでお手伝いします。
{{% /alert %}}

`wandb_artifact_configuration` 内の `serialization_module` 辞書で使用するシリアライズ方式を設定できます。モジュールが Dagster を実行するマシンで利用可能であることを確認してください。

この連携は、その Artifact を読む際にどのシリアライズモジュールを使うべきかを自動的に判断します。

現在サポートされているモジュールは `pickle`、`dill`、`cloudpickle`、`joblib` です。

以下は、joblib でシリアライズした “model” を作成し、その後推論に使用する簡略化した例です。

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
    # これは本物の ML モデルではありませんが、pickle モジュールでは不可能な例です
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

### 高度なシリアライズ形式（ONNX、PMML）
ONNX や PMML のような交換用ファイル形式を使うことは一般的です。この連携はそれらの形式もサポートしますが、Pickle 系に比べると少し手順が増えます。

これらの形式を使う方法は 2 つあります。
1. モデルを選択した形式に変換し、その形式の文字列表現を通常の Python オブジェクトのように返します。連携はその文字列を pickle 化します。その文字列からモデルを再構築できます。
2. シリアライズしたモデルを含むローカルファイルを作成し、add_file 設定を用いてそのファイルを含むカスタム Artifact を構築します。

以下は Scikit-learn のモデルを ONNX でシリアライズする例です。

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

    # モデルを学習
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 形式に変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # artifacts（model + test_set）を書き込み
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

    # ONNX Runtime で予測を実行
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### partitions の利用

この連携はネイティブに [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions) をサポートします。

asset のパーティションを 1 つ、複数、またはすべて選択的に読み取ることができます。

すべてのパーティションは辞書で提供され、キーがパーティションキー、値が Artifact の内容になります。


{{< tabpane text=true >}}
{{% tab "すべてのパーティションを読む" %}}
上流の `@asset` のすべてのパーティションを読み取り、辞書として渡されます。この辞書では、キーがパーティションキー、値が Artifact の内容に対応します。
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
{{% tab "特定のパーティションを読む" %}}
`AssetIn` の `partition_mapping` 設定で特定のパーティションを選べます。ここでは `TimeWindowPartitionMapping` を使用しています。
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

設定オブジェクト `metadata` は、W&B がプロジェクト内の各 artifact パーティションとどのようにやり取りするかを構成します。

`metadata` オブジェクトには `wandb_artifact_configuration` というキーが含まれ、その中に `partitions` という入れ子のオブジェクトが含まれます。

`partitions` オブジェクトは、各パーティション名をその設定にマップします。各パーティションの設定では、データの取得方法を指定できます。ユースケースに応じて、`get`、`version`、`alias` といったキーを含められます。

**設定キー**

1. `get`:
`get` キーは、データを取得する W&B オブジェクト（Table、Image など）の名前を指定します。
2. `version`:
`version` キーは、特定のバージョンの Artifact を取得したいときに使用します。
3. `alias`:
`alias` キーは、エイリアスで Artifact を取得することを可能にします。

**ワイルドカード設定**

ワイルドカード `"*"` は、個別設定されていないすべてのパーティションを表します。これは、`partitions` オブジェクトで明示されていないパーティションのデフォルト設定を提供します。

例えば、

```python
"*": {
    "get": "default_table_name",
},
```
この設定は、明示的に設定されていないすべてのパーティションについて、`default_table_name` という名前のテーブルからデータを取得することを意味します。

**特定パーティションの設定**

ワイルドカード設定は、パーティションのキーを使って個別の設定を与えることで上書きできます。

例えば、

```python
"yellow": {
    "get": "custom_table_name",
},
```

この設定は、`yellow` という名前のパーティションについて、`custom_table_name` というテーブルからデータを取得し、ワイルドカード設定を上書きすることを意味します。

**バージョニングとエイリアス**

バージョニングやエイリアスのために、設定に特定の `version` や `alias` キーを指定できます。

バージョンの例:

```python
"orange": {
    "version": "v0",
},
```

この設定は、`orange` の Artifact パーティションの `v0` バージョンからデータを取得します。

エイリアスの例:

```python
"blue": {
    "alias": "special_alias",
},
```

この設定は、`special_alias` というエイリアス（設定上は `blue` として参照）の Artifact パーティションの `default_table_name` というテーブルからデータを取得します。

### 高度な使用法
この連携の高度な使用法については、以下の完全なコード例を参照してください。
* [Advanced usage example for assets](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py) 
* [Partitioned job example](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [Linking a model to the Model Registry](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)


## W&B Launch の利用

{{% alert color="secondary" %}}
開発中のベータ製品
Launch に興味がありますか？W&B Launch のカスタマーパイロットプログラム参加について、アカウントチームにご連絡ください。
パイロットのお客様は、ベータプログラムの対象として AWS EKS または SageMaker を使用する必要があります。最終的には追加のプラットフォームをサポートする予定です。
{{% /alert %}}

先に、W&B Launch の使い方を理解しておくことをおすすめします。[Launch のガイド]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Dagster 連携は次のことに役立ちます。
* Dagster インスタンス内で 1 つまたは複数の Launch agent を実行
* Dagster インスタンス内でローカルの Launch ジョブを実行
* オンプレミスまたはクラウドでのリモート Launch ジョブ

### Launch agents
この連携は、`run_launch_agent` というインポート可能な `@op` を提供します。これは Launch Agent を起動し、手動で停止するまで長時間実行プロセスとして動作します。

Agent は Launch のキューをポーリングし、ジョブを順に実行（または外部サービスへ委譲）するプロセスです。

[Launch のページ]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad では、すべてのプロパティに関する有用な説明も確認できます。

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

シンプルな例
```python
# config.yaml にこれを追加してください
# あるいは Dagit の Launchpad または JobDefinition.execute_in_process で設定できます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # ここをあなたの W&B entity に置き換えてください
     project: my_project # ここをあなたの W&B project に置き換えてください
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

### Launch jobs
この連携は、`run_launch_job` というインポート可能な `@op` を提供します。これは Launch ジョブを実行します。

Launch ジョブは、実行のためにキューに割り当てられます。キューは既存のものを使うか、新たに作成できます。そのキューをリッスンするアクティブな agent がいることを確認してください。agent は Dagster インスタンスの中で動かすことも、Kubernetes にデプロイした agent を使うこともできます。

[Launch のページ]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad では、すべてのプロパティに関する有用な説明も確認できます。

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}


シンプルな例
```python
# config.yaml にこれを追加してください
# あるいは Dagit の Launchpad または JobDefinition.execute_in_process で設定できます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # ここをあなたの W&B entity に置き換えてください
     project: my_project # ここをあなたの W&B project に置き換えてください
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
   run_launch_job.alias("my_launched_job")() # エイリアスでジョブ名を変更しています
```

## ベストプラクティス

1. Artifacts の読み書きには IO Manager を使いましょう。 
[`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact#download" lang="ja" >}}) や [`Run.log_artifact()`]({{< relref path="/ref/python/sdk/classes/run#log_artifact" lang="ja" >}}) を直接使うのは避けてください。これらは連携で処理されます。代わりに、Artifact に保存したいデータを戻り値として返し、残りは連携に任せましょう。このやり方は Artifact のリネージをより良くします。

2. 複雑なユースケースに限って自分で Artifact オブジェクトを構築しましょう。
Python オブジェクトや W&B オブジェクトは ops／assets からそのまま返してください。連携が Artifact へのバンドルを処理します。
複雑なユースケースでは、Dagster のジョブ内で直接 Artifact を構築してもかまいません。ソース連携の名前とバージョン、使用した Python のバージョン、pickle プロトコルのバージョンなどのメタデータ拡充のため、Artifact オブジェクトを連携に渡すことをおすすめします。

3. ファイル、ディレクトリー、外部参照はメタデータ経由で Artifact に追加しましょう。
ファイル、ディレクトリー、外部参照（Amazon S3、GCS、HTTP など）の追加には、連携の `wandb_artifact_configuration` オブジェクトを使ってください。詳細は [Artifact 設定セクション]({{< relref path="#configuration-1" lang="ja" >}}) の高度な例を参照してください。

4. Artifact を生成する場合は、@op より @asset を使いましょう。
Artifacts は assets です。Dagster がその asset を管理する場合は asset を使うのが推奨です。これにより Dagit の Asset Catalog での可観測性が向上します。

5. Dagster の外部で作られた Artifact を消費するには SourceAsset を使いましょう。
これにより、連携を活用して外部作成の Artifacts を読み取れます。そうしない場合、連携が作成した Artifacts のみ利用可能になります。

6. 大規模なモデルには、専用コンピュートでのトレーニングをオーケストレーションするために W&B Launch を使いましょう。
小さなモデルは Dagster クラスター内で学習してもよく、GPU ノードを持つ Kubernetes クラスター上で Dagster を動かすこともできます。ただし大規模なモデル学習では W&B Launch の利用をおすすめします。これによりインスタンスの過負荷を防ぎ、より適切なコンピュートにアクセスできます。 

7. Dagster 内で実験管理を行うときは、W&B Run ID を Dagster Run ID に設定しましょう。
[Run の再開]({{< relref path="/guides/models/track/runs/resuming.md" lang="ja" >}}) を有効にし、W&B Run ID を Dagster Run ID または任意の文字列に設定することを推奨します。これにより、Dagster 内でモデルをトレーニングする際に、W&B のメトリクスと W&B Artifacts が同じ W&B Run に保存されます。

いずれかの方法で、W&B Run ID を Dagster Run ID に設定します。
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

または、自分で決めた W&B Run ID を IO Manager の設定に渡します。
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

8. 大きな W&B Artifacts では、必要なデータだけを get や get_path で取得しましょう。
デフォルトでは、連携は Artifact 全体をダウンロードします。非常に大きな Artifact を使う場合、必要なファイルやオブジェクトだけを取得するようにすると、速度とリソース使用効率が向上します。

9. Python オブジェクトでは、ユースケースに合わせて pickling モジュールを選びましょう。
デフォルトでは W&B 連携は標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使いますが、一部のオブジェクトは互換性がありません。例えば、yield を含む関数は pickle 化でエラーになります。W&B は他の Pickle 系シリアライズモジュール（[dill](https://github.com/uqfoundation/dill)、[cloudpickle](https://github.com/cloudpipe/cloudpickle)、[joblib](https://github.com/joblib/joblib)）もサポートします。 

また、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) のような高度なシリアライズも、シリアライズ済み文字列を返すか、直接 Artifact を作成することで利用できます。適切な選択はユースケースに依存します。関連文献を参照してください。