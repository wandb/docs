---
title: Dagster
description: W&B を Dagster と統合するためのガイド。
menu:
  launch:
    identifier: ja-launch-integration-guides-dagster
    parent: launch-integration-guides
url: /ja/guides/integrations/dagster
---

Dagster と W&B (W&B) を使用して MLOps パイプラインを調整し、ML アセットを維持します。W&B とのインテグレーションにより、Dagster 内で以下が簡単になります：

* [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の使用と作成。
* [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) で Registered Models の使用と作成。
* [W&B Launch]({{< relref path="/launch/" lang="ja" >}}) を使用して専用のコンピュートでトレーニングジョブを実行します。
* ops とアセットで [wandb]({{< relref path="/ref/python/" lang="ja" >}}) クライアントを使用します。

W&B Dagster インテグレーションは W&B 専用の Dagster リソースと IO マネージャーを提供します：

* `wandb_resource`: W&B API への認証と通信に使用される Dagster リソース。
* `wandb_artifacts_io_manager`: W&B Artifacts を処理するために使用される Dagster IO マネージャー。

以下のガイドでは、Dagster で W&B を使用するための前提条件の満たし方、ops とアセットで W&B Artifacts を作成して使用する方法、W&B Launch の利用方法、そして推奨されるベストプラクティスについて説明します。

## 始める前に
Dagster を Weights and Biases 内で使用するためには、以下のリソースが必要です：
1. **W&B API Key**。
2. **W&B entity (ユーザーまたはチーム)**: Entity は W&B Runs と Artifacts を送信する場所のユーザー名またはチーム名です。Runs をログに記録する前に、W&B App の UI でアカウントまたはチームエンティティを作成しておいてください。エンティティを指定しない場合、その run はデフォルトのエンティティに送信されます。通常、これはあなたのユーザー名です。設定の「Project Defaults」内でデフォルトのエンティティを変更できます。
3. **W&B project**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) が保存されるプロジェクトの名前。

W&B entity は、W&B App のそのユーザーまたはチームページのプロフィールページをチェックすることで見つけられます。既存の W&B project を使用するか、新しいものを作成することができます。新しいプロジェクトは、W&B App のホームページまたはユーザー/チームのプロフィールページで作成できます。プロジェクトが存在しない場合は、初回使用時に自動的に作成されます。以下の手順は API キーを取得する方法を示しています：

### APIキーの取得方法
1. [W&B にログインします](https://wandb.ai/login)。注：W&B サーバーを使用している場合は、管理者にインスタンスのホスト名を尋ねてください。
2. [認証ページ](https://wandb.ai/authorize) またはユーザー/チーム設定で APIキーを集めます。プロダクション環境では、そのキーを所有するために [サービスアカウント]({{< relref path="/support/kb-articles/service_account_useful.md" lang="ja" >}}) を使用することをお勧めします。
3. その APIキー用に環境変数を設定します。`WANDB_API_KEY=YOUR_KEY` をエクスポートします。

以下の例は、Dagster コード内で API キーを指定する場所を示しています。`wandb_config` のネストされた辞書内でエンティティとプロジェクト名を必ず指定してください。異なる W&B Project を使用したい場合は、異なる `wandb_config` の値を異なる ops/assets に渡すことができます。渡すことができる可能性のあるキーについての詳細は、以下の設定セクションを参照してください。

{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
例: `@job` の設定
```python
# これを config.yaml に追加します
# 代わりに、Dagit's Launchpad または JobDefinition.execute_in_process で設定することもできます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをあなたの W&B entity に置き換えます
     project: my_project # これをあなたの W&B project に置き換えます

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

例: アセットを使用する `@repository` の設定

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
                   {"cache_duration_in_minutes": 60} # ファイルを 1 時間だけキャッシュする
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # これをあなたの W&B entity に置き換えます
                       "project": "my_project", # これをあなたの W&B project に置き換えます
                   }
               }
           },
       ),
   ]
```
この例では @job の例と異なり IO Manager キャッシュ期間を設定しています。
{{% /tab %}}
{{< /tabpane >}}

### 設定
以下の設定オプションは、インテグレーションによって提供される W&B 専用 Dagster リソースと IO マネージャーの設定として使用されます。

* `wandb_resource`: W&B API と通信するために使用される Dagster [リソース](https://docs.dagster.io/concepts/resources)。提供された APIキー を使用して自動的に認証されます。プロパティ:
    * `api_key`: (ストリング, 必須): W&B API と通信するために必要な W&B APIキー。
    * `host`: (ストリング, オプショナル): 使用したい API ホストサーバー。W&B Server を使用している場合にのみ必要です。デフォルトはパブリッククラウドのホスト、`https://api.wandb.ai` です。
* `wandb_artifacts_io_manager`: W&B Artifacts を消費するための Dagster [IO マネージャー](https://docs.dagster.io/concepts/io-management/io-managers)。プロパティ：
    * `base_dir`: (整数, オプショナル) ローカルストレージとキャッシュに使用される基本ディレクトリ。W&B Artifacts と W&B Run のログはそのディレクトリから読み書きされます。デフォルトでは `DAGSTER_HOME` ディレクトリを使用します。
    * `cache_duration_in_minutes`: (整数, オプショナル) W&B Artifacts と W&B Run ログをローカルストレージに保持する時間。指定された時間が経過しアクセスされなかったファイルとディレクトリはキャッシュから削除されます。キャッシュのクリアは IO マネージャーの実行の終了時に行われます。キャッシュを無効にしたい場合は 0 に設定してください。キャッシュはジョブ間でアーティファクトが再利用されるときに速度を向上させます。デフォルトは30日間です。
    * `run_id`: (ストリング, オプショナル): この run の一意のIDで再開に使用されます。プロジェクト内で一意である必要があり、run を削除した場合、IDを再利用することはできません。短い説明名は name フィールドを使用し、ハイパーパラメーターを保存して runs 間で比較するために config を使用してください。IDには `/\#?%:` という特殊文字を含めることはできません。Dagster 内で実験管理を行う場合、IO マネージャーが run を再開できるように Run ID を設定する必要があります。デフォルトでは Dagster Run ID に設定されます。例：`7e4df022-1bf2-44b5-a383-bb852df4077e`。
    * `run_name`: (ストリング, オプショナル) この run を UI で識別しやすくするための短い表示名。デフォルトでは、以下の形式の文字列です：`dagster-run-[8最初のDagster Run IDの文字]`。たとえば、`dagster-run-7e4df022`。
    * `run_tags`: (list[str], オプショナル): この run の UI にタグ一覧を埋める文字列リスト。タグは runs をまとめて整理したり `baseline` や `production` など一時的なラベルを適用するのに便利です。UIでタグを追加・削除したり特定のタグを持つ run だけを絞り込むのは簡単です。インテグレーションで使用される W&B Run には `dagster_wandb` タグが付きます。

## W&B Artifacts を使用する

W&B Artifact とのインテグレーションは Dagster IO マネージャーに依存しています。

[IO マネージャー](https://docs.dagster.io/concepts/io-management/io-managers) は、アセットまたは op の出力を保存し、それを下流のアセットまたは ops への入力として読み込む責任を持つユーザ提供のオブジェクトです。たとえば、IO マネージャーはファイルシステム上のファイルからオブジェクトを保存および読み込む可能性があります。

今回のインテグレーションは W&B Artifacts 用のIO マネージャーを提供します。これにより Dagster の `@op` または `@asset` は W&B Artifacts をネイティブに作成および消費できます。ここに Python リストを含むデータセットタイプの W&B Artifact を生み出す `@asset` の簡単な例があります。

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

`@op`、`@asset`、`@multi_asset` をメタデータ設定で注釈を付けてアーティファクトを記述できます。同様に、W&B Artifacts を Dagster 外部で作成された場合でも消費できます。

## W&B Artifacts を書き込む
続行する前に、W&B Artifacts の使用方法について十分な理解を持っていることをお勧めします。[Guide on Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を検討してください。

Python 関数からオブジェクトを返すことで W&B Artifact を書き込みます。W&B でサポートされているオブジェクトは以下の通りです：
* Python オブジェクト (int, dict, list…)
* W&B オブジェクト (Table, Image, Graph…)
* W&B Artifact オブジェクト

以下の例は、Dagster アセット (`@asset`) を使用して W&B Artifacts を書き込む方法を示しています：

{{< tabpane text=true >}}
{{% tab "Python objects" %}}
[pickle](https://docs.python.org/3/library/pickle.html) モジュールでシリアライズできるものは何でも、インテグレーションによって作成された Artifact にピクルスされて追加されます。ダグスター内でその Artifact を読むときに内容が読み込まれます（さらなる詳細については [Read artifacts]({{< relref path="#read-wb-artifacts" lang="ja" >}}) を参照してください）。

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

W&B は複数のピクルスベースのシリアライズモジュール([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。また、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) といったより高度なシリアライズも利用できます。[Serialization]({{< relref path="#serialization-configuration" lang="ja" >}}) セクションを参照してください。
{{% /tab %}}
{{% tab "W&B Object" %}}
ネイティブ W&B オブジェクト (例: [Table]({{< relref path="/ref/python/data-types/table.md" lang="ja" >}}), [Image]({{< relref "/ref/python/data-types/image.md" >}}), or [Graph]({{< relref path="/ref/python/data-types/graph.md" lang="ja" >}})) のいずれかが作成された Artifact にインテグレーションによって追加されます。以下は Table を使った例です。

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

複雑なユースケースの場合、独自の Artifact オブジェクトを構築する必要があるかもしれません。インテグレーションは、統合の両側のメタデータを拡充するなど、便利な追加機能も提供しています。

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
`@op`、`@asset`、および `@multi_asset` の設定を行うために使用される辞書 wandb_artifact_configuration があり、この辞書はメタデータとしてデコレータの引数で渡される必要があります。この設定は、W&B Artifacts の IO マネージャーの読み取りと書き込みを制御するために必要です。

`@op` の場合、[Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) メタデータ引数を介して出力メタデータにあります。 
`@asset` の場合、アセットのメタデータ引数にあります。
`@multi_asset` の場合、[AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) メタデータ引数を介して各出力メタデータにあります。

以下のコード例は、`@op`、`@asset`、および `@multi_asset` 計算で辞書を構成する方法を示しています：

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

設定を通じて名前を渡す必要はありません。@asset にはすでに名前があります。インテグレーションはアセット名として Artifact 名を設定します。

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

サポートされたプロパティ：
* `name`: (str) このアーティファクトの人間が読み取り可能な名前で、その名前で UI内でこのアーティファクトを識別したり use_artifact 呼び出しで参照したりできます。名前には文字、数字、アンダースコア、ハイフン、ドットを含めることができます。プロジェクト内で一意である必要があります。`@op` に必須です。
* `type`: (str) アーティファクトのタイプで、アーティファクトを整理し差別化するために使用されます。一般的なタイプにはデータセットやモデルがありますが、任意の文字列を使用することができ、数字、アンダースコア、ハイフン、ドットを含めることができます。出力がすでにアーティファクトでない場合に必要です。
* `description`: (str) アーティファクトを説明するための自由なテキスト.説明は Markdownとして UIでレンダリングされるため,テーブル,リンクなどを配置するのに良い場所です。
* `aliases`: (list[str]) アーティファクトに適用したい 1つ以上のエイリアスを含む配列。インテグレーションは、それが設定されていようとなかろうと「最新」のタグもそのリストに追加します。これはモデルとデータセットのバージョン管理に効果的な方法です。
* [`add_dirs`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}): 配列（list[dict[str, Any]]）: Artifact に含める各ローカルディレクトリの設定を含む配列。SDK内の同名メソッドと同じ引数をサポートしています。
* [`add_files`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}): 配列（list[dict[str, Any]]）: Artifact に含める各ローカルファイルの設定を含む配列。SDK内の同名メソッドと同じ引数をサポートしています。
* [`add_references`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}): 配列（list[dict[str, Any]]）: Artifact に含める各外部リファレンスの設定を含む配列。SDK内の同名メソッドと同じ引数をサポートしています。
* `serialization_module`: (dict) 使用するシリアライズモジュールの設定。詳細については シリアル化 セクションを参照してください。
    * `name`: (str) シリアライズモジュールの名前。受け入れられる値: `pickle`, `dill`, `cloudpickle`, `joblib`。モジュールはローカルで使用可能である必要があります。
    * `parameters`: (dict[str, Any]) シリアライズ関数に渡されるオプション引数。モジュールの dump メソッドと同じ引数を受け入れます。例えば、`{"compress": 3, "protocol": 4}`。

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

アセットは統合の両側で有用なメタデータとともに実体化されます：
* W&B 側: ソースインテグレーション名とバージョン、使用された python バージョン、pickle プロトコルバージョンなど。
* Dagster 側:
    * Dagster Run ID
    * W&B Run: ID、名前、パス、URL
    * W&B Artifact: ID、名前、タイプ、バージョン、サイズ、URL
    * W&B エンティティ
    * W&B プロジェクト

以下の画像は、Dagster アセットに追加された W&B からのメタデータを示しています。この情報は、インテグレーションがなければ利用できませんでした。

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

以下の画像は、与えられた設定が W&B アーティファクト上の有用なメタデータでどのように充実されたかを示しています。この情報は、再現性とメンテナンスに役立ちます。インテグレーションがなければ利用できませんでした。

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
mypy のような静的型チェッカーを使用する場合は、以下の方法で設定タイプ定義オブジェクトをインポートしてください：

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### パーティションの利用

インテグレーションはネイティブに[Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)をサポートしています。

以下は `DailyPartitionsDefinition` を使用したパーティション化の例です。
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
このコードはパーティションごとに一つの W&B Artifact を生成します。アーティファクトは、アセット名の下にパーティションキーを追加して Artifact パネル (UI) で表示されます。例: `my_daily_partitioned_asset.2023-01-01`、`my_daily_partitioned_asset.2023-01-02`、または `my_daily_partitioned_asset.2023-01-03`。複数の次元でパーティション化されたアセットは、次元を点で区切った形式で表示されます。例: `my_asset.car.blue`。

{{% alert color="secondary" %}}  
インテグレーションによって、単一の run で複数のパーティションの実体化を許可することはできません。資産を実体化するためには、複数の run を実行する必要があります。これは、Dagit で資産を実体化するときに行うことができます。

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}  
{{% /alert %}}

#### 高度な使用法
- [パーティション化されたジョブ](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [シンプルなパーティション化アセット](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [マルチパーティション化アセット](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [高度なパーティション化の使用例](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts を読み取る
W&B Artifacts の読み取りは、それらを書くのと似ています。`@op` または `@asset` に `wandb_artifact_configuration` と呼ばれる設定辞書を設定することができます。唯一の違いは、その設定を出力ではなく入力に設定する必要がある点です。

`@op` の場合、[In](https://docs.dagster.io/_apidocs/ops#dagster.In) メタデータ引数を介して入力メタデータにあります。Artifact の名前を明示的に渡す必要があります。

`@asset` の場合、[Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) の In メタデータ引数の入力メタデータにあります。親アセットの名前がそれに一致する必要があるため、アーティファクトの名前を渡す必要はありません。

インテグレーションの外部で作成されたアーティファクトに依存関係を持たせたい場合は、[SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset) を使用する必要があります。それは常にそのアセットの最新バージョンを読み込みます。

次の例は、さまざまな ops から Artifact を読み取る方法を示しています。

{{< tabpane text=true >}}
{{% tab "From an @op" %}}
`@op` からアーティファクトを読み取る
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
別の `@asset` によって作成されたアーティファクトを読み取る
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 入力引数をリネームしたくない場合は 'key' を削除できます
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

Dagster の外部で作成された Artifact を読み取る：

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
以下の設定は、IO マネージャーが収集するものを装飾された関数への入力として提供するべきかを示すために使用されます。以下の読み取りパターンがサポートされています。

1. アーティファクト内にある名前付きオブジェクトを取得するには、get を使用します：

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

2. アーティファクト内にあるダウンロードされたファイルのローカルパスを取得するには、get_path を使用します：

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

3. アーティファクトオブジェクト全体を取得する（コンテンツをローカルでダウンロードします）：
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
* `get`: (str) アーティファクト相対の名前にある W&B オブジェクトを取得します。
* `get_path`: (str) アーティファクト相対の名前にあるファイルへのパスを取得します。

### シリアル化設定
デフォルトでは、インテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用しますが、一部のオブジェクトはこれと互換性がありません。たとえば、yield を持つ関数はシリアライズしようとした場合にエラーを発生させます。

より多くのピクルスベースのシリアライズモジュール ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。また、より高度なシリアル化を使用して [ONNX](https://onnx.ai/) または [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) など、シリアル化された文字列を返すか、直接アーティファクトを作成することもできます。あなたのユースケースに最適な選択肢は、利用可能な文献を参考にしてください。

### ピクルスベースのシリアル化モジュール

{{% alert color="secondary" %}}
ピクルスすることは安全性がないことが知られています。安全性が懸念される場合は、W&B オブジェクトのみを使用してください。データに署名し、ハッシュキーを独自のシステムで保存することをお勧めします。より複雑なユースケースに対しては、遠慮せずに私たちに連絡してください。私たちは喜んでお手伝いいたします。
{{% /alert %}}

使用するシリアル化を `wandb_artifact_configuration` 内の `serialization_module` 辞書を通じて設定することができます。Dagster を実行しているマシンでモジュールが利用可能であることを確認してください。

インテグレーションは、そのアーティファクトを読む際にどのシリアル化モジュールを使用するべきかを自動的に判断します。

現在サポートされているモジュールは `pickle`、`dill`、`cloudpickle`、および `joblib` です。

こちらが、joblib でシリアル化された「モデル」を作成し、推論に使用する例です。

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
    # これは本物の ML モデルではありませんが、pickle モジュールでは不可能であるものです
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
交換ファイル形式として ONNX や PMML を使用することは一般的です。インテグレーションはこれらの形式をサポートしていますが、Pickle ベースのシリアル化の場合よりも少し多くの作業が必要です。

これらの形式を使用する方法は 2 種類あります。
1. モデルを選択した形式に変換してから、通常の Python オブジェクトのようにその形式の文字列表現を返します。インテグレーションはその文字列をピクルスします。それから、その文字列を使用してモデルを再構築することができます。
2. シリアル化されたモデルを持つ新しいローカルファイルを作成し、そのファイルをカスタムアーティファクトに追加するために add_file 設定を実行します。

こちらは、Scikit-learn モデルを ONNX を使用してシリアル化する例です。

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
    # https://onnx.ai/sklearn-onnx/ からインスパイアされたサンプル

    # モデルのトレーニング
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 形式に変換
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # アーティファクトの書き込み（モデル + テストセット）
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
    # https://onnx.ai/sklearn-onnx/ からインスパイアされたサンプル

    # ONNX ランタイムを使用して予測を計算します
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

インテグレーションはネイティブに[Dagster パーティション](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)をサポートしています。

1つ、複数またはすべてのアセットパーティションを選別的に読み取ります。

すべてのパーティションは辞書で提供され、キーと値はそれぞれパーティションキーとアーティファクトコンテンツを表します。

{{< tabpane text=true >}}
{{% tab "Read all partitions" %}}
上流の `@asset` のすべてのパーティションを読み取り、それらは辞書として与えられます。この辞書で、キーはパーティションキー、値はアーティファクトコンテンツに関連しています。
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
指定したパーティションを選ぶために `AssetIn` の `partition_mapping` 設定を使用します。この例では `TimeWindowPartitionMapping` を使用しています。
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

設定オブジェクト `metadata` は、プロジェクト内の異なるアーティファクトパーティションと wandb のやり取りを設定するために使用されます。

オブジェクト `metadata` は、`wandb_artifact_configuration` というキーを含んでおり、さらに `partitions` というネストされたオブジェクトを含んでいます。

`partitions` オブジェクトは、各パーティションの名前とその設定をマッピングします。各パーティションの設定は、データの取得方法を指定でき、それには `get`、`version`、および `alias` のキーを含む場合があります。

**設定キー**

1. `get`:
`get` キーは、データを取得する W&B オブジェクト (テーブル、イメージなど) の名前を指定します。
2. `version`:
`version` キーは、特定のバージョンをアーティファクトから取得したいときに使用されます。
3. `alias`:
`alias` キーにより、エイリアスによってアーティファクトを取得することができます。

**ワイルドカード設定**

ワイルドカード `"*"` は、全ての非設定パーティションを表します。明示的に `partitions` オブジェクトに記載されていないパーティションに対するデフォルト設定を提供します。

例、

```python
"*": {
    "get": "default_table_name",
},
```
この設定は、明示的に設定されていないすべてのパーティションに対し、データが `default_table_name` というテーブルから取得されることを意味します。

**特定のパーティション設定**

ワイルドカード設定を、特定のキーを持つ特定のパーティション設定で上書きできます。

例、

```python
"yellow": {
    "get": "custom_table_name",
},
```

この設定は、`yellow` という名前のパーティションに対し、データが `custom_table_name` というテーブルから取得されることを意味し、ワイルドカード設定を上書きします。

**バージョニングとエイリアス**

バージョニングおよびエイリアスのために、設定で特定の `version` および `alias` のキーを指定することができます。

バージョンの場合、

```python
"orange": {
    "version": "v0",
},
```

この設定は、`orange` アーティファクトパーティションのバージョン `v0` からのデータを取得します。

エイリアスの場合、

```python
"blue": {
    "alias": "special_alias",
},
```

この設定は、アーティファクトパーティションのエイリアス `special_alias` (設定では `blue` として参照) の `default_table_name` テーブルからデータを取得します。

### 高度な使用法
インテグレーションの高度な使用法を確認するには、以下の完全なコード例を参照してください：
* [アセットに対する高度な使用例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [パーティション化されたジョブの例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [モデルをモデルレジストリにリンクする例](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch の使用

{{% alert color="secondary" %}}
ベータ版製品には積極的な開発が行われています。Launchに興味がありますか？ W&B Launch の顧客パイロットプログラムに参加するためにアカウントチームに連絡してください。
パイロット顧客は、ベータプログラムに適格となるためには AWS EKS もしくは SageMaker を使用する必要があります。最終的には追加のプラットフォームのサポートを計画しています。
{{% /alert %}}

継続する前に、W&B Launch の使用方法について十分な理解を持っていることをお勧めします。Launch のガイドを読むことを検討してください: /guides/launch。

Dagster インテグレーションは以下を補助します：
* Dagster インスタンス内での1つまたは複数の Launch エージェントの実行。
* あなたの Dagster インスタンス内でのローカル Launch ジョブの実行。
* オンプレミスまたはクラウドでのリモート Launch ジョブ。

### Launch エージェント
インテグレーションには `run_launch_agent` というインポート可能な `@op` が提供されます。この `@op` は Launch エージェントを起動し、手動で停止されるまで長時間実行プロセスとして実行します。

エージェントは launch キューをポールし、ジョブを（またはそれらを実行するために外部サービスにディスパッチ）発行するプロセスです。

設定については、[リファレンスドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください

Launchingpad で全プロパティに対する有用な説明を見ることもできます。

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

シンプルな例
```python
# これを config.yaml に追加します
# 代わりに、Dagit's Launchpad または JobDefinition.execute_in_process で設定することもできます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをあなたの W&B entity に置き換えます
     project: my_project # これをあなたの W&B project に置き換えます
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
インテグレーションには `run_launch_job` というインポート可能な `@op` が提供されます。この `@op` はあなたの Launch ジョブを実行します。

Launch ジョブは実行されるためにキューに割り当てられます。キューを作成するか、デフォルトのものを使用することができます。キューを監視する有効なエージェントがあることを確認します。あなたの Dagster インスタンス内でエージェントを実行するだけでなく、Kubernetes でデプロイ可能なエージェントを使用することも考慮に入れることができます。

設定については、[リファレンスドキュメント]({{< relref path="/launch/" lang="ja" >}}) を参照してください。

Launchpad では、すべてのプロパティに対する有用な説明も見ることができます。

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}

シンプルな例
```python
# これを config.yaml に追加します
# 代わりに、Dagit's Launchpad または JobDefinition.execute_in_process で設定することもできます
# 参考: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # これをあなたの W&B entity に置き換えます
     project: my_project # これをあなたの W&B project に置き換えます
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
   run_launch_job.alias("my_launched_job")() # 私たちはエイリアスを使ってジョブの名前を変更します。
```

## ベストプラクティス

1. IO マネージャーを使用して Artifacts を読み書きします。
[`Artifact.download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) や [`Run.log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) を直接使用する必要はありません。これらのメソッドはインテグレーションによって処理されます。Artifacts に保存したいデータを単に返し、インテグレーションに任せてください。これにより W&B での Artifact リネージが改善されます。

2. 複雑なユースケースのためにのみ Artifact オブジェクトを自分で構築します。
Python オブジェクトと W&B オブジェクトを ops/assets から返すべきです。インテグレーションは Artifact のバンドルを扱います。
複雑なユースケースに対しては、Dagster ジョブ内で直接 Artifact を構築できます。インテグレーション名とバージョン、使用された Python バージョン、ピクルスプロトコルバージョンなどのメタデータ拡充のために、インテグレーションに Artifact を渡すことをお勧めします。

3. メタデータを介してアーティファクトにファイル、ディレクトリ、外部リファレンスを追加します。
インテグレーション `wandb_artifact_configuration` オブジェクトを使用して、任意のファイル、ディレクトリ、外部リファレンス（Amazon S3、GCS、HTTP…）を追加します。詳細については [Artifact 設定セクション]({{< relref path="#configuration-1" lang="ja" >}}) の高度ない例を参照してください。

4. アーティファクトが生成される場合は、@op より @asset を使用してください。
Artifacts はなんらかのアセットです。Dagster がそのアセットを管理する場合は、アセットを使用することをお勧めします。これにより、Dagit Asset Catalog の可観測性が向上します。

5. Dagster 外部で作成されたアーティファクトを読み取るために SourceAsset を使用してください。
これにより、インテグレーションを活用して外部で作成されたアーティファクトを読むことができます。それ以外の場合、インテグレーションで作成されたアーティファクトのみを使用できます。

6. 大規模なモデルのための専用コンピュートでのトレーニングを調整するために W&B Launch を使用してください。
小さなモデルは Dagster クラスター内でトレーニングできますし、GPU ノードを持つ Kubernetes クラスターで Dagster を実行することもできます。W&B Launch を使用して大規模なモデルのトレーニングを行うことをお勧めします。これによりインスタンスの負荷が軽減され、より適切なコンピュートへのアクセスが得られます。

7. Dagster 内で実験管理を行う際は、W&B Run ID を Dagster Run ID の値に設定してください。
[Run を再開可能にする]({{< relref path="/guides/models/track/runs/resuming.md" lang="ja" >}}) ことと、W&B Run ID を Dagster Run ID またはお好みの文字列に設定することの両方をお勧めします。この推奨事項に従うことで、Dagster 内でモデルをトレーニングする際に W&B メトリクスと W&B Artifacts がすべて同じ W&B Run に格納されていることが保証されます。

W&B Run ID を Dagster Run ID に設定するか、

```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

独自の W&B Run ID を選び、それを IO マネージャー設定に渡します。

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

8. 大きな W&B Artifacts のために必要なデータだけを get や get_path で収集します。
デフォルトでインテグレーションはアーティファクト全体をダウンロードします。非常に大きなアーティファクトを使用している場合は、特定のファイルやオブジェクトだけを収集することをお勧めします。これにより速度が向上し、リソースの利用が向上します。

9. Python オブジェクトに対してユースケースに合わせてピクルスモジュールを適応させます。
デフォルトで W&Bインテグレーションは標準の [pickle](https://docs.python.org/3/library/pickle.html) モジュールを使用します。しかし、一部のオブジェクトはこれと互換性がありません。例えば、yield を持つ関数はシリアライズしようとするとエラーを発生します。W&B は他のピクルスベースのシリアライズモジュール([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)) をサポートしています。

また、[ONNX](https://onnx.ai/) や [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) など、より高度なシリアライズによってシリアライズされた文字列を返すか、直接 Artifact を作成することもできます。適切な選択はユースケースに依存します。このテーマに関しては、利用可能な文献を参考にしてください。