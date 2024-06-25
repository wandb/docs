---
description: W&B Artifacts に関するよくある質問への回答。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Artifacts FAQs

<head>
  <title>Artifacts に関するよくある質問</title>
</head>

以下の質問は [W&B Artifacts](#questions-about-artifacts) と [W&B Artifact ワークフロー](#questions-about-artifacts-workflows) についてよくある質問です。

## Questions about Artifacts

### 自分の artifacts には誰がアクセスできますか？

Artifacts は親プロジェクトのアクセス権を継承します:

* プロジェクトが非公開の場合、そのプロジェクトのチームメンバーのみが artifacts にアクセスできます。
* 公開プロジェクトの場合、すべてのユーザーが artifacts を読み取るアクセス権を持ちますが、作成や変更ができるのはプロジェクトのチームメンバーのみです。
* オープンプロジェクトの場合、すべてのユーザーが artifacts を読み書きできます。

## Questions about Artifacts workflows

このセクションでは Artifacts の管理と編集に関するワークフローについて説明します。多くのワークフローでは [W&B API](../track/public-api-guide.md) を使用します。これは [我々のクライアントライブラリ](../../ref/python/README.md) のコンポーネントで、W&B に保存されているデータへのアクセスを提供します。

### 既存の run に artifact をログするにはどうすれば良いですか？

時折、以前にログした run の出力として artifact をマークしたい場合があります。その場合、古い run を [再初期化](../runs/resuming.md) して新しい artifacts をログできます。次のように行います:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```

### アーティファクトに保存期限または削除ポリシーを設定するにはどうすれば良いですか？

PII を含むデータセットアーティファクトなど、データプライバシー規制の対象となる artifacts がある場合や、ストレージ管理のためにアーティファクトのバージョンの削除をスケジュールしたい場合は、TTL（タイム・トゥ・リブ）ポリシーを設定できます。詳細は [こちら](./ttl.md) のガイドを参照してください。

### run がログしたまたは消費した artifacts をどのように見つけることができますか？ また、それらの artifacts を生成または消費した run をどのように見つけることができますか？

W&B は特定の run がログした artifacts と使用した artifacts を自動的に追跡し、その情報を使用してアーティファクトグラフを構築します。これは、ノードが runs と artifacts である二部的有向非巡回グラフです。このグラフは [こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) （「Explode」をクリックするとグラフ全体が表示されます）などがあります。

このグラフは [Public API](../../ref/python/public-api/README.md) を使ってプログラム的にナビゲートできます。run または artifact から始めることができます。

<Tabs
  defaultValue="from_artifact"
  values={[
    {label: 'From an Artifact', value: 'from_artifact'},
    {label: 'From a Run', value: 'from_run'},
  ]}>
  <TabItem value="from_artifact">

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# アーティファクトからグラフを上方に移動:
producer_run = artifact.logged_by()
# アーティファクトからグラフを下方に移動:
consumer_runs = artifact.used_by()

# run からグラフを下方に移動:
next_artifacts = consumer_runs[0].logged_artifacts()
# run からグラフを上方に移動:
previous_artifacts = producer_run.used_artifacts()
```

  </TabItem>
  <TabItem value="from_run">

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run からグラフを下方に移動:
produced_artifacts = run.logged_artifacts()
# run からグラフを上方に移動:
consumed_artifacts = run.used_artifacts()

# アーティファクトからグラフを上方に移動:
earlier_run = consumed_artifacts[0].logged_by()
# アーティファクトからグラフを下方に移動:
consumer_runs = produced_artifacts[0].used_by()
```

  </TabItem>
</Tabs>

### sweep の run からモデルを最適にログする方法は？

[sweep](../sweeps/intro.md) のモデルをログする効果的なパターンの一つは、sweep のモデルアーティファクトを持ち、そのバージョンが sweep の異なる runs に対応するようにすることです。具体的には、次のようにします:

```python
wandb.Artifact(name="sweep_name", type="model")
```

### sweep の最適な run からアーティファクトを見つけるにはどうすれば良いですか？

以下のコードを使用して、sweep で最良のパフォーマンスを示す run に関連する artifacts を取得できます:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```

### コードを保存するにはどうすれば良いですか？‌

`wandb.init` に `save_code=True` を使用して、run を起動しているメインスクリプトやノートブックを保存します。すべてのコードを run に保存するには、Artifacts を使用してコードをバージョン管理します。以下はその例です:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```

### 複数のアーキテクチャと runs で artifacts を使用するには？

モデルのバージョン管理には様々な方法があります。Artifacts はあなたが適切だと思う方法でモデルのバージョン管理を実装するツールを提供します。複数のモデルアーキテクチャを多数の runs にわたって探索するプロジェクトの場合、アーキテクチャごとに artifacts を分けるのが一般的なパターンです。例えば、このように行うことができます:

1. 異なるモデルアーキテクチャ毎に新しいアーティファクトを作成します。アーティファクトの `metadata` 属性を使用してアーキテクチャの詳細を記述します（run で `config` を使用する方法に似ています）。
2. 各モデルに対して、定期的に `log_artifact` を使用してチェックポイントをログします。W&B は自動的にこれらのチェックポイントの履歴を構築し、最も最近のチェックポイントに `latest` エイリアスを付与するので、任意のモデルアーキテクチャに対して `architecture-name:latest` を使用して最新のチェックポイントを参照できます。

## Reference Artifact FAQs

### W&B でこれらのバージョンIDとETagを取得するにはどうすればよいですか？

W&B にアーティファクトリファレンスをログし、バケットでバージョン管理が有効になっている場合、バージョンIDは S3 UI で確認できます。これらのバージョンIDとETagを W&B で取得するには、アーティファクトを取得し、対応するマニフェストエントリを取得します。例えば:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = manifest_entry.extra.get("etag")
```
