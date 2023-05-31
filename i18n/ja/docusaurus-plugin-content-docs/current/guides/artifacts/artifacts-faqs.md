---
description: Answers to frequently asked question about W&B Artifacts.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# アーティファクトFAQ

<head>
  <title>アーティファクトに関するよくある質問</title>
</head>

以下の質問は、[W&B Artifacts](#questions-about-artifacts)および[W&B Artifactワークフロー](#questions-about-artifacts-workflows)に関してよくある質問です。

## アーティファクトに関する質問

### アーティファクトのファイルはいつ削除されますか？

W&Bは、上記のように、連続するアーティファクトのバージョン間での重複を最小限に抑える方法でアーティファクトのファイルを保存しています。

アーティファクトのバージョンを削除する際、W&Bは完全に削除が安全なファイルをチェックします。つまり、そのファイルが以前のアーティファクトのバージョンまたは後続のアーティファクトのバージョンで使用されていないことを保証します。削除が安全である場合、ファイルは直ちに削除され、当社のサーバーにはその痕跡は残りません。

### 誰が私のアーティファクトにアクセスできますか？

アーティファクトは、親プロジェクトのアクセス権を継承します。

* プロジェクトがプライベートの場合、プロジェクトのチームのメンバーのみがそのアーティファクトにアクセスできます。
* 公開プロジェクトの場合、すべてのユーザーはアーティファクトへの読み取りアクセスがありますが、プロジェクトのチームのメンバーのみがそれらを作成または変更できます。
* オープンプロジェクトの場合、すべてのユーザーはアーティファクトへの読み書きアクセスがあります。
## アーティファクトのワークフローに関する質問

このセクションでは、アーティファクトの管理と編集のワークフローについて説明します。これらのワークフローの多くは、W&Bで保存されたデータにアクセスを提供する[クライアントライブラリのコンポーネントであるW&B API](../track/public-api-guide.md)を使用しています。

### 既存のランにアーティファクトをログする方法は？

たまに、アーティファクトを以前にログしたランの出力としてマークしたい場合があります。そのようなシナリオでは、古いランを再初期化して、以下のように新しいアーティファクトをログすることができます。

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```

### ランでログしたアーティファクトを検索したり、ランが利用したアーティファクトを検索したりする方法はありますか？ アーティファクトを生成または使用したランを検索する方法は？

W&Bは、ランがログしたアーティファクトと、ランが使用したアーティファクトを自動的にトラッキングし、その情報をもとに、ランとアーティファクトで構成される二部有向非循環グラフであるアーティファクトグラフを構築します。[こちらのグラフ](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)のようなグラフです（フルグラフを見るには、「Explode」をクリックしてください）。

このグラフは、ランまたはアーティファクトから始めて、[Public API](../../ref/python/public-api/README.md)を使ってプログラムで探索することができます。

<Tabs
  defaultValue="from_artifact"
  values={[
    {label: 'アーティファクトから', value: 'from_artifact'},
    {label: 'ランから', value: 'from_run'},
  ]}>
  <TabItem value="from_artifact">

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# アーティファクトからグラフを遡る：
producer_run = artifact.logged_by()
# アーティファクトからグラフをたどる：
consumer_runs = artifact.used_by()

# runからグラフをたどる：
next_artifacts = consumer_runs[0].logged_artifacts()
# runからグラフを遡る：
previous_artifacts = producer_run.used_artifacts()
```

  </TabItem>
  <TabItem value="from_run">

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# runからグラフをたどる：
produced_artifacts = run.logged_artifacts()
# runからグラフを遡る：
consumed_artifacts = run.used_artifacts()

# アーティファクトからグラフを遡る：
earlier_run = consumed_artifacts[0].logged_by()
# アーティファクトからグラフをたどる：
consumer_runs = produced_artifacts[0].used_by()
```
</TabItem>
</Tabs>

### スイープ内のrunsからモデルを最適にログする方法は？

[sweep](../sweeps/intro.md)内でモデルをログする効果的な方法の1つは、スイープのモデルアーティファクトを持ち、バージョンがスイープからの異なるrunsに対応するようにすることです。具体的には、次のようになります。

```python
wandb.Artifact(name="sweep_name", type="model")
```

### スイープ内の最適なrunからアーティファクトを見つける方法は？

以下のコードを使用して、スイープ内で最も性能の良いrunに関連付けられたアーティファクトを取得できます。

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs,
              key=lambda run: run.summary.get("val_acc", 0), 
              reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
  artifact_path = artifact.download()
  print(artifact_path)
```

### コードを保存する方法は？

`wandb.init`で`save_code=True`を使用して、runを開始するメインスクリプトまたはノートブックを保存します。すべてのコードをrunに保存するには、アーティファクトを使ってコードをバージョン管理します。以下に例を示します。
```python
code_artifact = wandb.Artifact(type="code")

code_artifact.add_file("./train.py")

wandb.log_artifact(code_artifact)
```

### 複数のアーキテクチャーとrunsを使ったアーティファクトの使い方

モデルのバージョンを考える方法は多くありますが、アーティファクトはモデルのバージョン管理を自分で実装するためのツールを提供します。複数のモデルアーキテクチャーを探求し、いくつかのrunsでアーティファクトをアーキテクチャーごとに分ける典型的なパターンがあります。例えば以下のように行うことができます。

1. 異なるモデルアーキテクチャーごとに新しいアーティファクトを作成します。アーティファクトの`metadata`属性を使って、アーキテクチャーをより詳細に記述できます（runの`config`を使うのと同様に）。

2. 各モデルのチェックポイントを`log_artifact`で定期的に記録します。W&Bは自動的にこれらのチェックポイントの履歴を構築し、最新のチェックポイントを`latest`エイリアスで注釈付けします。これにより、任意のモデルアーキテクチャの最新のチェックポイントを`architecture-name:latest`で参照できます。

## アーティファクト参照FAQ

### W&BでこれらのバージョンIDとETagsをどのように取得できますか？

W&Bでアーティファクト参照をログし、バケットでバージョン管理が有効になっている場合、バージョンIDはS3 UIで表示できます。W&BでこれらのバージョンIDとETagsを取得するには、[公開API](../../ref/python/public-api/artifact.md) を使って対応するマニフェストエントリを取得できます。例えば以下のようになります。

```python
artifact = run.use_artifact('my_table:latest')

for entry in artifact.manifest.entries.values():

    versionID = entry.extra.get("versionID")

    etag = manifest_entry.extra.get("etag")
```