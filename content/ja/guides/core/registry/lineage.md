---
title: リネージ マップの作成と表示
description: W&B Registry でリネージマップを作成します。
menu:
  default:
    identifier: ja-guides-core-registry-lineage
    parent: registry
weight: 8
---

W&B Registry のコレクションでは、ML experiment で使用される Artifacts の履歴を表示できます。この履歴は _lineage graph_ と呼ばれます。
{{% pageinfo color="info" %}}
コレクションの一部ではない、W&B にログされた Artifacts の lineage graph も表示できます。
{{% /pageinfo %}}

Lineage graph は、Artifact をログした特定の Run を表示できます。さらに、どの Run が Artifact を入力として使用したかも表示できます。つまり、Lineage graph は Run の入力と出力の両方を可視化します。

例えば、次の画像は、ML experiment 全体で作成され使用された Artifacts を示しています。
{{< img src="/images/registry/registry_lineage.png" alt="Registry のリネージ" >}}

左から右へ、画像は以下を示します。
1. 複数の Runs が `split_zoo_dataset:v4` Artifact をログします。
2. 「rural-feather-20」Run は、`split_zoo_dataset:v4` Artifact をトレーニングに使用します。
3. 「rural-feather-20」Run の出力は、`zoo-ylbchv20:v0` という名前の Model Artifact です。
4. 「northern-lake-21」という名前の Run は、`zoo-ylbchv20:v0` Model Artifact を使用して Model を評価します。

## Run の入力を追跡する

`wandb.init.use_artifact` API を使用して、Artifact を Run の入力または依存関係としてマークします。

次のコードスニペットは、`use_artifact` の使用方法を示しています。山括弧 (`< >`) で囲まれた値をご自身の値に置き換えてください。
```python
import wandb

# Run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# Artifact を取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```

## Run の出力を追跡する

Run の出力として Artifact を宣言するには、([`wandb.init.log_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ja" >}})) を使用します。

次のコードスニペットは、`wandb.init.log_artifact` API の使用方法を示しています。山括弧 (`< >`) で囲まれた値をご自身の値に置き換えてください。
```python
import wandb

# Run を初期化
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# Artifact を Run の出力としてログ
run.log_artifact(artifact_or_path = artifact)
```

Artifacts の作成に関する詳細については、[Artifact を作成する]({{< relref path="guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。

## コレクション内の lineage graph を表示する

W&B Registry のコレクションにリンクされている Artifact のリネージを表示します。
1. W&B Registry に移動します。
2. Artifact を含むコレクションを選択します。
3. ドロップダウンから、lineage graph を表示したい Artifact のバージョンをクリックします。
4. 「Lineage」タブを選択します。

Artifact の lineage graph ページに入ると、その lineage graph 内の任意のノードに関する追加情報を表示できます。

Run ノードを選択すると、その Run の ID、Run の名前、Run のステータスなどの詳細を表示できます。例として、次の画像は `rural-feather-20` Run に関する情報を示しています。
{{< img src="/images/registry/lineage_expanded_node.png" alt="展開されたリネージ ノード" >}}

Artifact ノードを選択すると、その Artifact の詳細（完全な名前、タイプ、作成時刻、関連するエイリアス など）を表示できます。
{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="展開された Artifact ノードの詳細" >}}