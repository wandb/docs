---
title: Artifacts を更新する
description: W&B Run の内外で既存の Artifact を更新します。
menu:
  default:
    identifier: ja-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

`description`、`metadata`、`alias` を更新するために必要な値をアーティファクトに渡します。`save()` メソッドを呼び出して、W&B サーバー上のアーティファクトを更新します。アーティファクトは W&B Run 中に更新することも、Run の外部で更新することもできます。
{{% alert title="Artifact.save() または wandb.Run.log_artifact() を使用するタイミング" %}}
- 既存のアーティファクトを新しい run を作成せずに更新するには `Artifact.save()` を使用します。
- 新しいアーティファクトを作成し、特定の run に関連付けるには `wandb.Run.log_artifact()` を使用します。
{{% /alert %}}
run の外部でアーティファクトを更新するには W&B Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用します。run 中にアーティファクトを更新するには Artifact API ([`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}})) を使用します。
{{% alert color="secondary" %}}
Model Registry のモデルにリンクされているアーティファクトの alias は更新できません。
{{% /alert %}}
{{< tabpane text=true >}}
  {{% tab header="run 中" %}}
以下のコード例は、[`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) API を使用してアーティファクトの description を更新する方法を示しています。
```python
import wandb

run = wandb.init(project="<例>")
artifact = run.use_artifact("<アーティファクト名>:<エイリアス>")
artifact.description = "<説明>"
artifact.save()
```
  {{% /tab %}}
  {{% tab header="run の外部" %}}
以下のコード例は、`wandb.Api` API を使用してアーティファクトの description を更新する方法を示しています。
```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# description を更新
artifact.description = "My new description"

# メタデータキーを選択的に更新
artifact.metadata["oldKey"] = "new value"

# メタデータを完全に置き換える
artifact.metadata = {"newKey": "new value"}

# エイリアスを追加
artifact.aliases.append("best")

# エイリアスを削除
artifact.aliases.remove("latest")

# エイリアスを完全に置き換える
artifact.aliases = ["replaced"]

# すべてのアーティファクトの変更を永続化
artifact.save()
```
詳細については、Weights and Biases の [Artifact API]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="コレクションを使用する場合" %}}
単一のアーティファクトと同様に、Artifact コレクションも更新できます。
```python
import wandb
run = wandb.init(project="<例>")
api = wandb.Api()
artifact = api.artifact_collection(type="<タイプ名>", collection="<コレクション名>")
artifact.name = "<新しいコレクション名>"
artifact.description = "<ここにコレクションの目的を記述します。>"
artifact.save()
```
詳細については、[Artifacts コレクション]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) のリファレンスを参照してください。
  {{% /tab %}}
{{% /tabpane %}}