---
title: アーティファクトを更新する
description: W&B Run の内外で既存のアーティファクトを更新する方法をご紹介します。
menu:
  default:
    identifier: update-an-artifact
    parent: artifacts
weight: 4
---

希望する値を `description`、`metadata`、および `alias` に渡してアーティファクトを更新できます。`save()` メソッドを呼び出すことで、アーティファクトが W&B サーバー上で更新されます。アーティファクトは W&B Run の中でも、Run の外でも更新できます。

Run の外でアーティファクトを更新するには、W&B Public API（[`wandb.Api`]({{< relref "/ref/python/public-api/api.md" >}})）を使用します。Run の中でアーティファクトを更新するには、Artifact API（[`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}})）を使用します。

{{% alert color="secondary" %}}
モデルレジストリでモデルにリンクされたアーティファクトの alias は更新できません。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}

以下のコード例は、[`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) API を使ってアーティファクトの説明を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
以下のコード例は、`wandb.Api` API を使ってアーティファクトの説明を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# 説明を更新
artifact.description = "My new description"

# 特定の metadata キーを更新
artifact.metadata["oldKey"] = "new value"

# metadata を全て置き換える
artifact.metadata = {"newKey": "new value"}

# エイリアスを追加
artifact.aliases.append("best")

# エイリアスを削除
artifact.aliases.remove("latest")

# エイリアスをすべて置き換え
artifact.aliases = ["replaced"]

# すべての変更を保存
artifact.save()
```

詳細は Weights and Biases の [Artifact API]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) をご覧ください。  
  {{% /tab %}}
  {{% tab header="With collections" %}}
アーティファクトコレクションも、単体のアーティファクトと同じ方法で更新できます。

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
詳しくは [Artifacts Collection]({{< relref "/ref/python/public-api/api.md" >}}) リファレンスをご覧ください。
  {{% /tab %}}
{{% /tabpane %}}