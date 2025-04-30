---
title: アーティファクトを更新する
description: 既存のアーティファクトを W&B run の内外で更新します。
menu:
  default:
    identifier: ja-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

アーティファクトの `description`、`metadata`、および `alias` に希望する値を渡します。W&B サーバー上でアーティファクトを更新するには、`save()` メソッドを呼び出してください。W&B Run の間または Run の外でアーティファクトを更新することができます。

W&B Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用して、Run の外でアーティファクトを更新します。Artifact API ([`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})) を使用して、Run の間にアーティファクトを更新します。

{{% alert color="secondary" %}}
Model Registry にリンクされたアーティファクトのエイリアスを更新することはできません。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}

次のコード例は、[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用してアーティファクトの説明を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
次のコード例は、`wandb.Api` API を使用してアーティファクトの説明を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# 説明を更新する
artifact.description = "My new description"

# メタデータキーを選択的に更新する
artifact.metadata["oldKey"] = "new value"

# メタデータを完全に置き換える
artifact.metadata = {"newKey": "new value"}

# エイリアスを追加する
artifact.aliases.append("best")

# エイリアスを削除する
artifact.aliases.remove("latest")

# エイリアスを完全に置き換える
artifact.aliases = ["replaced"]

# すべてのアーティファクトの変更を保存する
artifact.save()
```

詳細は、Weights and Biases [Artifact API]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="With collections" %}}
コレクションも単一のアーティファクトと同様に更新することができます。

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
詳細は [Artifacts Collection]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) リファレンスを参照してください。
  {{% /tab %}}
{{% /tabpane %}}