---
description: 既存のアーティファクトを W&B run 内外で更新する。
displayed_sidebar: default
---


# アーティファクトを更新する

<head>
  <title>アーティファクトを更新する</title>
</head>

アーティファクトの `description`、`metadata`、および `alias` を更新するために、希望する値を渡します。アーティファクトを W&B サーバー上で更新するためには `save()` メソッドを呼び出します。W&B Run 中または Run の外でアーティファクトを更新することができます。

W&B Public API ([`wandb.Api`](../../ref/python/public-api/api.md)) を使用して Run の外でアーティファクトを更新します。Run 中にアーティファクトを更新するには Artifact API ([`wandb.Artifact`](../../ref/python/artifact.md)) を使用します。

:::caution
モデルレジストリにリンクされているアーティファクトのエイリアスは更新できません。
:::

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: 'Run 中', value: 'duringrun'},
    {label: 'Run の外', value: 'outsiderun'},
  ]}>
  <TabItem value="duringrun">

以下のコード例は、[`wandb.Artifact`](../../ref/python/artifact.md) API を使用してアーティファクトの説明を更新する方法を示しています:

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
artifact = run.use_artifact("<artifact-name>:<alias>")

artifact = wandb.Artifact("")
run.use_artifact(artifact)
artifact.description = "<description>"
artifact.save()
```
  </TabItem>
  <TabItem value="outsiderun">

以下のコード例は、`wandb.Api` API を使用してアーティファクトの説明を更新する方法を示しています:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# 説明を更新
artifact.description = "My new description"

# メタデータキーを選択的に更新
artifact.metadata["oldKey"] = "new value"

# メタデータを完全に置き換え
artifact.metadata = {"newKey": "new value"}

# エイリアスを追加
artifact.aliases.append("best")

# エイリアスを削除
artifact.aliases.remove("latest")

# エイリアスを完全に置き換え
artifact.aliases = ["replaced"]

# すべてのアーティファクト修正を保存
artifact.save()
```

詳細については、Weights and Biases [Artifact API](../../ref/python/artifact.md) を参照してください。
  </TabItem>
</Tabs>