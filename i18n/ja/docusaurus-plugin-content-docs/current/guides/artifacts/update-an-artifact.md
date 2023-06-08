---
description: Update an existing Artifact inside and outside of a W&B Run.
displayed_sidebar: ja
---

# アーティファクトの更新

<head>
  <title>アーティファクトの更新</title>
</head>
artifactの`description`、`metadata`、および`alias`を更新するために、望ましい値を渡してください。`save()`メソッドを呼び出して、Weights & Biasesサーバー上のartifactを更新します。W&B Run中またはRunの外でartifactを更新できます。

W&B Public API（[`wandb.Api`](https://docs.wandb.ai/ref/python/public-api/api)）を使用して、runの外でartifactを更新します。Artifact API（[`wandb.Artifact`](https://docs.wandb.ai/ref/python/artifact)）を使用して、runの間にartifactを更新します。

:::caution
モデルレジストリ内のモデルにリンクされているartifactのエイリアスは更新できません。
:::


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: 'ランの間', value: 'duringrun'},
    {label: 'ランの外', value: 'outsiderun'},
  ]}>
  <TabItem value="duringrun">

次のコード例は、[`wandb.Artifact`](https://docs.wandb.ai/ref/python/artifact) APIを使用してアーティファクトの説明を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
artifact = run.use_artifact('<artifact-name>:<alias>')


アーティファクト = wandb.Artifact('')
run.use_artifact(artifact)
アーティファクトの説明 = '<説明>'
アーティファクト.save()
```

  </TabItem>
  <TabItem value="外部実行">

以下のコード例では、`wandb.Api` APIを使用して、アーティファクトの説明を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact('entity/project/artifact:エイリアス')

# 説明を更新する
artifact.description = "新しい説明"

# メタデータキーを選択して更新する
artifact.metadata["oldKey"] = "新しい値"
# メタデータを完全に置き換える
artifact.metadata = {"newKey": "新しい値"}

# エイリアスを追加する
artifact.aliases.append('best')
# エイリアスを削除する
artifact.aliases.remove('latest')

# エイリアスを完全に置き換える
artifact.aliases = ['replaced']

# すべてのアーティファクトの変更を保存する
artifact.save()
```
詳細については、Weights and Biasesの [Public Artifact API](https://docs.wandb.ai/ref/python/public-api/artifact) をご覧ください。
  </TabItem>
</Tabs>