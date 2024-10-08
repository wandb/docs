---
title: Update an artifact
description: 기존 아티팩트를 W&B run 내외부에서 업데이트하세요.
displayed_sidebar: default
---

아티팩트의 `description`, `metadata`, `alias`를 업데이트하고 싶다면 원하는 값을 전달하세요. `save()` 메소드를 호출하여 W&B 서버에 아티팩트를 업데이트하세요. W&B Run 중 또는 Run 외부에서 아티팩트를 업데이트할 수 있습니다.

Run 외부에서 아티팩트를 업데이트하려면 W&B Public API ([`wandb.Api`](../../ref/python/public-api/api.md))를 사용하세요. Run 중에 아티팩트를 업데이트하려면 Artifact API ([`wandb.Artifact`](../../ref/python/artifact.md))를 사용하세요.

:::caution
Model Registry에 연결된 아티팩트의 alias는 업데이트할 수 없습니다.
:::

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: 'During a Run', value: 'duringrun'},
    {label: 'Outside of a Run', value: 'outsiderun'},
    {label: 'With Collections', value: 'withcollections'}
  ]}>
  <TabItem value="duringrun">

다음 코드 예제는 [`wandb.Artifact`](../../ref/python/artifact.md) API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```
  </TabItem>
  <TabItem value="outsiderun">

다음 코드 예제는 `wandb.Api` API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# 설명 업데이트
artifact.description = "My new description"

# 메타데이터 키 선택적 업데이트
artifact.metadata["oldKey"] = "new value"

# 메타데이터를 전부 교체
artifact.metadata = {"newKey": "new value"}

# 에일리어스 추가
artifact.aliases.append("best")

# 에일리어스 제거
artifact.aliases.remove("latest")

# 에일리어스를 완전히 교체
artifact.aliases = ["replaced"]

# 모든 아티팩트 수정 사항 저장
artifact.save()
```

자세한 정보는 Weights and Biases [Artifact API](../../ref/python/artifact.md) 참조를 참조하세요.
  </TabItem>
  
  <TabItem value="withcollections">
또한 단일 아티팩트와 동일한 방식으로 Artifact 컬렉션을 업데이트할 수 있습니다:

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
자세한 정보는 [Artifacts Collection](../../ref/python/public-api/api) 참조를 참조하세요.

  </TabItem>
</Tabs>