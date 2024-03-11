---
description: Update an existing Artifact inside and outside of a W&B Run.
displayed_sidebar: default
---

# 아티팩트 업데이트

<head>
  <title>아티팩트 업데이트</title>
</head>

아티팩트의 `description`, `metadata`, 그리고 `alias`를 업데이트하려는 값으로 전달하세요. W&B 서버에서 아티팩트를 업데이트하려면 `save()` 메소드를 호출하세요. W&B Run 중이거나 Run 외부에서 아티팩트를 업데이트할 수 있습니다.

Run 외부에서 아티팩트를 업데이트하려면 W&B Public API ([`wandb.Api`](../../ref/python/public-api/api.md))를 사용하세요. Run 중에 아티팩트를 업데이트하려면 Artifact API ([`wandb.Artifact`](../../ref/python/artifact.md))를 사용하세요.

:::caution
모델 레지스트리에 연결된 아티팩트의 에일리어스는 업데이트할 수 없습니다.
:::


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: 'Run 중', value: 'duringrun'},
    {label: 'Run 외부', value: 'outsiderun'},
  ]}>
  <TabItem value="duringrun">

다음 코드 예제는 [`wandb.Artifact`](../../ref/python/artifact.md) API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다:

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

다음 코드 예제는 `wandb.Api` API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# 설명 업데이트
artifact.description = "My new description"

# 메타데이터 키 선택적 업데이트
artifact.metadata["oldKey"] = "new value"

# 메타데이터 전체 교체
artifact.metadata = {"newKey": "new value"}

# 에일리어스 추가
artifact.aliases.append("best")

# 에일리어스 제거
artifact.aliases.remove("latest")

# 에일리어스 전체 교체
artifact.aliases = ["replaced"]

# 모든 아티팩트 수정 사항 저장
artifact.save()
```

자세한 정보는 Weights and Biases [Artifact API](../../ref/python/artifact.md)를 참조하세요.
  </TabItem>
</Tabs>