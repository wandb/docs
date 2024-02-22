---
description: Update an existing Artifact inside and outside of a W&B Run.
displayed_sidebar: default
---

# 아티팩트 업데이트

<head>
  <title>아티팩트 업데이트</title>
</head>

원하는 값들을 전달하여 아티팩트의 `description`, `metadata`, 및 `alias`를 업데이트하십시오. W&B 서버에서 아티팩트를 업데이트하기 위해 `save()` 메서드를 호출하십시오. W&B 실행 중이거나 실행 외부에서 아티팩트를 업데이트할 수 있습니다.

실행 외부에서 아티팩트를 업데이트하기 위해 W&B Public API([`wandb.Api`](../../ref/python/public-api/api.md))를 사용하십시오. 실행 도중에 아티팩트를 업데이트하기 위해 Artifact API([`wandb.Artifact`](../../ref/python/artifact.md))를 사용하십시오.

:::caution
모델 레지스트리에 연결된 아티팩트의 별칭은 업데이트할 수 없습니다.
:::


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: '실행 도중', value: 'duringrun'},
    {label: '실행 외부', value: 'outsiderun'},
  ]}>
  <TabItem value="duringrun">

다음 코드 예시는 [`wandb.Artifact`](../../ref/python/artifact.md) API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다:

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

다음 코드 예시는 `wandb.Api` API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다:

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

# 별칭 추가
artifact.aliases.append("best")

# 별칭 제거
artifact.aliases.remove("latest")

# 별칭 전체 교체
artifact.aliases = ["replaced"]

# 모든 아티팩트 수정 사항 저장
artifact.save()
```

더 많은 정보는 Weights and Biases [Artifact API](../../ref/python/artifact.md)를 참조하십시오.
  </TabItem>
</Tabs>