---
title: 아티팩트 업데이트
description: W&B Run 내부와 외부에서 기존 Artifact를 업데이트하세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

원하는 값들을 전달하여 아티팩트의 `description`, `metadata`, 그리고 `alias`를 업데이트할 수 있습니다. `save()` 메소드를 호출하면 W&B 서버에 아티팩트가 업데이트됩니다. W&B Run 중에도, Run 외부에서도 아티팩트를 업데이트할 수 있습니다.

Run 외부에서 아티팩트를 업데이트하려면 W&B Public API([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}))를 사용하세요. Run 중에 아티팩트를 업데이트하려면 Artifact API([`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}))를 사용하세요.

{{% alert color="secondary" %}}
Model Registry에 연결된 아티팩트의 alias는 업데이트할 수 없습니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}

다음 코드 예시는 [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}) API를 사용하여 아티팩트의 description을 업데이트하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
다음 코드 예시는 `wandb.Api` API를 사용하여 아티팩트의 description을 업데이트하는 방법을 보여줍니다:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# description 업데이트
artifact.description = "My new description"

# 메타데이터 키만 선택적으로 업데이트
artifact.metadata["oldKey"] = "new value"

# 메타데이터 전체를 대체
artifact.metadata = {"newKey": "new value"}

# alias 추가
artifact.aliases.append("best")

# alias 제거
artifact.aliases.remove("latest")

# alias 전체를 대체
artifact.aliases = ["replaced"]

# 모든 아티팩트 수정 사항 저장
artifact.save()
```

자세한 내용은 Weights and Biases [Artifact API]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}) 문서를 참고하세요.  
  {{% /tab %}}
  {{% tab header="With collections" %}}
아티팩트 컬렉션도 단일 아티팩트와 동일한 방법으로 업데이트할 수 있습니다:

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<이곳에 컬렉션의 목적을 설명하세요.>"
artifact.save()
```
자세한 내용은 [Artifacts Collection]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}) 레퍼런스를 참고하세요.
  {{% /tab %}}
{{% /tabpane %}}