---
title: Update an artifact
description: W&B Run 내부 및 외부에서 기존 아티팩트를 업데이트합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

아티팩트의 `description`, `metadata`, `alias`를 업데이트하기 위해 원하는 값을 전달하세요. W&B 서버에서 아티팩트를 업데이트하려면 `save()` 메서드를 호출하세요. W&B Run 중 또는 Run 외부에서 아티팩트를 업데이트할 수 있습니다.

Run 외부에서 아티팩트를 업데이트하려면 W&B Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}))를 사용하세요. Run 중에 아티팩트를 업데이트하려면 Artifact API ([`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}))를 사용하세요.

{{% alert color="secondary" %}}
Model Registry에서 모델에 연결된 아티팩트의 에일리어스는 업데이트할 수 없습니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Run 중" %}}

다음 코드 예제는 [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```
  {{% /tab %}}
  {{% tab header="Run 외부" %}}
다음 코드 예제는 `wandb.Api` API를 사용하여 아티팩트의 설명을 업데이트하는 방법을 보여줍니다.

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# Update the description
artifact.description = "My new description"

# Selectively update metadata keys
artifact.metadata["oldKey"] = "new value"

# Replace the metadata entirely
artifact.metadata = {"newKey": "new value"}

# Add an alias
artifact.aliases.append("best")

# Remove an alias
artifact.aliases.remove("latest")

# Completely replace the aliases
artifact.aliases = ["replaced"]

# Persist all artifact modifications
artifact.save()
```

자세한 내용은 Weights and Biases [Artifact API]({{< relref path="/ref/python/artifact.md" lang="ko" >}})를 참조하세요.
  {{% /tab %}}
  {{% tab header="컬렉션 사용" %}}
단일 아티팩트와 같은 방식으로 Artifact 컬렉션을 업데이트할 수도 있습니다.

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
자세한 내용은 [Artifacts Collection]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}) 참조를 참조하세요.
  {{% /tab %}}
{{% /tabpane %}}
