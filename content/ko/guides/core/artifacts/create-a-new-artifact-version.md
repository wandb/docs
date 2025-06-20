---
title: Create an artifact version
description: 단일 run 또는 분산된 process 에서 새로운 아티팩트 버전을 만드세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

단일 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에서 또는 분산된 run과 협업하여 새로운 아티팩트 버전을 만드세요. 선택적으로 [증분 아티팩트]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ko" >}})라고 알려진 이전 버전에서 새로운 아티팩트 버전을 만들 수 있습니다.

{{% alert %}}
원래 아티팩트의 크기가 상당히 큰 경우 아티팩트에서 파일의 서브셋에 변경 사항을 적용해야 할 때 증분 아티팩트를 만드는 것이 좋습니다.
{{% /alert %}}

## 처음부터 새로운 아티팩트 버전 만들기
새로운 아티팩트 버전을 만드는 방법에는 단일 run에서 만드는 방법과 분산된 run에서 만드는 두 가지 방법이 있습니다. 이들은 다음과 같이 정의됩니다.

* **단일 run**: 단일 run은 새로운 버전에 대한 모든 데이터를 제공합니다. 이것은 가장 일반적인 경우이며, run이 필요한 데이터를 완전히 재 생성할 때 가장 적합합니다. 예를 들어, 분석을 위해 테이블에 저장된 모델 또는 모델 예측값을 출력합니다.
* **분산된 run**: run 집합이 공동으로 새로운 버전에 대한 모든 데이터를 제공합니다. 이것은 여러 run이 데이터를 생성하는 분산 작업에 가장 적합하며, 종종 병렬로 수행됩니다. 예를 들어, 분산 방식으로 모델을 평가하고 예측값을 출력합니다.

W&B는 프로젝트에 존재하지 않는 이름을 `wandb.Artifact` API에 전달하면 새로운 아티팩트를 생성하고 `v0` 에일리어스를 할당합니다. 동일한 아티팩트에 다시 로그할 때 W&B는 콘텐츠의 체크섬을 계산합니다. 아티팩트가 변경되면 W&B는 새 버전 `v1`을 저장합니다.

W&B는 프로젝트에 있는 기존 아티팩트와 일치하는 이름과 아티팩트 유형을 `wandb.Artifact` API에 전달하면 기존 아티팩트를 검색합니다. 검색된 아티팩트의 버전은 1보다 큽니다.

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### 단일 run
아티팩트의 모든 파일을 생성하는 단일 run으로 Artifact의 새 버전을 기록합니다. 이 경우는 단일 run이 아티팩트의 모든 파일을 생성할 때 발생합니다.

유스 케이스에 따라 아래 탭 중 하나를 선택하여 run 내부 또는 외부에서 새로운 아티팩트 버전을 만드세요.

{{< tabpane text=true >}}
  {{% tab header="Run 내부" %}}
W&B run 내에서 아티팩트 버전을 만듭니다.

1. `wandb.init`으로 run을 만듭니다.
2. `wandb.Artifact`로 새로운 아티팩트를 만들거나 기존 아티팩트를 검색합니다.
3. `.add_file`로 아티팩트에 파일을 추가합니다.
4. `.log_artifact`로 아티팩트를 run에 기록합니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
    # 아티팩트에 파일 및 에셋을 추가합니다.
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```
  {{% /tab %}}
  {{% tab header="Run 외부" %}}
W&B run 외부에서 아티팩트 버전을 만듭니다.

1. `wanb.Artifact`로 새로운 아티팩트를 만들거나 기존 아티팩트를 검색합니다.
2. `.add_file`로 아티팩트에 파일을 추가합니다.
3. `.save`로 아티팩트를 저장합니다.

```python
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
# 아티팩트에 파일 및 에셋을 추가합니다.
artifact.add_file("image1.png")
artifact.save()
```
  {{% /tab %}}
{{< /tabpane  >}}

### 분산된 run

커밋하기 전에 run 컬렉션이 버전에서 공동 작업할 수 있도록 합니다. 이는 하나의 run이 새 버전에 대한 모든 데이터를 제공하는 위에서 설명한 단일 run 모드와 대조됩니다.

{{% alert %}}
1. 컬렉션의 각 run은 동일한 버전에 대해 공동 작업하기 위해 동일한 고유 ID ( `distributed_id`라고 함)를 인식해야 합니다. 기본적으로 W&B는 있는 경우 `wandb.init(group=GROUP)`에 의해 설정된 run의 `group`을 `distributed_id`로 사용합니다.
2. 해당 상태를 영구적으로 잠그는 버전을 "커밋"하는 최종 run이 있어야 합니다.
3. 협업 아티팩트에 추가하려면 `upsert_artifact`를 사용하고 커밋을 완료하려면 `finish_artifact`를 사용하세요.
{{% /alert %}}

다음 예제를 고려하십시오. 서로 다른 run (아래에 **Run 1**, **Run 2** 및 **Run 3**으로 표시됨)은 `upsert_artifact`를 사용하여 동일한 아티팩트에 다른 이미지 파일을 추가합니다.

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
    # 아티팩트에 파일 및 에셋을 추가합니다.
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
    # 아티팩트에 파일 및 에셋을 추가합니다.
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1과 Run 2가 완료된 후 실행해야 합니다. `finish_artifact`를 호출하는 Run은 아티팩트에 파일을 포함할 수 있지만 필요하지는 않습니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # 아티팩트에 파일 및 에셋을 추가합니다.
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 기존 버전에서 새로운 아티팩트 버전 만들기

변경되지 않은 파일을 다시 인덱싱할 필요 없이 이전 아티팩트 버전에서 파일의 서브셋을 추가, 수정 또는 제거합니다. 이전 아티팩트 버전에서 파일의 서브셋을 추가, 수정 또는 제거하면 *증분 아티팩트*라고 하는 새로운 아티팩트 버전이 생성됩니다.

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

다음은 발생할 수 있는 각 유형의 증분 변경에 대한 몇 가지 시나리오입니다.

- add: 새로운 배치를 수집한 후 데이터셋에 새로운 파일 서브셋을 주기적으로 추가합니다.
- remove: 여러 중복 파일을 발견했고 아티팩트에서 제거하고 싶습니다.
- update: 파일 서브셋에 대한 주석을 수정했고 이전 파일을 올바른 파일로 바꾸고 싶습니다.

처음부터 아티팩트를 만들어 증분 아티팩트와 동일한 기능을 수행할 수 있습니다. 그러나 처음부터 아티팩트를 만들면 로컬 디스크에 아티팩트의 모든 콘텐츠가 있어야 합니다. 증분 변경을 수행할 때 이전 아티팩트 버전의 파일을 변경하지 않고도 단일 파일을 추가, 제거 또는 수정할 수 있습니다.

{{% alert %}}
단일 run 또는 run 집합 (분산 모드) 내에서 증분 아티팩트를 만들 수 있습니다.
{{% /alert %}}

아래 절차에 따라 아티팩트를 증분 방식으로 변경합니다.

1. 증분 변경을 수행할 아티팩트 버전을 가져옵니다.

{{< tabpane text=true >}}
{{% tab header="Run 내부" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

{{% /tab %}}
{{% tab header="Run 외부" %}}

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

{{% /tab %}}
{{< /tabpane >}}

2. 다음으로 초안을 만듭니다.

```python
draft_artifact = saved_artifact.new_draft()
```

3. 다음 버전에서 보고 싶은 증분 변경 사항을 수행합니다. 기존 항목을 추가, 제거 또는 수정할 수 있습니다.

각 변경을 수행하는 방법에 대한 예제는 다음 탭 중 하나를 선택하십시오.

{{< tabpane text=true >}}
  {{% tab header="추가" %}}
`add_file` 메소드를 사용하여 기존 아티팩트 버전에 파일을 추가합니다.

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` 메소드를 사용하여 디렉토리를 추가하여 여러 파일을 추가할 수도 있습니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="제거" %}}
`remove` 메소드를 사용하여 기존 아티팩트 버전에서 파일을 제거합니다.

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
디렉토리 경로를 전달하여 `remove` 메소드로 여러 파일을 제거할 수도 있습니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="수정" %}}
초안에서 이전 콘텐츠를 제거하고 새 콘텐츠를 다시 추가하여 콘텐츠를 수정하거나 바꿉니다.

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```
  {{% /tab %}}
{{< /tabpane >}}

4. 마지막으로 변경 사항을 기록하거나 저장합니다. 다음 탭에서는 W&B run 내부 및 외부에서 변경 사항을 저장하는 방법을 보여줍니다. 유스 케이스에 적합한 탭을 선택하세요.

{{< tabpane text=true >}}
  {{% tab header="Run 내부" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="Run 외부" %}}
```python
draft_artifact.save()
```
  {{% /tab %}}
{{< /tabpane >}}

모두 합치면 위의 코드 예제는 다음과 같습니다.

{{< tabpane text=true >}}
  {{% tab header="Run 내부" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # 아티팩트를 가져와서 run에 입력합니다.
    draft_artifact = saved_artifact.new_draft()  # 초안 버전을 만듭니다.

    # 초안 버전에서 파일의 서브셋을 수정합니다.
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 변경 사항을 기록하여 새 버전을 만들고 run에 대한 출력으로 표시합니다.
```
  {{% /tab %}}
  {{% tab header="Run 외부" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # 아티팩트를 로드합니다.
draft_artifact = saved_artifact.new_draft()  # 초안 버전을 만듭니다.

# 초안 버전에서 파일의 서브셋을 수정합니다.
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 초안에 변경 사항을 커밋합니다.
```
  {{% /tab %}}
{{< /tabpane >}}
