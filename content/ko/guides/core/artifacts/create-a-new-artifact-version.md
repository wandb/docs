---
title: Create an artifact version
description: 단일 run 또는 분산된 process 에서 새로운 아티팩트 버전을 만드세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

단일 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})으로 또는 분산된 run으로 협업하여 새로운 artifact 버전을 생성하세요. 선택적으로 [증분 artifact]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ko" >}})이라고 하는 이전 버전의 artifact에서 새 artifact 버전을 생성할 수 있습니다.

{{% alert %}}
원본 artifact의 크기가 상당히 큰 경우 artifact의 파일 서브셋에 변경 사항을 적용해야 하는 경우 증분 artifact를 만드는 것이 좋습니다.
{{% /alert %}}

## 처음부터 새로운 artifact 버전 만들기
새로운 artifact 버전을 만드는 방법에는 단일 run에서 만드는 방법과 분산된 run에서 만드는 두 가지 방법이 있습니다. 이 방법들은 다음과 같이 정의됩니다.

* **단일 run**: 단일 run은 새로운 버전에 대한 모든 데이터를 제공합니다. 이는 가장 일반적인 경우이며, run이 필요한 데이터를 완전히 다시 생성할 때 가장 적합합니다. 예를 들어 분석을 위해 저장된 모델 또는 모델 예측값을 테이블로 출력하는 경우입니다.
* **분산된 run**: run 세트가 집합적으로 새로운 버전에 대한 모든 데이터를 제공합니다. 이는 분산된 방식으로 모델을 평가하고 예측값을 출력하는 등 여러 run이 데이터를 생성하는 분산 작업에 가장 적합합니다.

프로젝트에 존재하지 않는 이름을 `wandb.Artifact` API에 전달하면 W&B가 새로운 artifact를 생성하고 `v0` 에일리어스를 할당합니다. 동일한 artifact에 다시 로그하면 W&B가 콘텐츠의 체크섬을 계산합니다. artifact가 변경된 경우 W&B는 새로운 버전 `v1`을 저장합니다.

프로젝트에 있는 기존 artifact와 일치하는 이름과 artifact 유형을 `wandb.Artifact` API에 전달하면 W&B는 기존 artifact를 검색합니다. 검색된 artifact는 1보다 큰 버전을 갖습니다.

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### 단일 run
artifact의 모든 파일을 생성하는 단일 run으로 새로운 버전의 Artifact를 기록합니다. 이 경우는 단일 run이 artifact의 모든 파일을 생성할 때 발생합니다.

유스 케이스에 따라 아래 탭 중 하나를 선택하여 run 내부 또는 외부에서 새로운 artifact 버전을 만드세요.

{{< tabpane text=true >}}
  {{% tab header="Run 내부" %}}
W&B run 내에서 artifact 버전을 만드세요.

1. `wandb.init`으로 run을 만듭니다. (1번 라인)
2. `wandb.Artifact`로 새로운 artifact를 만들거나 기존 artifact를 검색합니다. (2번 라인)
3. `.add_file`로 artifact에 파일을 추가합니다. (9번 라인)
4. `.log_artifact`로 artifact를 run에 기록합니다. (10번 라인)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
    # artifact에 파일 및 자산 추가
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```
  {{% /tab %}}
  {{% tab header="Run 외부" %}}
W&B run 외부에서 artifact 버전을 만드세요.

1. `wanb.Artifact`로 새로운 artifact를 만들거나 기존 artifact를 검색합니다. (1번 라인)
2. `.add_file`로 artifact에 파일을 추가합니다. (4번 라인)
3. `.save`로 artifact를 저장합니다. (5번 라인)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
# artifact에 파일 및 자산 추가
artifact.add_file("image1.png")
artifact.save()
```
  {{% /tab %}}
{{< /tabpane  >}}

### 분산된 run

커밋하기 전에 run 컬렉션이 버전에서 협업하도록 허용합니다. 이는 하나의 run이 새로운 버전에 대한 모든 데이터를 제공하는 위에서 설명한 단일 run 모드와 대조됩니다.

{{% alert %}}
1. 컬렉션의 각 run은 동일한 버전에서 협업하려면 동일한 고유 ID(`distributed_id`라고 함)를 인식해야 합니다. 기본적으로 W&B는 있는 경우 `wandb.init(group=GROUP)`에 의해 설정된 run의 `group`을 `distributed_id`로 사용합니다.
2. 버전을 "커밋"하여 해당 상태를 영구적으로 잠그는 최종 run이 있어야 합니다.
3. 협업 artifact에 추가하려면 `upsert_artifact`를 사용하고 커밋을 완료하려면 `finish_artifact`를 사용합니다.
{{% /alert %}}

다음 예시를 고려해 보세요. 서로 다른 run(**Run 1**, **Run 2**, **Run 3**으로 레이블됨)은 `upsert_artifact`를 사용하여 동일한 artifact에 다른 이미지 파일을 추가합니다.

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
    # artifact에 파일 및 자산 추가
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`를 사용하여
    # artifact에 파일 및 자산 추가
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1과 Run 2가 완료된 후 실행해야 합니다. `finish_artifact`를 호출하는 Run은 artifact에 파일을 포함할 수 있지만 필수는 아닙니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # artifact에 파일 및 자산 추가
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 기존 버전에서 새로운 artifact 버전 만들기

변경되지 않은 파일을 다시 인덱싱할 필요 없이 이전 artifact 버전에서 파일 서브셋을 추가, 수정 또는 제거합니다. 이전 artifact 버전에서 파일 서브셋을 추가, 수정 또는 제거하면 *증분 artifact*라고 하는 새로운 artifact 버전이 생성됩니다.

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

다음은 발생할 수 있는 각 유형의 증분 변경에 대한 몇 가지 시나리오입니다.

- add: 새로운 배치를 수집한 후 데이터셋에 새로운 파일 서브셋을 주기적으로 추가합니다.
- remove: 여러 개의 중복 파일을 발견하여 artifact에서 제거하려고 합니다.
- update: 파일 서브셋에 대한 주석을 수정하고 이전 파일을 올바른 파일로 바꾸려고 합니다.

처음부터 artifact를 만들어 증분 artifact와 동일한 기능을 수행할 수 있습니다. 그러나 처음부터 artifact를 만들 때는 artifact의 모든 콘텐츠가 로컬 디스크에 있어야 합니다. 증분 변경을 수행할 때는 이전 artifact 버전의 파일을 변경하지 않고도 단일 파일을 추가, 제거 또는 수정할 수 있습니다.

{{% alert %}}
단일 run 또는 run 세트(분산 모드) 내에서 증분 artifact를 만들 수 있습니다.
{{% /alert %}}

다음 절차에 따라 artifact를 증분 방식으로 변경합니다.

1. 증분 변경을 수행할 artifact 버전을 가져옵니다.

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

2. 다음을 사용하여 초안을 만듭니다.

```python
draft_artifact = saved_artifact.new_draft()
```

3. 다음 버전에서 보려는 증분 변경을 수행합니다. 기존 항목을 추가, 제거 또는 수정할 수 있습니다.

이러한 각 변경을 수행하는 방법에 대한 예시는 다음 탭 중 하나를 선택하세요.

{{< tabpane text=true >}}
  {{% tab header="추가" %}}
`add_file` 메소드를 사용하여 기존 artifact 버전에 파일을 추가합니다.

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` 메소드를 사용하여 디렉토리를 추가하여 여러 파일을 추가할 수도 있습니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="제거" %}}
`remove` 메소드를 사용하여 기존 artifact 버전에서 파일을 제거합니다.

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
디렉토리 경로를 전달하여 `remove` 메소드를 사용하여 여러 파일을 제거할 수도 있습니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="수정" %}}
초안에서 이전 콘텐츠를 제거하고 새로운 콘텐츠를 다시 추가하여 콘텐츠를 수정하거나 바꿉니다.

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```
  {{% /tab %}}
{{< /tabpane >}}

4. 마지막으로 변경 사항을 기록하거나 저장합니다. 다음 탭은 W&B run 내부 및 외부에서 변경 사항을 저장하는 방법을 보여줍니다. 유스 케이스에 적합한 탭을 선택하세요.

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

모든 것을 종합하면 위의 코드 예시는 다음과 같습니다.

{{< tabpane text=true >}}
  {{% tab header="Run 내부" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # artifact를 가져와 run에 입력합니다.
    draft_artifact = saved_artifact.new_draft()  # 초안 버전을 만듭니다.

    # 초안 버전에서 파일 서브셋을 수정합니다.
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 변경 사항을 기록하여 새로운 버전을 만들고 run에 대한 출력으로 표시합니다.
```
  {{% /tab %}}
  {{% tab header="Run 외부" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # artifact를 로드합니다.
draft_artifact = saved_artifact.new_draft()  # 초안 버전을 만듭니다.

# 초안 버전에서 파일 서브셋을 수정합니다.
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 초안에 변경 사항을 커밋합니다.
```
  {{% /tab %}}
{{< /tabpane >}}
