---
title: 아티팩트 버전 생성
description: 단일 run 또는 분산 프로세스에서 새로운 아티팩트 버전을 생성합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

단일 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 또는 분산된 runs와 협업하여 새로운 artifact 버전을 생성할 수 있습니다. 이전 버전에서 파생된 새로운 artifact 버전(이를 [incremental artifact]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ko" >}})라고도 함)을 만들 수도 있습니다.

{{% alert %}}
원본 artifact의 크기가 상당히 크고, artifact 내 일부 파일만 변경해야 할 때에는 incremental artifact를 생성하는 것이 좋습니다.
{{% /alert %}}

## 새로운 artifact 버전 처음부터 생성하기
새로운 artifact 버전을 생성하는 방법에는 크게 두 가지가 있습니다: 단일 run에서 생성하거나 분산된 runs에서 협업하여 생성하는 방법입니다. 각각은 다음과 같이 정의됩니다:

* **Single run**: 단일 run이 새로운 버전에 필요한 모든 데이터를 제공합니다. 가장 일반적인 케이스로, run이 필요한 모든 데이터를 완전히 다시 생성할 수 있을 때 적합합니다. 예를 들어, 저장된 모델을 출력하거나 분석을 위한 테이블에 모델 예측값을 저장하는 경우가 이에 해당합니다.
* **Distributed runs**: 여러 run이 함께 새로운 버전에 필요한 모든 데이터를 제공합니다. 여러 run이 병렬로 데이터를 생성하는 분산 작업에 적합합니다. 예를 들어, 모델을 분산 방식으로 평가하여 예측값을 출력하는 경우가 이에 해당합니다.

W&B는 프로젝트 내에 존재하지 않는 이름을 `wandb.Artifact` API에 전달하면 새로운 artifact를 생성하고 `v0` 에일리어스를 할당합니다. 동일한 artifact에 로그를 다시 남길 경우 W&B는 내용의 체크섬을 계산해 변경 사항이 있으면 새로운 버전 `v1`을 저장합니다.

이미 프로젝트에 존재하는 artifact 이름과 타입을 `wandb.Artifact` API에 전달하면, W&B는 기존 artifact를 가져옵니다. 이때 버전은 1보다 큰 값이 됩니다.

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="Artifact workflow comparison" >}}

### Single run
단일 run으로 artifact의 모든 파일을 생성하여 새로운 Artifact 버전을 기록합니다. 즉, 한 번의 run이 artifact에 포함될 모든 파일을 생산하는 경우입니다.

유스 케이스에 따라 아래 탭 중에서 run 내부 또는 외부에서 새로운 artifact 버전을 생성하는 방법을 선택하세요:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
W&B run 내부에서 artifact 버전 생성하는 방법:

1. `wandb.init`을 사용하여 run을 만듭니다.
2. `wandb.Artifact`로 새로운 artifact를 생성하거나 기존 artifact를 불러옵니다.
3. `.add_file`로 파일을 artifact에 추가합니다.
4. `.log_artifact`로 artifact를 run에 로그합니다.

```python 
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, `.add_reference`로 파일 및 asset을 artifact에 추가
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
W&B run 외부에서 artifact 버전 생성하는 방법:

1. `wandb.Artifact`로 새로운 artifact를 생성하거나 기존 artifact를 불러옵니다.
2. `.add_file`로 파일을 artifact에 추가합니다.
3. `.save`로 artifact를 저장합니다.

```python 
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, `.add_reference`로 파일 및 asset을 artifact에 추가
artifact.add_file("image1.png")
artifact.save()
```
  {{% /tab %}}
{{< /tabpane  >}}



### Distributed runs

여러 runs가 협업하여 하나의 artifact 버전을 만들 수 있습니다. 위의 single run 모드와는 달리, 여러 run이 새로운 버전에 필요한 데이터를 분담합니다.

{{% alert %}}
1. 컬렉션에 속한 각 run이 동일한 고유 ID(즉, `distributed_id`)를 알아야 함께 버전을 생성할 수 있습니다. 별도로 지정하지 않으면 W&B는 기본적으로 run의 `group`(즉, `wandb.init(group=GROUP)` 사용 시 지정된 값)을 `distributed_id`로 사용합니다.
2. 버전을 최종적으로 "커밋"해서 상태를 고정시키는 run이 반드시 필요합니다.
3. `upsert_artifact`로 협업 artifact에 내용을 추가하고, `finish_artifact`로 최종 커밋을 완료하세요.
{{% /alert %}}

다음 예시를 참고하세요. 아래의 **Run 1**, **Run 2**, **Run 3**에서 각각 다른 이미지 파일을 `upsert_artifact`로 동일한 artifact에 추가합니다.

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, `.add_reference`로 파일 및 asset을 artifact에 추가
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, `.add_reference`로 파일 및 asset을 artifact에 추가
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

반드시 Run 1과 Run 2가 끝난 후 실행되어야 합니다. `finish_artifact`를 호출하는 run에서는 파일을 artifact에 추가해도 되고, 추가하지 않아도 됩니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, `.add_reference`로 파일 및 asset을 artifact에 추가
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```



## 기존 버전에서 새로운 artifact 버전 만들기

기존 artifact 버전에서 일부 파일을 추가, 수정, 삭제해 변경된 파일만 관리할 수 있습니다. 이전 artifact 버전의 일부 파일을 추가, 수정, 삭제하면 새로운 artifact 버전이 생성되며, 이를 *incremental artifact*라고 부릅니다.

{{< img src="/images/artifacts/incremental_artifacts.png" alt="Incremental artifact versioning" >}}

각 유형별 incremental 변경 예시는 아래와 같습니다:

- add: 새로운 데이터를 수집해 데이터셋에 주기적으로 파일 서브셋을 추가할 때
- remove: 중복된 파일을 발견하고 artifact에서 제거하고 싶을 때
- update: 일부 파일의 annotation을 수정해 기존 파일을 새 파일로 교체하고 싶을 때

새로운 artifact를 완전히 처음부터 생성해 incremental artifact와 동일한 효과를 얻을 수도 있습니다. 그러나 이렇게 하면 모든 artifact 내용을 로컬에 보관하고 있어야 합니다. 반면 incremental 변경은 이전 artifact 버전의 파일을 바꾸지 않고 한두 개의 파일만 추가, 삭제, 수정하면 됩니다.

{{% alert %}}
incremental artifact는 single run 또는 여러 run(분산 모드)에서 생성할 수 있습니다.
{{% /alert %}}

아래 절차에 따라 artifact에 incremental 변경을 적용하세요:

1. incremental 변경을 적용할 artifact 버전을 가져옵니다.

{{< tabpane text=true >}}
{{% tab header="Inside a run" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

{{% /tab %}}
{{% tab header="Outside of a run" %}}

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

{{% /tab %}}
{{< /tabpane >}}



2. 아래와 같이 draft를 만듭니다:

```python
draft_artifact = saved_artifact.new_draft()
```

3. 다음 버전에 반영하고 싶은 incremental 변경을 수행합니다. 파일을 추가, 삭제, 수정할 수 있습니다.

각 변경 예시는 아래 탭에서 확인할 수 있습니다:

{{< tabpane text=true >}}
  {{% tab header="Add" %}}
`add_file` 메소드로 기존 artifact 버전에 파일을 추가하기:

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
여러 개의 파일을 추가하고 싶으면, `add_dir` 메소드로 디렉토리를 추가할 수 있습니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="Remove" %}}
`remove` 메소드로 기존 artifact 버전에서 파일을 삭제하기:

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
디렉토리 경로를 넘기면 `remove` 메소드로 여러 파일을 한 번에 삭제할 수도 있습니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="Modify" %}}
draft에서 기존 파일을 삭제하고 새 파일을 추가해 내용을 변경하거나 교체하기:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```
  {{% /tab %}}
{{< /tabpane >}}



4. 마지막으로 변경 내용을 로그하거나 저장하세요. 아래 탭에서 W&B run 내부와 외부 각각의 저장/로그 방법을 확인할 수 있습니다. 유스 케이스에 맞는 탭을 선택하세요:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
draft_artifact.save()
```
  {{% /tab %}}
{{< /tabpane >}}


지금까지 살펴본 내용을 종합하면 코드 예시는 다음과 같습니다:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # artifact를 fetch하여 run에 입력
    draft_artifact = saved_artifact.new_draft()  # draft 버전 생성

    # draft 버전에서 일부 파일을 수정
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        draft_artifact
    )  # 변경 사항을 새로운 버전으로 기록하고 run의 output으로 표시
```
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # artifact 불러오기
draft_artifact = saved_artifact.new_draft()  # draft 버전 생성

# draft 버전에서 일부 파일을 수정
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # draft에 변경 사항 저장
```
  {{% /tab %}}
{{< /tabpane >}}