---
description: Create a new artifact version from a single run or from a distributed
  process.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 새 아티팩트 버전 생성하기

<head>
    <title>단일 및 멀티프로세스 실행으로부터 새 아티팩트 버전 생성하기.</title>
</head>

단일 [실행](../runs/intro.md) 또는 분산 실행으로 협업하여 새로운 아티팩트 버전을 생성하세요. 기존 버전에서 새 아티팩트 버전을 생성하는 것으로 알려진 [증분 아티팩트](#create-a-new-artifact-version-from-an-existing-version)에서 선택적으로 새 아티팩트 버전을 생성할 수도 있습니다.

:::tip
원본 아티팩트의 크기가 상당히 큰 경우, 아티팩트의 파일 서브세트에 변경을 적용해야 할 때 증분 아티팩트를 생성하는 것이 좋습니다.
:::

## 처음부터 새 아티팩트 버전 생성하기
새 아티팩트 버전을 생성하는 두 가지 방법이 있습니다: 단일 실행에서 및 분산 실행에서입니다. 다음과 같이 정의됩니다:

* **단일 실행**: 단일 실행은 새 버전에 필요한 모든 데이터를 제공합니다. 이것은 가장 일반적인 경우로, 실행이 필요한 데이터를 완전히 재생성할 때 가장 적합합니다. 예를 들어: 저장된 모델 또는 분석을 위한 테이블에서 모델 예측값을 출력하는 경우입니다.
* **분산 실행**: 여러 실행이 새 버전에 필요한 모든 데이터를 공동으로 제공합니다. 이는 병렬로 데이터를 생성하는 여러 실행이 있는 분산 작업에 가장 적합합니다. 예를 들어: 분산 방식으로 모델을 평가하고 예측값을 출력하는 경우입니다.

W&B는 프로젝트에 존재하지 않는 이름을 `wandb.Artifact` API에 전달하면 새 아티팩트를 생성하고 이에 `v0` 별칭을 할당합니다. 동일한 아티팩트에 다시 로그를 기록하면 W&B는 내용을 체크섬하고, 아티팩트가 변경된 경우 새 버전 `v1`을 저장합니다.

W&B는 프로젝트에 있는 기존 아티팩트와 일치하는 이름과 아티팩트 유형을 `wandb.Artifact` API에 전달하면 기존 아티팩트를 검색합니다. 검색된 아티팩트는 1보다 큰 버전을 가질 것입니다.

![](/images/artifacts/single_distributed_artifacts.png)

### 단일 실행
아티팩트의 모든 파일을 생성하는 단일 실행으로 새 버전의 아티팩트를 로그합니다. 이 경우는 단일 실행이 아티팩트의 모든 파일을 생성할 때 발생합니다.

사용 사례에 따라 아래 탭 중 하나를 선택하여 실행 내부 또는 외부에서 새 아티팩트 버전을 생성하세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: '실행 내부', value: 'inside'},
    {label: '실행 외부', value: 'outside'},
  ]}>
  <TabItem value="inside">

W&B 실행 내에서 아티팩트 버전 생성하기:

1. `wandb.init`으로 실행을 생성합니다. (1행)
2. `wandb.Artifact`으로 새 아티팩트를 생성하거나 기존 아티팩트를 검색합니다. (2행)
3. `.add_file`로 아티팩트에 파일 추가하기. (9행)
4. `.log_artifact`로 실행에 아티팩트 로그하기. (10행)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, 및 `.add_reference`를 사용하여
    # 아티팩트에 파일과 자산 추가하기
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

  </TabItem>
  <TabItem value="outside">

W&B 실행 외부에서 아티팩트 버전 생성하기:

1. `wanb.Artifact`로 새 아티팩트를 생성하거나 기존 아티팩트를 검색합니다. (1행)
2. `.add_file`로 아티팩트에 파일 추가하기. (4행)
3. `.save`로 아티팩트 저장하기. (5행)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, 및 `.add_reference`를 사용하여
# 아티팩트에 파일과 자산 추가하기
artifact.add_file("image1.png")
artifact.save()
```  
  </TabItem>
</Tabs>

### 분산 실행

버전을 커밋하기 전에 실행 모음이 버전 작업에 협력하도록 허용합니다. 이는 위에서 설명한 단일 실행 모드와 대조적으로 하나의 실행이 새 버전에 대한 모든 데이터를 제공하는 경우입니다.

:::info
1. 컬렉션의 각 실행은 동일한 버전에서 협력하기 위해 동일한 고유 ID(별칭 `distributed_id`로 불림)를 알고 있어야 합니다. 기본적으로, 존재한다면, W&B는 `wandb.init(group=GROUP)`으로 설정된 실행의 `group`을 `distributed_id`로 사용합니다.
2. 버전을 "커밋"하는 최종 실행이 있어야 하며, 그 상태를 영구적으로 잠급니다.
3. 협업 아티팩트에 추가하기 위해 `upsert_artifact`를 사용하고 커밋을 최종화하기 위해 `finish_artifact`를 사용합니다.
:::

다음 예를 고려하세요. 다른 실행들(**실행 1**, **실행 2**, 그리고 **실행 3**로 표시됨)이 `upsert_artifact`를 사용하여 동일한 아티팩트에 다른 이미지 파일을 추가합니다.

#### 실행 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, 및 `.add_reference`를 사용하여
    # 아티팩트에 파일과 자산 추가하기
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### 실행 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, 및 `.add_reference`를 사용하여
    # 아티팩트에 파일과 자산 추가하기
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### 실행 3

실행 1과 실행 2가 완료된 후 실행해야 합니다. `finish_artifact`를 호출하는 실행은 아티팩트에 파일을 포함할 수 있지만 필수는 아닙니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, 및 `.add_reference`를 사용하여
    # 아티팩트에 파일과 자산 추가하기
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 기존 버전에서 새 아티팩트 버전 생성하기

변경되지 않은 파일을 다시 색인화할 필요 없이 이전 아티팩트 버전에서 파일 서브세트를 추가, 수정 또는 제거합니다. 이전 아티팩트 버전에서 파일 서브세트를 추가, 수정 또는 제거하면 *증분 아티팩트*로 알려진 새 아티팩트 버전이 생성됩니다.

![](/images/artifacts/incremental_artifacts.png)

증분 변경을 위해 마주칠 수 있는 각 유형의 시나리오는 다음과 같습니다:

- 추가: 새 배치를 수집한 후 데이터세트에 새 서브세트 파일을 주기적으로 추가합니다.
- 제거: 여러 중복 파일을 발견하여 아티팩트에서 제거하고자 합니다.
- 업데이트: 파일 서브세트의 주석을 수정하여 오래된 파일을 올바른 파일로 교체하고자 합니다.

증분 아티팩트와 동일한 기능을 수행하기 위해 처음부터 아티팩트를 생성할 수 있습니다. 그러나 처음부터 아티팩트를 생성할 때는 아티팩트의 모든 내용을 로컬 디스크에 있어야 합니다. 증분 변경을 할 때는, 이전 아티팩트 버전의 파일을 변경하지 않고 단일 파일을 추가, 제거 또는 수정할 수 있습니다.

:::info
단일 실행이나 실행 모음(분산 모드) 내에서 증분 아티팩트를 생성할 수 있습니다.
:::

증분 변경을 수행하려는 아티팩트 버전을 얻으려면 아래 절차를 따르세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: '실행 내부', value: 'inside'},
    {label: '실행 외부', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

  </TabItem>
</Tabs>

2. 초안을 생성하세요:

```python
draft_artifact = saved_artifact.new_draft()
```

3. 다음 버전에서 보고 싶은 모든 증분 변경을 수행하세요. 기존 항목을 추가, 제거하거나 수정할 수 있습니다.

이러한 변경을 수행하는 방법에 대한 예제는 아래 탭 중 하나를 선택하세요:

<Tabs
  defaultValue="add"
  values={[
    {label: '추가', value: 'add'},
    {label: '제거', value: 'remove'},
    {label: '수정', value: 'modify'},
  ]}>
  <TabItem value="add">

기존 아티팩트 버전에 파일을 추가하려면 `add_file` 메서드를 사용하세요:

```python
draft_artifact.add_file("file_to_add.txt")
```

:::note
`add_dir` 메서드를 사용하여 디렉터리를 추가함으로써 여러 파일을 추가할 수도 있습니다.
:::

  </TabItem>
  <TabItem value="remove">

기존 아티팩트 버전에서 파일을 제거하려면 `remove` 메서드를 사용하세요:

```python
draft_artifact.remove("file_to_remove.txt")
```

:::note
디렉터리 경로를 전달하여 `remove` 메서드로 여러 파일을 제거할 수도 있습니다.
:::

  </TabItem>
  <TabItem value="modify">

이전 내용을 초안에서 제거하고 새 내용을 다시 추가하여 내용을 수정하거나 대체하세요:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>

4. 마지막으로, 변경 사항을 로그하거나 저장하세요. 다음 탭은 W&B 실행 내부 및 외부에서 변경 사항을 저장하는 방법을 보여줍니다. 사용 사례에 적합한 탭을 선택하세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: '실행 내부', value: 'inside'},
    {label: '실행 외부', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
run.log_artifact(draft_artifact)
```

  </TabItem>
  <TabItem value="outside">


```python
draft_artifact.save()
```

  </TabItem>
</Tabs>

모두 함께, 위의 코드 예제는 다음과 같습니다:

<Tabs
  defaultValue="inside"
  values={[
    {label: '실행 내부', value: 'inside'},
    {label: '실행 외부', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # 아티팩트를 가져와서 실행에 입력합니다
    draft_artifact = saved_artifact.new_draft()  # 초안 버전 생성

    # 초안 버전에서 파일 서브세트 수정
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 변경 사항을 로그하여 새 버전을 생성하고 실행의 출력으로 표시합니다
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # 아티팩트를 로드합니다
draft_artifact = saved_artifact.new_draft()  # 초안 버전 생성

# 초안 버전에서 파일 서브세트 수정
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 초안에 대한 변경 사항 커밋
```

  </TabItem>
</Tabs>