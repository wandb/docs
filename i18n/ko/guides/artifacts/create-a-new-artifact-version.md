---
description: Create a new artifact version from a single run or from a distributed
  process.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 새로운 아티팩트 버전 생성하기

<head>
    <title>단일 및 멀티프로세스 Runs에서 새로운 아티팩트 버전 생성하기.</title>
</head>

단일 [run](../runs/intro.md) 또는 분산된 runs와 협업하여 새로운 아티팩트 버전을 생성하세요. 이전 버전에서 알려진 [incremental artifact](#create-a-new-artifact-version-from-an-existing-version)에서 새로운 아티팩트 버전을 선택적으로 생성할 수 있습니다.

:::tip
원본 아티팩트의 크기가 상당히 큰 경우, 아티팩트의 파일 서브셋에 변경을 적용해야 할 때 incremental artifact를 생성하는 것이 좋습니다.
:::

## 처음부터 새로운 아티팩트 버전 생성하기
새로운 아티팩트 버전을 생성하는 데에는 두 가지 방법이 있습니다: 단일 run에서 생성하거나 분산된 runs에서 생성합니다. 다음과 같이 정의됩니다:


* **Single run**: Single run은 새 버전에 필요한 모든 데이터를 제공합니다. 이는 가장 일반적인 경우이며, run이 필요한 데이터를 완전히 재생성할 때 가장 적합합니다. 예를 들어: 분석을 위한 테이블에 저장된 모델이나 모델 예측값을 출력하는 경우.
* **Distributed runs**: 일련의 runs가 새 버전에 필요한 모든 데이터를 공동으로 제공합니다. 이는 여러 runs가 데이터를 생성하는 분산된 작업에 가장 적합하며, 종종 병렬로 수행됩니다. 예를 들어: 분산된 방식으로 모델을 평가하고 예측값을 출력하는 경우.


프로젝트에서 존재하지 않는 이름을 `wandb.Artifact` API에 전달하면 W&B는 새로운 아티팩트를 생성하고 `v0` 에일리어스를 할당합니다. 동일한 아티팩트에 다시 로그할 때 W&B는 내용을 체크섬하고, 아티팩트가 변경된 경우 새 버전 `v1`을 저장합니다.  

프로젝트의 기존 아티팩트와 일치하는 이름과 아티팩트 타입을 `wandb.Artifact` API에 전달하면 W&B는 기존 아티팩트를 검색합니다. 검색된 아티팩트는 버전이 1보다 큽니다. 

![](/images/artifacts/single_distributed_artifacts.png)

### Single run
Single run으로 아티팩트의 모든 파일을 생성하여 새로운 버전의 아티팩트를 로그합니다. 이 경우는 단일 run이 아티팩트의 모든 파일을 생성할 때 발생합니다. 

아래 탭 중 하나를 선택하여 run 내부 또는 외부에서 새로운 아티팩트 버전을 생성하는 방법을 선택하세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Run 내부', value: 'inside'},
    {label: 'Run 외부', value: 'outside'},
  ]}>
  <TabItem value="inside">

W&B run 내에서 아티팩트 버전 생성:

1. `wandb.init`으로 run을 생성합니다. (1번째 줄)
2. `wandb.Artifact`로 새로운 아티팩트를 생성하거나 기존 아티팩트를 검색합니다. (2번째 줄)
3. `.add_file`로 아티팩트에 파일을 추가합니다. (9번째 줄)
4. `.log_artifact`로 run에 아티팩트를 로그합니다. (10번째 줄)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # 아티팩트에 파일 및 에셋 추가하기
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 사용
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

  </TabItem>
  <TabItem value="outside">

W&B run 외부에서 아티팩트 버전 생성:

1. `wandb.Artifact`로 새로운 아티팩트를 생성하거나 기존 아티팩트를 검색합니다. (1번째 줄)
2. `.add_file`로 아티팩트에 파일을 추가합니다. (4번째 줄)
3. `.save`로 아티팩트를 저장합니다. (5번째 줄)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# 아티팩트에 파일 및 에셋 추가하기
# `.add`, `.add_file`, `.add_dir`, `.add_reference` 사용
artifact.add_file("image1.png")
artifact.save()
```  
  </TabItem>
</Tabs>

### Distributed runs

버전을 커밋하기 전에 runs의 집합이 협업하여 버전을 작업하도록 허용합니다. 이는 위에서 설명한 단일 run 모드와 대조적으로, 하나의 run이 새 버전에 대한 모든 데이터를 제공하는 경우와 다릅니다.


:::info
1. 컬렉션의 각 run은 동일한 버전에서 협업하기 위해 동일한 고유 ID(`distributed_id`라고 함)를 인식해야 합니다. 기본적으로, 존재하는 경우, W&B는 `wandb.init(group=GROUP)`으로 설정된 run의 `group`을 `distributed_id`로 사용합니다.
2. 버전을 "커밋"하는 최종 run이 있어야 하며, 그 상태를 영구적으로 잠급니다.
3. `upsert_artifact`를 사용하여 협력 아티팩트에 추가하고, `finish_artifact`를 사용하여 커밋을 완료합니다.
:::

다음 예를 고려하세요. 다른 runs(**Run 1**, **Run 2**, **Run 3**로 표시됨)이 `upsert_artifact`를 사용하여 동일한 아티팩트에 다른 이미지 파일을 추가합니다.

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # 아티팩트에 파일 및 에셋 추가하기
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 사용
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # 아티팩트에 파일 및 에셋 추가하기
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 사용
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1과 Run 2가 완료된 후에 실행해야 합니다. `finish_artifact`를 호출하는 Run은 아티팩트에 파일을 포함할 수 있지만 필수는 아닙니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # 아티팩트에 파일 및 에셋 추가하기
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 사용
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 기존 버전에서 새로운 아티팩트 버전 생성하기

이전 아티팩트 버전에서 파일 서브셋을 추가, 수정 또는 제거하여 변경된 파일이 아닌 파일을 다시 색인할 필요 없이 기능을 수행합니다. 이전 아티팩트 버전에서 파일 서브셋을 추가, 수정 또는 제거하면 *incremental artifact*로 알려진 새로운 아티팩트 버전이 생성됩니다.

![](/images/artifacts/incremental_artifacts.png)

여러분이 마주칠 수 있는 각각의 incremental 변경 유형에 대한 몇 가지 시나리오는 다음과 같습니다:

- add: 새 배치를 수집한 후 데이터셋에 새로운 파일 서브셋을 정기적으로 추가합니다.
- remove: 여러 중복 파일을 발견하여 아티팩트에서 제거하고자 합니다.
- update: 파일 서브셋의 주석을 수정하여 올바른 파일로 기존 파일을 대체하고자 합니다.

incremental artifact와 같은 기능을 수행하기 위해 아티팩트를 처음부터 생성할 수 있습니다. 그러나 처음부터 아티팩트를 생성할 때는 아티팩트의 모든 내용을 로컬 디스크에 가지고 있어야 합니다. incremental 변경을 수행할 때는 이전 아티팩트 버전의 파일을 변경하지 않고 단일 파일을 추가, 제거 또는 수정할 수 있습니다.


:::info
단일 run 내에서 또는 일련의 runs(분산 모드)로 incremental artifact를 생성할 수 있습니다.
:::


아래 절차를 따라 incremental 변경을 수행하는 아티팩트 버전을 얻으세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Run 내부', value: 'inside'},
    {label: 'Run 외부', value: 'outside'},
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





2. 다음과 같이 드래프트를 생성합니다:

```python
draft_artifact = saved_artifact.new_draft()
```

3. 다음 버전에서 보고 싶은 모든 incremental 변경을 수행합니다. 기존 항목을 추가, 제거 또는 수정할 수 있습니다.

다음 탭 중 하나를 선택하여 각 변경을 수행하는 방법에 대한 예시를 확인하세요:


<Tabs
  defaultValue="add"
  values={[
    {label: '추가', value: 'add'},
    {label: '제거', value: 'remove'},
    {label: '수정', value: 'modify'},
  ]}>
  <TabItem value="add">

기존 아티팩트 버전에 파일을 추가하려면 `add_file` 메소드를 사용하세요:

```python
draft_artifact.add_file("file_to_add.txt")
```

:::note
`add_dir` 메소드를 사용하여 여러 파일을 추가할 수도 있습니다.
:::

  </TabItem>
  <TabItem value="remove">

기존 아티팩트 버전에서 파일을 제거하려면 `remove` 메소드를 사용하세요:

```python
draft_artifact.remove("file_to_remove.txt")
```

:::note
디렉토리 경로를 전달하여 `remove` 메소드로 여러 파일을 제거할 수도 있습니다.
:::

  </TabItem>
  <TabItem value="modify">

드래프트에서 이전 내용을 제거하고 새로운 내용을 다시 추가하여 내용을 수정하거나 대체하세요:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>



4. 마지막으로, 변경 사항을 로그하거나 저장하세요. 다음 탭은 W&B run 내부 및 외부에서 변경 사항을 저장하는 방법을 보여줍니다. 사용 사례에 적합한 탭을 선택하세요:


<Tabs
  defaultValue="inside"
  values={[
    {label: 'Run 내부', value: 'inside'},
    {label: 'Run 외부', value: 'outside'},
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


모두 합치면, 위의 코드 예제는 다음과 같습니다:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Run 내부', value: 'inside'},
    {label: 'Run 외부', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # 아티팩트를 가져와서 run에 입력하기
    draft_artifact = saved_artifact.new_draft()  # 드래프트 버전 생성하기

    # 드래프트 버전에서 파일 서브셋을 수정하기
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 변경 사항을 로그하여 새 버전을 생성하고 run의 출력으로 표시하기
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # 아티팩트 로드하기
draft_artifact = saved_artifact.new_draft()  # 드래프트 버전 생성하기

# 드래프트 버전에서 파일 서브셋을 수정하기
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 드래프트에 대한 변경 사항 커밋하기
```

  </TabItem>
</Tabs>