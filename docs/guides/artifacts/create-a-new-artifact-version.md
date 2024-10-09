---
title: Create an artifact version
description: 단일 run 또는 분산 프로세스로부터 새로운 아티팩트 버전을 생성합니다.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

새로운 아티팩트 버전을 단일 [run](../runs/intro.md)으로 생성하거나 분산된 runs들과 함께 협력하여 생성하세요. 이전 버전에서 새로운 아티팩트 버전을 선택적으로 생성할 수 있으며, 이를 [증분 아티팩트](#create-a-new-artifact-version-from-an-existing-version)라고 합니다.

:::tip
아티팩트의 파일 서브셋에 변경을 적용해야 하고 원래 아티팩트의 크기가 상당히 클 때는 증분 아티팩트를 만드는 것이 좋습니다.
:::

## 새로운 아티팩트 버전 생성하기
새로운 아티팩트 버전을 생성하는 방법에는 두 가지가 있습니다: 단일 run에서 생성하거나 분산된 runs에서 생성하기. 이들은 다음과 같이 정의됩니다:

* **Single run**: 단일 run이 새로운 버전을 위한 모든 데이터를 제공합니다. 이는 가장 일반적인 경우이며, run이 필요한 데이터를 완전히 재생성할 때 가장 적합합니다. 예를 들어: 저장된 모델이나 모델 예측값을 분석을 위해 테이블에 출력하는 경우입니다.
* **Distributed runs**: 여러 runs가 함께 새로운 버전을 위한 모든 데이터를 제공합니다. 이는 다중 runs가 종종 병렬로 데이터를 생성하는 분산 작업에 가장 적합합니다. 예를 들어: 모델을 분산 방식으로 평가하고 예측값을 출력하는 경우입니다.

W&B는 프로젝트에 존재하지 않는 이름을 `wandb.Artifact` API에 전달하면 새로운 아티팩트를 생성하고 `v0` 에일리어스를 할당합니다. 동일한 아티팩트에 대해 다시 로그할 때 W&B는 내용을 체크썸합니다. 아티팩트가 변경되었을 때, W&B는 새로운 버전 `v1`을 저장합니다. 

W&B는 프로젝트에 있는 기존 아티팩트와 일치하는 이름과 아티팩트 유형을 `wandb.Artifact` API에 전달하면 해당 아티팩트를 검색합니다. 검색된 아티팩트의 버전은 1보다 클 것입니다.

![](/images/artifacts/single_distributed_artifacts.png)

### Single run
단일 run에서 아티팩트의 모든 파일을 생성하여 새로운 아티팩트 버전을 로그합니다. 이 경우는 단일 run이 아티팩트의 모든 파일을 생성할 때 발생합니다. 

유스 케이스에 따라, 다음 탭 중 하나를 선택하여 run 내부 또는 외부에서 새로운 아티팩트 버전을 생성하세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

W&B run 내에서 아티팩트 버전을 생성하세요:

1. `wandb.init`으로 run을 생성합니다. (Line 1)
2. `wandb.Artifact`로 새 아티팩트를 생성하거나 기존 아티팩트를 검색합니다. (Line 2)
3. `.add_file`로 아티팩트에 파일을 추가합니다. (Line 9)
4. `.log_artifact`로 run에 아티팩트를 로그합니다. (Line 10)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 를 사용하여
    # 파일 및 자산을 아티팩트에 추가합니다.
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

  </TabItem>
  <TabItem value="outside">

W&B run 외부에서 아티팩트 버전을 생성하세요:

1. `wandb.Artifact`로 새 아티팩트를 생성하거나 기존 아티팩트를 검색합니다. (Line 1)
2. `.add_file`로 아티팩트에 파일을 추가합니다. (Line 4)
3. `.save`로 아티팩트를 저장합니다. (Line 5)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, `.add_reference` 를 사용하여
# 파일 및 자산을 아티팩트에 추가합니다.
artifact.add_file("image1.png")
artifact.save()
```  
  </TabItem>
</Tabs>

### Distributed runs

runs의 모음을 통해 버전을 커밋하기 전에 협업할 수 있도록 허용합니다. 이는 위에서 설명한 단일 run 모드와 대조적이며, 하나의 run이 새로운 버전을 위한 모든 데이터를 제공합니다.

:::info
1. 컬렉션에 있는 각 run은 동일한 고유 ID(`distributed_id`)를 인식해야 협업이 가능합니다. 기본적으로 존재할 경우, W&B는 wandb.init(group=GROUP)으로 설정된 run의 `group`을 `distributed_id`로 사용합니다.
2. 버전을 "커밋"하여 상태를 영구적으로 잠그는 최종 run이 있어야 합니다.
3. 협업 아티팩트에 추가하려면 `upsert_artifact`를 사용하고 커밋을 완료하려면 `finish_artifact`를 사용하세요.
:::

다음 예를 고려합니다. 서로 다른 **Run 1**, **Run 2**, **Run 3**으로 표시된 runs는 `upsert_artifact`로 동일한 아티팩트에 서로 다른 이미지를 추가합니다.

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 를 사용하여
    # 파일 및 자산을 아티팩트에 추가합니다.
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 를 사용하여
    # 파일 및 자산을 아티팩트에 추가합니다.
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1과 Run 2가 완료된 후에 실행해야 합니다. `finish_artifact`를 호출하는 Run은 아티팩트에 파일을 포함할 수 있지만, 반드시 포함할 필요는 없습니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 를 사용하여
    # 파일 및 자산을 아티팩트에 추가합니다.
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 기존 버전에서 새로운 아티팩트 버전 생성하기

이전 아티팩트 버전의 파일 서브셋을 추가, 수정, 또는 제거하지만, 변경되지 않은 파일을 다시 인덱싱할 필요는 없습니다. 이전 아티팩트 버전의 파일 서브셋을 추가, 수정, 또는 제거하면 *증분 아티팩트*라고 불리는 새로운 아티팩트 버전이 생성됩니다.

![](/images/artifacts/incremental_artifacts.png)

각 증분 변경 유형에서 직면할 수 있는 몇 가지 시나리오는 다음과 같습니다:

- 추가: 새로운 배치를 수집한 후 정기적으로 데이터셋에 새로운 파일 서브셋을 추가합니다.
- 제거: 여러 중복 파일을 발견하여 아티팩트에서 제거하고자 합니다.
- 업데이트: 파일 서브셋에 대한 주석을 수정하였으며, 기존 파일을 올바른 파일로 교체하고자 합니다.

증분 아티팩트와 동일한 기능을 수행하기 위해 처음부터 아티팩트를 생성할 수도 있습니다. 그러나 처음부터 아티팩트를 생성하면 모든 내용을 로컬 디스크에 가져와야 합니다. 증분 변경을 수행할 때는 이전 아티팩트 버전의 파일을 변경하지 않고 단일 파일을 추가, 제거, 수정할 수 있습니다.

:::info
단일 run 또는 runs 모음(분산 모드)으로 증분 아티팩트를 생성할 수 있습니다.
:::

다음 절차를 따라 아티팩트를 증분 변경하세요:

1. 증분 변경을 수행할 아티팩트 버전을 획득합니다:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
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

2. 다음과 같이 초안을 만듭니다:

```python
draft_artifact = saved_artifact.new_draft()
```

3. 다음 버전에서 보기를 원하는 증분 변경을 수행하세요. 추가, 제거, 또는 기존 항목을 수정할 수 있습니다.

각 변경 사항을 수행하는 방법의 예제를 보려면 탭 중 하나를 선택하세요:

<Tabs
  defaultValue="add"
  values={[
    {label: 'Add', value: 'add'},
    {label: 'Remove', value: 'remove'},
    {label: 'Modify', value: 'modify'},
  ]}>
  <TabItem value="add">

`add_file` 메소드를 사용하여 기존 아티팩트 버전에 파일 추가:

```python
draft_artifact.add_file("file_to_add.txt")
```

:::note
`add_dir` 메소드를 사용하여 디렉토리를 추가하여 여러 파일을 추가할 수도 있습니다.
:::

  </TabItem>
  <TabItem value="remove">

`remove` 메소드를 사용하여 기존 아티팩트 버전에서 파일 제거:

```python
draft_artifact.remove("file_to_remove.txt")
```

:::note
디렉토리 경로를 전달하여 `remove` 메소드로 여러 파일을 제거할 수도 있습니다.
:::

  </TabItem>
  <TabItem value="modify">

기존 내용을 초안에서 제거하고 새로운 내용을 다시 추가하여 수정하거나 교체하세요:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>

4. 마지막으로, 변경사항을 로그 또는 저장하세요. 다음 탭에서는 W&B run 내와 외부에서 변경사항을 저장하는 방법을 보여줍니다. 유스 케이스에 맞는 탭을 선택하세요:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
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

모든 코드를 합치면, 위의 코드 예제는 다음과 같이 보입니다:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # 아티팩트를 가져와 run에 입력합니다.
    draft_artifact = saved_artifact.new_draft()  # 초안 버전을 생성합니다.

    # 초안 버전에서 파일 서브셋을 수정합니다.
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 변경사항을 로그하여 새로운 버전을 생성하고 run의 출력으로 표시합니다.
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # 아티팩트를 로드합니다.
draft_artifact = saved_artifact.new_draft()  # 초안 버전을 생성합니다.

# 초안 버전에서 파일 서브셋을 수정합니다.
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 초안에 변경사항을 커밋합니다.
```

  </TabItem>
</Tabs>