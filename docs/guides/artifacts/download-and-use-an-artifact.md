---
title: Download and use artifacts
description: 여러 프로젝트에서 Artifacts를 다운로드하고 사용하세요.
displayed_sidebar: default
---

아티팩트는 이미 W&B 서버에 저장되어 있거나, 아티팩트 오브젝트를 생성하여 필요한 경우 중복 제거를 위해 전달할 수 있습니다.

:::note
보기 전용 좌석을 가진 팀 멤버는 아티팩트를 다운로드할 수 없습니다.
:::

### W&B에 저장된 아티팩트 다운로드 및 사용

W&B에 저장된 아티팩트를 W&B Run 안이나 밖에서 다운로드하고 사용할 수 있습니다. Public API ([`wandb.Api`](../../ref/python/public-api/api.md))를 사용하여 이미 W&B에 저장된 데이터를 내보내거나 업데이트할 수 있습니다. 자세한 내용은 W&B [Public API Reference 가이드](../../ref/python/public-api/README.md)를 참조하세요.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="insiderun"
  values={[
    {label: 'During a run', value: 'insiderun'},
    {label: 'Outside of a run', value: 'outsiderun'},
    {label: 'W&B CLI', value: 'CLI'},
  ]}>
  <TabItem value="insiderun">

먼저, W&B Python SDK를 가져옵니다. 그런 다음, W&B [Run](../../ref/python/run.md)을 생성합니다:

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

[`use_artifact`](../../ref/python/run.md#use_artifact) 메소드를 사용하여 사용할 아티팩트를 지정합니다. 이는 run 오브젝트를 반환합니다. 다음 코드조각은 `'latest'` 에일리어스가 있는 `'bike-dataset'` 아티팩트를 지정합니다:

```python
artifact = run.use_artifact("bike-dataset:latest")
```

반환된 오브젝트를 사용하여 아티팩트의 모든 내용을 다운로드합니다:

```python
datadir = artifact.download()
```

옵션으로 특정 디렉토리에 아티팩트 내용을 다운로드하기 위해 root 파라미터에 경로를 전달할 수 있습니다. 자세한 내용은 [Python SDK Reference Guide](../../ref/python/artifact.md#download)를 참조하세요.

[`get_path`](../../ref/python/artifact.md#get_path) 메소드를 사용하여 파일 서브셋만 다운로드할 수 있습니다:

```python
path = artifact.get_path(name)
```

이는 경로 `name`에 있는 파일만 가져옵니다. 이는 다음 메소드를 가진 `Entry` 오브젝트를 반환합니다:

* `Entry.download`: 경로 `name`에 있는 아티팩트에서 파일을 다운로드합니다
* `Entry.ref`: `add_reference`가 엔트리를 참조로 저장한 경우 URI를 반환합니다

W&B가 처리할 수 있는 스키마를 가진 참조는 아티팩트 파일처럼 다운로드됩니다. 자세한 내용은 [외부 파일 추적](../../guides/artifacts/track-external-files.md)을 참조하세요.

  </TabItem>
  <TabItem value="outsiderun">
  
먼저, W&B SDK를 가져옵니다. 그런 다음, Public API 클래스에서 아티팩트를 생성합니다. 해당 아티팩트와 연관된 entity, project, artifact, 그리고 alias를 제공합니다:

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

반환된 오브젝트를 사용하여 아티팩트의 내용을 다운로드합니다:

```python
artifact.download()
```

옵션으로 특정 디렉토리에 아티팩트 내용을 다운로드하기 위해 `root` 파라미터에 경로를 전달할 수 있습니다. 자세한 내용은 [API Reference Guide](../../ref/python/artifact.md#download)를 참조하세요.
  
  </TabItem>
  <TabItem value="CLI">

`wandb artifact get` 코맨드를 사용하여 W&B 서버에서 아티팩트를 다운로드합니다.

```
$ wandb artifact get project/artifact:alias --root mnist/
```
  </TabItem>
</Tabs>

### 아티팩트 부분 다운로드

옵션으로 접두사를 기준으로 아티팩트의 일부를 다운로드할 수 있습니다. `path_prefix` 파라미터를 사용하여 단일 파일 또는 하위 폴더의 내용을 다운로드할 수 있습니다.

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png만 다운로드
```

또는 특정 디렉토리에서 파일을 다운로드할 수 있습니다:

```python
artifact.download(path_prefix="images/bikes/") # images/bikes 디렉토리의 파일 다운로드
```
### 다른 프로젝트에서 아티팩트를 사용

참조할 아티팩트의 이름과 프로젝트 이름을 지정하여 아티팩트를 참조할 수 있습니다. 또한 엔티티 이름과 함께 아티팩트의 이름을 지정하여 엔티티 간에 아티팩트를 참조할 수 있습니다.

다음 코드 예제는 현재 W&B run의 입력으로 다른 프로젝트의 아티팩트를 쿼리하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 다른 프로젝트에서 아티팩트를 쿼리하여 이를 이 run의 입력으로 표시합니다.
artifact = run.use_artifact("my-project/artifact:alias")

# 다른 엔티티에서 아티팩트를 사용하고 이를 이 run의 입력으로 표시합니다.
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### 동시에 아티팩트 생성 및 사용

아티팩트를 동시에 생성하고 사용합니다. 아티팩트 오브젝트를 생성하고 `use_artifact`에 전달합니다. W&B에 아티팩트가 아직 존재하지 않으면 이를 생성합니다. [`use_artifact`](../../ref/python/run.md#use_artifact) API는 멱등성을 가지므로 원하는 만큼 여러 번 호출할 수 있습니다.

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

아티팩트 생성에 대한 자세한 정보는 [아티팩트 생성](../../guides/artifacts/construct-an-artifact.md)을 참조하세요.