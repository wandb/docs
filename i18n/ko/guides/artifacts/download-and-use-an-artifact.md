---
description: Download and use Artifacts from multiple projects.
displayed_sidebar: default
---

# 아티팩트 다운로드 및 사용하기

<head>
  <title>아티팩트 다운로드 및 사용하기</title>
</head>

W&B 서버에 이미 저장된 아티팩트를 다운로드하거나 아티팩트 오브젝트를 구성하여 필요에 따라 중복 제거하여 전달하세요.

:::note
읽기 전용 자리가 있는 팀 멤버는 아티팩트를 다운로드할 수 없습니다.
:::

### W&B에 저장된 아티팩트 다운로드 및 사용하기

W&B Run 내부나 외부에서 W&B에 저장된 아티팩트를 다운로드하여 사용하세요. 이미 W&B에 저장된 데이터를 내보내거나(또는 데이터 업데이트)하기 위해 Public API([`wandb.Api`](../../ref/python/public-api/api.md))를 사용하세요. 자세한 내용은 W&B [Public API 참조 가이드](../../ref/python/public-api/README.md)를 참조하세요.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="insiderun"
  values={[
    {label: 'run 중에', value: 'insiderun'},
    {label: 'run 외부에서', value: 'outsiderun'},
    {label: 'wandb CLI', value: 'cli'},
  ]}>
  <TabItem value="insiderun">

먼저, W&B Python SDK를 import하세요. 다음으로, W&B [Run](../../ref/python/run.md)을 생성하세요:

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

[`use_artifact`](../../ref/python/run.md#use_artifact) 메소드를 사용하여 사용할 아티팩트를 지정하세요. 이는 run 오브젝트를 반환합니다. 다음 코드 조각에서는 `'bike-dataset'`이라는 아티팩트와 `'latest'`라는 에일리어스를 지정합니다:

```python
artifact = run.use_artifact("bike-dataset:latest")
```

반환된 오브젝트를 사용하여 아티팩트의 모든 내용을 다운로드하세요:

```python
datadir = artifact.download()
```

특정 디렉토리에 아티팩트의 내용을 다운로드하기 위해 `root` 파라미터에 경로를 선택적으로 전달할 수 있습니다. 자세한 내용은 [Python SDK 참조 가이드](../../ref/python/artifact.md#download)를 참조하세요.

파일의 서브셋만 다운로드하기 위해 [`get_path`](../../ref/python/artifact.md#get_path) 메소드를 사용하세요:

```python
path = artifact.get_path(name)
```

이는 `name` 경로에 있는 파일만 가져옵니다. 다음 메소드가 포함된 `Entry` 오브젝트를 반환합니다:

* `Entry.download`: `name` 경로에 있는 아티팩트의 파일을 다운로드합니다
* `Entry.ref`: 엔트리가 `add_reference`를 사용하여 참조로 저장된 경우 URI를 반환합니다

W&B가 처리할 수 있는 스키마를 가진 참조는 아티팩트 파일처럼 다운로드할 수 있습니다. 자세한 내용은 [외부 파일 추적하기](../../guides/artifacts/track-external-files.md)를 참조하세요.
  
  </TabItem>
  <TabItem value="outsiderun">
  
먼저, W&B SDK를 import하세요. 다음으로, Public API Class에서 아티팩트를 생성하세요. 해당 아티팩트와 관련된 엔터티, 프로젝트, 아티팩트, 에일리어스를 제공하세요:

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

반환된 오브젝트를 사용하여 아티팩트의 내용을 다운로드하세요:

```python
artifact.download()
```

특정 디렉토리에 아티팩트의 내용을 다운로드하기 위해 `root` 파라미터에 경로를 선택적으로 전달할 수 있습니다. 자세한 내용은 [API 참조 가이드](../../ref/python/artifact.md#download)를 참조하세요.
  
  </TabItem>
  <TabItem value="cli">

`wandb artifact get` 코맨드를 사용하여 W&B 서버에서 아티팩트를 다운로드하세요.

```
$ wandb artifact get project/artifact:alias --root mnist/
```
  </TabItem>
</Tabs>

### 다른 프로젝트의 아티팩트 사용하기

아티팩트를 참조하려면 아티팩트 이름과 프로젝트 이름을 지정하세요. 엔터티 간에 아티팩트를 참조하려면 아티팩트 이름과 그 엔터티 이름을 지정하세요.

다음 코드 예제는 다른 프로젝트의 아티팩트를 현재 W&B run의 입력으로 쿼리하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 다른 프로젝트의 아티팩트를 W&B에서 쿼리하여
# 이 run의 입력으로 표시합니다.
artifact = run.use_artifact("my-project/artifact:alias")

# 다른 엔터티의 아티팩트를 사용하여
# 이 run의 입력으로 표시합니다.
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### 아티팩트를 동시에 구성하고 사용하기

동시에 아티팩트를 구성하고 사용하세요. 아티팩트 오브젝트를 생성하고 use_artifact에 전달하세요. 이는 아직 존재하지 않는 경우 W&B에서 아티팩트를 생성합니다. [`use_artifact`](../../ref/python/run.md#use_artifact) API는 멱등성이 있으므로 원하는 만큼 여러 번 호출할 수 있습니다.

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

아티팩트를 구성하는 방법에 대한 자세한 내용은 [아티팩트 구성하기](../../guides/artifacts/construct-an-artifact.md)를 참조하세요.