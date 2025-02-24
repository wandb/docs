---
title: Download and use artifacts
description: 여러 프로젝트에서 Artifacts 를 다운로드하고 사용할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

W&B 서버에 이미 저장된 아티팩트를 다운로드하여 사용하거나, 필요에 따라 아티팩트 오브젝트를 구성하여 중복 제거를 위해 전달합니다.

{{% alert %}}
보기 전용 권한이 있는 팀 멤버는 아티팩트를 다운로드할 수 없습니다.
{{% /alert %}}

### W&B에 저장된 아티팩트 다운로드 및 사용

W&B [Run]({{< relref path="/ref/python/run.md" lang="ko" >}}) 내부 또는 외부에서 W&B에 저장된 아티팩트를 다운로드하여 사용합니다. W&B에 이미 저장된 데이터를 내보내거나 업데이트하려면 Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}))를 사용하세요. 자세한 내용은 W&B [Public API Reference guide]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 참조하세요.

{{< tabpane text=true >}}
  {{% tab header="Run 도중" %}}
먼저 W&B Python SDK를 임포트합니다. 다음으로 W&B [Run]({{< relref path="/ref/python/run.md" lang="ko" >}})을 생성합니다:

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ko" >}}) 메소드를 사용하여 사용할 아티팩트를 지정합니다. 그러면 run 오브젝트가 반환됩니다. 다음 코드 조각은 `'bike-dataset'`이라는 아티팩트를 에일리어스 `'latest'`로 지정합니다:

```python
artifact = run.use_artifact("bike-dataset:latest")
```

반환된 오브젝트를 사용하여 아티팩트의 모든 내용을 다운로드합니다:

```python
datadir = artifact.download()
```

선택적으로 `root` 파라미터에 경로를 전달하여 아티팩트의 내용을 특정 디렉토리에 다운로드할 수 있습니다. 자세한 내용은 [Python SDK Reference Guide]({{< relref path="/ref/python/artifact.md#download" lang="ko" >}})를 참조하세요.

파일의 서브셋만 다운로드하려면 [`get_path`]({{< relref path="/ref/python/artifact.md#get_path" lang="ko" >}}) 메소드를 사용합니다:

```python
path = artifact.get_path(name)
```

이것은 경로 `name`에 있는 파일만 가져옵니다. 다음 메소드를 가진 `Entry` 오브젝트를 반환합니다:

* `Entry.download`: 아티팩트에서 경로 `name`에 있는 파일을 다운로드합니다
* `Entry.ref`: `add_reference`가 엔트리를 참조로 저장한 경우 URI를 반환합니다

W&B가 처리 방법을 아는 스키마를 가진 참조는 아티팩트 파일처럼 다운로드됩니다. 자세한 내용은 [Track external files]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ko" >}})를 참조하세요.
  {{% /tab %}}
  {{% tab header="Run 외부" %}}
먼저 W&B SDK를 임포트합니다. 다음으로 Public API 클래스에서 아티팩트를 생성합니다. 해당 아티팩트와 연결된 엔티티, 프로젝트, 아티팩트 및 에일리어스를 제공합니다:

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

반환된 오브젝트를 사용하여 아티팩트의 내용을 다운로드합니다:

```python
artifact.download()
```

선택적으로 `root` 파라미터에 경로를 전달하여 아티팩트의 내용을 특정 디렉토리에 다운로드할 수 있습니다. 자세한 내용은 [API Reference Guide]({{< relref path="/ref/python/artifact.md#download" lang="ko" >}})를 참조하세요.
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` 코맨드를 사용하여 W&B 서버에서 아티팩트를 다운로드합니다.

```
$ wandb artifact get project/artifact:alias --root mnist/
```
  {{% /tab %}}
{{< /tabpane >}}

### 아티팩트 부분적으로 다운로드하기

선택적으로 접두사를 기반으로 아티팩트의 일부를 다운로드할 수 있습니다. `path_prefix` 파라미터를 사용하면 단일 파일 또는 하위 폴더의 내용을 다운로드할 수 있습니다.

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png만 다운로드합니다
```

또는 특정 디렉토리에서 파일을 다운로드할 수 있습니다:

```python
artifact.download(path_prefix="images/bikes/") # images/bikes 디렉토리의 파일을 다운로드합니다
```
### 다른 프로젝트의 아티팩트 사용

아티팩트 이름과 함께 프로젝트 이름을 지정하여 아티팩트를 참조합니다. 아티팩트 이름과 함께 엔티티 이름을 지정하여 엔티티 간에 아티팩트를 참조할 수도 있습니다.

다음 코드 예제는 다른 프로젝트의 아티팩트를 현재 W&B run의 입력으로 쿼리하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 다른 프로젝트의 아티팩트에 대해 W&B를 쿼리하고
# 이 run에 대한 입력으로 표시합니다.
artifact = run.use_artifact("my-project/artifact:alias")

# 다른 엔티티의 아티팩트를 사용하고 이를 입력으로 표시합니다
# 이 run에 대해.
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### 아티팩트 생성 및 동시에 사용

아티팩트를 동시에 생성하고 사용합니다. 아티팩트 오브젝트를 생성하고 `use_artifact`에 전달합니다. 아직 존재하지 않으면 W&B에 아티팩트가 생성됩니다. [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ko" >}}) API는 멱등성이므로 원하는 만큼 여러 번 호출할 수 있습니다.

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

아티팩트 구성에 대한 자세한 내용은 [Construct an artifact]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})를 참조하세요.
