---
title: Artifacts 다운로드 및 사용
description: 여러 Projects에서 Artifacts를 다운로드하고 사용하세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

이미 W&B 서버에 저장된 artifact를 다운로드하여 사용하거나, artifact 오브젝트를 생성해서 필요할 때 중복 저장 없이 사용할 수 있습니다.

{{% alert %}}
보기 전용 권한만 있는 팀 멤버는 artifact를 다운로드할 수 없습니다.
{{% /alert %}}

### W&B에 저장된 artifact 다운로드 및 사용

W&B에 저장된 artifact는 W&B Run 내부 또는 외부에서 다운로드해 사용할 수 있습니다. Public API([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}))를 활용하면 이미 W&B에 저장된 데이터를 내보내거나(또는 업데이트) 할 수 있습니다. 자세한 내용은 W&B [Public API Reference 가이드]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})를 참고하세요.

{{< tabpane text=true >}}
  {{% tab header="Run 중에 사용하기" %}}
먼저, W&B Python SDK를 임포트한 뒤, W&B [Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}})을 생성합니다:

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

사용할 artifact는 [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ko" >}}) 메소드로 지정합니다. 이 메소드는 run 오브젝트를 반환합니다. 아래 코드에서는 'bike-dataset'이라는 이름의 artifact와 'latest' 에일리어스를 지정합니다:

```python
artifact = run.use_artifact("bike-dataset:latest")
```

반환된 오브젝트를 이용해 해당 artifact의 모든 내용을 다운로드할 수 있습니다:

```python
datadir = artifact.download()
```

특정 디렉토리에 artifact의 내용을 저장하고 싶을 때는 `root` 파라미터에 경로를 입력할 수 있습니다. 자세한 내용은 [Python SDK Reference Guide]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ko" >}})에서 확인하세요.

파일의 일부분만 받고 싶다면 [`get_path`]({{< relref path="/ref/python/sdk/classes/artifact.md#get_path" lang="ko" >}}) 메소드를 이용해 원하는 일부만 다운로드할 수 있습니다:

```python
path = artifact.get_path(name)
```

이 메소드는 path `name`에 위치한 파일만 받아옵니다. 반환값은 다음과 같은 메소드를 포함하는 `Entry` 오브젝트입니다:

* `Entry.download`: path `name`에 있는 파일을 artifact에서 다운로드
* `Entry.ref`: 만약 `add_reference`로 entry를 참조로 저장했다면, URI 반환

W&B가 지원하는 scheme의 참조 파일도 일반 artifact 파일처럼 다운로드할 수 있습니다. 자세한 내용은 [외부 파일 추적하기]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ko" >}})를 참고하세요.  
  {{% /tab %}}
  {{% tab header="Run 외부에서 사용하기" %}}
먼저, W&B SDK를 임포트하고, Public API Class를 이용해 artifact를 가져옵니다. entity, project, artifact, alias를 지정합니다:

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

반환된 오브젝트로 artifact의 내용을 다운로드합니다:

```python
artifact.download()
```

특정 디렉토리에 artifact 파일을 받고 싶다면 `root` 파라미터에 경로를 지정할 수 있습니다. 자세한 사용법은 [API Reference Guide]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ko" >}})에서 확인하세요.  
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` 명령어를 이용해 W&B 서버에서 artifact를 다운로드할 수 있습니다.

```
$ wandb artifact get project/artifact:alias --root mnist/
```  
  {{% /tab %}}
{{< /tabpane >}}

### artifact 일부만 다운로드하기

`path_prefix` 파라미터를 사용하면 artifact의 특정 파일이나 폴더만 다운로드할 수 있습니다.

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png 파일만 다운로드
```

특정 디렉토리 아래의 여러 파일도 다운로드할 수 있습니다:

```python
artifact.download(path_prefix="images/bikes/") # images/bikes 폴더에 있는 파일들을 다운로드
```
### 다른 project의 artifact 사용하기

artifact 이름 앞에 project 이름을 명시하면 해당 project의 artifact를 참조할 수 있습니다. entity 이름까지 같이 쓰면 entity에 상관없이 artifact를 사용할 수도 있습니다.

아래 예시는 다른 project의 artifact를 현재 W&B run에 입력으로 연결하는 방법입니다.

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 다른 project에서 artifact를 찾아서 이 run의 input으로 사용합니다.
artifact = run.use_artifact("my-project/artifact:alias")

# 다른 entity의 artifact도 input으로 지정할 수 있습니다.
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### artifact를 동시에 생성하고 사용하기

artifact를 생성과 동시에 사용할 수도 있습니다. artifact 오브젝트를 만든 후 바로 `use_artifact`에 넘기면 됩니다. 해당 artifact가 W&B에 없다면 새로 만들어집니다. [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ko" >}}) API는 멱등성이 보장되므로 여러 번 호출해도 안전합니다.

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

artifact 생성에 대한 더 자세한 내용은 [artifact 생성하기]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})를 참고하세요.