---
title: Create a launch job
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch 작업은 W&B run을 재현하기 위한 청사진입니다. 작업은 워크로드를 실행하는 데 필요한 소스 코드, 종속성 및 입력을 캡처하는 W&B Artifacts입니다.

`wandb launch` 코맨드를 사용하여 작업을 생성하고 실행하세요.

{{% alert %}}
실행을 위해 제출하지 않고 작업을 생성하려면 `wandb job create` 코맨드를 사용하세요. 자세한 내용은 [코맨드 참조 문서]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ko" >}})를 참조하세요.
{{% /alert %}}

## Git 작업

W&B Launch를 사용하여 원격 git 저장소의 특정 커밋, 브랜치 또는 태그에서 코드 및 기타 추적된 자산을 복제하는 Git 기반 작업을 만들 수 있습니다. 코드 URI를 지정하려면 `--uri` 또는 `-u` 플래그를 사용하고, 하위 디렉토리를 지정하려면 선택적으로 `--build-context` 플래그를 사용하세요.

다음 코맨드를 사용하여 git 저장소에서 "hello world" 작업을 실행합니다.

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

이 코맨드는 다음을 수행합니다.
1. [W&B Launch 작업 저장소](https://github.com/wandb/launch-jobs)를 임시 디렉토리에 복제합니다.
2. **hello** 프로젝트에 **hello-world-git**이라는 작업을 만듭니다. 이 작업은 저장소의 기본 브랜치 헤드에 있는 커밋과 연결됩니다.
3. `jobs/hello_world` 디렉토리와 `Dockerfile.wandb`에서 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `python job.py`를 실행합니다.

특정 브랜치 또는 커밋 해시에서 작업을 빌드하려면 `-g`, `--git-hash` 인수를 추가하세요. 전체 인수 목록을 보려면 `wandb launch --help`를 실행하세요.

### 원격 URL 형식

Launch 작업과 연결된 git 원격은 HTTPS 또는 SSH URL일 수 있습니다. URL 유형은 작업 소스 코드를 가져오는 데 사용되는 프로토콜을 결정합니다.

| 원격 URL 유형 | URL 형식                                     | 엑세스 및 인증 요구 사항                                      |
| ---------- | -------------------------------------------- | ------------------------------------------------------------ |
| https      | `https://github.com/organization/repository.git` | git 원격으로 인증하기 위한 사용자 이름 및 비밀번호                   |
| ssh        | `git@github.com:organization/repository.git`    | git 원격으로 인증하기 위한 ssh 키                               |

정확한 URL 형식은 호스팅 공급자에 따라 다릅니다. `wandb launch --uri`로 생성된 작업은 제공된 `--uri`에 지정된 전송 프로토콜을 사용합니다.

## 코드 Artifact 작업

W&B Artifact에 저장된 소스 코드에서 작업을 만들 수 있습니다. `--uri` 또는 `-u` 인수로 로컬 디렉토리를 사용하여 새 코드 아티팩트 및 작업을 만드세요.

시작하려면 빈 디렉토리를 만들고 다음 내용으로 `main.py`라는 Python 스크립트를 추가합니다.

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

다음 내용으로 `requirements.txt` 파일을 추가합니다.

```txt
wandb>=0.17.1
```

다음 코맨드를 사용하여 디렉토리를 코드 Artifact로 기록하고 작업을 시작합니다.

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

위의 코맨드는 다음을 수행합니다.
1. 현재 디렉토리를 `hello-world-code`라는 코드 Artifact로 기록합니다.
2. `launch-quickstart` 프로젝트에 `hello-world-code`라는 작업을 만듭니다.
3. 현재 디렉토리와 Launch의 기본 Dockerfile에서 컨테이너 이미지를 빌드합니다. 기본 Dockerfile은 `requirements.txt` 파일을 설치하고 진입점을 `python main.py`로 설정합니다.

## 이미지 작업

또는 미리 만들어진 Docker 이미지에서 작업을 빌드할 수 있습니다. 이는 ML 코드에 대한 기존 빌드 시스템이 이미 있거나 작업에 대한 코드 또는 요구 사항을 조정할 필요는 없지만 하이퍼파라미터 또는 다른 인프라 규모를 실험하려는 경우에 유용합니다.

이미지는 Docker 레지스트리에서 가져와서 지정된 진입점 또는 지정되지 않은 경우 기본 진입점으로 실행됩니다. `--docker-image` 옵션에 전체 이미지 태그를 전달하여 Docker 이미지에서 작업을 생성하고 실행합니다.

미리 만들어진 이미지에서 간단한 작업을 실행하려면 다음 코맨드를 사용하세요.

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 자동 작업 생성

W&B는 추적된 소스 코드가 있는 모든 run에 대해 Launch로 생성되지 않은 경우에도 작업을 자동으로 생성하고 추적합니다. run은 다음 세 가지 조건 중 하나라도 충족되면 추적된 소스 코드가 있는 것으로 간주됩니다.
- run에 연결된 git 원격 및 커밋 해시가 있습니다.
- run이 코드 Artifact를 기록했습니다(자세한 내용은 [`Run.log_code`]({{< relref path="/ref/python/run.md#log_code" lang="ko" >}}) 참조).
- run이 `WANDB_DOCKER` 환경 변수가 이미지 태그로 설정된 Docker 컨테이너에서 실행되었습니다.

Launch 작업이 W&B run에 의해 자동으로 생성되는 경우 Git 원격 URL은 로컬 git 저장소에서 유추됩니다.

### Launch 작업 이름

기본적으로 W&B는 자동으로 작업 이름을 생성합니다. 이름은 작업이 생성된 방식(GitHub, 코드 Artifact 또는 Docker 이미지)에 따라 생성됩니다. 또는 환경 변수 또는 W&B Python SDK를 사용하여 Launch 작업의 이름을 정의할 수 있습니다.

다음 표는 작업 소스를 기반으로 기본적으로 사용되는 작업 명명 규칙을 설명합니다.

| 소스        | 명명 규칙                                 |
| ------------- | ---------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| 코드 Artifact | `job-<code-artifact-name>`             |
| Docker 이미지  | `job-<image-name>`                     |

W&B 환경 변수 또는 W&B Python SDK로 작업 이름을 지정하세요.

{{< tabpane text=true >}}
{{% tab "Environment variable" %}}
`WANDB_JOB_NAME` 환경 변수를 원하는 작업 이름으로 설정하세요. 예를 들어:

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings`로 작업 이름을 정의합니다. 그런 다음 `wandb.init`로 W&B를 초기화할 때 이 오브젝트를 전달합니다. 예를 들어:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
docker 이미지 작업의 경우 버전 에일리어스가 작업에 에일리어스로 자동 추가됩니다.
{{% /alert %}}

## 컨테이너화

작업은 컨테이너에서 실행됩니다. 이미지 작업은 미리 빌드된 Docker 이미지를 사용하는 반면 Git 및 코드 Artifact 작업에는 컨테이너 빌드 단계가 필요합니다.

작업 컨테이너화는 `wandb launch`에 대한 인수와 작업 소스 코드 내의 파일로 사용자 정의할 수 있습니다.

### 빌드 컨텍스트

빌드 컨텍스트라는 용어는 컨테이너 이미지를 빌드하기 위해 Docker 데몬으로 전송되는 파일 및 디렉토리 트리를 나타냅니다. 기본적으로 Launch는 작업 소스 코드의 루트를 빌드 컨텍스트로 사용합니다. 하위 디렉토리를 빌드 컨텍스트로 지정하려면 작업을 생성하고 시작할 때 `wandb launch`의 `--build-context` 인수를 사용하세요.

{{% alert %}}
`--build-context` 인수는 여러 프로젝트가 있는 모노레포로 작업하는 데 특히 유용합니다. 하위 디렉토리를 빌드 컨텍스트로 지정하면 모노레포 내의 특정 프로젝트에 대한 컨테이너 이미지를 빌드할 수 있습니다.

공식 W&B Launch 작업 저장소와 함께 `--build-context` 인수를 사용하는 방법에 대한 데모는 [위의 예]({{< relref path="#git-jobs" lang="ko" >}})를 참조하세요.
{{% /alert %}}

### Dockerfile

Dockerfile은 Docker 이미지를 빌드하기 위한 지침이 포함된 텍스트 파일입니다. 기본적으로 Launch는 `requirements.txt` 파일을 설치하는 기본 Dockerfile을 사용합니다. 사용자 정의 Dockerfile을 사용하려면 `wandb launch`의 `--dockerfile` 인수로 파일 경로를 지정하세요.

Dockerfile 경로는 빌드 컨텍스트를 기준으로 지정됩니다. 예를 들어 빌드 컨텍스트가 `jobs/hello_world`이고 Dockerfile이 `jobs/hello_world` 디렉토리에 있는 경우 `--dockerfile` 인수를 `Dockerfile.wandb`로 설정해야 합니다. 공식 W&B Launch 작업 저장소와 함께 `--dockerfile` 인수를 사용하는 방법에 대한 데모는 [위의 예]({{< relref path="#git-jobs" lang="ko" >}})를 참조하세요.

### Requirements 파일

사용자 정의 Dockerfile이 제공되지 않은 경우 Launch는 설치할 Python 종속성에 대한 빌드 컨텍스트를 찾습니다. `requirements.txt` 파일이 빌드 컨텍스트의 루트에 있는 경우 Launch는 파일에 나열된 종속성을 설치합니다. 그렇지 않고 `pyproject.toml` 파일이 발견되면 Launch는 `project.dependencies` 섹션에서 종속성을 설치합니다.
