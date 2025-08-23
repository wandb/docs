---
title: Launch 작업 생성
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch job은 W&B run 을 재현하기 위한 청사진입니다. job은 소스 코드, 의존성, 그리고 workload 실행에 필요한 입력값을 캡처하는 W&B Artifacts 중 하나입니다.

`wandb launch` 코맨드를 사용하여 job을 생성하고 실행할 수 있습니다.

{{% alert %}}
실행하지 않고 job만 생성하려면 `wandb job create` 코맨드를 사용하세요. 자세한 내용은 [코맨드 레퍼런스 문서]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

## Git jobs

코드와 기타 추적된 자산을 원격 git 저장소의 특정 커밋, 브랜치, 혹은 태그에서 복제(clone)하여 Git 기반의 job을 생성할 수 있습니다. `--uri` 또는 `-u` 플래그로 코드가 포함된 URI를 지정하고, 필요하다면 `--build-context` 플래그로 하위 디렉토리를 지정할 수 있습니다.

아래 명령어를 사용해 git 저장소에서 "hello world" job을 실행해 보세요.

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

이 명령어는 다음 작업을 수행합니다:
1. [W&B Launch jobs 저장소](https://github.com/wandb/launch-jobs)를 임시 디렉토리에 clone 합니다.
2. **hello** 프로젝트 안에 **hello-world-git**이라는 job을 생성합니다. 해당 job은 저장소의 기본 브랜치 HEAD에 있는 커밋과 연결됩니다.
3. `jobs/hello_world` 디렉토리와 `Dockerfile.wandb`로부터 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `python job.py`를 실행합니다.

특정 브랜치나 커밋 해시로부터 job을 빌드하려면 `-g`, `--git-hash` 인수를 추가하세요. 전체 인수 목록은 `wandb launch --help`를 실행하여 확인할 수 있습니다.

### 원격 URL 형식

Launch job에 연결되는 git remote는 HTTPS 또는 SSH URL 모두 사용할 수 있습니다. URL 유형이 job의 소스 코드를 가져오는 프로토콜을 결정합니다.

| Remote URL 타입 | URL 형식 | 엑세스 및 인증 요구 사항 |
| ----------| ------------------- | ------------------------------------------ |
| https      | `https://github.com/organization/repository.git`  | git remote 인증을 위한 username과 password 필요 |
| ssh        | `git@github.com:organization/repository.git` | git remote 인증을 위한 SSH key 필요 |

정확한 URL 포맷은 호스팅 제공자에 따라 다를 수 있습니다. `wandb launch --uri`로 생성된 job은 입력한 `--uri`에 지정된 전송 프로토콜을 그대로 사용합니다.


## Code artifact jobs

W&B Artifact에 저장된 소스 코드로 job을 생성할 수 있습니다. 로컬 디렉토리를 `--uri` 또는 `-u` 인수로 지정해 새로운 code artifact와 job을 생성할 수 있습니다.

먼저 빈 디렉토리를 만든 다음, `main.py`라는 Python 스크립트를 아래와 같이 작성하세요.

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

그리고 아래의 내용을 가진 `requirements.txt` 파일을 추가하세요.

```txt
wandb>=0.17.1
```

아래 명령어로 디렉토리를 코드 artifact로 기록(log)하고, job을 실행하세요.

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

위의 명령어는 다음과 같은 작업을 수행합니다:
1. 현재 디렉토리를 `hello-world-code`라는 code artifact로 기록합니다.
2. `launch-quickstart` 프로젝트에 `hello-world-code` 이름으로 job을 생성합니다.
3. 현재 디렉토리와 Launch의 기본 Dockerfile을 사용하여 컨테이너 이미지를 빌드합니다. 기본 Dockerfile은 `requirements.txt`를 설치하고, entry point를 `python main.py`로 설정합니다.

## Image jobs

또는, 미리 만들어진 Docker 이미지를 기반으로 job을 생성할 수도 있습니다. 이미 구축해둔 ML 코드 빌드 시스템이 있거나, 이번 job에서 코드나 requirements를 바꿀 필요 없이 하이퍼파라미터나 인프라만 실험하고 싶을 때 유용합니다.

이미지는 Docker 레지스트리에서 pull 하며, 지정한 entry point로 실행됩니다. entry point를 지정하지 않으면 기본 entry point가 사용됩니다. `--docker-image` 옵션에 전체 이미지 태그를 전달하여 Docker 이미지로부터 job을 생성/실행하세요.

아래 명령어로 미리 만든 이미지에서 간단한 job을 실행할 수 있습니다:

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 자동 job 생성

W&B는 추적 가능한 소스 코드가 있는 모든 run에 대해, Launch로 생성된 run이 아니더라도 자동으로 job을 생성하고 추적합니다. run이 아래 세 가지 조건 중 하나 이상에 해당되면 소스 코드가 추적됐다고 간주합니다:
- run에 연결된 git remote와 커밋 해시가 있음
- run에서 code artifact를 기록함 ([`Run.log_code`]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ko" >}}) 참고)
- Docker 컨테이너에서 `WANDB_DOCKER` 환경 변수가 이미지 태그로 설정된 채로 실행됨

Launch job이 W&B run에 의해 자동으로 생성되면 git remote URL은 로컬 git 저장소에서 자동으로 추론됩니다.

### Launch job 이름

기본적으로 W&B는 자동으로 job 이름을 생성해줍니다. 이름은 job이 생성되는 방식(GitHub, code artifact, Docker 이미지)에 따라 다르게 만들어집니다. 원한다면 환경 변수나 W&B Python SDK로 Launch job의 이름을 직접 지정할 수 있습니다.

아래 표는 job 소스에 따라 기본적으로 사용되는 job 이름 생성 규칙을 설명합니다.

| 소스        | 네이밍 규칙                               |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

다음 방법을 활용해 job 이름을 지정해보세요.

{{< tabpane text=true >}}
{{% tab "환경 변수" %}}
`WANDB_JOB_NAME` 환경 변수로 원하는 job 이름을 지정하세요. 예시:

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings`로 job 이름을 정의하고, `wandb.init`에 전달하여 W&B를 초기화하세요. 예시:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
docker image job의 경우, 버전 에일리어스가 자동으로 job에 에일리어스로 추가됩니다.
{{% /alert %}}

## 컨테이너화

job은 컨테이너에서 실행됩니다. image job은 미리 만들어진 Docker 이미지를 사용하고, Git 및 code artifact job은 컨테이너 이미지를 빌드하는 단계가 필요합니다.

컨테이너화 과정은 `wandb launch` 인수와 job 소스 코드 내 파일로 커스터마이즈할 수 있습니다.

### Build context

빌드 컨텍스트(build context)란 컨테이너 이미지를 빌드할 때 Docker 데몬에게 전달되는 파일과 디렉토리 구조 전체를 의미합니다. 기본적으로 Launch는 job 소스 코드의 루트를 빌드 컨텍스트로 사용합니다. 하위 디렉토리를 빌드 컨텍스트로 지정하고 싶다면 job 생성/실행 시 `wandb launch`의 `--build-context` 인수를 사용하세요.

{{% alert %}}
`--build-context` 인수는 여러 프로젝트가 한 monorepo(모노레포)에 담긴 Git job에서 특히 유용합니다. 하위 디렉토리를 빌드 컨텍스트로 지정하면, 모노레포 내 특정 프로젝트에 대한 컨테이너 이미지를 빌드할 수 있습니다.

[위 예시]({{< relref path="#git-jobs" lang="ko" >}})를 참고하세요. 공식 W&B Launch jobs 저장소에서 `--build-context` 인수 사용법을 확인할 수 있습니다.
{{% /alert %}}

### Dockerfile

Dockerfile은 Docker 이미지를 빌드하는 명령들이 적힌 텍스트 파일입니다. Launch는 기본적으로 `requirements.txt`를 설치하는 디폴트 Dockerfile을 사용합니다. 직접 만든 Dockerfile을 쓰고 싶다면, `wandb launch`의 `--dockerfile` 인수에 파일 경로를 지정하세요.

Dockerfile 경로는 빌드 컨텍스트 기준 상대 경로입니다. 예를 들어, 빌드 컨텍스트가 `jobs/hello_world`이고 Dockerfile이 해당 디렉토리에 있다면, `--dockerfile` 인수는 `Dockerfile.wandb`로 지정되어야 합니다. [위 예시]({{< relref path="#git-jobs" lang="ko" >}})를 참고해 공식 W&B Launch jobs 저장소에서 `--dockerfile` 인수 사용법을 살펴보세요.

### 요구 사항 파일

커스텀 Dockerfile을 별도로 지정하지 않은 경우, Launch는 빌드 컨텍스트 내 Python 의존성 파일을 찾아 설치합니다. 만약 빌드 컨텍스트의 루트에 `requirements.txt` 파일이 있으면 해당 파일의 모든 의존성을 설치합니다. 만약 없고, 대신 `pyproject.toml`이 있다면 `project.dependencies` 섹션의 의존성 패키지가 설치됩니다.