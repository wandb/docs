---
title: Create a launch job
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Launch job은 W&B run을 재현하기 위한 청사진입니다. Job은 워크로드를 실행하기 위해 필요한 소스 코드, 종속성, 입력을 캡처하는 W&B Artifacts입니다.

`wandb launch` 코맨드를 사용하여 job을 생성하고 실행하세요.

:::info
실행을 위해 제출하지 않고 job을 생성하려면 `wandb job create` 코맨드를 사용하세요. 더 많은 정보를 보려면 [코맨드 참고 문서](../../ref/cli/wandb-job/wandb-job-create.md)를 참조하세요.
:::

## Git job

W&B Launch와 함께 원격 git 저장소의 특정 커밋, 브랜치, 태그에서 코드 및 다른 추적 자산들이 복제된 Git 기반의 job을 생성할 수 있습니다. `--uri` 또는 `-u` 플래그를 사용하여 코드가 포함된 URI를 지정하고, 선택적으로 `--build-context` 플래그로 서브디렉토리를 지정할 수 있습니다.

다음 코맨드를 사용하여 git 저장소에서 "hello world" job을 실행하세요:

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

이 코맨드는 다음을 수행합니다:
1. [W&B Launch job 저장소](https://github.com/wandb/launch-jobs)를 임시 디렉토리에 복제합니다.
2. **hello** 프로젝트에 **hello-world-git**이라는 job을 생성합니다. 이 job은 저장소의 기본 브랜치의 헤드 커밋과 연관되어 있습니다.
3. `jobs/hello_world` 디렉토리와 `Dockerfile.wandb`로부터 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `python job.py`를 실행합니다.

특정 브랜치나 커밋 해시로부터 job을 빌드하려면 `-g`, `--git-hash` 인수를 추가하세요. 모든 인수 목록을 보려면 `wandb launch --help`를 실행하세요.

### 원격 URL 형식

Launch job에 연관된 git 원격은 HTTPS 또는 SSH URL일 수 있습니다. URL 형식은 job 소스 코드를 가져오는 데 사용되는 프로토콜을 결정합니다.

| 원격 URL 유형 | URL 형식 | 엑세스 및 인증 요구 사항 |
| -------- | ------------- | --------------------------- |
| https    | `https://github.com/organization/repository.git`  | git 원격에 인증하기 위한 사용자 이름과 비밀번호 |
| ssh      | `git@github.com:organization/repository.git` | git 원격에 인증하기 위한 ssh 키 |

호스팅 제공자에 따라 정확한 URL 형식이 다를 수 있습니다. `wandb launch --uri`로 생성된 job은 제공된 `--uri`에서 지정된 전송 프로토콜을 사용합니다.

## Code artifact job

어느 소스 코드로든 W&B Artifact에 저장된 코드에서 job을 생성할 수 있습니다. 로컬 디렉토리를 사용하여 `--uri` 또는 `-u` 인수로 새로운 코드 artifact와 job을 생성하세요.

시작하려면 빈 디렉토리를 생성하고 다음 내용을 가진 Python 스크립트 `main.py`를 추가하세요:

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

다음 내용을 가진 `requirements.txt` 파일을 추가하세요:

```txt
wandb>=0.17.1
```

코드 artifact로 디렉토리를 로그하고 다음 명령어로 job을 실행하세요:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

이 명령어는 다음을 수행합니다:
1. 현재 디렉토리를 `hello-world-code`라는 코드 artifact로 기록합니다.
2. `launch-quickstart` 프로젝트에 `hello-world-code`라는 job을 생성합니다.
3. 현재 디렉토리와 Launch의 기본 Dockerfile로부터 컨테이너 이미지를 빌드합니다. 기본 Dockerfile은 `requirements.txt` 파일을 설치하고 진입점을 `python main.py`로 설정합니다.

## 이미지 job

또는, 미리 생성된 Docker 이미지를 기반으로 job을 빌드할 수 있습니다. 이것은 ML 코드의 기존 빌드 시스템이 이미 있거나, job의 코드 또는 요구 사항을 조정할 필요는 없지만 하이퍼파라미터 또는 다른 인프라 규모로 실험해보고 싶을 때 유용합니다.

이미지는 Docker 레지스트리에서 가져와 지정된 진입점 또는 지정되지 않은 경우 기본 진입점과 함께 실행됩니다. Docker 이미지에서 full 이미지 태그를 `--docker-image` 옵션에 전달하여 Docker 이미지에서 job을 생성하고 실행하세요.

다음 코맨드를 사용하여 간단한 job을 미리 만들어진 이미지에서 실행하세요:

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 자동 job 생성

W&B는 Launch로 생성되지 않은 run이라도 추적된 소스 코드가 있는 모든 run에 대해 자동으로 job을 생성하고 추적합니다. 다음 세 가지 조건 중 하나라도 만족하면 run은 소스 코드가 추적되었다고 간주됩니다:
- run에는 관련된 git 원격 및 커밋 해시가 있습니다
- run은 코드 artifact를 기록했습니다 (자세한 정보는 [`Run.log_code`](../../ref/python/run.md#log_code)를 참조하세요)
- `WANDB_DOCKER` 환경 변수에 이미지 태그가 설정된 Docker 컨테이너에서 실행되었습니다

Git 원격 URL은 W&B run에 의해 Launch job이 자동으로 생성될 때 로컬 git 저장소에서 유추됩니다.

### Launch job 이름

기본적으로, W&B는 자동으로 job 이름을 생성합니다. 이름은 job이 생성된 방식에 따라(GitHub, 코드 artifact, Docker 이미지) 생성됩니다. 대신에, Launch job의 이름을 환경 변수나 W&B Python SDK로 정의할 수 있습니다.

다음 표는 job 원천에 따른 기본 사용되는 명명 규칙을 설명합니다:

| 출처        | 명명 규칙                                    |
| ------------| ------------------------------------------- |
| GitHub      | `job-<git-remote-url>-<path-to-script>`     |
| 코드 artifact| `job-<code-artifact-name>`                  |
| Docker 이미지| `job-<image-name>`                          |

환경 변수나 W&B Python SDK로 job의 이름을 지정하세요

<Tabs
defaultValue="env_var"
values={[
{label: '환경 변수', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

선호하는 job 이름으로 `WANDB_JOB_NAME` 환경 변수를 설정하세요. 예를 들어:

```bash
WANDB_JOB_NAME=awesome-job-name
```

  </TabItem>
  <TabItem value="python_sdk">

`wandb.Settings`로 job의 이름을 정의하세요. 그런 다음 이 오브젝트를 `wandb.init`로 W&B를 초기화할 때 전달하세요. 예를 들어:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
Docker 이미지 job의 경우, 버전 에일리어스는 자동으로 job에 에일리어스로 추가됩니다.
:::

## 컨테이너화

Job은 컨테이너에서 실행됩니다. 이미지 job은 미리 빌드된 Docker 이미지를 사용하고, Git 및 코드 artifact job은 컨테이너 빌드 단계가 필요합니다.

Job 컨테이너화는 `wandb launch` 인수와 job 소스 코드 내 파일을 통해 사용자 정의할 수 있습니다.

### 빌드 컨텍스트

빌드 컨텍스트라는 용어는 Docker 데몬에 컨테이너 이미지를 빌드하기 위해 전송되는 파일과 디렉토리의 트리를 의미합니다. 기본적으로, Launch는 job 소스 코드의 루트를 빌드 컨텍스트로 사용합니다. 서브디렉토리를 빌드 컨텍스트로 지정하려면 job을 생성하고 실행할 때 `wandb launch`의 `--build-context` 인수를 사용하세요.

:::tip
`--build-context` 인수는 여러 프로젝트가 있는 모노레포에 참조하는 Git job에서 작업할 때 특히 유용합니다. 서브디렉토리를 빌드 컨텍스트로 지정하여 모노레포 내 특정 프로젝트에 대한 컨테이너 이미지를 빌드할 수 있습니다.

공식 W&B Launch job 저장소와 함께 `--build-context` 인수를 사용하는 방법에 대한 예는 [위의 예제](#git-jobs)를 참조하세요.
:::

### Dockerfile

Dockerfile은 Docker 이미지를 빌드하기 위한 지침을 포함한 텍스트 파일입니다. 기본적으로, Launch는 `requirements.txt` 파일을 설치하는 기본 Dockerfile을 사용합니다. 사용자 지정 Dockerfile을 사용하려면 `wandb launch`의 `--dockerfile` 인수로 파일 경로를 지정하세요.

Dockerfile 경로는 빌드 컨텍스트를 기준으로 지정됩니다. 예를 들어, 빌드 컨텍스트가 `jobs/hello_world`이고 Dockerfile이 `jobs/hello_world` 디렉토리에 위치한다면 `--dockerfile` 인수는 `Dockerfile.wandb`로 설정해야 합니다. 공식 W&B Launch job 저장소와 함께 `--dockerfile` 인수를 사용하는 방법에 대한 예는 [위의 예제](#git-jobs)를 참조하세요.

### 요구 사항 파일

사용자 지정 Dockerfile이 제공되지 않으면 Launch는 설치할 Python 종속성을 빌드 컨텍스트에서 찾습니다. 빌드 컨텍스트의 루트에서 `requirements.txt` 파일이 발견되면 Launch는 파일에 나열된 종속성을 설치합니다. 그렇지 않으면 `pyproject.toml` 파일이 발견되면 `project.dependencies` 섹션에서 종속성을 설치합니다.