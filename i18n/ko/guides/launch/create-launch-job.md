---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 런치 작업 생성하기
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

작업은 W&B run에서 생성된 컨텍스트 정보를 포함하는 블루프린트입니다; 예를 들면, run의 소스 코드, 소프트웨어 의존성, 하이퍼파라미터, 아티팩트 버전 등을 포함합니다.

런치 작업을 가지고 있다면, 사전에 구성된 [런치 큐](./launch-terminology.md#launch-queue)에 추가할 수 있습니다. 당신이나 팀의 누군가가 배치한 런치 에이전트가 해당 큐를 폴링하고 작업(도커 이미지로)을 런치 큐에서 구성된 컴퓨트 리소스로 전송합니다.

런치 작업을 생성하는 세 가지 방법은 다음과 같습니다:

- [파이썬 스크립트로](#create-a-job-with-a-wb-artifact)
- [Docker 이미지로](#create-a-job-with-a-docker-image)
- [Git 저장소로](#create-a-job-with-git)

다음 섹션에서는 각 유스 케이스를 기반으로 작업을 생성하는 방법을 보여줍니다.

## 시작하기 전에

런치 작업을 생성하기 전에, 큐의 이름과 소속된 엔티티를 알아내세요. 그런 다음, 다음 지침을 따라 큐의 상태를 확인하고 에이전트가 해당 큐를 폴링하는지 확인하세요:

1. [wandb.ai/launch](https://wandb.ai/launch)로 이동하세요.
2. **모든 엔티티** 드롭다운에서 런치 큐가 속한 엔티티를 선택하세요.
3. 필터링된 결과에서 큐가 존재하는지 확인하세요.
4. 런치 큐의 오른쪽으로 마우스를 옮겨 `큐 보기`를 선택하세요.
5. **에이전트** 탭을 선택하세요. **에이전트** 탭에서는 에이전트 ID와 그 상태의 목록을 볼 수 있습니다. 하나의 에이전트 ID가 **폴링** 상태인지 확인하세요.

## W&B 아티팩트로 작업 생성하기

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'Python SDK', value: 'sdk'}
]}>
<TabItem value="cli">

W&B CLI로 런치 작업을 생성하세요.

파이썬 스크립트가 있는 경로에 코드를 실행하는 데 필요한 파이썬 의존성이 있는 `requirements.txt` 파일이 있는지 확인하세요. 파이썬 런타임도 필요합니다. 파이썬 런타임은 `runtime.txt` 또는 `.python-version 파일`에서 자동 감지되거나 런타임 파라미터로 수동으로 지정할 수 있습니다.

다음 코드 조각을 복사하고 붙여넣고 `"<>"` 안의 값을 귀하의 유스 케이스에 맞게 교체하세요:

```bash
wandb job create --project "<project-name>" -e "<your-entity>" \
--name "<name-for-job>" code "<path-to-script/code.py>"
```

사용할 수 있는 플래그의 전체 목록은 [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md) 코맨드 문서를 참조하세요.

:::note
W&B CLI로 런치 작업을 생성할 때 파이썬 스크립트 내에서 [`run.log_code()`](../../ref/python/run.md#log_code) 함수를 사용할 필요가 없습니다.
:::

  </TabItem>
  <TabItem value="sdk">

아티팩트로 코드를 로그하여 런치 작업을 생성하세요. 이렇게 하려면, 코드를 run에 아티팩트로 로그하세요 [`run.log_code()`](../../ref/python/run.md#log_code)로.

다음 샘플 파이썬 코드는 `run.log_code()` 함수(강조 표시된 부분 참조)를 파이썬 스크립트에 통합하는 방법을 보여줍니다.

```python title="create_simple_job.py"
import random
import wandb


def run_training_run(epochs, lr):
    settings = wandb.Settings(job_source="artifact")
    run = wandb.init(
        project="launch_demo",
        job_type="eval",
        settings=settings,
        entity="<your-entity>",
        # 하이퍼파라미터 트래킹 시뮬레이션
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # 트레이닝 run 시뮬레이션
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})

    # highlight-next-line
    run.log_code()
    run.finish()


run_training_run(epochs=10, lr=0.01)
```

`WANDB_JOB_NAME` 환경 변수로 작업의 이름을 지정할 수 있습니다. 또는 `wandb.Settings`에 `job_name` 파라미터를 설정하고 `wandb.init`에 전달할 수도 있습니다. 예를 들면:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

이름을 지정하지 않으면 W&B가 자동으로 런치 작업 이름을 생성해줍니다. 작업 이름은 다음과 같이 형식화됩니다: `job-<code-artifact-name>`.

[`run.log_code()`](../../ref/python/run.md#log_code) 명령에 대한 자세한 정보는 [API 참조 가이드](../../ref/README.md)를 참조하세요.

  </TabItem>
</Tabs>

## Docker 이미지로 작업 생성하기

W&B CLI 또는 이미지에서 Docker 컨테이너를 생성하여 Docker 이미지로 작업을 생성하세요. 이미지 기반 작업을 생성하려면 먼저 Docker 이미지를 생성해야 합니다. Docker 이미지는 W&B run을 실행하는 데 필요한 소스 코드(예: Dockerfile, requirements.txt 파일 등)를 포함해야 합니다.

예를 들어, 다음과 같은 디렉토리 구조를 가진 [`fashion_mnist_train`](https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train) 디렉토리가 있다고 가정해 봅시다:

```
fashion_mnist_train
│   data_loader.py
│   Dockerfile
│   job.py
│   requirements.txt
└───configs
│   │   example.yml
```

`docker build` 코맨드로 `fashion-mnist`라는 Docker 이미지를 생성할 수 있습니다:

```bash
docker build . -t fashion-mnist
```

Docker 이미지를 빌드하는 방법에 대한 자세한 정보는 [Docker build 참조 문서](https://docs.docker.com/engine/reference/commandline/build/)를 참조하세요.

<Tabs
defaultValue="cli"
values={[
{label: 'W&B CLI', value: 'cli'},
{label: 'Docker run', value: 'build'},
]}>
<TabItem value="cli">

W&B CLI로 런치 작업을 생성하세요. 다음 코드 조각을 복사하고 `"<>"` 안의 값을 귀하의 유스 케이스에 맞게 교체하세요:

```bash
wandb job create --project "<project-name>" --entity "<your-entity>" \
--name "<name-for-job>" image image-name:tag
```

사용할 수 있는 플래그의 전체 목록은 [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md) 코맨드 문서를 참조하세요.

  </TabItem>
  <TabItem value="build">

런과 Docker 이미지를 연결하세요. W&B는 `WANDB_DOCKER` 환경 변수에서 이미지 태그를 찾고, `WANDB_DOCKER`가 설정되어 있으면 지정된 이미지 태그에서 런치 작업을 생성합니다. `WANDB_DOCKER` 환경 변수가 전체 이미지 태그로 설정되어 있는지 확인하세요.

Docker 이미지에서 Docker 컨테이너를 빌드하여 런치 작업을 생성하세요. 다음 코드 조각을 복사하고 `"<>"` 안의 값을 귀하의 유스 케이스에 맞게 교체하세요:

```bash
docker run -e WANDB_PROJECT="<project-name>" \
-e WANDB_ENTITY="<your-entity>" \
-e WANDB_API_KEY="<your-w&B-api-key>" \
-e WANDB_DOCKER="<docker-image-name>" image:tag
```

`WANDB_JOB_NAME` 환경 변수로 작업의 이름을 지정할 수 있습니다. 이름을 지정하지 않으면 W&B가 자동으로 런치 작업 이름을 생성해줍니다. W&B는 다음 형식으로 작업 이름을 지정합니다: `job-<image>-<name>`.

:::tip
ECR 저장소에서 이미지를 실행하는 에이전트의 경우, `WANDB_DOCKER`를 ECR 저장소 URL을 포함한 전체 이미지 태그로 설정해야 합니다: `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:develop`. 이 경우 도커 태그 `'develop'`는 결과 작업에 에일리어스로 추가됩니다.
:::

  </TabItem>
</Tabs>

## Git으로 작업 생성하기

W&B Launch로 Git 기반 작업을 생성하세요. 코드와 기타 자산은 특정 커밋, 브랜치 또는 git 저장소의 태그에서 복제됩니다.

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'Autogenerate from git commit', value: 'git'},
]}>
<TabItem value="cli">

```bash
wandb job create --project "<project-name>" --entity "<your-entity>" \ 
--name "<name-for-job>" git https://github.com/org-name/repo-name.git \ 
--entry-point "<path-to-script/code.py>"
```

브랜치 또는 커밋 해시에서 빌드하려면 `-g` 인수를 추가하세요.

  </TabItem>
  <TabItem value="git">

파이썬 스크립트가 있는 경로에 코드를 실행하는 데 필요한 파이썬 의존성이 있는 `requirements.txt` 파일이 있는지 확인하세요. 파이썬 런타임도 필요합니다. 파이썬 런타임은 `runtime.txt` 또는 `.python-version 파일`에서 자동 감지되거나 런타임 파라미터로 수동으로 지정할 수 있습니다.

`WANDB_JOB_NAME` 환경 변수로 작업의 이름을 지정할 수 있습니다. 이름을 지정하지 않으면 W&B가 자동으로 런치 작업 이름을 생성해줍니다. 이 경우, W&B는 다음 형식으로 작업 이름을 지정합니다: `job-<git-remote-url>-<path-to-script>`.

</TabItem>
</Tabs>

### Git 원격 URL 처리

런치 작업과 연결된 Git 원격은 HTTPS 또는 SSH URL일 수 있습니다. Git 원격 URL은 일반적으로 다음 형식을 사용합니다:

- `https://github.com/organization/repository.git` (HTTPS)
- `git@github.com:organization/repository.git` (SSH)

정확한 형식은 git 호스팅 제공업체에 따라 다릅니다.

원격 URL 형식은 git 원격에 액세스하고 인증하는 방법을 결정합니다. 다음 표는 액세스 및 인증을 위해 충족해야 하는 요구 사항을 설명합니다:

| 원격 URL | 액세스 및 인증 요구 사항 |
| ---------- | ------------------------------------------ |
| HTTPS URL  | git 원격으로 인증하기 위한 사용자 이름과 비밀번호 |
| SSH URL    | git 원격으로 인증하기 위한 SSH 키 |

Git 원격 URL은 W&B run에 의해 자동으로 생성된 런치 작업의 경우 로컬 git 저장소에서 자동으로 추론됩니다. 

수동으로 작업을 생성하는 경우, 원하는 전송 프로토콜에 대한 URL을 제공하는 것은 당신의 책임입니다.

## 런치 작업 이름

기본적으로, W&B는 자동으로 작업 이름을 생성해줍니다. 이름은 작업 생성 방법(GitHub, 코드 아티팩트, 또는 Docker 이미지)에 따라 생성됩니다. 대안적으로, 환경 변수나 W&B Python SDK로 런치 작업의 이름을 정의할 수 있습니다.

### 기본 런치 작업 이름

다음 표는 작업 소스에 따라 기본적으로 사용되는 작업 명명 규칙을 설명합니다:

| 소스        | 명명 규칙                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| 코드 아티팩트 | `job-<code-artifact-name>`              |
| Docker 이미지  | `job-<image-name>`                      |

### 런치 작업 이름 지정하기

W&B 환경 변수나 W&B Python SDK로 작업 이름을 지정하세요

<Tabs
defaultValue="env_var"
values={[
{label: '환경 변수', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

`WANDB_JOB_NAME` 환경 변수를 원하는 작업 이름으로 설정하세요. 예를 들면:

```bash
WANDB_JOB_NAME=awesome-job-name
```

  </TabItem>
  <TabItem value="python_sdk">

`wandb.Settings`로 작업 이름을 정의하세요. 그런 다음 이 객체를 `wandb.init`할 때 W&B에 전달하세요. 예를 들면:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
도커 이미지 작업의 경우, 버전 에일리어스는 자동으로 작업에 에일리어스로 추가됩니다.
:::