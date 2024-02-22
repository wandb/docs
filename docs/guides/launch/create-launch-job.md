---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 런치 작업 생성하기
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

작업은 W&B 실행에서 생성된 컨텍스트 정보를 포함하는 청사진입니다. 예를 들면 실행의 소스 코드, 소프트웨어 의존성, 하이퍼파라미터, 아티팩트 버전 등이 있습니다.

런치 작업을 만들면 사전에 구성된 [런치 큐](./launch-terminology.md#launch-queue)에 추가할 수 있습니다. 여러분이나 팀의 누군가가 배포한 런치 에이전트는 해당 큐를 폴링하고 런치 큐에 구성된 컴퓨트 리소스로 작업(도커 이미지로)을 전송합니다.

런치 작업을 생성하는 세 가지 방법은 다음과 같습니다:

- [파이썬 스크립트로](#create-a-job-with-a-wb-artifact)
- [도커 이미지로](#create-a-job-with-a-docker-image)
- [Git 저장소로](#create-a-job-with-git)

다음 섹션에서는 각 사용 사례를 기반으로 작업을 생성하는 방법을 보여줍니다.

## 시작하기 전에

런치 작업을 생성하기 전에, 큐의 이름과 속한 엔티티를 확인하세요. 그런 다음, 다음 지침을 따라 큐의 상태를 확인하고 에이전트가 해당 큐를 폴링하는지 확인하세요:

1. [wandb.ai/launch](https://wandb.ai/launch)로 이동합니다.
2. **모든 엔티티** 드롭다운에서 런치 큐가 속한 엔티티를 선택합니다.
3. 필터링된 결과에서 큐가 존재하는지 확인합니다.
4. 런치 큐 오른쪽으로 마우스를 이동하고 `큐 보기`를 선택합니다.
5. **에이전트** 탭을 선택합니다. **에이전트** 탭에서 에이전트 ID와 그 상태의 목록을 볼 수 있습니다. 에이전트 ID 중 하나가 **폴링** 상태인지 확인하세요.

## W&B 아티팩트로 작업 생성하기

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'Python SDK', value: 'sdk'}
]}>
<TabItem value="cli">

W&B CLI로 런치 작업을 생성합니다.

여러분의 파이썬 스크립트가 있는 경로에 코드를 실행하는 데 필요한 파이썬 의존성이 포함된 `requirements.txt` 파일이 있는지 확인하세요. 파이썬 런타임이 필요합니다. 파이썬 런타임은 `runtime.txt` 또는 `.python-version 파일`에서 자동 감지되거나 런타임 파라미터로 수동으로 지정할 수 있습니다.

다음 코드 조각을 복사하여 붙여넣고 `"<>"` 안의 값을 여러분의 사용 사례에 맞게 대체하세요:

```bash
wandb job create --project "<project-name>" -e "<your-entity>" \
--name "<name-for-job>" code "<path-to-script/code.py>"
```

사용할 수 있는 플래그의 전체 목록은 [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md) 명령 문서를 참조하세요.

:::note
W&B CLI로 런치 작업을 생성할 때 파이썬 스크립트 내에서 [`run.log_code()`](../../ref/python/run.md#log_code) 함수를 사용할 필요가 없습니다.
:::

  </TabItem>
  <TabItem value="sdk">

코드를 아티팩트로 로깅하여 런치 작업을 생성합니다. 이를 위해, 코드를 실행과 함께 아티팩트로 로깅하세요 [`run.log_code()`](../../ref/python/run.md#log_code).

다음 샘플 파이썬 코드는 파이썬 스크립트에 `run.log_code()` 함수를 통합하는 방법을 보여줍니다(하이라이트된 부분 참조).

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
        # 하이퍼파라미터 추적 시뮬레이션
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # 학습 실행 시뮬레이션
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})

    # highlight-next-line
    run.log_code()
    run.finish()


run_training_run(epochs=10, lr=0.01)
```

`WANDB_JOB_NAME` 환경 변수로 작업의 이름을 지정할 수 있습니다. 또는 `wandb.Settings`에 `job_name` 파라미터를 설정하고 `wandb.init`에 전달함으로써 이름을 지정할 수 있습니다. 예를 들면:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

이름을 지정하지 않으면 W&B가 자동으로 런치 작업 이름을 생성해줍니다. 작업 이름은 다음과 같은 형식으로 생성됩니다: `job-<code-artifact-name>`.

[`run.log_code()`](../../ref/python/run.md#log_code) 명령에 대한 자세한 정보는 [API 참조 가이드](../../ref/README.md)를 참조하세요.

  </TabItem>
</Tabs>

## 도커 이미지로 작업 생성하기

W&B CLI나 도커 컨테이너를 생성하여 도커 이미지로 작업을 생성합니다. 이미지 기반 작업을 생성하려면 먼저 도커 이미지를 생성해야 합니다. 도커 이미지에는 W&B 실행을 수행하는 데 필요한 소스 코드(예: Dockerfile, requirements.txt 파일 등)가 포함되어야 합니다.

예를 들어, 다음과 같은 디렉터리 구조를 가진 [`fashion_mnist_train`](https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train)이라는 디렉터리가 있다고 가정해 보겠습니다:

```
fashion_mnist_train
│   data_loader.py
│   Dockerfile
│   job.py
│   requirements.txt
└───configs
│   │   example.yml
```

`docker build` 명령으로 `fashion-mnist`라는 도커 이미지를 생성할 수 있습니다:

```bash
docker build . -t fashion-mnist
```

도커 이미지를 빌드하는 방법에 대한 자세한 정보는 [도커 빌드 참조 문서](https://docs.docker.com/engine/reference/commandline/build/)를 참조하세요.

<Tabs
defaultValue="cli"
values={[
{label: 'W&B CLI', value: 'cli'},
{label: 'Docker run', value: 'build'},
]}>
<TabItem value="cli">

W&B CLI로 런치 작업을 생성합니다. 다음 코드 조각을 복사하여 붙여넣고 `"<>"` 안의 값을 여러분의 사용 사례에 맞게 대체하세요:

```bash
wandb job create --project "<project-name>" --entity "<your-entity>" \
--name "<name-for-job>" image image-name:tag
```

사용할 수 있는 플래그의 전체 목록은 [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md) 명령 문서를 참조하세요.

  </TabItem>
  <TabItem value="build">

도커 이미지와 연결된 실행을 생성합니다. W&B는 `WANDB_DOCKER` 환경 변수에서 이미지 태그를 찾고, `WANDB_DOCKER`가 설정되면 지정된 이미지 태그에서 런치 작업을 생성합니다. `WANDB_DOCKER` 환경 변수가 전체 이미지 태그로 설정되었는지 확인하세요.

도커 이미지에서 도커 컨테이너를 빌드하여 런치 작업을 생성합니다. 다음 코드 조각을 복사하여 붙여넣고 `"<>"` 안의 값을 여러분의 사용 사례에 맞게 대체하세요:

```bash
docker run -e WANDB_PROJECT="<project-name>" \
-e WANDB_ENTITY="<your-entity>" \
-e WANDB_API_KEY="<your-w&B-api-key>" \
-e WANDB_DOCKER="<docker-image-name>" image:tag
```

`WANDB_JOB_NAME` 환경 변수로 작업의 이름을 지정할 수 있습니다. 이름을 지정하지 않으면 W&B가 자동으로 런치 작업 이름을 생성해줍니다. W&B는 다음과 같은 형식으로 작업 이름을 지정합니다: `job-<image>-<name>`.

:::tip
전체 이미지 태그로 설정되었는지 확인하세요. 예를 들어, 에이전트가 ECR 저장소에서 이미지를 실행하는 경우, `WANDB_DOCKER`를 ECR 저장소 URL을 포함한 전체 이미지 태그로 설정해야 합니다: `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:develop`. 이 경우 도커 태그 `'develop'`은 결과적으로 생성된 작업에 별칭으로 추가됩니다.
:::

  </TabItem>
</Tabs>

## Git으로 작업 생성하기

W&B 런치로 Git 기반 작업을 생성합니다. 코드와 기타 자산은 git 저장소의 특정 커밋, 브랜치 또는 태그에서 클론됩니다.

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

브랜치나 커밋 해시에서 빌드하려면 `-g` 인수를 추가하세요.

  </TabItem>
  <TabItem value="git">

여러분의 파이썬 스크립트가 있는 경로에 코드를 실행하는 데 필요한 파이썬 의존성이 포함된 `requirements.txt` 파일이 있는지 확인하세요. 파이썬 런타임이 필요합니다. 파이썬 런타임은 `runtime.txt` 또는 `.python-version 파일`에서 자동 감지되거나 런타임 파라미터로 수동으로 지정할 수 있습니다.

`WANDB_JOB_NAME` 환경 변수로 작업의 이름을 지정할 수 있습니다. 이름을 지정하지 않으면 W&B가 자동으로 런치 작업 이름을 생성해줍니다. 이 경우, W&B는 다음과 같은 형식으로 작업 이름을 지정합니다: `job-<git-remote-url>-<path-to-script>`.

</TabItem>
</Tabs>

### Git 원격 URL 처리

런치 작업과 연결된 Git 원격은 HTTPS 또는 SSH URL일 수 있습니다. Git 원격 URL은 일반적으로 다음 형식을 사용합니다:

- `https://github.com/organization/repository.git` (HTTPS)
- `git@github.com:organization/repository.git` (SSH)

정확한 형식은 git 호스팅 제공업체에 따라 다릅니다.

원격 URL 형식은 git 원격에 접근하고 인증하는 방법을 결정하기 때문에 중요합니다. 다음 표는 접근 및 인증을 위해 충족해야 하는 요구 사항을 설명합니다:

| 원격 URL | 접근 및 인증을 위한 요구 사항 |
| ---------- | ------------------------------------------ |
| HTTPS URL  | git 원격에 인증하기 위한 사용자 이름과 비밀번호 |
| SSH URL    | git 원격에 인증하기 위한 SSH 키 |


런치 작업이 W&B 실행에 의해 자동으로 생성되는 경우 Git 원격 URL은 로컬 git 저장소에서 자동으로 추론됩니다.

직접 작업을 생성하는 경우, 원하는 전송 프로토콜에 대한 URL을 제공하는 책임이 있습니다.

## 런치 작업 이름

기본적으로, W&B는 작업 이름을 자동으로 생성해줍니다. 이름은 작업이 생성되는 방식(GitHub, 코드 아티팩트, 또는 도커 이미지)에 따라 생성됩니다. 대안적으로, 환경 변수나 W&B Python SDK를 사용하여 런치 작업의 이름을 정의할 수 있습니다.

### 기본 런치 작업 이름

다음 표는 작업 소스에 따라 기본적으로 사용되는 작업 명명 규칙을 설명합니다:

| 소스        | 명명 규칙                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| 코드 아티팩트 | `job-<code-artifact-name>`              |
| 도커 이미지  | `job-<image-name>`                      |

### 런치 작업 이름 지정하기

W&B 환경 변수나 W&B Python SDK로 작업 이름을 지정하세요

<Tabs
defaultValue="env_var"
values={[
{label: '환경 변수', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

`WANDB_JOB_NAME` 환경 변수를 선호하는 작업 이름으로 설정하세요. 예를 들면:

```bash
WANDB_JOB_NAME=awesome-job-name
```

  </TabItem>
  <TabItem value="python_sdk">

`wandb.Settings`로 작업 이름을 정의하세요. 그리고 이 객체를 `wandb.init`을 초기화할 때 전달하세요. 예를 들면:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
도커 이미지 작업의 경우, 버전 별칭이 자동으로 작업에 별칭으로 추가됩니다.
:::