---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 가이드

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>


이 가이드는 W&B 런치의 기본 구성요소인 **런치 작업**, **런치 큐**, **런치 에이전트** 설정 방법을 안내합니다. 이 가이드를 마치면 다음을 할 수 있게 됩니다:

1. 신경망을 훈련시키는 런치 작업을 생성합니다.
2. 로컬 머신에서 작업을 실행하기 위해 사용되는 런치 큐를 생성합니다.
3. 큐를 폴링하고 Docker로 런치 작업을 시작하는 런치 에이전트를 생성합니다.

:::note
이 페이지에 설명된 가이드는 Docker가 설치된 로컬 머신에서 실행됩니다.
:::

## 시작하기 전에

시작하기 전에 다음 사전 요구사항을 충족했는지 확인하세요:
1. W&B Python SDK 버전 0.14.0 이상을 설치합니다:
    ```bash
    pip install wandb>=0.14.0
    ```
2. https://wandb.ai/site 에서 무료 계정을 등록한 다음, W&B 계정에 로그인합니다.
3. Docker를 설치합니다. Docker 설치 방법에 대한 자세한 내용은 [Docker 문서](https://docs.docker.com/get-docker/)를 참조하세요. docker 데몬이 머신에서 실행 중인지 확인하세요.

## 런치 작업 생성하기

[런치 작업](./launch-terminology#launch-job)은 W&B 런치에서 작업의 기본 단위입니다. 다음 코드는 W&B Python SDK를 사용하여 W&B [run](../../ref/python/run.md)에서 런치 작업을 생성합니다.

1. 다음 Python 코드를 `train.py`라는 파일에 복사하여 로컬 머신에 저장하세요. `<your entity>`를 귀하의 W&B 엔티티로 교체하세요.

    ```python title="train.py"
    import wandb

    config = {"epochs": 10}

    entity = "<your entity>"
    project = "launch-quickstart"
    job_name = "walkthrough_example"

    settings = wandb.Settings(job_name=job_name)

    with wandb.init(
        entity=entity, config=config, project=project, settings=settings
    ) as run:
        config = wandb.config
        for epoch in range(1, config.epochs):
            loss = config.epochs / epoch
            accuracy = (1 + (epoch / config.epochs)) / 2
            wandb.log({"loss": loss, "accuracy": accuracy, "epoch": epoch})

        # highlight-next-line
        wandb.run.log_code()
    ```

2. Python 스크립트를 실행하고 스크립트가 완료될 때까지 실행하세요:
    ```bash
    python train.py
    ```

이렇게 하면 런치 작업이 생성됩니다. 위 예제에서 런치 작업은 `launch-quickstart` 프로젝트에 생성되었습니다.

다음으로, 새로 생성된 런치 작업을 *런치 큐*에 추가합니다.

:::tip
런치 작업을 생성하는 방법은 여러 가지가 있습니다. 런치 작업을 생성하는 다양한 방법에 대해 알아보려면 [런치 작업 생성하기](./create-launch-job.md) 페이지를 참조하세요.
:::

## 런치 작업을 큐에 추가하기
런치 작업을 생성한 후에는 해당 작업을 [런치 큐](./launch-terminology.md#launch-queue)에 추가하세요. 다음 단계는 Docker 컨테이너를 [목표 리소스](./launch-terminology.md#target-resources)로 사용할 기본 런치 큐를 생성하는 방법을 설명합니다:


1. W&B 프로젝트로 이동하세요.
2. 왼쪽 패널(번개 모양 아이콘)에서 Jobs 탭을 선택하세요.
3. 생성한 작업 이름 옆에 마우스를 가져가서 **Launch** 버튼을 선택하세요.
4. 화면 오른쪽에서 서랍이 열립니다. 다음을 선택하세요:
    1. **Job version**: 런치할 작업의 버전입니다. 우리는 하나의 버전만 가지고 있으므로 기본 **@latest** 버전을 선택합니다.
    2. **Overrides**: 런치 작업의 입력에 대한 새로운 값을 설정합니다. 우리의 run은 `wandb.config`에 하나의 값인 `epochs`를 가지고 있습니다. 이 값을 overrides 필드 내에서 재정의할 수 있습니다. 이 가이드에서는 에포크 수를 그대로 둡니다.
    3. **Queue**: run을 런치할 큐입니다. 드롭다운에서 **'Starter' 큐 생성**을 선택합니다.

![](/images/launch/starter-launch.gif)
5. 작업을 구성한 후에는 화면 오른쪽 하단에 있는 **Launch now** 버튼을 클릭하여 런치 작업을 큐에 추가하세요.


:::tip
런치 큐 구성의 내용은 큐의 목표 리소스에 따라 달라집니다.
:::

## 런치 에이전트 시작하기
런치 작업을 실행하려면 작업이 추가된 런치 큐를 폴링할 [런치 에이전트](./launch-terminology.md#launch-agent)가 필요합니다. 런치 에이전트를 생성하고 시작하는 단계는 다음과 같습니다:

1. [wandb.ai/launch](https://wandb.ai/launch)에서 런치 큐 페이지로 이동하세요.
2. **Add agent** 버튼을 클릭하세요.
3. 모달이 나타나면 W&B CLI 코맨드가 표시됩니다. 이 코맨드를 복사하여 터미널에 붙여넣으세요.

![](/images/launch/activate_starter_queue_agent.png)

일반적으로 런치 에이전트를 시작하는 코맨드는 다음과 같습니다:

```bash
wandb launch-agent -e <entity-name> -q <queue-name>
```

터미널에서 에이전트가 큐를 폴링하기 시작하는 것을 볼 수 있습니다. 몇 초에서 1분 정도 기다리면 에이전트가 큐에 추가한 런치 작업을 실행하는 것을 볼 수 있습니다.

:::tip
런치 에이전트는 쿠버네티스 클러스터와 같은 비-로컬 환경에서도 큐를 폴링할 수 있습니다.
:::

## 런치 작업 보기

W&B 계정에서 새로운 **launch-quickstart** 프로젝트로 이동하여 화면 왼쪽의 탐색에서 jobs 탭을 엽니다.

![](/images/launch/jobs-tab.png)

**Jobs** 페이지에는 이전에 실행된 run으로부터 생성된 W&B Jobs의 목록이 표시됩니다. **job-source-launch-quickstart-train.py:v0**라는 이름의 작업을 볼 수 있습니다. 런치 작업을 클릭하여 소스 코드 의존성과 런치 작업에 의해 생성된 run 목록을 확인하세요.

:::tip
작업 이름을 조금 더 기억하기 쉽게 만들고 싶다면 jobs 페이지에서 작업 이름을 편집할 수 있습니다.
:::