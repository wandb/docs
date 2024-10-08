---
title: Tutorial: W&B Launch basics
description: W&B Launch 시작 가이드.
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Launch 워크플로우의 기초를 설명하는 페이지입니다.

:::tip
W&B Launch는 컨테이너에서 기계학습 작업을 실행합니다. 컨테이너에 대한 기본적인 이해가 필수는 아니지만 이 과정을 이해하는 데 도움이 될 수 있습니다. 컨테이너에 대한 기본적인 내용은 [Docker 문서](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/)를 참조하세요.
:::

## 전제 조건

시작하기 전에 다음의 전제 조건을 충족했는지 확인하세요:

1. https://wandb.ai/site에서 계정을 등록하고 W&B 계정에 로그인하세요.
2. 이 워크스루에는 Docker CLI 및 엔진이 작동하는 컴퓨터에 대한 터미널 엑세스가 필요합니다. 자세한 내용은 [Docker 설치 가이드](https://docs.docker.com/engine/install/)를 참조하세요.
3. W&B Python SDK 버전 `0.17.1` 이상을 설치하세요:
```bash
pip install wandb>=0.17.1
```
4. 터미널 내에서 `wandb login`을 실행하거나 `WANDB_API_KEY` 환경 변수를 설정하여 W&B에 인증하세요.

    <Tabs
    defaultValue="login"
    values={[
        {label: 'W&B에 로그인', value: 'login'},
        {label: '환경 키', value: 'apikey'},
    ]}>
    <TabItem value="login">
    터미널 내에서 다음을 실행하세요:
    
    ```bash
    wandb login
    ```

    </TabItem>
    <TabItem value="apikey">

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>`를 W&B API 키로 대체하세요.

    </TabItem>
    </Tabs>

## Launch 작업 생성하기
Docker 이미지, git 리포지토리, 또는 로컬 소스 코드를 사용하여 [launch 작업](./launch-terminology.md#launch-job)을 생성할 수 있습니다:

<Tabs
  defaultValue="image"
  values={[
    {label: '이미지 기반 작업', value: 'image'},
    {label: 'Git 기반 작업', value: 'git'},
    {label: '코드 기반 작업', value: 'local'},
  ]}>
  <TabItem value="image">

W&B에 메시지를 로그하는 미리 만들어진 컨테이너를 실행하려면, 터미널을 열고 다음 코맨드를 실행하세요:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

위의 코맨드는 컨테이너 이미지 `wandb/job_hello_world:main`을 다운로드하고 실행합니다.

Launch는 컨테이너를 구성하여 W&B 및 `launch-quickstart` 프로젝트에 로그된 모든 것을 보고합니다. 컨테이너는 W&B에 메시지를 로그하고 W&B에서 새로 생성된 run에 대한 링크를 표시합니다. 링크를 클릭하여 W&B UI에서 run을 확인하세요.

  </TabItem>
  <TabItem value="git">

[W&B Launch jobs 리포지토리](https://github.com/wandb/launch-jobs)의 소스 코드에서 동일한 hello-world 작업을 실행하려면, 다음 코맨드를 실행하세요:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```

이 코맨드는 다음을 수행합니다:
1. [W&B Launch jobs 리포지토리](https://github.com/wandb/launch-jobs)를 임시 디렉토리로 클론합니다.
2. **hello** 프로젝트에 **hello-world-git**이라는 이름으로 작업을 생성합니다. 이 작업은 스크립트를 실행하는 데 사용되는 정확한 소스 코드와 설정을 추적합니다.
3. `jobs/hello_world` 디렉토리와 `Dockerfile.wandb`에서 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `job.py` 파이썬 스크립트를 실행합니다.

콘솔 출력은 이미지 빌드 및 실행을 보여줍니다. 컨테이너 출력은 이전 예제와 거의 동일해야 합니다.

  </TabItem>
  <TabItem value="local">

Git 리포지토리에 버전 관리를 하지 않은 코드는 로컬 디렉토리 경로를 `--uri` 인수로 지정하여 시작할 수 있습니다.

빈 디렉토리를 생성하고 다음 내용을 가진 `train.py`라는 이름의 파이썬 스크립트를 추가하세요:

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

다음 내용을 가진 `requirements.txt` 파일을 추가하세요:

```text
wandb>=0.17.1
```

디렉토리 내에서 다음 명령어를 실행하세요:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

이 명령어는 다음과 같은 작업을 수행합니다:
1. 현재 디렉토리의 내용을 W&B에 코드 아티팩트로 로그합니다.
2. **launch-quickstart** 프로젝트에 **hello-world-code**라는 이름의 작업을 생성합니다.
3. `train.py` 및 `requirements.txt`를 기본 이미지에 복사하고 `pip install`하여 요구 사항을 설치하여 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `python train.py`를 실행합니다.

  </TabItem>
</Tabs>

## 큐 생성하기

Launch는 Teams가 공유 컴퓨팅 자원을 활용하여 워크플로우를 구축할 수 있도록 설계되었습니다. 지금까지의 예제에서는 `wandb launch` 명령어가 로컬 머신에서 동기적으로 컨테이너를 실행했습니다. Launch 큐와 에이전트는 공유 자원에서 작업을 비동기적으로 실행할 수 있도록 하며 우선순위 지정 및 하이퍼파라미터 최적화와 같은 고급 기능을 지원합니다. 기본 큐를 생성하려면 다음 단계를 따르세요:

1. [wandb.ai/launch](https://wandb.ai/launch)로 이동하여 **Create a queue** 버튼을 클릭하세요.
2. 큐와 연결할 **Entity**를 선택하세요.
3. **Queue name**을 입력하세요.
4. **Resource**로 **Docker**를 선택하세요.
5. **Configuration**은 현재 비워 두세요.
6. **Create queue** 버튼을 클릭하세요 :rocket:

버튼을 클릭하면 브라우저가 큐 보기의 **Agents** 탭으로 리디렉션됩니다. 에이전트가 폴링을 시작할 때까지 큐는 **Not active** 상태로 유지됩니다.

![](/images/launch/create_docker_queue.gif)

고급 큐 설정 옵션은 [고급 큐 설정 페이지](./setup-queue-advanced.md)를 참조하세요.

## 에이전트를 큐에 연결하기

큐 보기에서 큐에 폴링 에이전트가 없으면 화면 상단의 빨간색 배너에 **Add an agent** 버튼이 표시됩니다. 버튼을 클릭하면 에이전트를 실행할 명령어를 복사할 수 있습니다. 명령어는 다음과 같이 보입니다:

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

터미널에서 명령어를 실행하여 에이전트를 시작하세요. 에이전트는 지정된 큐에서 실행할 작업을 폴링합니다. 작업을 받으면 에이전트는 컨테이너 이미지를 다운로드하거나 빌드하고 로컬에서 `wandb launch` 명령어를 실행한 것처럼 작동합니다.

[Launch 페이지](https://wandb.ai/launch)로 돌아가서 큐가 **Active**로 표시되는지 확인하세요.

## 큐에 작업 제출하기

W&B 계정의 새 **launch-quickstart** 프로젝트로 이동하여 화면 왼쪽의 탐색 메뉴에서 작업 탭을 엽니다.

**Jobs** 페이지에는 이전에 실행된 run에서 생성된 W&B Jobs 목록이 표시됩니다. Launch 작업을 클릭하여 소스 코드, 종속성 및 작업에서 생성된 run를 확인하세요. 이 워크스루를 완료하면 목록에 세 개의 작업이 나타납니다.

새 작업 중 하나를 선택하고 큐에 제출하려면 다음 지침을 따르세요:

1. **Launch** 버튼을 클릭하여 작업을 큐에 제출하세요. **Launch** 서랍이 나타납니다.
2. 이전에 생성한 **Queue**를 선택하고 **Launch**를 클릭하세요.

이 작업은 큐에 제출됩니다. 이 큐를 폴링하는 에이전트는 작업을 수신하고 실행합니다. 작업의 진행 상황은 W&B UI에서 확인하거나 터미널에서 에이전트 출력을 검사하여 모니터링할 수 있습니다.

`wandb launch` 명령어는 `--queue` 인수를 지정하여 작업을 큐에 직접 푸시할 수 있습니다. 예를 들어, hello-world 컨테이너 작업을 큐에 제출하려면 다음 명령어를 실행하세요:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```