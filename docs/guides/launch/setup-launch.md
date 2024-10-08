---
title: Launch quickstart
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

이 페이지는 W&B Launch 설정에 필요한 고급 단계들을 설명합니다:

1. **큐 설정하기**: 큐는 FIFO이며 큐 설정이 있습니다. 큐 설정은 작업이 실행되는 대상 리소스를 어디에서 어떻게 실행할지를 제어합니다.
2. **에이전트 설정하기**: 에이전트는 사용자의 머신/인프라에서 실행되며 하나 이상의 큐에서 실행 작업들을 폴링합니다. 작업이 가져와지면, 에이전트는 이미지가 빌드되고 사용 가능하도록 보장합니다. 그런 다음 에이전트는 작업을 대상 리소스에 제출합니다.

## 큐 설정하기
Launch 큐는 특정 대상 리소스에 대한 추가 설정과 함께 가리키도록 설정되어야 합니다. 예를 들어, Kubernetes 클러스터를 가리키는 Launch 큐는 환경 변수나 사용자 지정 네임스페이스를 큐 설정에 포함시킬 수 있습니다. 큐를 생성할 때, 사용하고자 하는 대상 리소스와 해당 리소스를 사용할 설정 양쪽 모두를 지정합니다.

에이전트가 큐에서 작업을 수신하면, 큐 설정도 함께 수신합니다. 에이전트가 작업을 대상으로 제출할 때, 작업 자체의 재정의를 포함한 큐 설정을 함께 포함합니다. 예를 들어, 작업 설정을 사용하여 해당 작업 인스턴스에만 Amazon SageMaker 인스턴스 유형을 지정할 수 있습니다. 이 경우, 일반적으로 최종 사용자 인터페이스로 [큐 설정 템플릿](./setup-queue-advanced.md#configure-queue-template)을 사용합니다.

### 큐 생성하기
1. [wandb.ai/launch](https://wandb.ai/launch)에서 Launch App으로 이동합니다.
2. 화면 우측 상단의 **create queue** 버튼을 클릭합니다.

![](/images/launch/create-queue.gif)

3. **Entity** 드롭다운 메뉴에서 큐가 속할 엔터티를 선택합니다.
4. **Queue** 필드에 큐의 이름을 입력합니다.
5. **Resource** 드롭다운에서 이 큐에 추가된 작업이 사용할 컴퓨팅 리소스를 선택합니다.
6. 이 큐에 대해 **Prioritization** 허용 여부를 선택합니다. 우선순위가 활성화되면 팀의 사용자가 작업의 실행 순위를 정의할 수 있습니다. 높은 우선순위를 가진 작업은 낮은 우선순위 작업보다 먼저 실행됩니다.
7. **Configuration** 필드에 JSON 또는 YAML 형식으로 리소스 구성을 제공합니다. 큐가 지시하는 리소스 유형에 따라 구성 문서의 구조와 의미가 달라집니다. 자세한 내용은 대상 리소스에 대한 전용 설정 페이지를 참조하세요.

## Launch 에이전트 설정하기
Launch 에이전트는 작업을 찾기 위해 하나 이상의 Launch 큐에서 작업을 폴링하는 장기 실행 프로세스입니다. Launch 에이전트는 첫 번째 입력, 첫 번째 출력(FIFO) 순서로 또는 우선순위 순서로 큐에서 작업을 제거하거나 작업을 제거합니다. 에이전트가 큐에서 작업을 꺼내면, 해당 작업을 위한 이미지를 빌드할 수 있습니다. 그런 다음 에이전트는 큐 설정에서 지정한 설정 옵션과 함께 작업을 대상 리소스에 제출합니다.

:::info
에이전트는 매우 유연하여 다양한 유스 케이스를 지원하도록 구성할 수 있습니다. 에이전트에 요구되는 설정은 특정 유스 케이스에 따라 다릅니다. [Docker](./setup-launch-docker.md), [Amazon SageMaker](./setup-launch-sagemaker.md), [Kubernetes](./setup-launch-kubernetes.md), 또는 [Vertex AI](./setup-vertex.md)에 대한 전용 페이지를 참조하세요.
:::

:::tip
W&B는 특정 사용자의 API 키보다 [서비스 계정](/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)의 API 키로 에이전트를 시작할 것을 권장합니다. 서비스 계정의 API 키를 사용하는 두 가지 이점이 있습니다:
1. 에이전트는 개별 사용자에게 의존하지 않습니다.
2. Launch를 통해 생성된 run과 관련된 작성자는 에이전트와 관련된 사용자가 아닌 Launch 작업을 제출한 사용자로 표시됩니다.
:::

### 에이전트 설정
`launch-config.yaml`이라는 YAML 파일로 Launch 에이전트를 설정하세요. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 설정 파일을 확인합니다. Launch 에이전트를 활성화할 때 다른 디렉토리를 선택적으로 지정할 수 있습니다.

Launch 에이전트의 설정 파일 내용은 Launch 에이전트의 환경, Launch 큐의 대상 리소스, Docker 빌더 요구사항, 클라우드 레지스트리 요구사항 등에 따라 달라집니다.

유스 케이스와 상관없이, Launch 에이전트에 대해 핵심적으로 설정할 수 있는 옵션은 다음과 같습니다:
* `max_jobs`: 에이전트가 병렬로 실행할 수 있는 최대 작업 수
* `entity`: 큐가 속한 엔터티
* `queues`: 에이전트가 감시할 하나 이상의 큐 이름

:::tip
Launch 에이전트에 대한 범용 설정 옵션을 W&B CLI를 사용하여 지정할 수 있습니다 (설정 YAML 파일 대신): 최대 작업 수, W&B 엔터티, Launch 큐. 자세한 내용은 [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) 명령어를 참조하세요.
:::

다음 YAML 코드조각은 핵심 Launch 에이전트 설정 키를 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
# 동시에 수행할 최대 run 수. -1 = 제한 없음
max_jobs: -1

entity: <entity-name>

# 폴링할 큐 목록
queues:
  - <queue-name>
```

### 컨테이너 빌더 설정
Launch 에이전트를 사용하면 이미지를 빌드할 수 있습니다. launch 작업을 git 리포지토리나 코드 아티팩트에서 생성하려는 경우, 컨테이너 빌더를 사용하도록 에이전트를 설정해야 합니다. Launch 작업을 만드는 방법에 대한 자세한 내용은 [Create a launch job](./create-launch-job.md)을 참조하세요.

W&B Launch는 세 가지 빌더 옵션을 지원합니다:

* Docker: Docker 빌더는 로컬 Docker 데몬을 사용하여 이미지를 빌드합니다.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko는 Docker 데몬이 없는 환경에서 이미지 빌딩을 가능하게 하는 Google 프로젝트입니다.
* Noop: 에이전트는 작업을 빌드하려고 하지 않으며, 미리 빌드된 이미지만 가져옵니다.

:::tip
에이전트가 Docker 데몬이 없는 환경(예: Kubernetes 클러스터)에서 폴링 중인 경우 Kaniko 빌더를 사용하세요.

Kaniko 빌더에 대한 자세한 내용은 [Set up Kubernetes](./setup-launch-kubernetes.md)를 참조하세요.
:::

이미지 빌더를 지정하려면, 에이전트 설정에 빌더 키를 포함하십시오. 예를 들어, 다음 코드조각은 Docker 또는 Kaniko를 사용하도록 지정하는 launch 설정 (`launch-config.yaml`)의 일부를 보여줍니다:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### 컨테이너 레지스트리 설정
일부 경우에 클라우드 레지스트리에 Launch 에이전트를 연결하고 싶을 수 있습니다. 클라우드 레지스트리에 Launch 에이전트를 연결하고 싶을 때 일반적인 시나리오는 다음과 같습니다:

* 빌드된 환경이 아닌 강력한 워크스테이션이나 클러스터 같은 환경에서 작업을 실행하고 싶을 때.
* 에이전트를 사용하여 이미지를 빌드하고 이 이미지를 Amazon SageMaker나 VertexAI에서 사용하려고 할 때.
* Launch 에이전트가 이미지 저장소에서 가져오기 위한 자격을 제공해주기 원할 때.

컨테이너 레지스트리와 상호 작용하도록 에이전트를 설정하는 방법에 대한 자세한 내용은 [Advanced agent set](./setup-agent-advanced.md) 페이지를 참조하세요.

## Launch 에이전트 활성화하기
`launch-agent` W&B CLI 명령으로 Launch 에이전트를 활성화합니다:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

어떤 유스 케이스에서는 Kubernetes 클러스터 내부에서 큐를 폴링하는 Launch 에이전트를 원할 수도 있습니다. 자세한 내용은 [Advanced queue set up page](./setup-queue-advanced.md)를 참조하세요.