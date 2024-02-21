---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch 설정하기

이 페이지는 W&B Launch를 설정하기 위한 고수준 단계를 설명합니다:

1. **큐 설정하기**: 큐는 FIFO이며 큐 구성을 가지고 있습니다. 큐의 구성은 작업이 대상 리소스에서 어떻게 실행되는지와 어디에서 실행되는지를 제어합니다.
2. **에이전트 설정하기**: 에이전트는 사용자의 기계/인프라에서 실행되며 하나 이상의 큐에서 Launch 작업을 폴링합니다. 작업이 추출되면, 에이전트는 이미지가 구축되고 사용 가능한지를 보장합니다. 그런 다음 에이전트는 작업을 대상 리소스로 제출합니다.

## 큐 설정하기
Launch 큐는 특정 대상 리소스를 가리키도록 구성되어야 하며 그 리소스에 특정한 추가 구성이 필요할 수 있습니다. 예를 들어, Kubernetes 클러스터를 가리키는 Launch 큐는 환경 변수를 포함하거나 Launch 큐 구성에서 사용자 지정 네임스페이스를 설정할 수 있습니다. 큐를 생성할 때 사용하고자 하는 대상 리소스와 그 리소스가 사용할 구성을 모두 지정합니다.

에이전트가 큐에서 작업을 받으면 큐 구성도 함께 받습니다. 에이전트가 작업을 대상 리소스에 제출할 때, 큐 구성과 작업 자체에서 오는 모든 오버라이드를 포함합니다. 예를 들어, 작업 구성을 사용하여 해당 작업 인스턴스에 대한 Amazon SageMaker 인스턴스 유형을 지정할 수 있습니다. 이 경우, [큐 구성 템플릿](./setup-queue-advanced.md#configure-queue-template)을 최종 사용자 인터페이스로 사용하는 것이 일반적입니다.

### 큐 생성하기
1. [wandb.ai/launch](https://wandb.ai/launch)에서 Launch 앱으로 이동합니다.
2. 화면 오른쪽 상단에 있는 **큐 생성** 버튼을 클릭합니다.

![](/images/launch/create-queue.gif)

3. **엔티티** 드롭다운 메뉴에서 큐가 속할 엔티티를 선택합니다.
  :::tip
  팀 엔티티를 선택하면 팀의 모든 멤버가 이 큐에 작업을 보낼 수 있습니다. 개인 엔티티(사용자 이름과 연관된)를 선택하면 W&B는 해당 사용자만 사용할 수 있는 개인 큐를 생성합니다.
  :::
4. **큐** 필드에 큐의 이름을 제공합니다.
5. **리소스** 드롭다운에서 이 큐에 추가된 작업이 사용할 컴퓨팅 리소스를 선택합니다.
6. 이 큐에 대해 **우선 순위 설정**을 허용할지 선택합니다. 우선 순위가 활성화되면, 팀의 사용자가 작업을 큐에 추가할 때 그들의 Launch 작업에 대한 우선 순위를 정의할 수 있습니다. 더 높은 우선 순위의 작업이 더 낮은 우선 순위의 작업보다 먼저 실행됩니다.
7. **구성** 필드에 JSON 또는 YAML 형식으로 리소스 구성을 제공합니다. 구성 문서의 구조와 의미는 큐가 가리키는 리소스 유형에 따라 달라집니다. 대상 리소스에 대한 전용 설정 페이지에서 자세한 내용을 확인하세요.

## Launch 에이전트 설정하기
Launch 에이전트는 작업을 폴링하기 위해 하나 이상의 Launch 큐에서 긴 시간 동안 실행되는 프로세스입니다. Launch 에이전트는 큐에서 작업을 FIFO 순서 또는 우선 순위에 따라 큐에서 작업을 데큐합니다. 에이전트가 큐에서 작업을 데큐할 때, 선택적으로 해당 작업에 대한 이미지를 구축합니다. 그런 다음 에이전트는 큐 구성에서 지정한 구성 옵션과 함께 대상 리소스에 작업을 제출합니다.



:::info
에이전트는 매우 유연하며 다양한 사용 사례를 지원하도록 구성할 수 있습니다. 에이전트에 필요한 구성은 구체적인 사용 사례에 따라 다릅니다. [Docker](./setup-launch-docker.md), [Amazon SageMaker](./setup-launch-sagemaker.md), [Kubernetes](./setup-launch-kubernetes.md), 또는 [Vertex AI](./setup-vertex.md)에 대한 전용 페이지를 참조하세요.
:::

:::tip
W&B는 특정 사용자의 API 키가 아닌 [서비스 계정의](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) API 키로 에이전트를 시작하는 것이 좋습니다. 서비스 계정의 API 키를 사용하면 두 가지 이점이 있습니다:
1. 에이전트가 개별 사용자에게 의존하지 않습니다.
2. Launch를 통해 생성된 실행과 관련된 작성자는 에이전트와 연관된 사용자가 아니라 Launch 작업을 제출한 사용자로 Launch에 의해 보여집니다.
:::

### 에이전트 구성
`launch-config.yaml`이라는 이름의 YAML 파일로 launch 에이전트를 구성합니다. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. Launch 에이전트를 활성화할 때 다른 디렉터리를 선택적으로 지정할 수 있습니다.

Launch 에이전트의 구성 파일 내용은 Launch 에이전트의 환경, Launch 큐의 대상 리소스, Docker 빌더 요구 사항, 클라우드 레지스트리 요구 사항 등에 따라 다릅니다.

사용 사례와 관계없이 Launch 에이전트에 대한 핵심 구성 옵션이 있습니다:
* `max_jobs`: 에이전트가 동시에 실행할 수 있는 최대 작업 수
* `entity`: 큐가 속한 엔티티
* `queues`: 에이전트가 관찰할 하나 이상의 큐의 이름

:::tip
W&B CLI를 사용하여 launch 에이전트에 대한 보편적인 구성 옵션(최대 작업 수, W&B 엔티티 및 Launch 큐 대신 구성 YAML 파일)을 지정할 수 있습니다. 자세한 정보는 [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) 명령을 참조하세요.
:::


다음 YAML 조각은 핵심 Launch 에이전트 구성 키를 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
# 동시에 수행할 최대 실행 수. -1 = 제한 없음
max_jobs: -1

entity: <엔티티-이름>

# 폴링할 큐 목록.
queues:
  - <큐-이름>
```

### 컨테이너 빌더 구성하기
Launch 에이전트는 이미지를 구축하도록 구성할 수 있습니다. git 저장소나 코드 아티팩트에서 생성된 Launch 작업을 사용하려면 에이전트를 컨테이너 빌더를 사용하도록 구성해야 합니다. Launch 작업을 생성하는 방법에 대한 자세한 내용은 [Launch 작업 생성하기](./create-launch-job.md)를 참조하세요.

W&B Launch는 세 가지 빌더 옵션을 지원합니다:

* Docker: Docker 빌더는 로컬 Docker 데몬을 사용하여 이미지를 구축합니다.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko는 Docker 데몬이 사용할 수 없는 환경에서 이미지를 구축할 수 있도록 하는 Google 프로젝트입니다.
* Noop: 에이전트는 작업을 구축하려고 시도하지 않으며 대신 사전 구축된 이미지만을 가져옵니다.

:::tip
에이전트가 Docker 데몬이 사용할 수 없는 환경(예: Kubernetes 클러스터)에서 폴링하는 경우 Kaniko 빌더를 사용하세요.

Kaniko 빌더에 대한 자세한 내용은 [Kubernetes 설정하기](./setup-launch-kubernetes.md)를 참조하세요.
:::

이미지 빌더를 지정하려면 에이전트 구성에 빌더 키를 포함시킵니다. 예를 들어, 다음 코드 조각은 Docker 또는 Kaniko를 사용하도록 지정하는 launch 구성(`launch-config.yaml`)의 일부를 보여줍니다:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### 컨테이너 레지스트리 구성하기
일부 경우에는 Launch 에이전트를 클라우드 레지스트리에 연결하고 싶을 수 있습니다. Launch 에이전트를 클라우드 레지스트리에 연결하고 싶은 일반적인 시나리오는 다음과 같습니다:

* 강력한 워크스테이션 또는 클러스터와 같은 다른 환경에서 작업을 실행하고 싶습니다.
* 에이전트를 사용하여 이미지를 구축하고 이러한 이미지를 Amazon SageMaker 또는 VertexAI에서 실행하고 싶습니다.
* 이미지 저장소에서 가져오기 위해 에이전트가 자격 증명을 제공하길 원합니다.

에이전트가 컨테이너 레지스트리와 상호 작용하도록 구성하는 방법에 대해 자세히 알아보려면 [고급 에이전트 설정](./setup-agent-advanced.md) 페이지를 참조하세요.

## Launch 에이전트 활성화하기
`launch-agent` W&B CLI 명령으로 Launch 에이전트를 활성화합니다:

```bash
wandb launch-agent -q <큐-1> -q <큐-2> --max-jobs 5
```

일부 사용 사례에서는 Kubernetes 클러스터 내부에서 큐를 폴링하는 Launch 에이전트를 가질 수 있습니다. 자세한 내용은 [고급 큐 설정 페이지](./setup-queue-advanced.md)를 참조하세요.