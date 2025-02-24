---
title: Set up Launch
menu:
  launch:
    identifier: ko-launch-set-up-launch-_index
    parent: launch
weight: 3
---

이 페이지에서는 W&B Launch를 설정하는 데 필요한 높은 수준의 단계를 설명합니다.

1. **대기열 설정**: 대기열은 FIFO이며 대기열 설정을 갖습니다. 대기열의 설정은 대상 리소스에서 작업을 실행하는 위치와 방법을 제어합니다.
2. **에이전트 설정**: 에이전트는 사용자 머신/인프라에서 실행되며 Launch 작업을 위해 하나 이상의 대기열을 폴링합니다. 작업이 풀되면 에이전트는 이미지가 빌드되어 사용 가능한지 확인합니다. 그런 다음 에이전트는 작업을 대상 리소스에 제출합니다.

## 대기열 설정
Launch 대기열은 특정 대상 리소스와 해당 리소스에 특정한 추가 설정을 가리키도록 구성해야 합니다. 예를 들어, Kubernetes 클러스터를 가리키는 Launch 대기열은 환경 변수를 포함하거나 Launch 대기열 구성에 사용자 지정 네임스페이스를 설정할 수 있습니다. 대기열을 생성할 때 사용하려는 대상 리소스와 해당 리소스에 사용할 구성을 모두 지정합니다.

에이전트가 대기열에서 작업을 수신하면 대기열 구성도 수신합니다. 에이전트가 작업을 대상 리소스에 제출할 때 작업 자체의 재정의와 함께 대기열 구성을 포함합니다. 예를 들어, 작업 구성을 사용하여 해당 작업 인스턴스에만 Amazon SageMaker 인스턴스 유형을 지정할 수 있습니다. 이 경우 최종 사용자 인터페이스로 [대기열 구성 템플릿]({{< relref path="./setup-queue-advanced.md#configure-queue-template" lang="ko" >}})을 사용하는 것이 일반적입니다.

### 대기열 생성
1. [wandb.ai/launch](https://wandb.ai/launch)에서 Launch 앱으로 이동합니다.
2. 화면 오른쪽 상단에 있는 **대기열 생성** 버튼을 클릭합니다.

{{< img src="/images/launch/create-queue.gif" alt="" >}}

3. **Entity** 드롭다운 메뉴에서 대기열이 속할 entity를 선택합니다.
4. **대기열** 필드에 대기열 이름을 입력합니다.
5. **리소스** 드롭다운에서 이 대기열에 추가할 작업에 사용할 컴퓨팅 리소스를 선택합니다.
6. 이 대기열에 대해 **우선 순위 지정**을 허용할지 여부를 선택합니다. 우선 순위 지정이 활성화되면 팀의 사용자는 작업을 대기열에 넣을 때 Launch 작업에 대한 우선 순위를 정의할 수 있습니다. 우선 순위가 높은 작업은 우선 순위가 낮은 작업보다 먼저 실행됩니다.
7. **구성** 필드에 JSON 또는 YAML 형식으로 리소스 구성을 제공합니다. 구성 문서의 구조와 의미는 대기열이 가리키는 리소스 유형에 따라 달라집니다. 자세한 내용은 대상 리소스에 대한 전용 설정 페이지를 참조하세요.

## Launch 에이전트 설정
Launch 에이전트는 작업을 위해 하나 이상의 Launch 대기열을 폴링하는 장기 실행 프로세스입니다. Launch 에이전트는 선입선출(FIFO) 순서 또는 대기열에서 가져오는 우선 순위에 따라 작업을 대기열에서 제거합니다. 에이전트가 대기열에서 작업을 제거하면 해당 작업에 대한 이미지를 선택적으로 빌드합니다. 그런 다음 에이전트는 대기열 구성에 지정된 구성 옵션과 함께 작업을 대상 리소스에 제출합니다.

{{% alert %}}
에이전트는 매우 유연하며 광범위한 유스 케이스를 지원하도록 구성할 수 있습니다. 에이전트에 필요한 구성은 특정 유스 케이스에 따라 달라집니다. [Docker]({{< relref path="./setup-launch-docker.md" lang="ko" >}}), [Amazon SageMaker]({{< relref path="./setup-launch-sagemaker.md" lang="ko" >}}), [Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ko" >}}), 또는 [Vertex AI]({{< relref path="./setup-vertex.md" lang="ko" >}})에 대한 전용 페이지를 참조하십시오.
{{% /alert %}}

{{% alert %}}
W&B는 특정 user의 API 키가 아닌 서비스 계정의 API 키로 에이전트를 시작하는 것이 좋습니다. 서비스 계정의 API 키를 사용하면 다음과 같은 두 가지 이점이 있습니다.
1. 에이전트는 개별 user에 의존하지 않습니다.
2. Launch를 통해 생성된 run과 연결된 작성자는 에이전트와 연결된 user가 아닌 Launch 작업을 제출한 user로 Launch에서 간주합니다.
{{% /alert %}}

### 에이전트 구성
`launch-config.yaml`이라는 YAML 파일로 Launch 에이전트를 구성합니다. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. Launch 에이전트를 활성화할 때 다른 디렉토리를 선택적으로 지정할 수 있습니다.

Launch 에이전트의 구성 파일 내용은 Launch 에이전트의 환경, Launch 대기열의 대상 리소스, Docker 빌더 요구 사항, cloud 레지스트리 요구 사항 등에 따라 달라집니다.

유스 케이스와 관계없이 Launch 에이전트에 대한 핵심 구성 가능 옵션은 다음과 같습니다.
* `max_jobs`: 에이전트가 병렬로 실행할 수 있는 최대 작업 수
* `entity`: 대기열이 속한 entity
* `queues`: 에이전트가 감시할 하나 이상의 대기열 이름

{{% alert %}}
W&B CLI를 사용하여 Launch 에이전트에 대한 일반적인 구성 가능 옵션(구성 YAML 파일 대신): 최대 작업 수, W&B entity 및 Launch 대기열을 지정할 수 있습니다. 자세한 내용은 [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ko" >}}) 코맨드를 참조하십시오.
{{% /alert %}}

다음 YAML 코드 조각은 핵심 Launch 에이전트 구성 키를 지정하는 방법을 보여줍니다.

```yaml title="launch-config.yaml"
# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

entity: <entity-name>

# List of queues to poll.
queues:
  - <queue-name>
```

### 컨테이너 빌더 구성
Launch 에이전트는 이미지를 빌드하도록 구성할 수 있습니다. git 저장소 또는 code Artifacts에서 생성된 Launch 작업을 사용하려는 경우 에이전트가 컨테이너 빌더를 사용하도록 구성해야 합니다. Launch 작업 생성 방법에 대한 자세한 내용은 [Launch 작업 생성]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ko" >}})을 참조하십시오.

W&B Launch는 세 가지 빌더 옵션을 지원합니다.

* Docker: Docker 빌더는 로컬 Docker 데몬을 사용하여 이미지를 빌드합니다.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko는 Docker 데몬을 사용할 수 없는 환경에서 이미지 빌드를 지원하는 Google 프로젝트입니다.
* Noop: 에이전트는 작업을 빌드하려고 시도하지 않고 미리 빌드된 이미지만 가져옵니다.

{{% alert %}}
에이전트가 Docker 데몬을 사용할 수 없는 환경(예: Kubernetes 클러스터)에서 폴링하는 경우 Kaniko 빌더를 사용하십시오.

Kaniko 빌더에 대한 자세한 내용은 [Kubernetes 설정]({{< relref path="./setup-launch-kubernetes.md" lang="ko" >}})을 참조하십시오.
{{% /alert %}}

이미지 빌더를 지정하려면 에이전트 구성에 빌더 키를 포함하십시오. 예를 들어, 다음 코드 조각은 Docker 또는 Kaniko를 사용하도록 지정하는 Launch 구성(`launch-config.yaml`)의 일부를 보여줍니다.

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### 컨테이너 레지스트리 구성
경우에 따라 Launch 에이전트를 cloud 레지스트리에 연결할 수 있습니다. Launch 에이전트를 cloud 레지스트리에 연결하려는 일반적인 시나리오는 다음과 같습니다.

* 강력한 워크스테이션 또는 클러스터와 같이 빌드한 환경 이외의 환경에서 작업을 실행하려는 경우.
* 에이전트를 사용하여 이미지를 빌드하고 이러한 이미지를 Amazon SageMaker 또는 VertexAI에서 실행하려는 경우.
* Launch 에이전트가 이미지 저장소에서 가져올 자격 증명을 제공하도록 하려는 경우.

에이전트가 컨테이너 레지스트리와 상호 작용하도록 구성하는 방법에 대한 자세한 내용은 [고급 에이전트 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}}) 페이지를 참조하십시오.

## Launch 에이전트 활성화
`launch-agent` W&B CLI 코맨드를 사용하여 Launch 에이전트를 활성화합니다.

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

일부 유스 케이스에서는 Kubernetes 클러스터 내에서 Launch 에이전트가 대기열을 폴링하도록 할 수 있습니다. 자세한 내용은 [고급 대기열 설정 페이지]({{< relref path="./setup-queue-advanced.md" lang="ko" >}})를 참조하십시오.
