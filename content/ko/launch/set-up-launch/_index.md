---
title: Launch 설정
menu:
  launch:
    identifier: ko-launch-set-up-launch-_index
    parent: launch
weight: 3
---

이 페이지에서는 W&B Launch를 설정하는 데 필요한 주요 단계를 설명합니다.

1. **큐 생성**: 큐는 FIFO 방식으로 동작하며 큐 설정을 가집니다. 큐의 설정은 어떤 리소스에서, 그리고 어떻게 job들이 실행될지를 제어합니다.
2. **에이전트 설정**: 에이전트는 사용자의 머신이나 인프라에서 동작하며 하나 이상의 큐에서 Launch job을 폴링합니다. job이 전달되면, 에이전트는 이미지를 빌드하고 사용할 수 있는지 확인합니다. 이후 에이전트가 해당 job을 목표 리소스에 제출합니다.

## 큐 생성

Launch 큐는 특정 목표 리소스를 지정하고, 해당 리소스에 맞는 추가 설정과 함께 구성되어야 합니다. 예를 들어, Kubernetes 클러스터를 대상으로 하는 Launch 큐라면 환경 변수나 커스텀 네임스페이스 같은 정보가 큐 설정에 포함될 수 있습니다. 큐를 생성할 때는 사용하고자 하는 목표 리소스와, 그 리소스를 위한 설정을 모두 지정해야 합니다.

에이전트가 큐에서 job을 받을 때는 큐 설정도 함께 전달받습니다. 에이전트가 job을 목표 리소스에 제출할 때, 큐 설정과 job에서 override된 값들을 같이 포함시킵니다. 예를 들어, job 설정을 사용해 해당 job 인스턴스만 Amazon SageMaker 인스턴스 타입을 지정할 수 있습니다. 이 경우 [queue config templates]({{< relref path="./setup-queue-advanced.md#configure-queue-template" lang="ko" >}})를 사용자 인터페이스로 활용하는 것이 일반적입니다.

### 큐 생성 방법
1. [wandb.ai/launch](https://wandb.ai/launch)에서 Launch App으로 이동하세요. 
2. 화면 우측 상단의 **create queue** 버튼을 클릭하세요.

{{< img src="/images/launch/create-queue.gif" alt="Creating a Launch queue" >}}

3. **Entity** 드롭다운 메뉴에서 큐가 속할 Entity를 선택하세요.
4. **Queue** 필드에 큐의 이름을 입력하세요.
5. **Resource** 드롭다운에서 이 큐에 추가된 job이 사용할 컴퓨트 리소스를 선택하세요.
6. **Prioritization** 기능을 사용할지 선택하세요. Prioritization이 활성화되면 팀 내 사용자가 job을 큐에 넣을 때 우선순위를 지정할 수 있습니다. 우선순위가 높은 job이 낮은 job보다 먼저 실행됩니다.
7. **Configuration** 필드에 JSON 또는 YAML 형식으로 리소스 설정을 입력하세요. 설정 문서의 구조와 의미는 큐가 지정하는 리소스 타입에 따라 달라집니다. 자세한 내용은 해당 리소스 유형에 맞는 별도의 설정 페이지를 참고하세요.

## Launch 에이전트 설정

Launch 에이전트는 하나 이상의 Launch 큐에서 job을 폴링하는 장시간 실행되는 프로세스입니다. Launch 에이전트는 FIFO 방식 혹은 큐에서 지정된 우선순위 순서에 따라 job을 처리합니다. 에이전트가 큐에서 job을 가져오면, 필요 시 해당 job의 이미지를 빌드할 수도 있습니다. 이후에는 큐 설정에서 정의된 옵션과 함께 job을 목표 리소스에 제출합니다.

{{% alert %}}
에이전트는 매우 유연하게 다양한 유스 케이스를 지원하도록 설정할 수 있습니다. 필요한 설정은 각각의 유스 케이스에 따라 달라집니다. [Docker]({{< relref path="./setup-launch-docker.md" lang="ko" >}}), [Amazon SageMaker]({{< relref path="./setup-launch-sagemaker.md" lang="ko" >}}), [Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ko" >}}), 또는 [Vertex AI]({{< relref path="./setup-vertex.md" lang="ko" >}}) 관련 페이지에서 자세한 설정 방법을 확인하세요.
{{% /alert %}}

{{% alert %}}
W&B에서는 개별 사용자 계정의 API 키 대신 서비스 계정의 API 키로 에이전트를 시작할 것을 권장합니다. 서비스 계정의 API 키를 사용할 경우 두 가지 이점이 있습니다:
1. 에이전트가 특정 사용자에게 의존적이지 않습니다.
2. Launch를 통해 생성된 run의 작성자는 에이전트 사용자가 아닌 launch job을 제출한 사용자로 Launch에서 표시됩니다.
{{% /alert %}}

### 에이전트 설정

`launch-config.yaml`이라는 이름으로 Launch 에이전트 설정 파일(YAML)을 구성합니다. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml` 위치에서 설정 파일을 확인하며, 필요하다면 에이전트 활성화 시 다른 디렉토리도 지정할 수 있습니다.

에이전트 설정 파일의 내용은 에이전트가 배포된 환경, Launch 큐의 목표 리소스, Docker 빌더 설정, 클라우드 레지스트리 요구사항 등 여러 요소에 따라 달라집니다.

유스 케이스와 무관하게 모든 Launch 에이전트에서 필수 핵심 옵션은 다음과 같습니다:
* `max_jobs`: 에이전트가 병렬로 실행할 수 있는 job의 최대 개수
* `entity`: 큐가 속한 Entity
* `queues`: 에이전트가 감시할 큐 이름(들)

{{% alert %}}
설정 YAML 파일 대신 W&B CLI를 사용하여 Launch 에이전트의 공통 핵심 옵션(max_jobs, Entity, Launch 큐)도 지정할 수 있습니다. 자세한 내용은 [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ko" >}}) 커맨드 문서를 참고하세요.
{{% /alert %}}

다음은 핵심 Launch 에이전트 설정 키를 사용하는 예시 YAML 코드입니다:

```yaml title="launch-config.yaml"
# 동시에 실행 가능한 run의 최대 개수. -1 = 제한 없음
max_jobs: -1

entity: <entity-name>

# 폴링할 큐 목록
queues:
  - <queue-name>
```

### 컨테이너 빌더 설정

Launch 에이전트는 이미지를 빌드할 수 있도록 설정할 수 있습니다. git 저장소나 코드 아티팩트에서 생성된 launch job을 사용하고 싶다면, 반드시 에이전트에 컨테이너 빌더 설정이 필요합니다. Launch job 생성 방법에 대한 자세한 안내는 [Create a launch job]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ko" >}})를 참고하세요.

W&B Launch에서 지원하는 빌더 옵션은 세 가지입니다:

* Docker: 로컬 Docker 데몬을 사용해 이미지를 빌드합니다.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Docker 데몬이 없는 환경(예: Kubernetes 클러스터)에서도 이미지 빌드를 가능하게 해주는 Google 프로젝트입니다.
* Noop: 에이전트가 이미지를 빌드하지 않고, 미리 빌드된 이미지만 가져옵니다.

{{% alert %}}
에이전트가 Docker 데몬이 없는 환경(예: Kubernetes 클러스터)에서 돌고 있다면 Kaniko 빌더를 사용하세요.

Kaniko 빌더에 대한 자세한 설정 방법은 [Set up Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ko" >}})에서 확인할 수 있습니다.
{{% /alert %}}

이미지 빌더를 지정하려면, 에이전트 설정에 builder 키를 포함시키면 됩니다. 아래 코드조각은 launch 설정 파일(`launch-config.yaml`)에서 Docker나 Kaniko 빌더를 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### 컨테이너 레지스트리 설정

경우에 따라 Launch 에이전트를 클라우드 레지스트리에 연결해야 할 수 있습니다. 대표적인 상황은 다음과 같습니다:

* 이미지를 빌드한 환경과 다른 환경(예: 강력한 워크스테이션이나 클러스터)에서 job을 실행하고 싶은 경우
* 에이전트로 이미지를 빌드하고, Amazon SageMaker 또는 Vertex AI에서 이미지를 실행하려는 경우
* 이미지 레포지토리에서 이미지를 가져올 때 인증 정보를 에이전트가 제공해야 하는 경우

에이전트를 컨테이너 레지스트리와 연동하는 방법은 [Advanced agent set]({{< relref path="./setup-agent-advanced.md" lang="ko" >}}) up 페이지에서 자세히 알아볼 수 있습니다.

## Launch 에이전트 활성화

`launch-agent` W&B CLI 커맨드로 Launch 에이전트를 활성화하세요:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

특정 유스 케이스에서는 Launch 에이전트가 Kubernetes 클러스터 내에서 큐를 폴링하도록 할 수도 있습니다. 자세한 내용은 [Advanced queue set up page]({{< relref path="./setup-queue-advanced.md" lang="ko" >}})를 참고하세요.