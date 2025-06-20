---
title: Launch terms and concepts
menu:
  launch:
    identifier: ko-launch-launch-terminology
    parent: launch
url: /ko/guides//launch/launch-terminology
weight: 2
---

W&B Launch를 사용하면 [jobs]({{< relref path="#launch-job" lang="ko" >}})를 [queues]({{< relref path="#launch-queue" lang="ko" >}})에 대기열에 추가하여 runs를 생성할 수 있습니다. Jobs는 W&B로 계측된 Python 스크립트입니다. Queues는 [target resource]({{< relref path="#target-resources" lang="ko" >}})에서 실행할 jobs 목록을 보관합니다. [Agents]({{< relref path="#launch-agent" lang="ko" >}})는 queues에서 jobs를 가져와 target resource에서 jobs를 실행합니다. W&B는 W&B가 [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})를 추적하는 방식과 유사하게 Launch jobs를 추적합니다.

### Launch job
Launch job은 완료할 작업을 나타내는 특정 유형의 [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}})입니다. 예를 들어 일반적인 Launch jobs에는 model 트레이닝 또는 model 평가 트리거가 포함됩니다. Job 정의에는 다음이 포함됩니다.

- 하나 이상의 실행 가능한 진입점을 포함하여 Python 코드 및 기타 파일 자산.
- 입력 (config parameter) 및 출력 (metrics 로깅)에 대한 정보.
- 환경에 대한 정보. (예: `requirements.txt`, 기본 `Dockerfile`).

Job 정의에는 세 가지 주요 종류가 있습니다.

| Job 유형 | 정의 | 이 Job 유형을 실행하는 방법 |
| ---------- | --------- | -------------- |
|Artifact 기반 (또는 코드 기반) jobs| 코드 및 기타 자산은 W&B artifact로 저장됩니다.| Artifact 기반 jobs를 실행하려면 Launch agent가 builder로 구성되어야 합니다. |
|Git 기반 jobs| 코드 및 기타 자산은 Git 저장소의 특정 커밋, branch 또는 tag에서 복제됩니다. | Git 기반 jobs를 실행하려면 Launch agent가 builder 및 Git 저장소 자격 증명으로 구성되어야 합니다. |
|Image 기반 jobs| 코드 및 기타 자산은 Docker 이미지에 포함됩니다. | Image 기반 jobs를 실행하려면 Launch agent가 이미지 저장소 자격 증명으로 구성되어야 할 수 있습니다. |

{{% alert %}}
Launch jobs는 model 트레이닝과 관련이 없는 활동 (예: model을 Triton 추론 서버에 배포)을 수행할 수 있지만 모든 jobs는 성공적으로 완료하려면 `wandb.init`를 호출해야 합니다. 이렇게 하면 W&B workspace에서 추적 목적으로 run이 생성됩니다.
{{% /alert %}}

프로젝트 workspace의 `Jobs` 탭 아래의 W&B App에서 생성한 jobs를 찾습니다. 여기에서 jobs를 구성하고 [launch queue]({{< relref path="#launch-queue" lang="ko" >}})로 보내 다양한 [target resource]({{< relref path="#target-resources" lang="ko" >}})에서 실행할 수 있습니다.

### Launch queue
Launch *queues*는 특정 target resource에서 실행할 jobs의 정렬된 목록입니다. Launch queues는 선입선출 (FIFO) 방식입니다. 가질 수 있는 queues 수에는 실제적인 제한이 없지만 target resource당 하나의 queue를 사용하는 것이 좋습니다. Jobs는 W&B App UI, W&B CLI 또는 Python SDK를 사용하여 대기열에 추가할 수 있습니다. 그런 다음 하나 이상의 Launch agents를 구성하여 queue에서 항목을 가져와 queue의 target resource에서 실행할 수 있습니다.

### Target resources
Launch queue가 jobs를 실행하도록 구성된 컴퓨팅 환경을 *target resource*라고 합니다.

W&B Launch는 다음 target resources를 지원합니다.

- [Docker]({{< relref path="/launch/set-up-launch/setup-launch-docker.md" lang="ko" >}})
- [Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ko" >}})
- [AWS SageMaker]({{< relref path="/launch/set-up-launch/setup-launch-sagemaker.md" lang="ko" >}})
- [GCP Vertex]({{< relref path="/launch/set-up-launch/setup-vertex.md" lang="ko" >}})

각 target resource는 *resource configurations*라는 서로 다른 configuration parameter 집합을 허용합니다. Resource configurations는 각 Launch queue에서 정의한 기본값을 사용하지만 각 job에서 독립적으로 재정의할 수 있습니다. 자세한 내용은 각 target resource에 대한 설명서를 참조하십시오.

### Launch agent
Launch agents는 실행할 jobs에 대해 Launch queues를 주기적으로 확인하는 가벼운 영구 프로그램입니다. Launch agent가 job을 수신하면 먼저 job 정의에서 이미지를 빌드하거나 가져온 다음 target resource에서 실행합니다.

하나의 agent가 여러 queues를 폴링할 수 있지만 agent는 폴링하는 각 queue에 대해 모든 지원 target resources를 지원하도록 적절하게 구성되어야 합니다.

### Launch agent environment
Agent 환경은 Launch agent가 실행 중인 jobs를 폴링하는 환경입니다.

{{% alert %}}
Agent의 런타임 환경은 queue의 target resource와 독립적입니다. 즉, agents는 필요한 target resources에 엑세스할 수 있도록 충분히 구성되어 있는 한 어디든 배포할 수 있습니다.
{{% /alert %}}
