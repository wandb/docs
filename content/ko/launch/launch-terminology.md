---
title: Launch 용어 및 개념
menu:
  launch:
    identifier: ko-launch-launch-terminology
    parent: launch
url: guides/launch/launch-terminology
weight: 2
---

W&B Launch를 사용하면 [jobs]({{< relref path="#launch-job" lang="ko" >}})를 [queues]({{< relref path="#launch-queue" lang="ko" >}})에 등록하여 runs를 생성할 수 있습니다. Job은 W&B로 계측된 Python 스크립트입니다. Queue는 [target resource]({{< relref path="#target-resources" lang="ko" >}})에서 실행될 job의 목록을 보유합니다. [Agents]({{< relref path="#launch-agent" lang="ko" >}})는 queue에서 job을 가져와 target resource에서 실행합니다. W&B는 launch job을 W&B에서 [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})를 추적하는 방식과 유사하게 추적합니다.

### Launch job
Launch job은 특정 작업을 나타내는 [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}})의 한 형태입니다. 예를 들어, 일반적인 launch job에는 모델 트레이닝이나 모델 평가 트리거링 등이 있습니다. Job 정의에는 아래 항목이 포함됩니다.

- Python 코드 및 실행 가능한 entrypoint가 반드시 하나 이상 포함된 기타 파일 에셋
- 입력(config 파라미터) 및 출력(로그된 메트릭)에 대한 정보
- 환경에 대한 정보(예: `requirements.txt`, base `Dockerfile` 등)

Job 정의는 세 가지 주요 유형이 있습니다.

| Job types | Definition | How to run this job type | 
| ---------- | --------- | -------------- |
|Artifact-based (or code-based) jobs| 코드 및 기타 에셋이 W&B artifact로 저장됩니다. | Artifact-based job을 실행하려면 Launch agent에 builder가 설정되어야 합니다. |
|Git-based jobs| 코드 및 기타 에셋이 git 저장소의 특정 커밋, 브랜치, 태그에서 클론됩니다. | Git-based job을 실행하려면 Launch agent에 builder와 git 저장소 자격 증명이 필요합니다. |
|Image-based jobs| 코드 및 기타 에셋이 Docker 이미지로 만들어집니다. | Image-based job을 실행하려면 Launch agent에 이미지 저장소 자격 증명이 필요할 수 있습니다. |

{{% alert %}}
Launch jobs는 모델 트레이닝과 관련 없는 작업(예: Triton 추론 서버에 모델 배포 등)도 수행할 수 있지만, 모든 job은 반드시 `wandb.init`을 호출해야 정상적으로 완료됩니다. 이를 통해 run이 생성되어 W&B workspace에서 추적할 수 있습니다.
{{% /alert %}}

생성한 job은 W&B App의 프로젝트 workspace 내 `Jobs` 탭에서 확인할 수 있습니다. 여기서 job을 설정하고 [launch queue]({{< relref path="#launch-queue" lang="ko" >}})로 전송하여 다양한 [target resources]({{< relref path="#target-resources" lang="ko" >}})에서 실행할 수 있습니다.

### Launch queue
Launch *queues*는 특정 target resource에서 실행할 job을 순서대로 담는 리스트입니다. Launch queue는 선입선출(FIFO) 방식으로 작동합니다. Queue의 개수에는 사실상 제한이 없으나, 일반적으로 target resource별로 하나씩 두는 것이 좋습니다. Job은 W&B App UI, W&B CLI 또는 Python SDK로 queue에 등록할 수 있습니다. 이후 한 개 또는 여러 개의 Launch agent를 구성하여 queue에서 job을 받아 해당 target resource에서 실행할 수 있습니다.

### Target resources
Launch queue가 job을 실행하도록 설정된 컴퓨팅 환경을 *target resource*라고 합니다.

W&B Launch는 다음과 같은 target resources를 지원합니다.

- [Docker]({{< relref path="/launch/set-up-launch/setup-launch-docker.md" lang="ko" >}})
- [Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ko" >}})
- [AWS SageMaker]({{< relref path="/launch/set-up-launch/setup-launch-sagemaker.md" lang="ko" >}})
- [GCP Vertex]({{< relref path="/launch/set-up-launch/setup-vertex.md" lang="ko" >}})

각 target resource는 *resource configurations*라 불리는 서로 다른 설정 파라미터 세트를 받습니다. Resource configuration은 각 Launch queue에서 기본값이 정해지지만, 각 job별로 개별적으로 오버라이드할 수 있습니다. 더 자세한 내용은 각 target resource의 문서를 참고하세요.

### Launch agent
Launch agent는 실행할 job이 있는지 Launch queue를 주기적으로 확인하는 가벼운 상시 실행 프로그램입니다. Launch agent가 job을 받으면, 먼저 job 정의로부터 이미지를 빌드하거나 받아온 후 target resource에서 job을 실행합니다.

하나의 agent가 여러 queue를 감시할 수 있지만, 해당 agent는 자신이 감시하는 각 queue의 모든 target resource를 지원하도록 올바르게 구성되어야 합니다.

### Launch agent environment
Agent 환경이란, launch agent가 실행되어 job을 감시하는 환경을 의미합니다.

{{% alert %}}
Agent의 런타임 환경은 queue의 target resource와 독립적입니다. 즉, 필요한 target resource에 접근할 수 있도록 충분히 설정만 되어 있다면 agent는 어디서나 배포될 수 있습니다.
{{% /alert %}}