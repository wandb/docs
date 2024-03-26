---
displayed_sidebar: default
---

# 용어 및 개념
W&B Launch에서는 [job](#launch-job)을 [큐](#launch-queue)에 추가하여 run을 생성합니다. Job은 W&B로 계측된 파이썬 스크립트입니다. 큐는 [대상 리소스](#target-resources)에서 실행할 job 목록을 보유합니다. [에이전트](#launch-agent)는 큐에서 job을 가져와 대상 리소스에서 job을 실행합니다. W&B는 launch job을 [run](../runs/intro.md)을 추적하는 방식과 유사하게 추적합니다.

### Launch job
Launch job은 완료할 작업을 나타내는 특정 유형의 [W&B Artifacts](../artifacts/intro.md)입니다. 예를 들어, 일반적인 launch job에는 모델 트레이닝 또는 모델 평가를 트리거하는 것이 포함됩니다. Job 정의는 다음을 포함합니다:

- 파이썬 코드 및 기타 파일 자산, 적어도 하나의 실행 가능한 진입점 포함.
- 입력(설정 파라미터) 및 출력(로그된 메트릭)에 대한 정보.
- 환경에 대한 정보. (예: `requirements.txt`, 기본 `Dockerfile`).

Job 정의의 세 가지 주요 유형이 있습니다:

| job 유형 | 정의 | 이 job 유형을 실행하는 방법 | 
| ---------- | --------- | -------------- |
|아티팩트 기반 (또는 코드 기반) job| 코드 및 기타 자산이 W&B 아티팩트로 저장됩니다.| 아티팩트 기반 job을 실행하려면, Launch 에이전트가 빌더로 구성되어야 합니다. |
|Git 기반 job| 코드 및 기타 자산이 git 저장소의 특정 커밋, 분기 또는 태그에서 복제됩니다. | Git 기반 job을 실행하려면, Launch 에이전트가 빌더와 git 저장소 자격 증명으로 구성되어야 합니다. |
|이미지 기반 job|코드 및 기타 자산이 Docker 이미지로 구워집니다. | 이미지 기반 job을 실행하려면, Launch 에이전트가 이미지 저장소 자격 증명으로 구성될 수 있어야 합니다. | 

:::tip
Launch job은 모델 트레이닝과 관련되지 않은 활동을 수행할 수 있지만--예를 들어, 모델을 Triton 추론 서버에 배포하는 것과 같은--모든 job은 성공적으로 완료되기 위해 `wandb.init`을 호출해야 합니다. 이는 W&B 워크스페이스에서 추적 목적으로 run을 생성합니다.
:::

W&B App에서 `Jobs` 탭 아래에 있는 프로젝트 워크스페이스에서 생성한 job을 찾을 수 있습니다. 거기서 job을 구성하고 다양한 [대상 리소스](#target-resources)에서 실행하기 위해 [launch 큐](#launch-queue)로 보낼 수 있습니다.

### Launch 큐
Launch *큐*는 특정 대상 리소스에서 실행할 job의 순서가 지정된 목록입니다. Launch 큐는 선입선출(FIFO)입니다. 큐의 수에 실질적인 제한은 없지만, 좋은 지침은 대상 리소스마다 하나의 큐입니다. job은 W&B App UI, W&B CLI 또는 Python SDK를 통해 큐에 추가할 수 있습니다. 그런 다음 하나 이상의 Launch 에이전트가 큐에서 항목을 가져와 큐의 대상 리소스에서 실행할 수 있도록 구성될 수 있습니다.

### 대상 리소스
Launch 큐가 job을 실행하도록 구성된 컴퓨팅 환경을 *대상 리소스*라고 합니다.

W&B Launch는 다음과 같은 대상 리소스를 지원합니다:

- [Docker](./setup-launch-docker.md)
- [Kubernetes](./setup-launch-kubernetes.md)
- [AWS SageMaker](./setup-launch-sagemaker.md)
- [GCP Vertex](./setup-vertex.md)

각 대상 리소스는 *리소스 설정*이라고 하는 다른 설정 파라미터 세트를 수용합니다. 리소스 설정은 각 launch 큐에 의해 정의된 기본값을 취하지만, 각 job에 의해 독립적으로 재정의될 수 있습니다. 각 대상 리소스에 대한 자세한 내용은 해당 문서를 참조하십시오.

### Launch 에이전트
Launch 에이전트는 주기적으로 launch 큐를 확인하여 실행할 job을 검색하는 가벼운, 지속적인 프로그램입니다. Launch 에이전트가 job을 수신하면, 먼저 job 정의에서 이미지를 빌드하거나 가져온 다음 대상 리소스에서 실행합니다.

한 에이전트는 여러 큐를 폴링할 수 있지만, 에이전트는 폴링하는 각 큐를 위한 모든 지원 대상 리소스를 지원하도록 올바르게 구성되어야 합니다.

### Launch 에이전트 환경
에이전트 환경은 launch 에이전트가 실행되어 job을 폴링하는 환경입니다.

:::info
에이전트의 런타임 환경은 큐의 대상 리소스와 독립적입니다. 즉, 에이전트는 필요한 대상 리소스에 엑세스하기 위해 충분히 구성되어 있는 한 어디에서나 배포될 수 있습니다.
:::