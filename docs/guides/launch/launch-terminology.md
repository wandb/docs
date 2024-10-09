---
title: Launch terms and concepts
displayed_sidebar: default
---

W&B Launch를 사용하면 [job](#launch-job)을 [queue](#launch-queue)에 등록하여 run을 생성할 수 있습니다. Job은 W&B가 도입된 Python 스크립트입니다. Queue는 [target resource](#target-resources)에서 실행할 job 목록을 저장합니다. [Agent](#launch-agent)는 queue에서 job을 가져와 target resource에서 실행합니다. W&B는 launch job을 W&B가 [run](../runs/intro.md)을 추적하는 방식과 유사하게 추적합니다.

### Launch job
Launch job은 완료할 과제를 나타내는 특정 유형의 [W&B Artifact](../artifacts/intro.md)입니다. 예를 들어, 일반적인 launch job에는 모델 트레이닝이나 모델 평가 트리거가 포함됩니다. Job 정의에는 다음이 포함됩니다:

- runnable 엔트리포인트가 하나 이상 포함된 Python 코드 및 기타 파일 에셋.
- 입력(config 파라미터)과 출력(로그된 메트릭)에 대한 정보.
- 환경에 대한 정보. (예: `requirements.txt`, 기본 `Dockerfile`)

세 가지 주요 job 정의 유형이 있습니다:

| Job types | Definition | How to run this job type | 
| ---------- | --------- | -------------- |
|Artifact-based (or code-based) jobs| 코드 및 기타 에셋은 W&B 아티팩트로 저장됩니다.| Artifact 기반 job을 실행하려면 Launch agent를 builder로 설정해야 합니다. |
|Git-based jobs| 코드 및 기타 에셋은 특정 커밋, 브랜치 또는 Git 리포지토리의 태그에서 클론됩니다. | Git 기반 job을 실행하려면 Launch agent를 builder와 git 리포지토리 크레덴셜로 설정해야 합니다. |
|Image-based jobs|코드 및 기타 에셋은 Docker 이미지에 포함됩니다. | 이미지 기반 job을 실행하려면 Launch agent에 이미지 리포지토리 크레덴셜을 설정해야 할 수 있습니다. | 

:::tip
Launch job은 모델 트레이닝과 관련 없는 활동을 수행할 수 있지만, 예를 들어 모델을 Triton 추론 서버에 배포하는 경우에도 모든 job은 `wandb.init`을 호출해야 성공적으로 완료됩니다. 이 작업은 W&B 워크스페이스에서 추적 목적으로 run을 생성합니다.
:::

생성한 job은 W&B App의 프로젝트 워크스페이스의 `Jobs` 탭에서 찾을 수 있습니다. 거기에서 job은 [launch queue](#launch-queue)에 설정되어 다양한 [target resources](#target-resources)에서 실행될 수 있도록 전송됩니다.

### Launch queue
Launch *queue*는 특정 target resource에서 실행할 job의 정렬된 목록입니다. Launch queue는 선입선출(FIFO)입니다. 큐 개수에 실질적인 제한은 없지만, target resource당 큐 하나가 좋은 기준입니다. Job은 W&B App UI, W&B CLI 또는 Python SDK를 사용하여 queue에 등록할 수 있습니다. 그런 다음 하나 이상의 Launch agent가 queue에서 항목을 가져와 queue의 target resource에서 실행되도록 설정할 수 있습니다.

### Target resources
Launch queue가 job을 실행하도록 설정된 컴퓨팅 환경을 *target resource*라고 합니다.

W&B Launch는 다음과 같은 target resources를 지원합니다:

- [Docker](./setup-launch-docker.md)
- [Kubernetes](./setup-launch-kubernetes.md)
- [AWS SageMaker](./setup-launch-sagemaker.md)
- [GCP Vertex](./setup-vertex.md)

각 target resource는 *resource configurations*이라는 서로 다른 설정 파라미터 집합을 허용합니다. Resource configurations는 각 Launch queue에 의해 정의된 기본 값을 취하지만 각 job에 의해 독립적으로 재정의될 수 있습니다. 각 target resource에 대한 자세한 내용은 설명서를 참조하세요.

### Launch agent
Launch agent는 주기적으로 Launch queue에서 실행할 job을 확인하는 가볍고 지속 가능한 프로그램입니다. Launch agent가 job을 수신하면 먼저 job 정의에서 이미지를 빌드하거나 가져온 다음 target resource에서 실행합니다.

하나의 에이전트는 여러 queue에 대해 폴링할 수 있지만, 에이전트는 폴링하는 각 queue의 모든 백업 target resources를 지원하도록 적절히 설정되어야 합니다.

### Launch agent environment
에이전트 환경은 launch agent가 실행되고, job을 폴링하는 환경입니다.

:::info
에이전트의 런타임 환경은 queue의 target resource와 독립적입니다. 즉, 에이전트는 필요한 target resources에 접근할 수 있도록 충분히 구성되면 어디에서나 배포할 수 있습니다.
:::