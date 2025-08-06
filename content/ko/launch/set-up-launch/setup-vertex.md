---
title: '튜토리얼: Vertex AI에서 W&B Launch 설정하기'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch 를 사용하면 Vertex AI 트레이닝 job 으로 실행할 job 을 제출할 수 있습니다. Vertex AI 트레이닝 job 을 이용하면 Vertex AI 플랫폼에서 제공하는 알고리즘이나 커스텀 알고리즘으로 기계학습 모델을 트레이닝할 수 있습니다. launch job 이 시작되면, Vertex AI 가 기본 인프라, 스케일링, 오케스트레이션을 자동으로 관리합니다.

W&B Launch 는 `google-cloud-aiplatform` SDK 의 `CustomJob` 클래스를 통해 Vertex AI 와 연동됩니다. `CustomJob` 의 파라미터는 launch queue 설정에서 제어할 수 있습니다. Vertex AI 는 GCP 외부의 프라이빗 레지스트리에서 이미지를 가져올 수 없습니다. 따라서, Vertex AI 와 W&B Launch 를 함께 사용하려면 컨테이너 이미지를 GCP 또는 퍼블릭 레지스트리에 저장해야 합니다. 컨테이너 이미지를 Vertex job 에서 엑세스할 수 있도록 만드는 방법은 Vertex AI 공식 문서를 참고하세요.

## 사전 준비 사항

1. **Vertex AI API 가 활성화된 GCP 프로젝트를 생성하거나 엑세스하세요.** API 활성화 방법은 [GCP API 콘솔 문서](https://support.google.com/googleapi/answer/6158841?hl=ko) 를 참고하세요.
2. **실행할 이미지를 저장할 GCP Artifact Registry 저장소를 생성하세요.** 자세한 내용은 [GCP Artifact Registry 문서](https://cloud.google.com/artifact-registry/docs/overview) 를 참고하세요.
3. **Vertex AI 가 메타데이터를 저장할 스테이징 GCS 버킷을 생성하세요.** 이 버킷은 Vertex AI workload 와 동일한 리전에 위치해야 합니다. 하나의 버킷을 staging 및 build context 용으로 함께 사용할 수 있습니다.
4. **Vertex AI job 을 실행할 권한이 있는 서비스 계정을 생성하세요.** 서비스 계정에 권한을 할당하는 방법에 대한 자세한 정보는 [GCP IAM 문서](https://cloud.google.com/iam/docs/creating-managing-service-accounts) 를 참고하세요.
5. **서비스 계정에 Vertex job 관리 권한을 부여하세요.**

| 권한                             | 리소스 범위                | 설명                                                                                 |
| ------------------------------ | ------------------------- | ----------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | 지정된 GCP 프로젝트        | 프로젝트 내에서 새로운 기계학습 job 을 생성할 수 있습니다.                             |
| `aiplatform.customJobs.list`   | 지정된 GCP 프로젝트        | 프로젝트 내의 기계학습 job 목록을 조회할 수 있습니다.                                 |
| `aiplatform.customJobs.get`    | 지정된 GCP 프로젝트        | 프로젝트 내 특정 기계학습 job 의 정보를 가져올 수 있습니다.                           |

{{% alert %}}
Vertex AI workload 가 비표준 서비스 계정의 ID 를 사용하려면 Vertex AI 문서의 서비스 계정 생성 및 권한 안내를 참고하세요. launch queue 설정의 `spec.service_account` 필드를 사용해 W&B run 별로 커스텀 서비스 계정을 선택할 수 있습니다.
{{% /alert %}}

## Vertex AI 용 queue 설정하기

Vertex AI 리소스에 대한 queue 설정은 Vertex AI Python SDK 의 `CustomJob` 생성자와 `CustomJob` 클래스의 `run` 메소드에 입력값을 지정합니다. 리소스 설정은 `spec` 및 `run` 키 하위에 저장됩니다:

- `spec` 키에는 Vertex AI Python SDK 의 [`CustomJob` 생성자](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) 에 대한 네임드 인수 값이 들어갑니다.
- `run` 키에는 Vertex AI Python SDK 의 `CustomJob` 클래스의 `run` 메소드에 대한 네임드 인수 값이 들어갑니다.

실행 환경에 대한 커스터마이징은 주로 `spec.worker_pool_specs` 리스트에서 이루어집니다. 워커 풀 스펙은 job 을 실행할 워커 그룹을 정의합니다. 기본 설정에서는 가속기가 없는 단일 `n1-standard-4` 머신을 요청합니다. 필요에 따라 머신 타입, 가속기 종류, 가속기 개수를 변경할 수 있습니다.

사용 가능한 머신 타입 및 가속기 종류에 대한 자세한 내용은 [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) 를 참고하세요.

## 큐 생성하기

W&B App 에서 Vertex AI 를 컴퓨트 리소스로 사용하는 Queue 를 생성할 수 있습니다:

1. [Launch 페이지](https://wandb.ai/launch) 로 이동하세요.
2. **Create Queue** 버튼을 클릭하세요.
3. 큐를 만들고자 하는 **Entity** 를 선택하세요.
4. **Name** 필드에 큐의 이름을 입력하세요.
5. **Resource** 로 **GCP Vertex** 를 선택하세요.
6. **Configuration** 필드에는 이전 단계에서 정의한 Vertex AI `CustomJob` 정보를 입력하세요. 기본적으로 W&B 는 아래와 유사한 YAML 및 JSON request body 예시를 제공합니다.

```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
        accelerator_type: ACCELERATOR_TYPE_UNSPECIFIED
        accelerator_count: 0
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: <REQUIRED>
run:
  restart_job_on_worker_restart: false
```

7. Queue 설정을 마쳤다면 **Create Queue** 버튼을 클릭하세요.

최소한 아래 항목을 반드시 명시해야 합니다:

- `spec.worker_pool_specs` : 워커 풀 스펙이 포함된 비어있지 않은 리스트여야 합니다.
- `spec.staging_bucket` : Vertex AI 자산 및 메타데이터를 staging 할 GCS 버킷입니다.

{{% alert color="secondary" %}}
일부 Vertex AI 문서에서는 워커 풀 스펙의 키가 카멜 케이스(`workerPoolSpecs`)로 표기되어 있습니다. 그러나 Vertex AI Python SDK 에서는 이 키들을 snake case, 예를 들어 `worker_pool_specs` 형식으로 사용해야 합니다.

launch queue 설정에서 모든 키는 snake case 를 사용해야 합니다.
{{% /alert %}}

## launch agent 설정

launch agent 는 기본적으로 `~/.config/wandb/launch-config.yaml` 경로에 위치한 설정 파일을 통해 구성할 수 있습니다.

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Vertex AI 에서 실행될 이미지를 launch agent 가 빌드하도록 설정하려면 [Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ko" >}}) 문서를 참고하세요.

## agent 권한 설정

서비스 계정으로 인증하는 방법은 여러 가지가 있습니다. Workload Identity, 서비스 계정 JSON 파일 다운로드, 환경 변수, Google Cloud Platform CLI 툴 또는 이 방법들을 조합해서 인증할 수 있습니다.