---
displayed_sidebar: default
---

# Vertex AI 설정하기

W&B Launch를 사용하여 Vertex AI 트레이닝 작업으로 실행 작업을 제출할 수 있습니다. Vertex AI 트레이닝 작업을 사용하면 제공되거나 사용자 지정 알고리즘을 사용하여 Vertex AI 플랫폼에서 기계학습 모델을 트레이닝할 수 있습니다. Launch 작업이 시작되면 Vertex AI는 기본 인프라, 확장 및 오케스트레이션을 관리합니다.

W&B Launch는 `google-cloud-aiplatform` SDK의 `CustomJob` 클래스를 통해 Vertex AI와 작동합니다. `CustomJob`의 파라미터는 Launch 큐 설정으로 제어될 수 있습니다. GCP 외부의 개인 레지스트리에서 이미지를 가져올 수 있도록 Vertex AI를 구성할 수 없습니다. 이는 W&B Launch와 Vertex AI를 사용하려면 컨테이너 이미지를 GCP나 공개 레지스트리에 저장해야 함을 의미합니다. Vertex 작업에 컨테이너 이미지를 접근 가능하게 하는 방법에 대한 자세한 정보는 Vertex AI 문서를 참조하세요.

## 사전 요구 사항

1. **Vertex AI API가 활성화된 GCP 프로젝트를 생성하거나 엑세스합니다.** API를 활성화하는 방법에 대한 자세한 정보는 [GCP API 콘솔 문서](https://support.google.com/googleapi/answer/6158841?hl=en)를 참조하세요.
2. **Vertex에서 실행하려는 이미지를 저장할 GCP 아티팩트 레지스트리 리포지토리를 생성합니다.** 자세한 정보는 [GCP 아티팩트 레지스트리 문서](https://cloud.google.com/artifact-registry/docs/overview)를 참조하세요.
3. **Vertex AI가 메타데이터를 저장할 스테이징 GCS 버킷을 생성합니다.** 이 버킷은 스테이징 버킷으로 사용되기 위해 Vertex AI 작업과 같은 지역에 있어야 합니다. 동일한 버킷을 스테이징 및 빌드 컨텍스트에 사용할 수 있습니다.
4. **Vertex AI 작업을 시작하는 데 필요한 권한이 있는 서비스 계정을 생성합니다.** 서비스 계정에 권한을 할당하는 방법에 대한 자세한 정보는 [GCP IAM 문서](https://cloud.google.com/iam/docs/creating-managing-service-accounts)를 참조하세요.
5. **서비스 계정에 Vertex 작업을 관리할 권한 부여**

| 권한                            | 리소스 범위           | 설명                                                                                   |
| ------------------------------ | --------------------- | -------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | 지정된 GCP 프로젝트    | 프로젝트 내에서 새로운 기계학습 작업을 생성할 수 있습니다.                              |
| `aiplatform.customJobs.list`   | 지정된 GCP 프로젝트    | 프로젝트 내에서 기계학습 작업을 나열할 수 있습니다.                                      |
| `aiplatform.customJobs.get`    | 지정된 GCP 프로젝트    | 프로젝트 내 특정 기계학습 작업에 대한 정보를 검색할 수 있습니다.                         |

:::info
Vertex AI 작업이 비표준 서비스 계정의 신원을 가정하길 원한다면, 서비스 계정 생성 및 필요한 권한에 대한 지침은 Vertex AI 문서를 참조하세요. Launch 큐 설정의 `spec.service_account` 필드를 사용하여 W&B 실행에 대한 사용자 지정 서비스 계정을 선택할 수 있습니다.
:::

## Vertex AI용 큐 구성하기

Vertex AI 리소스에 대한 큐 설정은 Vertex AI Python SDK의 `CustomJob` 생성자에 대한 입력과 `CustomJob`의 `run` 메소드를 지정합니다. 리소스 설정은 `spec` 및 `run` 키 아래에 저장됩니다:

- `spec` 키는 Vertex AI Python SDK의 [`CustomJob` 생성자](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec)의 명명된 인수에 대한 값이 포함됩니다.
- `run` 키는 Vertex AI Python SDK의 `CustomJob` 클래스의 `run` 메소드의 명명된 인수에 대한 값이 포함됩니다.

실행 환경의 사용자 지정은 주로 `spec.worker_pool_specs` 리스트에서 발생합니다. Worker pool spec은 작업을 실행할 일련의 워커를 정의합니다. 기본 설정의 워커 스펙은 가속기 없이 단일 `n1-standard-4` 기계를 요청합니다. 필요에 따라 기계 유형, 가속기 유형 및 수를 변경할 수 있습니다.

사용 가능한 기계 유형 및 가속기 유형에 대한 자세한 정보는 [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec)를 참조하세요.

## 큐 생성하기

W&B 앱에서 Vertex AI를 컴퓨트 리소스로 사용하는 큐를 생성합니다:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **큐 생성** 버튼을 클릭합니다.
3. 큐를 생성하려는 **엔티티**를 선택합니다.
4. **이름** 필드에 큐의 이름을 입력합니다.
5. **리소스**로 **GCP Vertex**를 선택합니다.
6. **설정** 필드에 이전 섹션에서 정의한 Vertex AI `CustomJob`에 대한 정보를 제공합니다. 기본적으로 W&B는 다음과 같은 YAML 및 JSON 요청 본문을 채웁니다:

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

7. 큐를 구성한 후 **큐 생성** 버튼을 클릭합니다.

최소한 다음을 지정해야 합니다:

- `spec.worker_pool_specs` : 비어 있지 않은 worker pool 사양 목록.
- `spec.staging_bucket` : Vertex AI 자산 및 메타데이터를 스테이징하기 위한 GCS 버킷.

:::caution
일부 Vertex AI 문서는 모든 키를 카멜 케이스로 표시하는 worker pool 사양을 보여줍니다. 예를 들어, `workerPoolSpecs`. Vertex AI Python SDK는 이러한 키를 스네이크 케이스로 사용합니다. 예를 들어 `worker_pool_specs`.

Launch 큐 설정의 모든 키는 스네이크 케이스를 사용해야 합니다.
:::

## Launch 에이전트 구성하기

Launch 에이전트는 기본적으로 `~/.config/wandb/launch-config.yaml`에 위치한 설정 파일을 통해 구성할 수 있습니다.

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Launch 에이전트가 Vertex AI에서 실행될 이미지를 빌드하길 원한다면 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.

## 에이전트 권한 설정하기

이 서비스 계정으로 인증하는 여러 방법이 있습니다. 이는 Workload Identity, 다운로드된 서비스 계정 JSON, 환경 변수, Google Cloud Platform 커맨드라인 툴 또는 이러한 방법의 조합을 통해 달성될 수 있습니다.