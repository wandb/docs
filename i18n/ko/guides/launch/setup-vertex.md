---
displayed_sidebar: default
---

# Vertex AI 설정하기

W&B Launch를 사용하여 실행 작업을 Vertex AI 학습 작업으로 제출할 수 있습니다. Vertex AI 학습 작업을 통해 제공되거나 사용자 정의 알고리즘을 사용하여 Vertex AI 플랫폼에서 머신 러닝 모델을 학습할 수 있습니다. Launch 작업이 시작되면 Vertex AI가 기본 인프라, 확장 및 오케스트레이션을 관리합니다.

W&B Launch는 `google-cloud-aiplatform` SDK의 `CustomJob` 클래스를 통해 Vertex AI와 작동합니다. `CustomJob`의 파라미터는 launch 큐 구성을 통해 제어할 수 있습니다. Vertex AI는 GCP 외부의 개인 레지스트리에서 이미지를 가져올 수 있도록 구성할 수 없습니다. 이는 W&B Launch와 Vertex AI를 사용하려면 컨테이너 이미지를 GCP 또는 공개 레지스트리에 저장해야 함을 의미합니다. 컨테이너 이미지를 Vertex 작업에 액세스할 수 있도록 하는 자세한 정보는 Vertex AI 문서를 참조하십시오.

## 전제 조건

1. **Vertex AI API가 활성화된 GCP 프로젝트를 생성하거나 액세스합니다.** API를 활성화하는 방법에 대한 자세한 정보는 [GCP API 콘솔 문서](https://support.google.com/googleapi/answer/6158841?hl=en)를 참조하십시오.
2. **Vertex에서 실행하려는 이미지를 저장할 GCP Artifact Registry 저장소를 생성합니다.** 자세한 정보는 [GCP Artifact Registry 문서](https://cloud.google.com/artifact-registry/docs/overview)를 참조하십시오.
3. **Vertex AI가 메타데이터를 저장할 스테이징 GCS 버킷을 생성합니다.** 이 버킷은 스테이징 버킷으로 사용되려면 Vertex AI 작업과 같은 지역에 있어야 합니다. 같은 버킷을 스테이징 및 빌드 컨텍스트에 사용할 수 있습니다.
4. **Vertex AI 작업을 시작하는 데 필요한 권한이 있는 서비스 계정을 생성합니다.** 권한을 서비스 계정에 할당하는 방법에 대한 자세한 정보는 [GCP IAM 문서](https://cloud.google.com/iam/docs/creating-managing-service-accounts)를 참조하십시오.
5. **서비스 계정에 Vertex 작업 관리 권한 부여**

|    권한    |    리소스 범위     |      설명      | 
| ---------------- | --------------------- | --------------------- |
| `ml.jobs.create` | 지정된 GCP 프로젝트 | 프로젝트 내에서 새로운 머신 러닝 작업 생성을 허용합니다.    |
| `ml.jobs.list`   | 지정된 GCP 프로젝트 | 프로젝트 내에서 머신 러닝 작업 목록을 허용합니다.  |
| `ml.jobs.get`    | 지정된 GCP 프로젝트 | 프로젝트 내에서 특정 머신 러닝 작업에 대한 정보 검색을 허용합니다. |

:::info
Vertex AI 작업이 비표준 서비스 계정의 ID를 사용하도록 하려면 Vertex AI 문서를 참조하여 서비스 계정 생성 및 필요한 권한에 대한 지침을 참조하십시오. Launch 큐 구성의 `spec.service_account` 필드를 사용하여 W&B 실행에 대한 사용자 정의 서비스 계정을 선택할 수 있습니다.
:::

## Vertex AI용 큐 구성하기
Vertex AI 리소스에 대한 큐 구성은 Vertex AI Python SDK의 `CustomJob` 생성자 및 `CustomJob`의 `run` 메서드에 대한 입력을 지정합니다. 리소스 구성은 `spec` 및 `run` 키 아래에 저장됩니다:

- `spec` 키는 Vertex AI Python SDK의 [`CustomJob` 생성자](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec)의 명명된 인수에 대한 값이 포함됩니다.
- `run` 키는 Vertex AI Python SDK의 `CustomJob` 클래스의 `run` 메서드의 명명된 인수에 대한 값이 포함됩니다.

실행 환경의 사용자 지정은 주로 `spec.worker_pool_specs` 리스트에서 이루어집니다. Worker pool spec은 작업을 실행할 작업자 그룹을 정의합니다. 기본 설정의 worker spec은 가속기가 없는 하나의 `n1-standard-4` 기계를 요청합니다. 필요에 따라 기계 유형, 가속기 유형 및 수를 변경할 수 있습니다.

사용 가능한 기계 유형 및 가속기 유형에 대한 자세한 정보는 [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec)를 참조하십시오.

## 큐 생성하기

Vertex AI를 컴퓨팅 리소스로 사용하는 W&B 앱에서 큐를 생성합니다:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **큐 생성** 버튼을 클릭합니다.
3. 큐를 생성하고자 하는 **엔터티**를 선택합니다.
4. **이름** 필드에 큐의 이름을 입력합니다.
5. **리소스**로 **GCP Vertex**를 선택합니다.
6. **구성** 필드에 이전 섹션에서 정의한 Vertex AI `CustomJob`에 대한 정보를 제공합니다. 기본적으로 W&B는 다음과 같은 YAML 및 JSON 요청 본문을 채웁니다:
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
* `spec.worker_pool_specs` : 비어 있지 않은 worker pool 명세 목록.
* `spec.staging_bucket` : Vertex AI 자산 및 메타데이터를 스테이징하는 데 사용될 GCS 버킷.

:::caution
일부 Vertex AI 문서는 모든 키를 camel case로 표시한 worker pool 명세를 보여주며, 예를 들어 `workerPoolSpecs`입니다. Vertex AI Python SDK는 이러한 키에 대해 snake case를 사용하며, 예를 들어 `worker_pool_specs`입니다.

launch 큐 구성의 모든 키는 snake case를 사용해야 합니다.
:::

## Launch 에이전트 구성하기
launch 에이전트는 기본적으로 `~/.config/wandb/launch-config.yaml`에 위치한 구성 파일을 통해 구성할 수 있습니다.

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

launch 에이전트가 Vertex AI에서 실행될 이미지를 빌드하도록 하려면 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하십시오.

## 에이전트 권한 설정하기
이 서비스 계정으로 인증하는 여러 방법이 있습니다. 이는 Workload Identity, 다운로드된 서비스 계정 JSON, 환경 변수, Google Cloud Platform 명령줄 도구 또는 이러한 방법의 조합을 통해 달성할 수 있습니다.