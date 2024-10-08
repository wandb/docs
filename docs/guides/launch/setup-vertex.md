---
title: Tutorial: Set up W&B Launch on Vertex AI
displayed_sidebar: default
---

W&B Launch를 사용하여 Vertex AI 트레이닝 작업으로 실행할 작업을 제출할 수 있습니다. Vertex AI 트레이닝 작업을 통해 제공된 알고리즘이나 사용자 정의 알고리즘을 사용하여 Vertex AI 플랫폼에서 기계학습 모델을 트레이닝할 수 있습니다. 런치 작업이 시작되면, Vertex AI는 기본 인프라, 확장성 및 오케스트레이션을 관리합니다.

W&B Launch는 `google-cloud-aiplatform` SDK의 `CustomJob` 클래스를 통해 Vertex AI와 연동됩니다. `CustomJob`의 파라미터는 런치 큐 설정으로 제어할 수 있습니다. Vertex AI는 GCP 외부의 개인 레지스트리에서 이미지를 가져오도록 설정될 수 없습니다. 이는 GCP 또는 공개 레지스트리에 컨테이너 이미지를 저장해야 Vertex AI를 W&B Launch와 함께 사용할 수 있다는 것을 의미합니다. 컨테이너 이미지를 Vertex 작업이 엑세스할 수 있도록 만드는 방법에 대한 자세한 내용은 Vertex AI 문서를 참조하세요.

## 사전 준비 사항

1. **Vertex AI API가 활성화된 GCP 프로젝트를 생성하거나 엑세스하세요.** API를 활성화하는 방법에 대한 자세한 내용은 [GCP API 콘솔 문서](https://support.google.com/googleapi/answer/6158841?hl=en)를 참조하세요.
2. **이미지를 저장할 GCP 아티팩트 레지스트리 저장소를 생성하세요.** Vertex에서 실행하려는 이미지를 저장하기 위해 [GCP 아티팩트 레지스트리 문서](https://cloud.google.com/artifact-registry/docs/overview)를 참조하세요.
3. **Vertex AI가 메타데이터를 저장할 스테이징 GCS 버킷을 만들기** Vertex AI 워크로드와 동일한 지역에 있어야 스테이징 버킷으로 사용할 수 있습니다. 동일한 버킷을 스테이징 및 빌드 컨텍스트에 사용할 수 있습니다.
4. **Vertex AI 작업을 스핀업할 권한을 가진 서비스 계정을 생성하세요.** 서비스 계정에 권한을 할당하는 방법에 대한 자세한 내용은 [GCP IAM 문서](https://cloud.google.com/iam/docs/creating-managing-service-accounts)를 참조하세요.
5. **서비스 계정에 Vertex 작업을 관리할 권한을 부여하세요.**

| 권한                            | 리소스 범위            | 설명                                                                                            |
| ------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create`  | 지정된 GCP 프로젝트    | 프로젝트 내에서 새로운 기계학습 작업을 생성할 수 있습니다.                                       |
| `aiplatform.customJobs.list`    | 지정된 GCP 프로젝트    | 프로젝트 내의 기계학습 작업 목록을 열람할 수 있습니다.                                           |
| `aiplatform.customJobs.get`     | 지정된 GCP 프로젝트    | 특정 기계학습 작업에 대한 정보를 검색할 수 있도록 허용합니다.                                    |

:::info
Vertex AI 워크로드의 아이덴티티를 비표준 서비스 계정으로 할당하려면 필요한 권한 및 서비스 계정 생성에 대한 지침은 Vertex AI 문서를 참조하세요. 런치 큐 설정의 `spec.service_account` 필드를 사용하여 W&B Runs에 사용자 정의 서비스 계정을 선택할 수 있습니다.
:::

## Vertex AI를 위한 큐 설정

Vertex AI 리소스에 대한 큐 설정은 Vertex AI Python SDK의 `CustomJob` 생성자 및 `CustomJob` 클래스의 `run` 메소드에 입력을 지정합니다. 리소스 설정은 `spec` 및 `run` 키 아래에 저장됩니다:

- `spec` 키는 Vertex AI Python SDK에서 [`CustomJob` 생성자](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec)의 명명된 인수를 위한 값을 포함합니다.
- `run` 키는 Vertex AI Python SDK의 `CustomJob` 클래스의 `run` 메소드의 명명된 인수를 위한 값을 포함합니다.

주로 `spec.worker_pool_specs` 리스트에서 실행 환경을 사용자 정의합니다. 워커 풀 스펙은 작업을 실행할 워커 그룹을 정의합니다. 기본 설정의 워커 스펙은 가속기가 없는 `n1-standard-4` 머신 하나를 요구합니다. 필요에 따라 머신 타입, 가속기 타입 및 수량을 변경할 수 있습니다.

사용 가능한 머신 타입 및 가속기 타입에 대한 자세한 내용은 [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec)를 참조하세요.

## 큐 생성하기

Vertex AI를 컴퓨팅 리소스로 사용하는 W&B 앱 내에서 큐를 생성하세요:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 큐를 생성할 **Entity**를 선택합니다.
4. **Name** 필드에 큐의 이름을 입력합니다.
5. **Resource**로 **GCP Vertex**를 선택합니다.
6. **Configuration** 필드에 이전 섹션에서 정의한 Vertex AI `CustomJob`에 대한 정보를 제공합니다. 기본적으로, W&B는 다음과 유사한 YAML 및 JSON 요청 본문을 자동으로 채웁니다:

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

7. 큐 설정 완료 후, **Create Queue** 버튼을 클릭합니다.

최소한 다음 사항을 지정해야 합니다:

- `spec.worker_pool_specs` : 워커 풀 사양의 비어 있지 않은 목록.
- `spec.staging_bucket` : Vertex AI 자산 및 메타데이터를 스테이징하기 위한 GCS 버킷.

:::caution
일부 Vertex AI 문서는 모든 키를 camel case로 나타낸 워커 풀 사양을 보여줍니다, 예를 들어, `workerPoolSpecs`. Vertex AI Python SDK는 이 키들에 대해 snake case를 사용합니다, 예를 들어 `worker_pool_specs`.

런치 큐 설정에서 모든 키는 snake case를 사용해야 합니다.
:::

## Launch 에이전트 설정

런치 에이전트는 기본적으로 `~/.config/wandb/launch-config.yaml`에 위치한 설정 파일을 통해 구성할 수 있습니다.

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Vertex AI에서 실행되는 이미지를 빌드하도록 런치 에이전트를 원하면 [고급 에이전트 설정](./setup-agent-advanced.md)을 참고하세요.

## 에이전트 권한 설정

이 서비스 계정으로 인증하는 여러 메소드를 사용할 수 있습니다. 워크로드 아이덴티티, 다운로드한 서비스 계정 JSON, 환경 변수, Google Cloud Platform 명령줄 툴 또는 이러한 방법의 조합을 통해 이를 수행할 수 있습니다.