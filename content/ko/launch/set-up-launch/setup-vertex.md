---
title: 'Tutorial: Set up W&B Launch on Vertex AI'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch를 사용하여 Vertex AI 트레이닝 작업으로 실행하기 위한 작업을 제출할 수 있습니다. Vertex AI 트레이닝 작업을 통해 Vertex AI 플랫폼에서 제공되거나 사용자 정의된 알고리즘을 사용하여 기계 학습 모델을 트레이닝할 수 있습니다. Launch 작업이 시작되면 Vertex AI는 기본 인프라, 확장 및 오케스트레이션을 관리합니다.

W&B Launch는 `google-cloud-aiplatform` SDK의 `CustomJob` 클래스를 통해 Vertex AI와 연동됩니다. `CustomJob`의 파라미터는 Launch 대기열 설정으로 제어할 수 있습니다. Vertex AI는 GCP 외부의 개인 레지스트리에서 이미지를 가져오도록 구성할 수 없습니다. 즉, W&B Launch와 함께 Vertex AI를 사용하려면 컨테이너 이미지를 GCP 또는 공개 레지스트리에 저장해야 합니다. 컨테이너 이미지를 Vertex 작업에 엑세스할 수 있도록 설정하는 방법에 대한 자세한 내용은 Vertex AI 문서를 참조하십시오.

## 전제 조건

1. **Vertex AI API가 활성화된 GCP 프로젝트를 만들거나 엑세스합니다.** API 활성화에 대한 자세한 내용은 [GCP API Console 문서](https://support.google.com/googleapi/answer/6158841?hl=en)를 참조하십시오.
2. **Vertex에서 실행할 이미지를 저장할 GCP Artifact Registry 저장소를 만듭니다.** 자세한 내용은 [GCP Artifact Registry 문서](https://cloud.google.com/artifact-registry/docs/overview)를 참조하십시오.
3. Vertex AI가 메타데이터를 저장할 **스테이징 GCS 버킷을 만듭니다**. 이 버킷은 스테이징 버킷으로 사용하려면 Vertex AI 워크로드와 동일한 리전에 있어야 합니다. 동일한 버킷을 스테이징 및 빌드 컨텍스트에 사용할 수 있습니다.
4. Vertex AI 작업을 스핀업하는 데 필요한 권한이 있는 **서비스 계정을 만듭니다**. 서비스 계정에 권한을 할당하는 방법에 대한 자세한 내용은 [GCP IAM 문서](https://cloud.google.com/iam/docs/creating-managing-service-accounts)를 참조하십시오.
5. **Vertex 작업을 관리할 수 있는 권한을 서비스 계정에 부여합니다.**

| 권한                         | 리소스 범위           | 설명                                                                                       |
| ------------------------------ | --------------------- | ------------------------------------------------------------------------------------------ |
| `aiplatform.customJobs.create` | 지정된 GCP 프로젝트 | 프로젝트 내에서 새로운 기계 학습 작업을 생성할 수 있습니다.                                 |
| `aiplatform.customJobs.list`   | 지정된 GCP 프로젝트 | 프로젝트 내에서 기계 학습 작업 목록을 볼 수 있습니다.                                     |
| `aiplatform.customJobs.get`    | 지정된 GCP 프로젝트 | 프로젝트 내에서 특정 기계 학습 작업에 대한 정보를 검색할 수 있습니다.                         |

{{% alert %}}
Vertex AI 워크로드가 표준이 아닌 서비스 계정의 ID를 사용하도록 하려면 서비스 계정 생성 및 필요한 권한에 대한 지침은 Vertex AI 문서를 참조하십시오. Launch 대기열 설정의 `spec.service_account` 필드를 사용하여 W&B Runs에 대한 사용자 정의 서비스 계정을 선택할 수 있습니다.
{{% /alert %}}

## Vertex AI용 대기열 구성하기

Vertex AI 리소스에 대한 대기열 설정은 Vertex AI Python SDK의 `CustomJob` 생성자와 `CustomJob`의 `run` 메소드에 대한 입력을 지정합니다. 리소스 설정은 `spec` 및 `run` 키 아래에 저장됩니다.

- `spec` 키에는 Vertex AI Python SDK의 [`CustomJob` 생성자](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component)의 명명된 인수에 대한 값이 포함되어 있습니다.
- `run` 키에는 Vertex AI Python SDK의 `CustomJob` 클래스의 `run` 메소드의 명명된 인수에 대한 값이 포함되어 있습니다.

실행 환경의 사용자 정의는 주로 `spec.worker_pool_specs` 목록에서 이루어집니다. 작업자 풀 사양은 작업을 실행할 작업자 그룹을 정의합니다. 기본 설정의 작업자 사양은 가속기 없이 단일 `n1-standard-4` 머신을 요청합니다. 필요에 따라 머신 유형, 가속기 유형 및 수를 변경할 수 있습니다.

사용 가능한 머신 유형 및 가속기 유형에 대한 자세한 내용은 [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec)를 참조하십시오.

## 대기열 만들기

Vertex AI를 컴퓨팅 리소스로 사용하는 대기열을 W&B App에서 만듭니다.

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 대기열을 만들 **Entity**를 선택합니다.
4. **Name** 필드에 대기열 이름을 입력합니다.
5. **Resource**로 **GCP Vertex**를 선택합니다.
6. **Configuration** 필드 내에서 이전 섹션에서 정의한 Vertex AI `CustomJob`에 대한 정보를 제공합니다. 기본적으로 W&B는 다음과 유사한 YAML 및 JSON 요청 본문을 채웁니다.

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

7. 대기열을 구성한 후 **Create Queue** 버튼을 클릭합니다.

최소한 다음을 지정해야 합니다.

- `spec.worker_pool_specs` : 비어 있지 않은 작업자 풀 사양 목록.
- `spec.staging_bucket` : Vertex AI 자산 및 메타데이터를 스테이징하는 데 사용할 GCS 버킷.

{{% alert color="secondary" %}}
일부 Vertex AI 문서에서는 모든 키가 카멜 케이스인 작업자 풀 사양(예: ` workerPoolSpecs`)을 보여줍니다. Vertex AI Python SDK는 이러한 키에 대해 스네이크 케이스(예: `worker_pool_specs`)를 사용합니다.

Launch 대기열 설정의 모든 키는 스네이크 케이스를 사용해야 합니다.
{{% /alert %}}

## Launch 에이전트 구성하기

Launch 에이전트는 기본적으로 `~/.config/wandb/launch-config.yaml`에 있는 구성 파일을 통해 구성할 수 있습니다.

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Launch 에이전트가 Vertex AI에서 실행되는 이미지를 빌드하도록 하려면 [고급 에이전트 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})을 참조하십시오.

## 에이전트 권한 설정하기

이 서비스 계정으로 인증하는 방법에는 여러 가지가 있습니다. 이는 워크로드 ID, 다운로드된 서비스 계정 JSON, 환경 변수, Google Cloud Platform 코맨드 라인 툴 또는 이러한 방법의 조합을 통해 수행할 수 있습니다.
