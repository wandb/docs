---
displayed_sidebar: default
---

# 고급 큐 설정
다음 페이지에서는 추가적인 실행 대기열 옵션을 구성하는 방법에 대해 설명합니다.

## 큐 설정 템플릿 설정
큐 설정 템플릿을 사용하여 컴퓨팅 소비에 대한 가드레일을 관리하고 관리하세요. 메모리 소비, GPU, 실행 시간 등의 필드에 대한 기본값, 최소값, 최대값을 설정합니다.

큐를 설정 템플릿으로 구성한 후에는 팀 구성원이 정의한 범위 내에서만 정의한 필드를 변경할 수 있습니다.

### 큐 템플릿 구성
기존 큐에 큐 템플릿을 구성하거나 새 큐를 생성할 수 있습니다.

1. [https://wandb.ai/launch](https://wandb.ai/launch)에서 Launch App으로 이동합니다.
2. 템플릿을 추가하려는 큐 이름 옆에 있는 **View queue**를 선택합니다.
3. **Config** 탭을 선택합니다. 이는 큐가 생성된 시간, 큐 구성 및 기존 실행 시간 재정의에 대한 정보를 표시합니다.
4. **Queue config** 섹션으로 이동합니다.
5. 템플릿을 생성하려는 구성 키-값을 확인합니다.
6. 구성의 값을 템플릿 필드로 대체합니다. 템플릿 필드는 `{{variable-name}}`의 형태를 취합니다.
7. **Parse configuration** 버튼을 클릭합니다. 구성을 분석하면 W&B는 생성한 각 템플릿에 대해 큐 구성 아래에 타일을 자동으로 생성합니다.
8. 생성된 각 타일에 대해 먼저 큐 구성이 허용할 수 있는 데이터 유형(문자열, 정수, 또는 실수)을 지정해야 합니다. 이를 위해 **Type** 드롭다운 메뉴에서 데이터 유형을 선택합니다.
9. 데이터 유형을 기반으로 각 타일 내에 나타나는 필드를 완성합니다.
10. **Save config**를 클릭합니다.

예를 들어, 팀이 사용할 수 있는 AWS 인스턴스를 제한하는 템플릿을 생성하려는 경우 템플릿 필드를 추가하기 전에 큐 구성은 다음과 비슷할 수 있습니다:

```yaml title="launch config"
RoleArn: arn:aws:iam:region:account-id:resource-type/resource-id
ResourceConfig:
  InstanceType: ml.m4.xlarge
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://bucketname
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

`InstanceType`에 대해 템플릿 필드를 추가하면 구성은 다음과 같이 보일 것입니다:

```yaml title="launch config"
RoleArn: arn:aws:iam:region:account-id:resource-type/resource-id
ResourceConfig:
  InstanceType: "{{aws_instance}}"
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://bucketname
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

다음으로, **Parse configuration**을 클릭합니다. **Queue config** 아래에 `aws-instance`라고 레이블이 붙은 새 타일이 나타날 것입니다.

그 후, **Type** 드롭다운에서 String을 데이터 유형으로 선택합니다. 이렇게 하면 사용자가 선택할 수 있는 값들을 지정할 수 있는 필드가 나타납니다. 예를 들어, 다음 이미지에서는 팀의 관리자가 사용자가 선택할 수 있는 두 가지 다른 AWS 인스턴스 유형(`ml.m4.xlarge` 및 `ml.p3.xlarge`)을 구성했습니다:

![](/images/launch/aws_template_example.png)

## 동적으로 실행 작업 구성
에이전트가 큐에서 작업을 디큐할 때 평가되는 매크로를 사용하여 큐 구성을 동적으로 구성할 수 있습니다. 다음 매크로를 설정할 수 있습니다:

| 매크로             | 설명                                         |
|-------------------|-------------------------------------------------|
| `${project_name}` | 실행이 시작되는 프로젝트의 이름입니다.       |
| `${entity_name}`  | 실행이 시작되는 프로젝트의 소유자입니다.     |
| `${run_id}`       | 시작되는 실행의 ID입니다.                    |
| `${run_name}`     | 시작하는 실행의 이름입니다.                  |
| `${image_uri}`    | 이 실행에 대한 컨테이너 이미지의 URI입니다.  |

:::info
위의 표에 나열되지 않은 사용자 정의 매크로(예: `${MY_ENV_VAR}`)는 에이전트의 환경에서 환경 변수로 대체됩니다.
:::

## 가속기(GPU)에서 실행되는 이미지를 빌드하기 위해 실행 에이전트 사용
가속 환경에서 실행되는 이미지를 빌드하기 위해 가속기 기반 이미지를 지정해야 할 수 있습니다.

이 가속기 기반 이미지는 다음 요구 사항을 충족해야 합니다:

- 데비안 호환성(Launch Dockerfile은 python을 가져오기 위해 apt-get을 사용합니다)
- CPU 및 GPU 하드웨어 명령 세트 호환성(사용하려는 GPU가 지원하는 CUDA 버전인지 확인하세요)
- 제공하는 가속기 버전과 ML 알고리즘에 설치된 패키지 간의 호환성
- 하드웨어와의 호환성 설정을 위한 추가 단계가 필요한 설치된 패키지들

### TensorFlow와 함께 GPU 사용하기

TensorFlow가 GPU를 올바르게 사용하도록 합니다. 이를 위해 큐 리소스 구성에서 `builder.accelerator.base_image` 키에 Docker 이미지와 이미지 태그를 지정합니다.

예를 들어, `tensorflow/tensorflow:latest-gpu` 기본 이미지는 TensorFlow가 GPU를 올바르게 사용하도록 보장합니다. 이는 큐 구성에서 리소스 구성을 사용하여 구성할 수 있습니다.

다음 JSON 스니펫은 큐 구성에서 TensorFlow 기본 이미지를 지정하는 방법을 보여줍니다:

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```