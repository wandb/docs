---
title: Configure launch queue
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

다음 페이지에서는 Launch Queue 옵션을 구성하는 방법을 설명합니다.

## Queue Config 템플릿 설정
Queue Config 템플릿으로 컴퓨팅 소비에 대한 안전 장치를 관리하세요. 메모리 소비, GPU, 런타임 지속 시간과 같은 필드에 대한 기본값, 최소값 및 최대값을 설정합니다.

Config 템플릿으로 Queue를 구성한 후에는 팀 구성원이 지정한 범위 내에서만 정의한 필드를 변경할 수 있습니다.

### Queue 템플릿 구성
기존 Queue에서 Queue 템플릿을 구성하거나 새 Queue를 만들 수 있습니다.

1. [https://wandb.ai/launch](https://wandb.ai/launch)의 Launch App으로 이동합니다.
2. 템플릿을 추가할 Queue 이름 옆에 있는 **Queue 보기**를 선택합니다.
3. **Config** 탭을 선택합니다. 그러면 Queue 생성 시기, Queue Config, 기존 Launch 시 재정의와 같은 Queue에 대한 정보가 표시됩니다.
4. **Queue Config** 섹션으로 이동합니다.
5. 템플릿을 만들 Config 키-값을 식별합니다.
6. Config의 값을 템플릿 필드로 바꿉니다. 템플릿 필드는 `{{variable-name}}` 형식을 취합니다.
7. **구성 파싱** 버튼을 클릭합니다. 구성을 파싱하면 W&B에서 생성한 각 템플릿에 대해 Queue Config 아래에 타일을 자동으로 만듭니다.
8. 생성된 각 타일에 대해 먼저 Queue Config에서 허용할 수 있는 데이터 유형(문자열, 정수 또는 부동 소수점)을 지정해야 합니다. 이렇게 하려면 **유형** 드롭다운 메뉴에서 데이터 유형을 선택합니다.
9. 데이터 유형에 따라 각 타일에 나타나는 필드를 완성합니다.
10. **Config 저장**을 클릭합니다.

예를 들어 팀에서 사용할 수 있는 AWS 인스턴스를 제한하는 템플릿을 만들려고 한다고 가정합니다. 템플릿 필드를 추가하기 전에 Queue Config는 다음과 유사하게 보일 수 있습니다.

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

`InstanceType`에 대한 템플릿 필드를 추가하면 Config는 다음과 같이 표시됩니다.

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

다음으로 **구성 파싱**을 클릭합니다. `aws-instance`라는 새 타일이 **Queue Config** 아래에 나타납니다.

거기에서 **유형** 드롭다운에서 문자열을 데이터 유형으로 선택합니다. 그러면 사용자가 선택할 수 있는 값을 지정할 수 있는 필드가 채워집니다. 예를 들어 다음 이미지에서 팀의 관리자는 사용자가 선택할 수 있는 두 가지 다른 AWS 인스턴스 유형(`ml.m4.xlarge` 및 `ml.p3.xlarge`)을 구성했습니다.

{{< img src="/images/launch/aws_template_example.png" alt="" >}}

## Launch 작업을 동적으로 구성
Queue Config는 에이전트가 Queue에서 작업을 뺄 때 평가되는 매크로를 사용하여 동적으로 구성할 수 있습니다. 다음 매크로를 설정할 수 있습니다.

| 매크로             | 설명                                                         |
|-------------------|-------------------------------------------------------------|
| `${project_name}` | Run이 시작되는 프로젝트의 이름입니다.                        |
| `${entity_name}`  | Run이 시작되는 프로젝트의 소유자입니다.                        |
| `${run_id}`       | 시작되는 Run의 ID입니다.                                      |
| `${run_name}`     | 시작되는 Run의 이름입니다.                                     |
| `${image_uri}`    | 이 Run에 대한 컨테이너 이미지의 URI입니다.                     |

{{% alert %}}
이전 표에 나열되지 않은 사용자 지정 매크로(예: `${MY_ENV_VAR}`)는 에이전트 환경의 환경 변수로 대체됩니다.
{{% /alert %}}

## Launch 에이전트를 사용하여 가속기(GPU)에서 실행되는 이미지를 빌드합니다.
Launch를 사용하여 가속기 환경에서 실행되는 이미지를 빌드하는 경우 가속기 기본 이미지를 지정해야 할 수 있습니다.

이 가속기 기본 이미지는 다음 요구 사항을 충족해야 합니다.

- Debian 호환성(Launch Dockerfile은 apt-get을 사용하여 Python을 가져옴)
- 호환 가능한 CPU 및 GPU 하드웨어 명령어 세트(CUDA 버전이 사용하려는 GPU에서 지원되는지 확인)
- 제공하는 가속기 버전과 ML 알고리즘에 설치된 패키지 간의 호환성
- 하드웨어와의 호환성을 설정하기 위해 추가 단계가 필요한 설치된 패키지

### TensorFlow와 함께 GPU를 사용하는 방법

TensorFlow가 GPU를 제대로 활용하는지 확인합니다. 이를 위해 Queue 리소스 구성에서 `builder.accelerator.base_image` 키에 대한 Docker 이미지와 해당 이미지 태그를 지정합니다.

예를 들어 `tensorflow/tensorflow:latest-gpu` 기본 이미지는 TensorFlow가 GPU를 제대로 사용하는지 확인합니다. 이는 Queue의 리소스 구성을 사용하여 구성할 수 있습니다.

다음 JSON 스니펫은 Queue Config에서 TensorFlow 기본 이미지를 지정하는 방법을 보여줍니다.

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```