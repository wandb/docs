---
title: Configure launch queue
displayed_sidebar: default
---

다음 페이지에서는 `launch` 큐 옵션을 설정하는 방법을 설명합니다.

## 큐 설정 템플릿 구성
Queue Config Templates를 사용하여 컴퓨팅 소비에 대한 가드레일을 관리합니다. 메모리 소비, GPU, 런타임 지속 시간과 같은 필드에 대한 기본값, 최소값 및 최대값을 설정할 수 있습니다.

큐를 설정 템플릿으로 구성한 후 팀 구성원은 지정한 범위 내에서만 정의한 필드를 변경할 수 있습니다.

### 큐 템플릿 구성
기존 큐에서 큐 템플릿을 구성하거나 새 큐를 생성할 수 있습니다.

1. [https://wandb.ai/launch](https://wandb.ai/launch)에서 Launch App으로 이동합니다.
2. 템플릿을 추가하려는 큐 이름 옆의 **View queue**를 선택합니다.
3. **Config** 탭을 선택합니다. 여기에는 큐의 생성 시기, 큐 설정 및 기존 런치 타임 재정의와 같은 정보가 표시됩니다.
4. **Queue config** 섹션으로 이동합니다.
5. 템플릿을 생성하려는 설정 키-값을 식별합니다.
6. 설정의 값을 템플릿 필드로 교체합니다. 템플릿 필드는 `{{variable-name}}`의 형태를 취합니다.
7. **Parse configuration** 버튼을 클릭합니다. 설정을 구문 분석하면 W&B가 생성한 각 템플릿에 대해 큐 설정 아래에 타일을 자동으로 생성합니다.
8. 생성된 각 타일에 대해 큐 설정에서 허용할 수 있는 데이터 유형(문자열, 정수 또는 부동 소수점)을 먼저 지정해야 합니다. 이를 위해 **Type** 드롭다운 메뉴에서 데이터 유형을 선택합니다.
9. 데이터 유형에 따라 각 타일 내에서 나타나는 필드를 작성합니다.
10. **Save config**를 클릭합니다.

예를 들어, 팀이 사용할 수 있는 AWS 인스턴스를 제한하는 템플릿을 만들고자 한다고 가정해 보겠습니다. 템플릿 필드를 추가하기 전, 큐 설정은 다음과 유사하게 보일 것입니다:

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

`InstanceType`에 대한 템플릿 필드를 추가하면 설정은 다음과 같이 보입니다:

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

그 다음에, **Parse configuration**을 클릭합니다. 그러면 **Queue config** 아래에 `aws-instance`라는 새 타일이 나타납니다.

여기서 **Type** 드롭다운에서 데이터 유형으로 문자열을 선택합니다. 그러면 사용자가 선택할 수 있는 값을 지정할 수 있는 필드가 나타납니다. 예를 들어, 다음 이미지에서 팀 관리자는 다른 두 가지 AWS 인스턴스 유형을 구성하여 사용자가 선택할 수 있도록 했습니다 (`ml.m4.xlarge` 및 `ml.p3.xlarge`):

![](/images/launch/aws_template_example.png)

## 런치 작업을 동적으로 구성
큐 설정은 에이전트가 큐에서 작업을 디큐할 때 평가되는 매크로를 사용하여 동적으로 구성될 수 있습니다. 다음과 같은 매크로를 설정할 수 있습니다:

| Macro             | 설명                                                   |
|-------------------|-------------------------------------------------------|
| `${project_name}` | run이 런치될 프로젝트의 이름                            |
| `${entity_name}`  | run이 런치될 프로젝트의 소유자                         |
| `${run_id}`       | 런치될 run의 ID                                       |
| `${run_name}`     | 런치되는 run의 이름                                    |
| `${image_uri}`    | 이 run에 대한 컨테이너 이미지의 URI                   |

:::info
위에 나열되지 않은 사용자 정의 매크로(예: `${MY_ENV_VAR}`)는 에이전트의 환경에서 환경 변수로 대체됩니다.
:::

## Launch 에이전트를 사용하여 가속기(GPU)에서 실행하는 이미지를 빌드
Launch를 사용하여 가속기 환경에서 실행할 이미지를 빌드하는 경우 가속기 베이스 이미지를 지정해야 할 수 있습니다.

이 가속기 베이스 이미지는 다음 요구 사항을 충족해야 합니다:

- Debian 호환성(Launch Dockerfile은 python을 가져오기 위해 apt-get을 사용)
- 호환 가능한 CPU & GPU 하드웨어 명령어 집합(사용할 GPU에서 지원하는 CUDA 버전인지 확인)
- 제공한 가속기 버전과 ML 알고리즘에서 설치한 패키지 간의 호환성
- 하드웨어와의 호환성을 설정하기 위해 추가 단계가 필요한 패키지 설치

### TensorFlow와 GPU 사용 방법

TensorFlow가 GPU를 제대로 활용하게 하려면, `builder.accelerator.base_image` 키에 대한 Docker 이미지와 그 이미지 태그를 큐 리소스 설정에 지정하십시오.

예를 들어, `tensorflow/tensorflow:latest-gpu` 베이스 이미지는 TensorFlow가 GPU를 제대로 사용할 수 있도록 보장합니다. 이는 큐 리소스 설정을 사용하여 구성할 수 있습니다.

다음 JSON 스니펫은 큐 설정에서 TensorFlow 베이스 이미지를 지정하는 방법을 보여줍니다:

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```