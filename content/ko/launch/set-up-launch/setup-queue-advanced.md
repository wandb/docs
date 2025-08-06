---
title: 런치 큐 구성하기
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

다음 페이지에서는 launch queue 옵션 설정 방법을 설명합니다.

## Queue config 템플릿 설정
Queue Config Templates를 사용하여 컴퓨트 사용량에 대한 가드레일을 관리하고 설정할 수 있습니다. 메모리 사용량, GPU, 런타임 시간과 같은 필드의 기본값, 최소값, 최대값을 지정할 수 있습니다.

queue에 config 템플릿을 설정하면, 팀 구성원들은 지정한 범위 내에서만 필드를 변경할 수 있습니다.

### Queue 템플릿 구성하기
기존 queue에 템플릿을 추가하거나 새 queue를 생성할 때 템플릿을 구성할 수 있습니다.  

1. [W&B Launch App](https://wandb.ai/launch)으로 이동합니다.
2. 템플릿을 추가하려는 queue 이름 옆의 **View queue**를 선택합니다.
3. **Config** 탭을 선택합니다. 이 탭에서는 queue가 생성된 시점, queue 설정, 기존 launch-time 오버라이드 등 queue 정보가 표시됩니다.
4. **Queue config** 섹션으로 이동합니다.
5. 템플릿을 만들고 싶은 config key-value를 찾습니다.
6. config의 값을 템플릿 필드로 교체합니다. 템플릿 필드는 `{{variable-name}}` 형태로 작성됩니다.
7. **Parse configuration** 버튼을 클릭합니다. 설정을 파싱하면, W&B가 queue 설정 아래에 생성한 각 템플릿에 대한 타일을 자동으로 만듭니다.
8. 생성된 각 타일에서 queue config가 허용할 데이터 타입(string, integer, float) 중 하나를 먼저 지정해야 합니다. **Type** 드롭다운에서 데이터 타입을 선택하세요.
9. 선택한 데이터 타입에 따라 각 타일 내에 나타나는 필드를 완성합니다.
10. **Save config**를 클릭합니다.

예를 들어, 팀이 사용할 수 있는 AWS 인스턴스 종류를 제한하는 템플릿을 만들고 싶다고 가정해보겠습니다. 템플릿 필드를 추가하기 전 queue config는 다음과 같을 수 있습니다:

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

`InstanceType`에 템플릿 필드를 추가하면 config는 다음과 같이 됩니다:

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

이제 **Parse configuration**을 클릭하면, **Queue config** 아래에 `aws-instance`라는 이름의 새로운 타일이 나타납니다.

여기서 **Type** 드롭다운에서 String을 데이터 타입으로 선택합니다. 그러면 사용자가 선택할 수 있는 값을 지정할 수 있는 필드가 추가됩니다. 예시 이미지처럼, 팀의 관리자는 사용자가 선택할 수 있도록 두 가지 AWS 인스턴스 타입(`ml.m4.xlarge` 및 `ml.p3.xlarge`)을 설정할 수 있습니다.

{{< img src="/images/launch/aws_template_example.png" alt="AWS CloudFormation template" >}}



## Launch job 동적 구성하기
queue config는 에이전트가 queue에서 job을 디큐할 때 평가되는 매크로를 사용하여 동적으로 지정할 수 있습니다. 다음 매크로들을 사용할 수 있습니다:

| 매크로             | 설명                                                     |
|-------------------|---------------------------------------------------------|
| `${project_name}` | run이 실행될 project의 이름입니다.                         |
| `${entity_name}`  | run이 실행될 project의 소유자입니다.                      |
| `${run_id}`       | 실행되는 run의 ID입니다.                                 |
| `${run_name}`     | 실행되는 run의 이름입니다.                               |
| `${image_uri}`    | 이 run에서 사용할 컨테이너 이미지의 URI입니다.             |

{{% alert %}}
위 표에 없는 임의의 커스텀 매크로(예: `${MY_ENV_VAR}`)는, 에이전트의 환경에서 환경 변수로 대체되어 사용됩니다.
{{% /alert %}}

## launch agent를 사용해 accelerator (GPU)에서 실행할 이미지를 빌드하기
launch로 accelerator 환경에서 실행할 이미지를 빌드할 경우, accelerator base image를 지정해야 할 수도 있습니다.

이 base image는 아래 조건을 만족해야 합니다:

- Debian 호환 가능(Launch Dockerfile에서 python을 가져올 때 apt-get 사용)
- CPU & GPU 하드웨어 명령어셋 호환 가능(사용하고자 하는 GPU가 현재 CUDA 버전을 지원하는지 체크)
- 제공한 accelerator 버전과 ML 알고리즘에 설치된 패키지 간의 호환성
- 하드웨어 호환을 위해 추가 설정이 필요한 패키지 설치

### TensorFlow에서 GPU 사용하기

TensorFlow가 GPU를 제대로 사용할 수 있도록 하려면, queue 리소스 설정에서 `builder.accelerator.base_image` 키에 사용할 Docker 이미지와 해당 이미지 태그를 지정하세요.

예를 들어, `tensorflow/tensorflow:latest-gpu` base image를 사용하면 TensorFlow가 GPU를 제대로 사용할 수 있습니다. 이는 queue의 리소스 설정을 통해 구성할 수 있습니다.

아래 JSON 예시는 queue config에서 TensorFlow base image를 지정하는 방법을 보여줍니다:

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```