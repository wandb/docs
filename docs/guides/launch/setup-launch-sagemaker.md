---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SageMaker 설정하기

W&B Launch를 사용하여 제공되거나 사용자 정의 알고리즘을 사용하여 SageMaker 플랫폼에서 머신 러닝 모델을 학습하기 위한 작업을 Amazon SageMaker에 제출할 수 있습니다. SageMaker는 컴퓨팅 리소스를 확보하고 해제하는 작업을 관리하므로, EKS 클러스터가 없는 팀에게 좋은 선택이 될 수 있습니다.

Amazon SageMaker로 전송된 작업은 [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)를 사용하여 SageMaker 학습 작업으로 실행됩니다. `CreateTrainingJob` API에 대한 인수는 실행 큐 구성으로 제어됩니다.

Amazon SageMaker는 [Docker 이미지를 사용하여 학습 작업을 실행합니다](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html). SageMaker에서 가져온 이미지는 Amazon Elastic Container Registry(ECR)에 저장되어야 합니다. 이는 학습에 사용하는 이미지가 ECR에 저장되어야 함을 의미합니다. ECR로 Launch를 설정하는 방법에 대한 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.

Amazon SageMaker는 IAM 실행 역할을 요구합니다. IAM 역할은 SageMaker 학습 작업 인스턴스 내에서 ECR 및 Amazon S3와 같은 필요한 리소스에 대한 엑세스를 제어하는 데 사용됩니다. IAM 역할 ARN을 기록해 두세요. 큐 구성에서 IAM 역할 ARN을 지정해야 합니다.

## 사전 요구 사항
다음 AWS 리소스를 생성하고 기록하세요:

1. **AWS 계정에서 SageMaker 설정.** 자세한 내용은 [SageMaker 개발자 가이드](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html)를 참조하세요.
2. **Amazon SageMaker에서 실행할 이미지를 저장할 Amazon ECR 리포지토리 생성.** 자세한 내용은 [Amazon ECR 문서](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)를 참조하세요.
3. **Amazon S3 버킷 생성**하여 SageMaker 학습 작업의 입력 및 출력을 저장합니다. 자세한 내용은 [Amazon S3 문서](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)를 참조하세요. S3 버킷 URI 및 디렉터리를 기록하세요.
4. **IAM 실행 역할 생성.** SageMaker 학습 작업에서 사용되는 역할은 작동하기 위해 다음 권한이 필요합니다. 이러한 권한은 로깅 이벤트, ECR에서 가져오기, 입력 및 출력 버킷과의 상호 작용을 허용합니다. (참고: SageMaker 학습 작업에 대해 이미 이 역할을 가지고 있다면, 다시 생성할 필요가 없습니다.)
  ```json title="IAM 역할 정책"
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "cloudwatch:PutMetricData",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:CreateLogGroup",
          "logs:DescribeLogStreams",
          "ecr:GetAuthorizationToken"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::<input-bucket>"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject"
        ],
        "Resource": [
          "arn:aws:s3:::<input-bucket>/<object>",
          "arn:aws:s3:::<output-bucket>/<path>"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repo>"
      }
    ]
  }
  ```
  이 단계에서 생성한 역할 ARN을 기록하세요. 큐를 구성할 때 이 역할 ARN을 제공합니다.
5. **런치 에이전트용 IAM 역할 생성** 런치 에이전트는 SageMaker 학습 작업을 생성할 수 있는 권한이 필요합니다. 런치 에이전트에 사용할 IAM 역할에 다음 정책을 연결하세요. 런치 에이전트용으로 생성한 IAM 역할 ARN을 기록하세요:

  ```yaml
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "sagemaker:AddTags",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:sagemaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<RoleArn-from-queue-config>"
      },
      {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-OF-KMS-KEY>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "sagemaker.<region>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```

:::note
* 런치 에이전트가 이미지를 빌드하도록 하려면 추가 권한이 필요합니다. 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.
* SageMaker 큐에 대한 `kms:CreateGrant` 권한은 관련 ResourceConfig에 VolumeKmsKeyId가 지정되어 있고 관련 역할에 이 작업을 허용하는 정책이 없는 경우에만 필요합니다.
:::

## SageMaker에 대한 큐 구성
W&B 앱에서 SageMaker를 계산 리소스로 사용하는 큐를 생성하세요:

1. [Launch App](https://wandb.ai/launch)으로 이동하세요.
3. **Create Queue** 버튼을 클릭하세요.
4. 큐를 생성하고자 하는 **Entity**를 선택하세요.
5. **Name** 필드에 큐의 이름을 입력하세요.
6. **Resource**로 **SageMaker**를 선택하세요.
7. **Configuration** 필드에서 SageMaker 작업에 대한 정보를 제공하세요. 기본적으로, W&B는 YAML 및 JSON `CreateTrainingJob` 요청 본문을 채워 넣습니다:
```json
{
  "RoleArn": "<REQUIRED>",
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
      "S3OutputPath": "<REQUIRED>"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
  }
}
```
최소한 다음을 지정해야 합니다:

- `RoleArn` : [사전 요구 사항](#사전-요구-사항)에서 생성한 IAM 역할의 ARN입니다.
- `OutputDataConfig.S3OutputPath` : SageMaker 출력이 저장될 Amazon S3 URI입니다.
- `ResourceConfig`: 리소스 구성의 필요한 사양입니다. 리소스 구성에 대한 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)에서 설명합니다.
- `StoppingCondition`: 학습 작업의 중지 조건에 대한 필요한 사양입니다. 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)에서 설명합니다.
7. **Create Queue** 버튼을 클릭하세요.

## 런치 에이전트 구성
`launch-config.yaml`이라는 이름의 YAML 구성 파일로 런치 에이전트를 구성하세요. 기본적으로, W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. 런치 에이전트를 활성화할 때 다른 디렉터리를 선택적으로 지정할 수 있습니다.

다음 YAML 스니펫은 핵심 구성 에이전트 옵션을 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

:::tip
Amazon SageMaker에서 런치를 제출하는 두 가지 방법이 있습니다:
* 옵션 1: 자신의 이미지(BYOI)를 가져와 Amazon ECR 리포지토리에 푸시합니다.
* 옵션 2: W&B 런치 에이전트가 컨테이너를 빌드하고 ECR 리포지토리에 푸시하도록 합니다.

런치 에이전트가 이미지를 빌드하도록 하려는 경우(옵션 2), 런치 에이전트 구성에 추가 정보를 제공해야 합니다. 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.
:::

## Amazon SageMaker에 대한 에이전트 권한 설정
런치 에이전트와 관련된 IAM 역할은 다양한 방식으로 연결될 수 있습니다. 이러한 역할을 구성하는 방법은 부분적으로 런치 에이전트가 폴링하는 위치에 따라 다릅니다.


사용 사례에 따라 다음 가이드를 참조하세요.

### 에이전트가 로컬 기계에서 폴링하는 경우

`~/.aws/config` 및 `~/.aws/credentials`에 위치한 AWS 구성 파일을 사용하여 로컬 기계에서 폴링하는 에이전트와 역할을 연결하세요. 이전 단계에서 생성한 런치 에이전트용 IAM 역할 ARN을 제공하세요.

```yaml title="~/.aws/config"
[profile sagemaker-agent]
role_arn = arn:aws:iam::<account-id>:role/<agent-role-name>
source_profile = default                                                                   
```

```yaml title="~/.aws/credentials"
[default]
aws_access_key_id=<access-key-id>
aws_secret_access_key=<secret-access-key>
aws_session_token=<session-token>
```

### 에이전트가 AWS 내부(EC2와 같은)에서 폴링하는 경우
EC2와 같은 AWS 서비스 내에서 에이전트를 실행하려는 경우 인스턴스 역할을 사용하여 에이전트에 권한을 제공할 수 있습니다.