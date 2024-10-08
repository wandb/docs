---
title: Tutorial: Set up W&B Launch on SageMaker
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Launch를 사용하여 Amazon SageMaker에 launch 작업을 제출하여 제공된 알고리즘이나 사용자 정의 알고리즘을 사용해 기계학습 모델을 트레이닝할 수 있습니다. SageMaker는 컴퓨팅 리소스를 설정하고 해제하는 작업을 처리하므로 EKS 클러스터가 없는 팀에게 좋은 선택이 될 수 있습니다.

Amazon SageMaker에 연결된 W&B Launch 대기열로 전송된 launch 작업은 [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)를 사용하여 SageMaker Training Jobs로 실행됩니다. `CreateTrainingJob` API로 전송되는 인수는 launch 대기열 설정을 사용하여 제어할 수 있습니다.

Amazon SageMaker는 [Docker 이미지를 사용하여 트레이닝 작업을 실행](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)합니다. SageMaker에 의해 가져오는 이미지는 Amazon Elastic Container Registry (ECR)에 저장되어야 합니다. 따라서 트레이닝에 사용할 이미지는 ECR에 저장되어야 합니다.

:::note
이 가이드는 SageMaker Training Jobs를 실행하는 방법을 보여줍니다. Amazon SageMaker에서 추론을 위해 모델을 배포하는 방법에 대한 정보는 [이 Launch 작업 예제](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)를 참조하세요.
:::

## 전제 조건

시작하기 전에 다음 전제 조건을 충족했는지 확인하십시오:

* [Launch 에이전트가 Docker 이미지를 빌드하도록 할 것인지 결정하십시오.](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
* [AWS 리소스를 설정하고 S3, ECR, SageMaker IAM 역할에 대한 정보를 수집하십시오.](#set-up-aws-resources)
* [Launch 에이전트를 위한 IAM 역할을 생성하십시오.](#create-an-iam-role-for-launch-agent)

### Launch 에이전트가 Docker 이미지를 빌드하도록 할 것인지 결정

W&B Launch 에이전트가 Docker 이미지를 빌드하도록 할 것인지를 결정하십시오. 선택할 수 있는 두 가지 옵션이 있습니다:

* Launch 에이전트가 Docker 이미지를 빌드하고, Amazon ECR에 이미지를 푸시하며, [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) 작업을 대신 제출하도록 허용합니다. 이 옵션은 ML 엔지니어들이 트레이닝 코드에 빠르게 반영할 수 있는 간편함을 제공할 수 있습니다.
* Launch 에이전트가 트레이닝 또는 추론 스크립트를 포함하는 기존 Docker 이미지를 사용합니다. 이 옵션은 기존 CI 시스템과 잘 작동합니다. 이 옵션을 선택한 경우 Docker 이미지를 Amazon ECR의 컨테이너 레지스트리에 수동으로 업로드해야 합니다.

### AWS 리소스 설정

선호하는 AWS 지역에 다음의 AWS 리소스가 설정되어 있는지 확인하세요:

1. 컨테이너 이미지를 저장할 [ECR 저장소](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html).
2. SageMaker Training 작업을 위한 입력 및 출력을 저장하기 위한 하나 이상의 [S3 버킷](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html).
3. SageMaker가 트레이닝 작업을 실행하고 ECR과 S3와 상호작용할 수 있도록 허용하는 Amazon SageMaker IAM 역할.

이 리소스의 ARN을 기록해 두세요. SageMaker를 위한 [Launch 대기열 설정](#configure-launch-queue-for-sagemaker)을 정의할 때 ARN이 필요합니다.

### Launch 에이전트를 위한 IAM 정책 생성

1. AWS의 IAM 화면에서 새 정책을 생성하세요.
2. JSON 정책 편집기로 전환한 다음, 자신의 유스 케이스에 기반한 다음 정책을 붙여넣습니다. `<>`로 묶인 값을 자신의 값으로 대체하세요:

<Tabs
defaultValue="build"
values={[
{label: 'Agent builds and submits Docker image', value: 'build' },
{label: 'Agent submits pre-built Docker image', value: 'no-build' },
]}>

<TabItem value="no-build">

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:DescribeLogStreams",
        "SageMaker:AddTags",
        "SageMaker:CreateTrainingJob",
        "SageMaker:DescribeTrainingJob"
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
          "kms:ViaService": "SageMaker.<region>.amazonaws.com",
          "kms:GrantIsForAWSResource": "true"
        }
      }
    }
  ]
}
```

</TabItem>
<TabItem value="build">

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:DescribeLogStreams",
        "SageMaker:AddTags",
        "SageMaker:CreateTrainingJob",
        "SageMaker:DescribeTrainingJob"
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
    "Action": [
      "ecr:CreateRepository",
      "ecr:UploadLayerPart",
      "ecr:PutImage",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchDeleteImage"
    ],
    "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repository>"
  },
  {
    "Effect": "Allow",
    "Action": "ecr:GetAuthorizationToken",
    "Resource": "*"
  },
  {
      "Effect": "Allow",
      "Action": "kms:CreateGrant",
      "Resource": "<ARN-OF-KMS-KEY>",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "SageMaker.<region>.amazonaws.com",
          "kms:GrantIsForAWSResource": "true"
        }
      }
    }
  ]
}
```

</TabItem>
</Tabs>

3. **Next**를 클릭합니다.
4. 정책에 이름과 설명을 부여합니다.
5. **Create policy**를 클릭합니다.

### Launch 에이전트를 위한 IAM 역할 생성

Launch 에이전트는 Amazon SageMaker 트레이닝 작업을 생성할 권한이 필요합니다. 다음 절차를 따라 IAM 역할을 생성하세요:

1. AWS의 IAM 화면에서 새 역할을 생성합니다. 
2. **Trusted Entity**로 **AWS Account** (또는 조직의 정책에 맞는 다른 옵션)를 선택합니다.
3. 권한 화면을 스크롤하여 위에서 생성한 정책 이름을 선택합니다. 
4. 역할에 이름과 설명을 부여합니다.
5. **Create role**을 선택합니다.
6. 역할의 ARN을 기록해 두세요. launch 에이전트를 설정할 때 이 ARN을 지정해야 합니다.

IAM 역할을 생성하는 방법에 대한 자세한 정보는 [AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)을 참조하세요.

:::info
* Launch 에이전트가 이미지를 빌드하도록 하려면, 추가 권한이 필요한 [Advanced agent 설정](./setup-agent-advanced.md)을 참조하세요.
* SageMaker 대기열의 `kms:CreateGrant` 권한은 관련된 ResourceConfig에 명시된 VolumeKmsKeyId가 있고, 관련된 역할에 이 작업을 허용하는 정책이 없는 경우에만 필요합니다.
:::

## SageMaker를 위한 launch 대기열 설정

다음으로, SageMaker를 컴퓨트 리소스로 사용하는 W&B 애플리케이션의 대기열을 생성합니다:

1. [Launch App](https://wandb.ai/launch)으로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 대기열을 만들고자 하는 **Entity**를 선택합니다.
4. **Name** 필드에 대기열 이름을 제공합니다.
5. **Resource**로는 **SageMaker**를 선택합니다.
6. **Configuration** 필드에 SageMaker 작업에 대한 정보를 입력합니다. 기본적으로, W&B는 YAML과 JSON `CreateTrainingJob` 요청 본문을 채웁니다:
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

- `RoleArn` : SageMaker 실행 IAM 역할의 ARN (자세한 정보는 [전제 조건](#prerequisites) 참조). launch **agent** IAM 역할과 혼동하지 마세요.
- `OutputDataConfig.S3OutputPath` : SageMaker 출력이 저장될 Amazon S3 URI.
- `ResourceConfig`: 리소스 설정의 필수 명세. 리소스 설정에 대한 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)에 설명되어 있습니다.
- `StoppingCondition`: 트레이닝 작업의 중지 조건의 필수 명세. [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)에 설명된 옵션.
7. **Create Queue** 버튼을 클릭합니다.

## Launch 에이전트 설정

다음 섹션에서는 에이전트를 배포할 수 있는 위치와 배포 위치에 따라 에이전트를 어떻게 구성할지 설명합니다.

Amazon SageMaker 대기열에 대한 Launch 에이전트를 배포하는 [여러 방법](#decide-where-to-run-the-launch-agent)들이 존재합니다: 로컬 머신, EC2 인스턴스, 또는 EKS 클러스터에서. 에이전트를 배포하는 위치에 따라 적절히 [launch 에이전트를 구성](#configure-a-launch-agent)하세요.

### Launch 에이전트를 실행할 위치 결정

프로덕션 워크로드와 이미 EKS 클러스터를 보유한 고객의 경우, W&B는 EKS 클러스터에 Launch 에이전트를 배포하기 위해 이 Helm 차트를 사용할 것을 권장합니다.

현재 EKS 클러스터가 없는 프로덕션 워크로드의 경우, EC2 인스턴스가 좋은 옵션입니다. Launch 에이전트 인스턴스는 계속 실행되지만, 에이전트는 상대적으로 저렴한 `t2.micro` 크기의 EC2 인스턴스 이상이 필요하지 않습니다.

실험적이거나 개인 유스케이스의 경우, 로컬 머신에서 Launch 에이전트를 실행하는 것이 시작하는 빠른 방법이 될 수 있습니다.

유스케이스에 따라, launch 에이전트를 적절하게 구성하기 위해 제공된 경로의 지침을 따르세요:
<Tabs
defaultValue="eks"
values={[
{label: 'EKS', value: 'eks'},
{label: 'EC2', value: 'ec2'},
{label: 'Local machine', value: 'local'},
]}>

<TabItem value="eks">

W&B는 [W&B 관리 Helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 에이전트를 EKS 클러스터에 설치할 것을 강력히 권장합니다.

</TabItem>
<TabItem value="ec2">

Amazon EC2 대시보드로 이동하여 다음 단계를 완료합니다:

1. **Launch instance**를 클릭합니다.
2. **Name** 필드에 이름을 제공하고, 태그를 추가할 수도 있습니다.
3. **Instance type**에서 EC2 컨테이너 유형을 선택합니다. 1vCPU와 1GiB 메모리 이상이 필요하지 않습니다 (예: t2.micro).
4. **Key pair (login)** 필드에서 조직을 위한 키 쌍을 생성합니다. 나중에 SSH 클라이언트를 사용하여 EC2 인스턴스에 [연결](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)할 때 이 키 쌍을 사용합니다.
5. **Network settings**에서 조직에 적절한 보안 그룹을 선택합니다. 
6. **Advanced details**를 확장합니다. **IAM instance profile**에서 위에서 생성한 launch 에이전트 IAM 역할을 선택합니다.
7. **Summary** 필드를 검토합니다. 올바르다면 **Launch instance**를 선택합니다.

AWS의 EC2 대시보드 왼쪽 패널에서 **Instances**로 이동합니다. 생성한 EC2 인스턴스가 실행 중인지 확인합니다 (**Instance state** 열 참조). EC2 인스턴스가 실행 중임을 확인한 후, 로컬 머신의 터미널로 이동하여 다음을 완료하세요:

1. **Connect**를 선택합니다.
2. **SSH client** 탭을 선택하고 EC2 인스턴스에 연결하기 위한 지침을 따릅니다.
3. EC2 인스턴스 내에서 다음 패키지를 설치합니다:
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 다음으로, EC2 인스턴스 내에서 Docker를 설치하고 시작합니다:
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

이제 Launch 에이전트 설정을 진행할 수 있습니다.

</TabItem>
<TabItem value="local">

AWS config 파일 `~/.aws/config` 및 `~/.aws/credentials`를 사용하여 로컬 머신에서 에이전트를 폴링하는 역할을 에이전트와 연결합니다. 이전 단계에서 생성한 launch 에이전트의 IAM 역할 ARN을 제공하십시오.

```yaml title="~/.aws/config"
[profile SageMaker-agent]
role_arn = arn:aws:iam::<account-id>:role/<agent-role-name>
source_profile = default                                                                   
```

```yaml title="~/.aws/credentials"
[default]
aws_access_key_id=<access-key-id>
aws_secret_access_key=<secret-access-key>
aws_session_token=<session-token>
```

세션 토큰은 관련 주체에 따라 1시간 또는 3일의 [최대 길이](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)를 가질 수 있습니다.

</TabItem>
</Tabs>

### Launch 에이전트 구성

YAML 구성 파일 `launch-config.yaml`을 사용하여 launch 에이전트를 구성합니다. 

기본적으로 W&B는 구성 파일을 `~/.config/wandb/launch-config.yaml`에서 자동으로 확인합니다. 옵션으로, launch 에이전트를 활성화할 때 `-c` 플래그를 사용하여 다른 디렉토리를 지정할 수 있습니다.

다음 YAML 스니펫은 기본 구성 에이전트 옵션을 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
max_jobs: -1
queues:
  - <queue-name>
environment:
  type: aws
  region: <your-region>
registry:
  type: ecr
  uri: <ecr-repo-arn>
builder: 
  type: docker
```

이제 `wandb launch-agent`를 사용하여 에이전트를 시작하세요.

## (선택 사항) Docker 이미지를 Amazon ECR에 푸시하기

:::info
이 섹션은 기존 Docker 이미지를 사용하는 launch 에이전트에만 적용됩니다. [Launch 에이전트가 작동하는 두 가지 옵션](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)이 있습니다.  
:::

Amazon ECR 저장소로 launch 작업을 포함한 Docker 이미지를 업로드하세요. 이미지 기반 작업을 사용하는 경우 Docker 이미지는 새로운 launch 작업을 제출하기 전에 ECR 레지스트리에 있어야 합니다.