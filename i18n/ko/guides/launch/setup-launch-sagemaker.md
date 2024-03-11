---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SageMaker 설정하기

W&B Launch를 사용하여 제공되거나 사용자 지정 알고리즘을 사용하여 기계학습 모델을 트레이닝하기 위해 Amazon SageMaker에 런치 작업을 제출할 수 있습니다. SageMaker는 컴퓨팅 리소스를 생성하고 해제하는 작업을 관리하므로, EKS 클러스터가 없는 팀에게 좋은 선택이 될 수 있습니다.

Amazon SageMaker에 연결된 W&B Launch 큐로 전송된 런치 작업은 [CreateTrainingJob API](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_CreateTrainingJob.html)를 사용하여 SageMaker 트레이닝 작업으로 실행됩니다. `CreateTrainingJob` API에 전송할 인수를 제어하려면 런치 큐 설정을 사용하십시오.

Amazon SageMaker는 [트레이닝 작업을 실행하기 위해 도커 이미지를 사용합니다](https://docs.aws.amazon.com/SageMaker/latest/dg/your-algorithms-training-algo-dockerfile.html). SageMaker가 가져오는 이미지는 Amazon Elastic Container Registry(ECR)에 저장되어야 합니다. 이는 트레이닝에 사용하는 이미지가 ECR에 저장되어야 함을 의미합니다.

:::note
이 가이드는 SageMaker 트레이닝 작업을 실행하는 방법을 보여줍니다. Amazon SageMaker에서 모델을 추론에 배포하는 방법에 대한 정보는 [이 예제 런치 작업](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_SageMaker_endpoints)을 참조하십시오.
:::

## 사전 요구사항

시작하기 전에 다음 사전 요구사항을 충족하는지 확인하십시오:

* [런치 에이전트가 도커 이미지를 대신 빌드하게 할지 결정합니다.](#런치-에이전트가-도커-이미지를-빌드하게-할지-결정하기)
* [AWS 리소스를 설정하고 S3, ECR 및 SageMaker IAM 역할에 대한 정보를 수집합니다.](#aws-리소스-설정하기)
* [런치 에이전트를 위한 IAM 역할을 생성합니다.](#런치-에이전트를-위한-iam-역할-생성하기)

### 런치 에이전트가 도커 이미지를 빌드하게 할지 결정하기

W&B Launch 에이전트가 도커 이미지를 대신 빌드하게 할지 결정하십시오. 선택할 수 있는 두 가지 옵션이 있습니다:

* 런치 에이전트가 도커 이미지를 빌드하고, Amazon ECR에 이미지를 푸시하며, [SageMaker 트레이닝](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) 작업을 대신 제출하도록 허용합니다. 이 옵션은 트레이닝 코드를 빠르게 반복하는 ML 엔지니어에게 일부 단순성을 제공할 수 있습니다.
* 런치 에이전트가 트레이닝 또는 추론 스크립트가 포함된 기존 도커 이미지를 사용합니다. 이 옵션은 기존 CI 시스템과 잘 작동합니다. 이 옵션을 선택한 경우, 도커 이미지를 Amazon ECR에 있는 컨테이너 레지스트리에 수동으로 업로드해야 합니다.

### AWS 리소스 설정하기

선호하는 AWS 리전에서 다음 AWS 리소스가 구성되어 있는지 확인하십시오:

1. 컨테이너 이미지를 저장하기 위한 [ECR 리포지토리](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html).
2. SageMaker 트레이닝 작업의 입력 및 출력을 저장하기 위한 하나 이상의 [S3 버킷](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html).
3. SageMaker가 트레이닝 작업을 실행하고 Amazon ECR 및 Amazon S3와 상호 작용할 수 있도록 허용하는 Amazon SageMaker용 IAM 역할.

이 리소스의 ARN을 기록해 두십시오. [SageMaker용 큐 구성](#sagemaker를-위한-큐-구성하기)을 정의할 때 ARN이 필요합니다.

### 런치 에이전트를 위한 IAM 역할 생성하기

런치 에이전트가 Amazon SageMaker 트레이닝 작업을 생성할 수 있도록 권한이 필요합니다. 아래 절차를 따라 IAM 역할을 생성하십시오:

1. AWS의 IAM 화면에서 새 역할을 생성합니다.
2. **신뢰할 수 있는 엔터티**에서 **AWS 계정**을 선택합니다(또는 조직의 정책에 적합한 다른 옵션 선택).
3. 권한 화면을 스크롤하고 **다음**을 클릭합니다.
4. 역할에 이름과 설명을 지정합니다.
5. **역할 생성**을 선택합니다.
6. **권한 추가**에서 **인라인 정책 생성**을 선택합니다.
7. JSON 정책 편집기로 전환한 다음, 다음 정책을 붙여넣습니다. `<>`로 묶인 값은 자신의 값으로 대체하십시오:

<Tabs
  defaultValue="build"
  values={[
    {label: '에이전트가 도커 이미지를 빌드하고 제출', value: 'build'},
    {label: '에이전트가 사전 빌드된 도커 이미지를 제출', value: 'no-build'},
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
        "Resource": "arn:aws:SageMaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<큐-설정에서-RoleArn>"
      },
    {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<KMS-키-ARN>",
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
        "Resource": "arn:aws:SageMaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<큐-설정에서-RoleArn>"
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
      "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<리포지토리>"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<KMS-키-ARN>",
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

8. **다음**을 클릭합니다.
9. 역할의 ARN을 기록해 둡니다. 런치 에이전트를 설정할 때 ARN을 지정해야 합니다.

IAM 역할을 생성하는 방법에 대한 자세한 내용은 [AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)을 참조하십시오.

:::info
* 이미지를 빌드하려는 경우 추가 필요한 권한에 대해서는 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하십시오.
* SageMaker 큐에 연결된 ResourceConfig에 VolumeKmsKeyId가 지정되어 있고 관련 역할에 이 작업을 허용하는 정책이 없는 경우에만 `kms:CreateGrant` 권한이 필요합니다.
:::

## SageMaker를 위한 큐 구성하기

다음으로, W&B 앱에서 SageMaker를 컴퓨팅 리소스로 사용하는 큐를 생성하십시오:

1. [Launch App](https://wandb.ai/launch)으로 이동합니다.
3. **큐 생성** 버튼을 클릭합니다.
4. 큐를 생성할 **엔터티**를 선택합니다.
5. **이름** 필드에 큐의 이름을 제공합니다.
6. **리소스**로 **SageMaker**를 선택합니다.
7. **설정** 필드에서 SageMaker 작업에 대한 정보를 제공합니다. 기본적으로 W&B는 YAML 및 JSON `CreateTrainingJob` 요청 본문을 채웁니다:
```json
{
  "RoleArn": "<필수>", 
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
      "S3OutputPath": "<필수>"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
  }
}
```
최소한 다음을 지정해야 합니다:

- `RoleArn`: SageMaker 실행 IAM 역할의 ARN([사전 요구사항](#사전-요구사항) 참조). 런치 **에이전트** IAM 역할과 혼동하지 마십시오.
- `OutputDataConfig.S3OutputPath`: SageMaker 출력이 저장될 Amazon S3 URI.
- `ResourceConfig`: 리소스 구성의 필수 사양. 리소스 구성 옵션은 [여기](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_ResourceConfig.html)에 설명되어 있습니다.
- `StoppingCondition`: 트레이닝 작업의 중지 조건 사양이 필요합니다. 옵션은 [여기](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_StoppingCondition.html)에 설명되어 있습니다.
7. **큐 생성** 버튼을 클릭합니다.

## 런치 에이전트 설정하기

다음 섹션에서는 에이전트를 배포할 위치와 해당 위치에 따라 에이전트를 어떻게 구성해야 하는지 설명합니다.

Amazon SageMaker 큐에 대한 런치 에이전트 배포에는 [여러 옵션이 있습니다](#런치-에이전트를-실행할-위치-결정하기): 로컬 기계, EC2 인스턴스 또는 EKS 클러스터에 배포할 수 있습니다. 에이전트를 배포하는 위치에 따라 [런치 에이전트를 적절하게 구성합니다](#런치-에이전트-구성하기).

### 런치 에이전트를 실행할 위치 결정하기

프로덕션 워크로드와 이미 EKS 클러스터를 보유한 고객의 경우, 이 Helm 차트를 사용하여 EKS 클러스터에 런치 에이전트를 배포하는 것이 W&B에서 권장합니다.

현재 EKS 클러스터가 없는 프로덕션 워크로드의 경우, EC2 인스턴스가 좋은 옵션일 수 있습니다. 런치 에이전트 인스턴스가 항상 실행되지만, `t2.micro` 크기의 EC2 인스턴스는 비교적 저렴하므로 에이전트에는 그 이상이 필요하지 않습니다.

실험적이거나 개인적인 사용 사례의 경우, 로컬 기계에서 런치 에이전트를 실행하는 것이 시작하는 데 빠른 방법일 수 있습니다.

사용 사례에 따라 다음 탭의 지침을 따라 런치 에이전트를 올바르게 구성하십시오:
<Tabs
  defaultValue="eks"
  values={[
    {label: 'EKS', value: 'eks'},
    {label: 'EC2', value: 'ec2'},
    {label: '로컬 기계', value: 'local'},
  ]}>
  <TabItem value="eks">

W&B는 [W&B 관리 helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 EKS 클러스터에 에이전트를 설치하는 것을 강력히 권장합니다.

</TabItem>
  <TabItem value="ec2">

Amazon EC2 대시보드로 이동하여 다음 단계를 완료하십시오:

1. **인스턴스 시작**을 클릭합니다.
2. **이름** 필드에 이름을 제공합니다. 선택적으로 태그를 추가합니다.
2. **인스턴스 유형**에서 EC2 컨테이너에 대한 인스턴스 유형을 선택합니다. 1vCPU와 1GiB의 메모리(예: t2.micro) 이상이 필요하지 않습니다.
3. **키 쌍(로그인)** 필드 내에서 조직에 대한 키 쌍을 생성합니다. 이 키 쌍을 사용하여 나중에 SSH 클라이언트로 [EC2 인스턴스에 연결](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)합니다.
2. **네트워크 설정** 내에서 조직에 적합한 보안 그룹을 선택합니다.
3. **고급 세부 정보**를 확장합니다. **IAM 인스턴스 프로필**에서 위에서 생성한 런치 에이전트 IAM 역할을 선택합니다.
2. **요약** 필드를 검토합니다. 맞으면 **인스턴스 시작**을 선택하십시오.

AWS의 EC2 대시보드 왼쪽 패널에서 **인스턴스**로 이동합니다. 생성한 EC2 인스턴스가 실행 중인지 확인합니다(**인스턴스 상태** 열 참조). EC2 인스턴스가 실행 중임을 확인한 후 로컬 기계의 터미널로 이동하여 다음을 완료하십시오:

1. **연결**을 선택합니다.
2. **SSH 클라이언트** 탭을 선택하고 설명된 지침을 따라