---
title: 'Tutorial: Set up W&B Launch on SageMaker'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B Launch 를 사용하여 Amazon SageMaker에 launch 작업을 제출하여 제공된 또는 사용자 지정 알고리즘을 사용하여 SageMaker 플랫폼에서 기계 학습 모델을 트레이닝할 수 있습니다. SageMaker는 컴퓨팅 리소스를 시작하고 해제하는 것을 관리하므로 EKS 클러스터가 없는 팀에게 좋은 선택이 될 수 있습니다.

Amazon SageMaker에 연결된 W&B Launch 대기열로 전송된 Launch 작업은 [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)를 통해 SageMaker Training 작업으로 실행됩니다. Launch 대기열 설정을 사용하여 `CreateTrainingJob` API로 전송되는 인수를 제어합니다.

Amazon SageMaker는 [Docker 이미지를 사용하여 트레이닝 작업을 실행합니다](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html). SageMaker에서 가져온 이미지는 Amazon Elastic Container Registry (ECR)에 저장해야 합니다. 즉, 트레이닝에 사용하는 이미지는 ECR에 저장해야 합니다.

{{% alert %}}
이 가이드는 SageMaker Training 작업을 실행하는 방법을 보여줍니다. Amazon SageMaker에서 추론을 위해 모델을 배포하는 방법에 대한 자세한 내용은 [이 예제 Launch 작업](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)을 참조하십시오.
{{% /alert %}}

## 전제 조건

시작하기 전에 다음 전제 조건을 충족하는지 확인하십시오.

* [Launch 에이전트가 Docker 이미지를 빌드하도록 할지 여부를 결정합니다.]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ko" >}})
* [AWS 리소스를 설정하고 S3, ECR 및 Sagemaker IAM 역할에 대한 정보를 수집합니다.]({{< relref path="#set-up-aws-resources" lang="ko" >}})
* [Launch 에이전트에 대한 IAM 역할을 만듭니다.]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ko" >}})

### Launch 에이전트가 Docker 이미지를 빌드하도록 할지 여부를 결정합니다.

W&B Launch 에이전트가 Docker 이미지를 빌드하도록 할지 여부를 결정합니다. 다음 두 가지 옵션 중에서 선택할 수 있습니다.

* Launch 에이전트가 Docker 이미지를 빌드하고 이미지를 Amazon ECR에 푸시한 다음 [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) 작업을 제출하도록 허용합니다. 이 옵션은 ML 엔지니어가 트레이닝 코드를 빠르게 반복하는 데 어느 정도의 단순성을 제공할 수 있습니다.
* Launch 에이전트는 트레이닝 또는 추론 스크립트가 포함된 기존 Docker 이미지를 사용합니다. 이 옵션은 기존 CI 시스템과 잘 작동합니다. 이 옵션을 선택하는 경우 Docker 이미지를 Amazon ECR의 컨테이너 레지스트리에 수동으로 업로드해야 합니다.

### AWS 리소스 설정

선호하는 AWS 리전에서 다음 AWS 리소스가 구성되어 있는지 확인합니다.

1. 컨테이너 이미지를 저장할 [ECR 레포지토리](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html).
2. SageMaker Training 작업에 대한 입력 및 출력을 저장할 하나 이상의 [S3 버킷](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html).
3. SageMaker가 트레이닝 작업을 실행하고 Amazon ECR 및 Amazon S3와 상호 작용할 수 있도록 허용하는 Amazon SageMaker용 IAM 역할.

이러한 리소스의 ARN을 기록해 두십시오. [Launch 대기열 구성]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ko" >}})을 정의할 때 ARN이 필요합니다.

### Launch 에이전트에 대한 IAM 정책 만들기

1. AWS의 IAM 화면에서 새 정책을 만듭니다.
2. JSON 정책 편집기로 전환한 다음 유스 케이스에 따라 다음 정책을 붙여넣습니다. `<>`로 묶인 값을 자신의 값으로 대체합니다.

{{< tabpane text=true >}}
{{% tab "에이전트가 미리 빌드된 Docker 이미지 제출" %}}
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
{{% /tab %}}
{{% tab "에이전트가 Docker 이미지를 빌드하고 제출" %}}
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
{{% /tab %}}
{{< /tabpane >}}

3. **다음**을 클릭합니다.
4. 정책 이름과 설명을 지정합니다.
5. **정책 만들기**를 클릭합니다.

### Launch 에이전트에 대한 IAM 역할 만들기

Launch 에이전트에는 Amazon SageMaker 트레이닝 작업을 생성할 수 있는 권한이 필요합니다. 아래 절차에 따라 IAM 역할을 만듭니다.

1. AWS의 IAM 화면에서 새 역할을 만듭니다.
2. **신뢰할 수 있는 엔터티**의 경우 **AWS 계정** (또는 조직의 정책에 적합한 다른 옵션)을 선택합니다.
3. 권한 화면을 스크롤하여 위에서 만든 정책 이름을 선택합니다.
4. 역할 이름과 설명을 지정합니다.
5. **역할 만들기**를 선택합니다.
6. 역할의 ARN을 기록해 둡니다. Launch 에이전트를 설정할 때 ARN을 지정합니다.

IAM 역할을 만드는 방법에 대한 자세한 내용은 [AWS Identity and Access Management 설명서](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)를 참조하십시오.

{{% alert %}}
* Launch 에이전트가 이미지를 빌드하도록 하려면 필요한 추가 권한에 대한 [고급 에이전트 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})을 참조하십시오.
* SageMaker 대기열에 대한 `kms:CreateGrant` 권한은 연결된 ResourceConfig에 지정된 VolumeKmsKeyId가 있고 연결된 역할에 이 작업을 허용하는 정책이 없는 경우에만 필요합니다.
{{% /alert %}}

## SageMaker용 Launch 대기열 구성

다음으로, SageMaker를 컴퓨팅 리소스로 사용하는 W&B App에서 대기열을 만듭니다.

1. [Launch App](https://wandb.ai/launch)으로 이동합니다.
3. **대기열 만들기** 버튼을 클릭합니다.
4. 대기열을 만들 **Entities**를 선택합니다.
5. **이름** 필드에 대기열 이름을 제공합니다.
6. **리소스**로 **SageMaker**를 선택합니다.
7. **설정** 필드 내에서 SageMaker 작업에 대한 정보를 제공합니다. 기본적으로 W&B는 YAML 및 JSON `CreateTrainingJob` 요청 본문을 채웁니다.
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
최소한 다음을 지정해야 합니다.

- `RoleArn`: SageMaker 실행 IAM 역할의 ARN ([전제 조건]({{< relref path="#prerequisites" lang="ko" >}}) 참조). Launch **에이전트** IAM 역할과 혼동하지 마십시오.
- `OutputDataConfig.S3OutputPath`: SageMaker 출력이 저장될 위치를 지정하는 Amazon S3 URI입니다.
- `ResourceConfig`: 리소스 구성의 필수 사양입니다. 리소스 구성에 대한 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)에 설명되어 있습니다.
- `StoppingCondition`: 트레이닝 작업에 대한 중지 조건의 필수 사양입니다. 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)에 설명되어 있습니다.
7. **대기열 만들기** 버튼을 클릭합니다.

## Launch 에이전트 설정

다음 섹션에서는 에이전트를 배포할 수 있는 위치와 배포 위치에 따라 에이전트를 구성하는 방법을 설명합니다.

[Amazon SageMaker에 Launch 에이전트를 배포하는 방법에 대한 여러 옵션](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)이 있습니다. 로컬 머신, EC2 인스턴스 또는 EKS 클러스터에서 실행할 수 있습니다. 에이전트를 배포하는 위치에 따라 [Launch 에이전트를 적절하게 구성합니다.]({{< relref path="#configure-a-launch-agent" lang="ko" >}})

### Launch 에이전트를 실행할 위치 결정

프로덕션 워크로드 및 이미 EKS 클러스터가 있는 고객의 경우 W&B는 이 Helm 차트를 사용하여 Launch 에이전트를 EKS 클러스터에 배포하는 것이 좋습니다.

현재 EKS 클러스터가 없는 프로덕션 워크로드의 경우 EC2 인스턴스가 좋은 옵션입니다. Launch 에이전트 인스턴스는 항상 실행되지만 에이전트에는 비교적 저렴한 `t2.micro` 크기의 EC2 인스턴스 이상이 필요하지 않습니다.

실험적 또는 개인용 유스 케이스의 경우 로컬 머신에서 Launch 에이전트를 실행하는 것이 시작하는 빠른 방법이 될 수 있습니다.

유스 케이스에 따라 다음 탭에 제공된 지침에 따라 Launch 에이전트를 적절하게 구성합니다.
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B는 [W&B 관리 헬름 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 에이전트를 EKS 클러스터에 설치하는 것이 좋습니다.
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 대시보드로 이동하여 다음 단계를 완료합니다.

1. **인스턴스 시작**을 클릭합니다.
2. **이름** 필드에 이름을 제공합니다. 선택적으로 태그를 추가합니다.
2. **인스턴스 유형**에서 EC2 컨테이너의 인스턴스 유형을 선택합니다. 1vCPU 및 1GiB 이상의 메모리가 필요하지 않습니다 (예: t2.micro).
3. **키 페어(로그인)** 필드 내에서 조직의 키 페어를 만듭니다. 이 키 페어를 사용하여 나중에 SSH 클라이언트를 사용하여 [EC2 인스턴스에 연결](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)합니다.
2. **네트워크 설정** 내에서 조직에 적합한 보안 그룹을 선택합니다.
3. **고급 세부 정보**를 확장합니다. **IAM 인스턴스 프로필**의 경우 위에서 만든 Launch 에이전트 IAM 역할을 선택합니다.
2. **요약** 필드를 검토합니다. 올바른 경우 **인스턴스 시작**을 선택합니다.

AWS의 EC2 대시보드의 왼쪽 패널에서 **인스턴스**로 이동합니다. 생성한 EC2 인스턴스가 실행 중인지 확인합니다 (**인스턴스 상태** 열 참조). EC2 인스턴스가 실행 중인지 확인했으면 로컬 머신의 터미널로 이동하여 다음을 완료합니다.

1. **연결**을 선택합니다.
2. **SSH 클라이언트** 탭을 선택하고 설명된 지침에 따라 EC2 인스턴스에 연결합니다.
3. EC2 인스턴스 내에서 다음 패키지를 설치합니다.
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 다음으로 EC2 인스턴스 내에서 Docker를 설치하고 시작합니다.
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

이제 Launch 에이전트 구성을 설정할 수 있습니다.

{{% /tab %}}
{{% tab "로컬 머신" %}}

`~/.aws/config` 및 `~/.aws/credentials`에 있는 AWS 구성 파일을 사용하여 로컬 머신에서 폴링하는 에이전트와 역할을 연결합니다. 이전 단계에서 Launch 에이전트에 대해 만든 IAM 역할 ARN을 제공합니다.
 
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

세션 토큰의 [최대 길이](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)는 연결된 주체에 따라 1시간 또는 3일입니다.
{{% /tab %}}
{{< /tabpane >}}

### Launch 에이전트 구성

`launch-config.yaml`이라는 YAML 구성 파일로 Launch 에이전트를 구성합니다.

기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. `-c` 플래그로 Launch 에이전트를 활성화할 때 다른 디렉토리를 선택적으로 지정할 수 있습니다.

다음 YAML 스니펫은 핵심 구성 에이전트 옵션을 지정하는 방법을 보여줍니다.

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

이제 `wandb launch-agent`로 에이전트를 시작합니다.

## (선택 사항) Launch 작업 Docker 이미지를 Amazon ECR에 푸시합니다.

{{% alert %}}
이 섹션은 Launch 에이전트가 트레이닝 또는 추론 로직이 포함된 기존 Docker 이미지를 사용하는 경우에만 적용됩니다. [Launch 에이전트가 동작하는 방식에는 두 가지 옵션이 있습니다.]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ko" >}})
{{% /alert %}}

Launch 작업이 포함된 Docker 이미지를 Amazon ECR 레포지토리에 업로드합니다. 이미지 기반 작업을 사용하는 경우 새 Launch 작업을 제출하기 전에 Docker 이미지가 ECR 레지스트리에 있어야 합니다.
```
