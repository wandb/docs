---
title: '튜토리얼: SageMaker에서 W&B Launch 설정하기'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B Launch 를 사용하면 Amazon SageMaker 에 launch job 을 제출하여 SageMaker 플랫폼에서 제공되는 알고리즘이나 커스텀 알고리즘으로 기계학습 모델을 트레이닝할 수 있습니다. SageMaker 가 컴퓨트 리소스를 자동으로 할당하고 해제해주므로, EKS 클러스터가 없는 팀에게도 적합한 선택이 될 수 있습니다.

Amazon SageMaker 와 연결된 W&B Launch 큐에 전송된 launch job 은 [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) 를 통해 SageMaker Training Job 으로 실행됩니다. launch 큐의 설정을 사용해 `CreateTrainingJob` API 로 전달되는 인수를 제어할 수 있습니다.

Amazon SageMaker 는 [트레이닝 job 실행에 Docker 이미지를 사용](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html) 합니다. SageMaker 에서 사용하는 이미지는 Amazon Elastic Container Registry (ECR)에 저장되어 있어야 하므로, 트레이닝에 사용할 이미지는 반드시 ECR에 업로드되어야 합니다.

{{% alert %}}
이 가이드는 SageMaker Training Job 실행 방법을 설명합니다. Amazon SageMaker 에 모델을 배포(Inference)하는 방법은 [이 Launch 예제 job](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints) 을 참고하세요.
{{% /alert %}}


## 사전 준비사항

시작하기 전에 다음 사전 준비사항을 모두 충족했는지 확인하세요:

* [Launch agent 가 Docker 이미지를 빌드하도록 할지 결정하기]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ko" >}})
* [AWS 리소스 설정 및 S3, ECR, Sagemaker IAM 역할 관련 정보 수집]({{< relref path="#set-up-aws-resources" lang="ko" >}})
* [Launch agent용 IAM 역할 생성]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ko" >}})

### Launch agent가 Docker 이미지를 빌드할지 결정하기

W&B Launch agent가 Docker 이미지를 직접 빌드할지 결정하세요. 선택지는 두 가지입니다:

* launch agent가 Docker 이미지를 빌드, Amazon ECR에 이미지를 푸시, 그리고 [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) job 제출까지 해줍니다. 트레이닝 코드를 빠르게 반복하는 ML 엔지니어에게 적합한 단순함을 제공합니다.  
* launch agent가 트레이닝 혹은 추론 스크립트가 포함된 기존 Docker 이미지를 사용합니다. 기존 CI 시스템과 연동할 때 적합하며, 이 옵션을 선택한다면 Docker 이미지를 직접 ECR 레지스트리에 업로드해야 합니다.


### AWS 리소스 설정

선호하는 AWS 리전에 다음 리소스가 준비되어 있는지 확인하세요:

1. 컨테이너 이미지를 저장할 [ECR 레포지토리](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)
2. SageMaker Training job의 입력과 출력을 저장할 하나 이상의 [S3 버킷](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)
3. SageMaker가 트레이닝 job 을 실행하고 Amazon ECR 및 S3와 상호작용할 수 있도록 권한이 부여된 SageMaker 전용 IAM 역할

이들 리소스의 ARN을 별도로 기록해 두세요. [Launch 큐 설정]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ko" >}})에서 필요합니다.

### Launch agent를 위한 IAM Policy 생성

1. AWS의 IAM 콘솔에서 새로운 정책을 만듭니다.
2. JSON 정책 편집기로 전환한 뒤, 아래의 가이드에 맞는 정책을 복사/붙여넣기 하세요. `<>` 로 감싸진 값은 본인 값으로 바꿔야 합니다.

{{< tabpane text=true >}}
{{% tab "사전 구축 Docker 이미지를 제출하는 agent" %}}
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
{{% tab "Docker 이미지를 빌드 및 제출하는 agent" %}}
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

3. **Next** 를 클릭합니다.
4. 정책 이름과 설명을 입력합니다.
5. **Create policy** 를 클릭합니다.


### Launch agent용 IAM 역할 생성

Launch agent 가 Amazon SageMaker 트레이닝 job 을 생성할 수 있어야 합니다. 아래 절차를 따라 새로운 IAM 역할을 만드세요.

1. AWS IAM 콘솔에서 새 역할(Role)을 만듭니다.
2. **Trusted Entity**는 **AWS Account** (또는 조직 정책에 적합한 항목)를 선택하세요.
3. 권한 목록 화면에서 앞서 만든 정책 이름을 선택합니다.
4. 역할의 이름과 설명을 입력합니다.
5. **Create role** 을 선택합니다.
6. 이 역할의 ARN을 메모해두세요. launch agent 설정 시 필요합니다.

IAM 역할 생성 방법은 [AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)를 참고하세요.

{{% alert %}}
* launch agent가 이미지를 빌드해야 한다면, [고급 agent 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})의 추가 권한도 부여해야 합니다.
* SageMaker 큐의 `kms:CreateGrant` 권한은 연결된 ResourceConfig에 VolumeKmsKeyId가 지정되어 있고, 해당 역할에 관련 권한이 없을 때만 필요합니다.
{{% /alert %}}



## SageMaker 용 launch 큐 설정

이제 W&B App에서 SageMaker를 컴퓨트 리소스로 사용하는 큐를 만듭니다.

1. [Launch App](https://wandb.ai/launch) 으로 이동하세요.
2. **Create Queue** 버튼을 클릭합니다.
3. 큐를 만들고자 하는 **Entity** 를 선택하세요.
4. **Name** 필드에 큐의 이름을 입력하세요.
5. **Resource** 항목에서 **SageMaker** 를 선택합니다.
6. **Configuration** 필드에 SageMaker job 관련 정보를 입력하세요. 기본적으로 W&B에서 YAML 및 JSON 형태의 `CreateTrainingJob` 요청 본문이 자동 입력됩니다:
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
최소한 다음 요소들은 반드시 입력해야 합니다.

- `RoleArn`: SageMaker 실행용 IAM 역할의 ARN ([사전 준비사항]({{< relref path="#prerequisites" lang="ko" >}}) 참고). launch **agent** IAM 역할과 혼동하지 마세요.
- `OutputDataConfig.S3OutputPath`: SageMaker 출력 결과를 저장할 Amazon S3 URI
- `ResourceConfig`: 리소스 설정은 필수입니다. 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html) 참고
- `StoppingCondition`: 트레이닝 job의 종료 조건 지정. 옵션은 [여기](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html) 참고
7. **Create Queue** 버튼을 클릭하세요.


## launch agent 설정

아래 섹션에서는 agent를 어디에 배포할 수 있는지와 배포 위치에 따라 agent를 어떻게 설정할지 설명합니다.

Amazon SageMaker 큐에 대해 [Launch agent 배포 방식은 여러가지]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ko" >}})가 있습니다: 로컬 머신, EC2 인스턴스, EKS 클러스터. agent가 배포되는 위치에 따라 [launch agent를 적절히 설정]({{< relref path="#configure-a-launch-agent" lang="ko" >}})하세요.


### Launch agent 실행 위치 결정

프로덕션 워크로드 또는 이미 EKS 클러스터를 보유한 고객의 경우, Launch agent를 Helm chart로 EKS 클러스터에 배포하는 것을 권장합니다.

EKS 클러스터가 없는 환경이라면 EC2 인스턴스가 적합한 선택입니다. launch agent 인스턴스는 계속 구동되지만, `t2.micro`와 같이 비교적 저렴한 EC2 인스턴스면 충분합니다.

실험적 사용이나 개인 유스 케이스라면, 로컬 머신에 Launch agent를 실행하는 것이 빠른 시작 방법입니다.

유스 케이스에 따라 아래 탭에서 연결된 안내를 참고해서 Launch agent 를 적절히 설정해주세요.
{{< tabpane text=true >}}
{{% tab "EKS" %}}
EKS 클러스터에 agent를 설치할 땐 [W&B 제공 helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) 사용을 강력히 권장합니다.
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 대시보드에서 다음 단계를 진행하세요:

1. **Launch instance** 버튼을 클릭합니다.
2. **Name** 필드에 인스턴스 이름을 입력하고 필요에 따라 태그를 추가합니다.
3. **Instance type**에서 EC2 컨테이너에 맞는 인스턴스 타입을 선택하세요. 1vCPU, 1GiB 메모리면 충분합니다(예: t2.micro).
4. **Key pair (login)** 필드에서 조직용 키 페어를 만드세요. 이 키 페어로 [EC2 인스턴스에 SSH 접속](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html) 하게 됩니다.
5. **Network settings**에서 조직에 맞는 보안 그룹(Security Group)을 선택하세요.
6. **Advanced details**를 펼쳐, **IAM instance profile**에서 위에서 만든 launch agent IAM role을 선택하세요.
7. **Summary**를 확인 후, 맞으면 **Launch instance** 를 클릭합니다.

EC2 대시보드 왼쪽 패널의 **Instances**로 이동하여, 인스턴스 상태(**Instance state** 컬럼)가 "실행 중"인지 확인하세요. 실행 중임을 확인하면, 로컬 머신 터미널에서 아래 단계를 진행합니다.

1. **Connect** 를 클릭합니다.
2. **SSH client** 탭을 선택하고 안내대로 EC2에 접속하세요.
3. EC2 인스턴스에서 다음 패키지들을 설치합니다:
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 다음으로, Docker 설치 및 실행을 진행합니다:
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

이제 Launch agent 설정(config)으로 넘어갈 수 있습니다.

{{% /tab %}}
{{% tab "로컬 머신" %}}

로컬 머신에서 agent를 사용할 경우, 역할(Role)을 연결하려면 `~/.aws/config` 와 `~/.aws/credentials`에 있는 AWS 설정 파일을 사용하세요. 위 단계에서 만든 launch agent IAM 역할 ARN을 넣으세요.
 
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

세션 토큰은 연관된 프린시펄(principal)에 따라 [최대 1시간 또는 3일](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)까지 유효합니다.
{{% /tab %}}
{{< /tabpane >}}


### Launch agent 구성

`launch-config.yaml` 이라는 YAML 설정 파일로 launch agent를 설정하세요.

W&B는 기본적으로 `~/.config/wandb/launch-config.yaml` 에서 설정 파일을 찾습니다. agent를 실행할 때 `-c` 플래그로 다른 디렉터리를 지정할 수도 있습니다.

아래 YAML 예시는 core agent 옵션을 지정하는 방법을 보여줍니다:

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

이제 `wandb launch-agent` 명령어로 agent를 시작하세요.


 ## (선택) Launch job용 Docker 이미지를 Amazon ECR에 업로드하기

{{% alert %}}
이 섹션은 launch agent가 트레이닝 또는 추론 로직이 담긴 기존 Docker 이미지를 사용하는 경우에만 해당합니다. [launch agent의 동작 방식 두 가지]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ko" >}}) 중 한 가지를 참고하세요.
{{% /alert %}}

launch job이 포함된 Docker 이미지를 Amazon ECR 레포지토리에 업로드하세요. 이미지 기반 job을 사용하는 경우, launch job을 제출하기 전에 Docker 이미지가 ECR registry에 있어야 합니다.