---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 고급 에이전트 설정

런치 에이전트를 구성하는 방법은 여러 요소에 따라 달라집니다. 이러한 요소 중 하나는 런치 에이전트가 이미지를 대신 빌드해주는지 여부입니다.

:::tip
Git 저장소 기반 또는 [아티팩트 기반 작업](./create-launch-job.md#create-a-job-with-a-wb-artifact)을 제공하는 경우 W&B 런치 에이전트가 이미지를 빌드해줍니다.
:::

가장 간단한 사용 사례에서는 이미지 기반 런치 작업을 제공하며, 이미지 저장소에 엑세스할 수 있는 런치 대기열 대상 환경에서 실행됩니다.

다음 섹션에서는 런치 에이전트가 이미지를 빌드할 때 충족해야 하는 요구 사항을 설명합니다.

## 빌더

런치 에이전트는 W&B 아티팩트 및 Git 저장소 기반 작업에서 이미지를 빌드할 수 있습니다. 이는 머신 러닝 엔지니어가 도커 이미지를 직접 재빌드할 필요 없이 코드를 신속하게 반복할 수 있음을 의미합니다. 이러한 빌더 동작을 허용하려면 런치 에이전트 구성 파일(`launch-config.yaml`)에 빌더 옵션이 지정되어 있어야 합니다. W&B 런치는 Kaniko와 Docker 빌더를 지원하며, 사전 빌드된 이미지만 사용하도록 에이전트에 지시하는 `noop` 옵션도 지원합니다.

* Kaniko: 에이전트가 쿠버네티스 클러스터에서 런치 대기열을 폴링할 때 Kaniko를 사용하세요.
* Docker: 이미지를 자동으로 빌드하고 싶은 모든 다른 경우에 Docker를 사용하세요.
* Noop: 사전 빌드된 이미지만 사용하고자 할 때 사용하세요. (Kaniko와 Docker 빌더 모두 사전 빌드된 이미지를 사용하거나 새 이미지를 빌드할 수 있습니다.)

### Docker

에이전트가 로컬 머신(도커가 설치된)에서 이미지를 빌드하도록 하고 싶다면 Docker 빌더를 사용하는 것이 좋습니다. 런치 에이전트 구성에서 Docker 빌더를 빌더 키로 지정하세요.

예를 들어, 다음 YAML 스니펫은 런치 에이전트 구성 파일(`launch-config.yaml`)에서 이를 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
builder:
  type: docker
```

### Kaniko

Kaniko 빌더를 사용하려면 컨테이너 레지스트리 및 환경 옵션을 지정해야 합니다.

예를 들어, 다음 YAML 스니펫은 런치 에이전트 구성 파일(`launch-config.yaml`)에서 Kaniko를 지정하는 방법을 보여줍니다:

```yaml title="launch-config.yaml"
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/
  build-job-name: wandb-image-build # 모든 빌드에 대한 쿠버네티스 작업 이름 접두사
```

AKS, EKS, GKE 이외의 쿠버네티스 클러스터를 실행하는 경우 클라우드 환경에 대한 자격 증명을 포함하는 쿠버네티스 시크릿을 생성해야 합니다.

- GCP에 엑세스를 부여하려면 이 시크릿에 [서비스 계정 JSON](https://cloud.google.com/iam/docs/keys-create-delete#creating)이 포함되어야 합니다.
- AWS에 엑세스를 부여하려면 이 시크릿에 [AWS 자격 증명 파일](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html)이 포함되어야 합니다.

에이전트 구성 파일 내에서 빌더 섹션 내에 `secret-name` 및 `secret-key` 키를 설정하여 Kaniko가 시크릿을 사용할 수 있도록 합니다:

```yaml title="launch-config.yaml"
builder:
	type: kaniko
  build-context-store: <my-build-context-store>
  secret-name: <Kubernetes-secret-name>
  secret-key: <secret-file-name>
```

:::note
Kaniko 빌더는 Amazon S3와 같은 클라우드 스토리지에 데이터를 넣을 권한을 요구합니다. 자세한 내용은 [에이전트 권한](#agent-permissions) 섹션을 참조하세요.
:::

## 컨테이너 레지스트리에 에이전트 연결
런치 에이전트를 Amazon Elastic Container Registry (Amazon ECR), Google Artifact Registry on GCP 또는 Azure Container Registry와 같은 컨테이너 레지스트리에 연결할 수 있습니다. 다음은 런치 에이전트를 클라우드 컨테이너 레지스트리에 연결하고자 할 수 있는 일반적인 사용 사례를 설명합니다:

- 로컬 머신에 빌드 중인 이미지를 저장하고 싶지 않은 경우
- 여러 머신에서 이미지를 공유하고 싶은 경우
- 에이전트가 이미지를 빌드하고 Amazon SageMaker 또는 VertexAI와 같은 클라우드 컴퓨팅 리소스를 사용하는 경우

런치 에이전트를 컨테이너 레지스트리에 연결하려면 사용하고자 하는 환경 및 레지스트리에 관한 추가 정보를 런치 에이전트 구성에 제공하세요. 또한, 사용 사례에 따라 필요한 구성 요소와 상호 작용할 수 있도록 환경 내에서 에이전트에 권한을 부여하세요.

:::note
런치 에이전트는 작업이 실행되는 노드가 엑세스할 수 있는 모든 컨테이너 레지스트리에서 *당기기(pulling)*를 지원합니다. 여기에는 비공개 Dockerhub, JFrog, Quay 등이 포함됩니다. 레지스트리에 이미지를 *푸시하는(push)* 기능은 현재 ECR, ACR 및 GCR에 대해서만 지원됩니다.
:::

### 에이전트 구성

런치 에이전트 구성(`launch-config.yaml`)에서 대상 리소스 환경의 이름과 `environment` 및 `registry` 키에 대한 컨테이너 레지스트리를 각각 제공하세요.

다음 탭은 환경 및 레지스트리에 따라 런치 에이전트를 구성하는 방법을 보여줍니다.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS 환경 구성에는 `region` 키가 필요합니다. 리전은 에이전트가 실행되는 AWS 리전이어야 합니다. 에이전트는 `boto3`을 사용하여 기본 AWS 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
registry:
  type: ecr
  # 에이전트가 이미지를 저장할 ECR 저장소의 URI입니다.
  # 환경에서 구성한 리전과 일치해야 합니다.
  uri: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # 또는, 단순히 저장소 이름을 설정할 수 있습니다.
  # repository: my-repository-name
```

기본 AWS 자격 증명을 구성하는 방법에 대한 자세한 내용은 [boto3 문서](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)를 참조하세요.

  </TabItem>
  <TabItem value="gcp">

GCP 환경에는 `region` 및 `project` 키가 필요합니다. `region`을 에이전트가 실행되는 GCP 리전으로 설정하세요. GCP `project`를 에이전트가 실행되는 프로젝트로 설정하세요. 에이전트는 `google.auth.default()`을 사용하여 기본 GCP 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
registry:
  # GCP 환경 구성이 필요합니다.
  type: gcr
  # 에이전트가 이미지를 저장할 아티팩트 레지스트리 저장소 및 이미지 이름의 URI입니다.
  # 환경에서 구성한 리전과 프로젝트와 일치해야 합니다.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # 또는, 저장소 및 이미지 이름 키를 설정할 수 있습니다.
  # repository: my-artifact-repo
  # image-name: my-image-name
```

기본 GCP 자격 증명을 구성하는 방법에 대한 자세한 내용은 [`google-auth` 문서](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)를 참조하세요.

  </TabItem>
  <TabItem value="azure">

Azure 환경에는 추가 키가 필요하지 않습니다. 에이전트가 시작될 때 `azure.identity.DefaultAzureCredential()`을 사용하여 기본 Azure 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: azure
registry:
  type: acr
  uri: https://my-registry.azurecr.io/my-repository
```

기본 Azure 자격 증명을 구성하는 방법에 대한 자세한 내용은 [`azure-identity` 문서](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python)를 참조하세요.

  </TabItem>
</Tabs>

## 에이전트 권한

에이전트 권한은 사용 사례에 따라 다릅니다. 아래에는 런치 에이전트가 사용하는 정책이 개요되어 있습니다.

### 클라우드 레지스트리 권한

아래는 런치 에이전트가 클라우드 레지스트리와 상호 작용하는 데 일반적으로 필요한 권한입니다.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

```yaml
{
  'Version': '2012-10-17',
  'Statement':
    [
      {
        'Effect': 'Allow',
        'Action':
          [
            'ecr:CreateRepository',
            'ecr:UploadLayerPart',
            'ecr:PutImage',
            'ecr:CompleteLayerUpload',
            'ecr:InitiateLayerUpload',
            'ecr:DescribeRepositories',
            'ecr:DescribeImages',
            'ecr:BatchCheckLayerAvailability',
            'ecr:BatchDeleteImage',
          ],
        'Resource': 'arn:aws:ecr:<region>:<account-id>:repository/<repository>',
      },
      {
        'Effect': 'Allow',
        'Action': 'ecr:GetAuthorizationToken',
        'Resource': '*',
      },
    ],
}
```

  </TabItem>
  <TabItem value="gcp">

```js
artifactregistry.dockerimages.list;
artifactregistry.repositories.downloadArtifacts;
artifactregistry.repositories.list;
artifactregistry.repositories.uploadArtifacts;
```

  </TabItem>
  <TabItem value="azure">

Kaniko 빌더를 사용하는 경우 [`AcrPush` 역할](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush)을 추가하세요.

</TabItem>
</Tabs>

### Kaniko 권한

런치 에이전트는 Kaniko 빌더를 사용하는 경우 클라우드 스토리지에 푸시할 권한이 필요합니다. Kaniko는 빌드 작업을 실행하는 파드 외부의 컨텍스트 스토어를 사용합니다.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS에서 Kaniko 빌더에 권장되는 컨텍스트 스토어는 Amazon S3입니다. 다음 정책은 에이전트가 S3 버킷에 접근할 수 있도록 사용될 수 있습니다:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>/*"]
    }
  ]
}
```

  </TabItem>
  <TabItem value="gcp">

GCP에서는 빌드 컨텍스트를 GCS에 업로드하기 위해 에이전트에 필요한 다음 IAM 권한이 있습니다:

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

  </TabItem>
  <TabItem value="azure">

에이전트가 Azure Blob Storage에 빌드 컨텍스트를 업로드할 수 있도록 [Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) 역할이 필요합니다.

  </TabItem>
</Tabs>

### 작업 실행 권한

에이전트는 AWS나 GCP 클라우드에서 Amazon SageMaker 또는 Vertex AI에서 작업을 시작할 수 있는 권한이 필요합니다.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon SageMaker', value: 'aws'},
{label: 'Vertex AI', value: 'vertex'},
]}>
<TabItem value="aws">

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:CreateTrainingJob",
      "Resource": "arn:aws:sagemaker:<REGION>:<ACCOUNT-ID>/*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "<ARN-OF-ROLE-TO-PASS>"
    },
    {
      "Effect": "Allow",
      "Action": "kms:CreateGrant",
      "Resource": "<ARN-OF-KMS-KEY>",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "sagemaker.<REGION>.amazonaws.com",
          "kms:GrantIsForAWSResource": "true"
        }
      }
    }
  ]
}
```

:::note
SageMaker 대기열과 관련된 ResourceConfig에 특정 VolumeKmsKeyId가 지정되어 있고 관련 역할에 이 작업을 허용하는 정책이 없는 경우에만 `kms:CreateGrant` 권한이 필요합니다.
:::

  </TabItem>
  <TabItem value="vertex">

vertex AI에서 작업을 실행하려면 GCS 버킷을 설정하고 위에서 설명한 권한을 에이전트에 부여해야 합니다.

```js
ml.jobs.create;
ml.jobs.list;
ml.jobs.get;
```

  </TabItem>
</Tabs>

### Git