---
title: Set up launch agent
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 고급 에이전트 설정

이 가이드는 다양한 환경에서 컨테이너 이미지를 빌드하기 위해 W&B Launch 에이전트를 설정하는 방법에 대한 정보를 제공합니다.

:::info
빌드는 git 및 코드 아티팩트 작업에 대해서만 필요합니다. 이미지 작업에는 빌드가 필요하지 않습니다.

작업 유형에 대한 자세한 내용은 [런치 작업 생성](./create-launch-job.md)를 참조하세요.
:::

## 빌더

Launch 에이전트는 [Docker](https://docs.docker.com/) 또는 [Kaniko](https://github.com/GoogleContainerTools/kaniko)를 사용하여 이미지를 빌드할 수 있습니다.

* Kaniko: Kubernetes에서 빌드를 특권 컨테이너로 실행하지 않고 컨테이너 이미지를 빌드합니다.
* Docker: 로컬에서 `docker build` 명령을 실행하여 컨테이너 이미지를 빌드합니다.

빌더 유형은 launch 에이전트 설정의 `builder.type` 키를 통해 `docker`, `kaniko` 또는 `noop`으로 설정하여 빌드를 비활성화할 수 있습니다. 기본적으로 에이전트 헬름 차트는 `builder.type`을 `noop`으로 설정합니다. `builder` 섹션의 추가 키는 빌드 프로세스를 설정하는 데 사용됩니다.

에이전트 설정에 빌더가 지정되지 않았고 작동 가능한 `docker` CLI가 발견되면 에이전트는 기본적으로 Docker를 사용합니다. Docker가 사용 가능하지 않으면 에이전트는 `noop`을 기본적으로 사용합니다.

:::tip
Kubernetes 클러스터에서는 이미지를 빌드하기 위해 Kaniko를 사용하세요. 그 외의 경우에는 Docker를 사용하세요.
:::

## 컨테이너 레지스트리에 푸시하기

Launch 에이전트는 모든 빌드된 이미지에 고유한 소스 해시를 태그합니다. 에이전트는 `builder.destination` 키에 지정된 레지스트리에 이미지를 푸시합니다.

예를 들어, `builder.destination` 키가 `my-registry.example.com/my-repository`로 설정된 경우 에이전트는 `my-registry.example.com/my-repository:<source-hash>`로 이미지를 태그하고 푸시합니다. 이미지가 레지스트리에 이미 존재하면 빌드는 건너뜁니다.

### 에이전트 구성

헬름 차트를 통해 에이전트를 배포하는 경우, 에이전트 설정은 `values.yaml` 파일의 `agentConfig` 키에 제공되어야 합니다.

`wandb launch-agent`를 사용하여 직접 에이전트를 호출하는 경우, 에이전트 설정을 YAML 파일 경로로 `--config` 플래그와 함께 제공할 수 있습니다. 기본적으로 설정은 `~/.config/wandb/launch-config.yaml`에서 로드됩니다.

`launch-config.yaml` 내의 launch 에이전트 설정에는 대상 리소스 환경의 이름과 `environment` 및 `registry` 키에 대한 컨테이너 레지스트리를 제공하세요.

다음 탭들은 환경과 레지스트리에 기반한 launch 에이전트 구성을 보여줍니다.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS 환경 설정은 지역(region) 키가 필요합니다. 지역은 에이전트가 실행되는 AWS 지역이어야 합니다.

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 ECR 리포지토리의 URI입니다.
  # region이 환경에 구성한 것과 일치하는지 확인하세요.
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko를 사용하는 경우 에이전트가 빌드 컨텍스트를 저장할 S3 버킷 지정
  build-context-store: s3://<bucket-name>/<path>
```

에이전트는 기본 AWS 자격 증명을 로드하기 위해 boto3 를 사용합니다. 기본 AWS 자격 증명을 설정하는 방법에 대한 더 많은 정보는 [boto3 설명서](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)를 참조하세요.

  </TabItem>
  <TabItem value="gcp">

Google Cloud 환경은 지역(region) 및 프로젝트(project) 키가 필요합니다. `region`을 에이전트가 실행되는 Google Cloud 프로젝트에 설정하세요. 에이전트는 `google.auth.default()`를 통해 Python에서 기본 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 Artifact Registry 리포지토리와 이미지 이름의 URI
  # region과 project가 환경에 구성한 것과 일치하는지 확인하세요.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko를 사용하는 경우 에이전트가 빌드 컨텍스트를 저장할 GCS 버킷 지정
  build-context-store: gs://<bucket-name>/<path>
```

에이전트에서 사용할 수 있도록 기본 GCP 자격 증명을 구성하는 방법에 대한 더 많은 정보는 [`google-auth` 설명서](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) 를 참조하세요.

  </TabItem>
  <TabItem value="azure">

Azure 환경은 추가적인 키가 필요하지 않습니다. 에이전트가 시작될 때 `azure.identity.DefaultAzureCredential()`을 사용하여 기본 Azure 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 Azure Container Registry 리포지토리의 URI
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko를 사용하는 경우 에이전트가 빌드 컨텍스트를 저장할 Azure Blob Storage 컨테이너 지정
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

기본 Azure 자격 증명을 설정하는 방법에 대한 더 많은 정보는 [`azure-identity` 설명서](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python)를 참조하세요.

  </TabItem>
</Tabs>

## 에이전트 권한

에이전트 권한 요구 사항은 유스 케이스에 따라 다릅니다.

### 클라우드 레지스트리 권한

아래는 launch 에이전트가 클라우드 레지스트리와 상호작용하기 위해 일반적으로 요구되는 권한입니다.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
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

Kaniko 빌더를 사용하는 경우, [`AcrPush` 역할](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush)을 추가하세요.

</TabItem>
</Tabs>

### Kaniko를 위한 스토리지 권한

Kaniko 빌더를 사용하는 경우 launch 에이전트는 클라우드 스토리지에 푸시할 수 있는 권한이 필요합니다. Kaniko는 빌드 작업을 실행하는 pod 외부에 컨텍스트 스토어를 사용합니다.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS에서 Kaniko 빌더에 추천되는 컨텍스트 스토어는 Amazon S3입니다. 다음 정책은 에이전트에게 S3 버킷에 대한 엑세스 권한을 부여하기 위해 사용할 수 있습니다:

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

GCP에서 에이전트가 GCS에 빌드 컨텍스트를 업로드할 수 있는 IAM 권한은 다음과 같습니다:

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

  </TabItem>
  <TabItem value="azure">

에이전트가 Azure Blob Storage에 빌드 컨텍스트를 업로드할 수 있게 하려면 [Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) 역할이 필요합니다.

  </TabItem>
</Tabs>

## Kaniko 빌드 커스터마이징

에이전트 설정의 `builder.kaniko-config` 키에서 Kaniko 작업에 사용할 Kubernetes Job 사양을 지정하세요. 예를 들어:

```yaml title="launch-config.yaml"
builder:
  type: kaniko
  build-context-store: <my-build-context-store>
  destination: <my-image-destination>
  build-job-name: wandb-image-build
  kaniko-config:
    spec:
      template:
        spec:
          containers:
          - args:
            - "--cache=false" # Args는 "key=value" 형식이어야 함
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## Launch 에이전트를 CoreWeave에 배포
W&B Launch 에이전트를 선택적으로 CoreWeave 클라우드 인프라에 배포할 수 있습니다. CoreWeave는 GPU 가속 워크로드에 맞게 설계된 클라우드 인프라입니다.

Launch 에이전트를 CoreWeave에 배포하는 방법에 대한 정보는 [CoreWeave 설명서](https://docs.coreweave.com/partners/weights-and-biases#integration)를 참조하세요.

:::note
CoreWeave 인프라에 Launch 에이전트를 배포하려면 [CoreWeave 계정](https://cloud.coreweave.com/login)을 생성해야 합니다.
:::
