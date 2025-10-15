---
title: Set up launch agent
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: /ko/guides//launch/setup-agent-advanced
---

# 고급 에이전트 설정

본 가이드는 다양한 환경에서 컨테이너 이미지를 빌드하기 위해 W&B Launch 에이전트를 설정하는 방법에 대한 정보를 제공합니다.

{{% alert %}}
빌드는 git 및 코드 아티팩트 job에만 필요합니다. 이미지 job은 빌드가 필요하지 않습니다.

job 유형에 대한 자세한 내용은 [Launch job 생성]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ko" >}})을 참조하세요.
{{% /alert %}}

## 빌더

Launch 에이전트는 [Docker](https://docs.docker.com/) 또는 [Kaniko](https://github.com/GoogleContainerTools/kaniko)를 사용하여 이미지를 빌드할 수 있습니다.

*   Kaniko: 권한이 필요한 컨테이너로 빌드를 실행하지 않고 Kubernetes에서 컨테이너 이미지를 빌드합니다.
*   Docker: 로컬에서 `docker build` 코맨드를 실행하여 컨테이너 이미지를 빌드합니다.

빌더 유형은 launch 에이전트 설정에서 `builder.type` 키를 `docker`, `kaniko` 또는 `noop`(빌드 해제)으로 설정하여 제어할 수 있습니다. 기본적으로 에이전트 helm chart는 `builder.type`을 `noop`로 설정합니다. `builder` 섹션의 추가 키는 빌드 프로세스를 구성하는 데 사용됩니다.

에이전트 설정에 빌더가 지정되지 않고 작동하는 `docker` CLI가 발견되면 에이전트는 기본적으로 Docker를 사용합니다. Docker를 사용할 수 없으면 에이전트는 기본적으로 `noop`를 사용합니다.

{{% alert %}}
Kubernetes 클러스터에서 이미지를 빌드하려면 Kaniko를 사용하세요. 다른 모든 경우에는 Docker를 사용하세요.
{{% /alert %}}

## 컨테이너 레지스트리에 푸시

Launch 에이전트는 빌드하는 모든 이미지에 고유한 소스 해시로 태그를 지정합니다. 에이전트는 `builder.destination` 키에 지정된 레지스트리에 이미지를 푸시합니다.

예를 들어, `builder.destination` 키가 `my-registry.example.com/my-repository`로 설정된 경우 에이전트는 이미지를 `my-registry.example.com/my-repository:<source-hash>`로 태그 지정하고 푸시합니다. 이미지가 레지스트리에 존재하면 빌드는 건너뜁니다.

### 에이전트 설정

Helm chart를 통해 에이전트를 배포하는 경우 에이전트 설정은 `values.yaml` 파일의 `agentConfig` 키에 제공되어야 합니다.

`wandb launch-agent`로 에이전트를 직접 호출하는 경우 `--config` 플래그를 사용하여 에이전트 설정을 YAML 파일 경로로 제공할 수 있습니다. 기본적으로 설정은 `~/.config/wandb/launch-config.yaml`에서 로드됩니다.

launch 에이전트 설정(`launch-config.yaml`) 내에서 대상 리소스 환경의 이름과 `environment` 및 `registry` 키에 대한 컨테이너 레지스트리를 각각 제공합니다.

다음 탭은 환경 및 레지스트리를 기반으로 launch 에이전트를 구성하는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 환경 설정에는 region 키가 필요합니다. region은 에이전트가 실행되는 AWS region이어야 합니다.

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 ECR 리포지토리의 URI입니다.
  # region이 환경에 구성한 region과 일치하는지 확인하십시오.
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko를 사용하는 경우 에이전트가 빌드 컨텍스트를 저장할 S3 버킷을 지정합니다.
  build-context-store: s3://<bucket-name>/<path>
```

에이전트는 boto3을 사용하여 기본 AWS 자격 증명을 로드합니다. 기본 AWS 자격 증명을 구성하는 방법에 대한 자세한 내용은 [boto3 설명서](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)를 참조하세요.
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 환경에는 region 및 project 키가 필요합니다. `region`을 에이전트가 실행되는 region으로 설정합니다. `project`를 에이전트가 실행되는 Google Cloud 프로젝트로 설정합니다. 에이전트는 Python에서 `google.auth.default()`를 사용하여 기본 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 Artifact Registry 리포지토리 및 이미지 이름의 URI입니다.
  # region 및 프로젝트가 환경에 구성한 것과 일치하는지 확인하십시오.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko를 사용하는 경우 에이전트가 빌드 컨텍스트를 저장할 GCS 버킷을 지정합니다.
  build-context-store: gs://<bucket-name>/<path>
```

에이전트에서 사용할 수 있도록 기본 GCP 자격 증명을 구성하는 방법에 대한 자세한 내용은 [`google-auth` 설명서](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)를 참조하세요.

{{% /tab %}}
{{% tab "Azure" %}}

Azure 환경은 추가 키가 필요하지 않습니다. 에이전트가 시작되면 `azure.identity.DefaultAzureCredential()`을 사용하여 기본 Azure 자격 증명을 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 Azure Container Registry 리포지토리의 URI입니다.
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko를 사용하는 경우 에이전트가 빌드 컨텍스트를 저장할 Azure Blob Storage 컨테이너를 지정합니다.
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

기본 Azure 자격 증명을 구성하는 방법에 대한 자세한 내용은 [`azure-identity` 설명서](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python)를 참조하세요.
{{% /tab %}}
{{< /tabpane >}}

## 에이전트 권한

필요한 에이전트 권한은 유스 케이스에 따라 다릅니다.

### 클라우드 레지스트리 권한

다음은 클라우드 레지스트리와 상호 작용하기 위해 launch 에이전트에서 일반적으로 요구하는 권한입니다.

{{< tabpane text=true >}}
{{% tab "AWS" %}}
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
{{% /tab %}}
{{% tab "GCP" %}}
```js
artifactregistry.dockerimages.list;
artifactregistry.repositories.downloadArtifacts;
artifactregistry.repositories.list;
artifactregistry.repositories.uploadArtifacts;
```

{{% /tab %}}
{{% tab "Azure" %}}

Kaniko 빌더를 사용하는 경우 [`AcrPush` 역할](https://learn.microsoft.com/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush)을 추가합니다.
{{% /tab %}}
{{< /tabpane >}}

### Kaniko의 스토리지 권한

에이전트가 Kaniko 빌더를 사용하는 경우 launch 에이전트는 클라우드 스토리지에 푸시할 수 있는 권한이 필요합니다. Kaniko는 빌드 job을 실행하는 pod 외부의 컨텍스트 저장소를 사용합니다.

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS에서 Kaniko 빌더에 권장되는 컨텍스트 저장소는 Amazon S3입니다. 다음 정책을 사용하여 에이전트에 S3 버킷에 대한 액세스 권한을 부여할 수 있습니다.

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
{{% /tab %}}
{{% tab "GCP" %}}
GCP에서 에이전트가 빌드 컨텍스트를 GCS에 업로드하려면 다음 IAM 권한이 필요합니다.

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

에이전트가 빌드 컨텍스트를 Azure Blob Storage에 업로드하려면 [Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) 역할이 필요합니다.
{{% /tab %}}
{{< /tabpane >}}

## Kaniko 빌드 사용자 정의

에이전트 설정의 `builder.kaniko-config` 키에서 Kaniko job이 사용하는 Kubernetes Job 사양을 지정합니다. 예:

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
            - "--cache=false" # Args must be in the format "key=value"
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## CoreWeave에 Launch 에이전트 배포
선택적으로 W&B Launch 에이전트를 CoreWeave Cloud 인프라에 배포합니다. CoreWeave는 GPU 가속 워크로드를 위해 특별히 구축된 클라우드 인프라입니다.

Launch 에이전트를 CoreWeave에 배포하는 방법에 대한 자세한 내용은 [CoreWeave 설명서](https://docs.coreweave.com/partners/weights-and-biases#integration)를 참조하세요.

{{% alert %}}
Launch 에이전트를 CoreWeave 인프라에 배포하려면 [CoreWeave 계정](https://cloud.coreweave.com/login)을 만들어야 합니다.
{{% /alert %}}
