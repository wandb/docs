---
title: 런치 에이전트 설정
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: guides/launch/setup-agent-advanced
---

# 고급 에이전트 설정

이 가이드는 다양한 환경에서 W&B Launch 에이전트가 컨테이너 이미지를 빌드하도록 설정하는 방법을 안내합니다.

{{% alert %}}
빌드는 git 및 코드 아티팩트 job 에서만 필요합니다. 이미지 job 에서는 빌드가 필요하지 않습니다.

job 유형에 대한 자세한 내용은 [런치 job 생성]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ko" >}})을 참고하세요.
{{% /alert %}}

## 빌더

Launch 에이전트는 [Docker](https://docs.docker.com/) 또는 [Kaniko](https://github.com/GoogleContainerTools/kaniko)를 사용해 이미지를 빌드할 수 있습니다.

* Kaniko: Kubernetes 내에서 빌드 job 을 루트 권한 없이 컨테이너 이미지로 생성합니다.
* Docker: 로컬에서 `docker build` 명령어를 실행해 컨테이너 이미지를 만듭니다.

빌더 타입은 launch 에이전트 설정의 `builder.type` 키를 통해 `docker`, `kaniko`, 또는 `noop`(빌드 비활성화)로 제어할 수 있습니다. 기본적으로 에이전트 헬름 차트는 `builder.type` 을 `noop` 으로 설정합니다. `builder` 섹션의 추가 키들은 빌드 프로세스 설정에 사용됩니다.

에이전트 설정에 빌더가 지정되어 있지 않고, 동작 가능한 `docker` CLI가 발견되면, 에이전트는 기본적으로 Docker 를 사용합니다. Docker 가 없을 경우에는 자동으로 `noop` 으로 처리됩니다.

{{% alert %}}
이미지 빌드가 Kubernetes 클러스터 내에서 필요하다면 Kaniko 를 사용하세요. 그 외의 환경에서는 Docker 를 사용하세요.
{{% /alert %}}


## 컨테이너 레지스트리에 푸시하기

launch 에이전트는 빌드한 모든 이미지에 고유한 source 해시로 태그를 달아 저장합니다. 에이전트는 `builder.destination` 키에 지정된 레지스트리에 이미지를 푸시합니다.

예를 들어, `builder.destination` 키가 `my-registry.example.com/my-repository`로 설정되어 있으면, 에이전트는 이미지를 `my-registry.example.com/my-repository:<source-hash>`로 태그 및 푸시합니다. 만약 이미지가 이미 레지스트리에 있으면, 빌드는 생략됩니다.

### 에이전트 설정

에이전트를 우리 헬름 차트로 배포하는 경우, 에이전트 설정은 `values.yaml` 파일의 `agentConfig` 키에 입력하면 됩니다.

직접 `wandb launch-agent` 명령어로 에이전트를 실행할 경우, `--config` 플래그를 사용하여 YAML 파일 경로로 에이전트 설정을 지정할 수 있습니다. 기본적으로는 `~/.config/wandb/launch-config.yaml` 경로에서 설정을 불러옵니다.

launch 에이전트 설정(`launch-config.yaml`) 파일에는 대상 리소스 환경 이름과 컨테이너 레지스트리를 각각 `environment` 와 `registry` 키로 지정하세요.

아래 탭에서는 사용 환경과 레지스트리에 따라 launch 에이전트를 어떻게 설정하는지 예시를 보여줍니다.

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 환경 설정에는 region 키가 필요합니다. region 값은 에이전트가 동작하는 AWS 리전을 입력해야 합니다.

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 ECR 리포지토리 URI입니다.
  # region 값이 환경에 설정한 값과 일치하는지 확인하세요.
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko 사용 시, 에이전트가 빌드 컨텍스트를 저장할 S3 버킷을 지정합니다.
  build-context-store: s3://<bucket-name>/<path>
```

에이전트는 boto3 로 기본 AWS 인증정보를 불러옵니다. 기본 AWS 인증정보 설정 방법은 [boto3 공식 문서](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)를 참고하세요.
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 환경은 region 와 project 키가 필요합니다. `region`에 에이전트가 동작하는 리전을, `project`에 실행되는 Google Cloud 프로젝트를 입력합니다. 에이전트는 Python 환경의 `google.auth.default()`를 이용하여 기본 인증정보를 로드합니다.

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 Artifact Registry 리포지토리 및 이미지 이름의 URI를 입력하세요.
  # region 및 project가 환경에 설정한 값과 일치하는지 확인하세요.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko 사용 시, 에이전트가 빌드 컨텍스트를 저장할 GCS 버킷을 지정합니다.
  build-context-store: gs://<bucket-name>/<path>
```

기본 GCP 인증정보가 에이전트에서 사용 가능하도록 설정하는 방법은 [`google-auth` 공식 문서](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)를 참고하세요.

{{% /tab %}}
{{% tab "Azure" %}}

Azure 환경에는 추가적인 키가 필요하지 않습니다. 에이전트가 시작될 때 `azure.identity.DefaultAzureCredential()`을 사용해 기본 Azure 인증정보를 불러옵니다.

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # 에이전트가 이미지를 저장할 Azure Container Registry 리포지토리 URI입니다.
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko 사용 시, 에이전트가 빌드 컨텍스트를 저장할 Azure Blob Storage 컨테이너를 지정합니다.
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

기본 Azure 인증정보 설정 방법은 [`azure-identity` 공식 문서](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python)를 참고하세요.
{{% /tab %}}
{{< /tabpane >}}

## 에이전트 권한

에이전트가 필요로 하는 권한은 유스 케이스에 따라 다릅니다.

### 클라우드 레지스트리 권한

아래는 launch 에이전트가 클라우드 레지스트리와 연동할 때 일반적으로 요구되는 권한들입니다.

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

Kaniko 빌더 사용 시 [`AcrPush` 역할](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles/containers#acrpush)을 추가하세요.
{{% /tab %}}
{{< /tabpane >}}

### Kaniko 를 위한 스토리지 권한

launch 에이전트가 Kaniko 빌더를 사용할 경우에는 클라우드 스토리지에 푸시할 수 있는 권한이 필요합니다. Kaniko 는 빌드 작업을 실행하는 pod 외부의 컨텍스트 스토어를 이용합니다.

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS에서 Kaniko 빌더의 권장 컨텍스트 스토어는 Amazon S3입니다. 아래 정책을 통해 에이전트에 S3 버킷에 대한 엑세스를 부여할 수 있습니다.

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
GCP에서는 에이전트가 빌드 컨텍스트를 GCS에 업로드하려면 다음 IAM 권한이 필요합니다.

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

에이전트가 Azure Blob Storage에 빌드 컨텍스트를 업로드할 수 있으려면 [Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) 역할이 필요합니다.
{{% /tab %}}
{{< /tabpane >}}


## Kaniko 빌드 커스터마이징

에이전트 설정의 `builder.kaniko-config` 키에 Kaniko job 에서 사용할 Kubernetes Job 스펙을 지정하세요. 예시:

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
            - "--cache=false" # Args 값은 반드시 "key=value" 형식이어야 합니다
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## CoreWeave 에 Launch 에이전트 배포하기
필요하다면 W&B Launch 에이전트를 CoreWeave 클라우드 인프라에 배포할 수 있습니다. CoreWeave는 GPU 가속 워크로드에 최적화된 클라우드 인프라입니다.

Launch 에이전트를 CoreWeave에 배포하는 방법은 [CoreWeave 공식 문서](https://docs.coreweave.com/partners/weights-and-biases#integration)에서 확인하세요.

{{% alert %}}
Launch 에이전트를 CoreWeave 인프라에 배포하려면 [CoreWeave 계정](https://cloud.coreweave.com/login)을 생성해야 합니다.
{{% /alert %}}