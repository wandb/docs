---
title: 직접 버킷(BYOB) 사용하기
menu:
  default:
    identifier: ko-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

## 개요
BYOB(Bring Your Own Bucket)를 사용하면 W&B Artifacts 및 기타 민감한 데이터를 귀하의 클라우드 또는 온프레미스 인프라에 직접 저장할 수 있습니다. [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우, 귀하의 버킷에 저장된 데이터는 W&B에서 관리하는 인프라로 복사되지 않습니다.

{{% alert %}}
* W&B SDK / CLI / UI와 버킷 간의 통신은 [pre-signed URL]({{< relref path="./presigned-urls.md" lang="ko" >}})을 사용하여 이뤄집니다.
* W&B는 W&B Artifacts를 삭제하기 위해 가비지 컬렉션 프로세스를 사용합니다. 자세한 내용은 [Artifacts 삭제]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ko" >}})를 참고하세요.
* 버킷을 설정할 때 하위 경로(sub-path)를 지정할 수 있으므로, W&B가 버킷의 루트 폴더에는 파일을 저장하지 않도록 할 수 있습니다. 이는 조직의 버킷 거버넌스 정책을 더 쉽게 준수하도록 도와줍니다.
{{% /alert %}}

### 중앙 데이터베이스 vs 버킷에 저장되는 데이터
BYOB 기능을 사용할 경우, 일부 데이터는 W&B 중앙 데이터베이스에 저장되고, 다른 일부는 귀하의 버킷에 저장됩니다.

#### 데이터베이스
- 사용자, 팀, Artifacts, Experiments, Projects에 대한 메타데이터
- Reports
- Experiment 로그
- 시스템 메트릭
- 콘솔 로그

#### 버킷
- Experiment 파일 및 메트릭
- Artifact 파일
- 미디어 파일
- Run 파일
- Parquet 형식으로 내보낸 히스토리 메트릭 및 시스템 이벤트

### 버킷 범위
스토리지 버킷의 설정 범위는 두 가지가 있습니다:

| 범위           | 설명 |
|----------------|-------------|
| 인스턴스 레벨 | [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 및 [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 환경에서는, 조직 또는 인스턴스 내에서 필요한 권한이 있는 사용자는 인스턴스의 스토리지 버킷에 저장된 파일에 엑세스할 수 있습니다. [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에는 해당하지 않습니다. |
| 팀 레벨     | W&B Team이 팀 레벨 스토리지 버킷을 사용하도록 설정된 경우, 팀 멤버들은 해당 버킷에 저장된 파일에 엑세스할 수 있습니다. 팀 레벨 스토리지 버킷은 높은 수준의 데이터 접근 제어와 팀간 데이터 분리를 지원하며, 민감 데이터나 엄격한 컴플라이언스 요구 사항이 있는 팀에 적합합니다.<br><br>팀 레벨 스토리지는 하나의 인스턴스를 여러 부서나 업무 단위가 공유할 때 인프라 및 관리 자원 활용 면에서 효율을 높일 수 있습니다. 별도의 프로젝트 팀이 서로 다른 고객 업무를 각각 운영하는 경우에도 적합합니다. 모든 배포 타입에서 사용 가능합니다. 팀 레벨 BYOB는 팀 생성 시 설정합니다. |

이러한 유연한 설계로 인해, 조직의 요구 사항에 따라 다양한 스토리지 토폴로지를 구성할 수 있습니다. 예시:

- 하나의 동일한 버킷을 인스턴스 및 하나 이상의 팀이 함께 사용할 수 있습니다.
- 각 팀마다 별도의 버킷을 사용하거나, 일부 팀은 인스턴스 버킷을 사용하도록 할 수 있고, 또는 여러 팀이 하위 경로(subpaths)를 나눠 하나의 버킷을 공유할 수도 있습니다.
- 서로 다른 팀의 버킷을 서로 다른 클라우드 인프라 환경 또는 리전(region)에 호스팅하고 별도의 스토리지 관리 팀이 관리할 수도 있습니다.

예를 들어, 조직에 Kappa라는 팀이 있다고 가정합시다. 조직(및 Kappa 팀)은 기본적으로 인스턴스 레벨 스토리지 버킷을 사용합니다. 이후 Omega라는 팀을 새로 만들고, 이 때 Omega 팀은 팀 레벨 스토리지 버킷을 설정합니다. Omega 팀에서 생성한 파일은 Kappa 팀에서는 접근할 수 없습니다. 반대로, Kappa 팀에서 생성한 파일은 Omega 팀도 엑세스할 수 있습니다. 만약 Kappa 팀의 데이터를 완전히 분리하고 싶다면, Kappa 팀에도 팀 레벨 스토리지 버킷을 설정해야 합니다.

### 지원 매트릭스
W&B는 다음과 같은 스토리지 제공자를 지원합니다:
- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage): AI 워크로드에 최적화된 S3-호환 오브젝트 스토리지 서비스입니다.
- [Amazon S3](https://aws.amazon.com/s3/): 업계 최고 수준의 확장성, 가용성, 보안, 성능을 제공하는 오브젝트 스토리지 서비스입니다.
- [Google Cloud Storage](https://cloud.google.com/storage): 비정형 데이터를 대규모로 저장할 수 있는 관리형 서비스입니다.
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs): 텍스트, 바이너리 데이터, 이미지, 동영상, 로그 등 대용량 비정형 데이터를 저장하는 클라우드 오브젝트 스토리지 솔루션입니다.
- [MinIO](https://github.com/minio/minio) 등 S3 호환 스토리지를 자체 클라우드 또는 온프레미스 인프라에 호스팅할 수 있습니다.

아래 표는 각 W&B 배포 유형과 스토리지 범위별로 BYOB 지원 여부를 보여줍니다.

| W&B 배포 타입        | 인스턴스 레벨   | 팀 레벨 | 추가 정보 |
|----------------------|----------------|---------|----------------|
| Dedicated Cloud      | &check;        | &check;| 인스턴스 및 팀 레벨 BYOB는 CoreWeave AI Object Storage, Amazon S3, GCP Storage, Microsoft Azure Blob Storage, MinIO 등 S3 호환 스토리지 (자체 클라우드 또는 온프레미스)에 대해 지원됨. |
| Multi-tenant Cloud   | 해당 없음       | &check;| 팀 레벨 BYOB는 CoreWeave AI Object Storage, Amazon S3, GCP Storage를 지원. Microsoft Azure는 기본 및 유일 스토리지 버킷만 W&B가 관리. |
| Self-Managed         | &check;        | &check;| 인스턴스 및 팀 레벨 BYOB는 CoreWeave AI Object Storage, Amazon S3, GCP Storage, Microsoft Azure Blob Storage, MinIO 등 S3 호환 스토리지를 자체 클라우드 또는 온프레미스에 호스팅 가능. |

다음 섹션에서 BYOB 설정 과정을 안내합니다.

## 버킷을 준비하세요 {#provision-your-bucket}

[지원 여부 확인]({{< relref path="#availability-matrix" lang="ko" >}}) 이후에는, 엑세스 정책 및 CORS를 포함한 스토리지 버킷을 준비할 수 있습니다. 계속하려면 탭을 선택하세요.

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
<a id="coreweave-requirements"></a>**필수 사항**:
- **Dedicated Cloud** 또는 **Self-Hosted** v0.70.0 이상, 혹은 **Multi-tenant Cloud**.
- AI Object Storage가 활성화되어 있고 버킷, API 엑세스 키, 시크릿 키를 생성할 권한이 있는 CoreWeave 계정.
- W&B 인스턴스가 CoreWeave 네트워크 엔드포인트에 접근 가능해야 합니다.

자세한 내용은 CoreWeave 공식 문서의 [Create a CoreWeave AI Object Storage bucket](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/create-bucket)을 참고하세요.

1. **Multi-tenant Cloud**: 버킷 정책에 필요한 조직 ID를 확인하세요.
    1. [W&B App](https://wandb.ai/)에 로그인합니다.
    1. 왼쪽 네비게이션에서 **Create a new team**을 클릭하세요.
    1. 열리는 패널에서, **Invite team members** 위에 표시된 W&B 조직 ID를 복사하세요.
    1. 이 페이지는 열어둡니다. 이 ID는 이후 [W&B 설정]({{< relref path="#configure-byob" lang="ko" >}})에 사용됩니다.
1. CoreWeave에서 원하는 버킷 이름과 가용 영역을 지정하여 버킷을 생성합니다. 필요시, W&B 파일의 하위 경로로 사용할 폴더를 추가로 생성하세요. 버킷 이름, 가용 영역, API 엑세스 키, 시크릿 키, 하위 경로 정보를 메모해 두세요.
1. Cross-origin resource sharing (CORS) 정책을 버킷에 다음과 같이 설정하세요:
    ```json
    [
      {
        "AllowedHeaders": [
          "*"
        ],
        "AllowedMethods": [
          "GET",
          "HEAD",
          "PUT"
        ],
        "AllowedOrigins": [
          "*"
        ],
        "ExposeHeaders": [
          "ETag"
        ],
        "MaxAgeSeconds": 3000
      }
    ]
    ```
    CoreWeave 스토리지는 S3 와 호환됩니다. CORS에 대한 자세한 내용은 [AWS CORS 설정 방법](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html)을 참고하세요.
1. **Multi-tenant Cloud**: W&B 배포 환경이 버킷에 엑세스하고, 클라우드 인프라나 사용자 브라우저에서 버킷 엑세스 시 사용하는 [pre-signed URL]({{< relref path="./presigned-urls.md" lang="ko" >}})을 생성하는 데 필요한 권한을 부여하는 버킷 정책을 구성하세요. 자세한 내용은 CoreWeave 공식 문서의 [Bucket Policy Reference](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy)를 참고하세요.

    `<cw-bucket>`에는 CoreWeave 버킷명을, `<wb-org-id>`에는 앞 단계에서 확인한 W&B 조직 ID를 입력하세요.

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
      {
        "Sid": "AllowWandbUser",
        "Action": [
          "s3:GetObject*",
          "s3:GetEncryptionConfiguration",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
          "s3:PutObject",
          "s3:GetBucketCORS",
          "s3:GetBucketLocation",
          "s3:GetBucketVersioning"
        ],
        "Effect": "Allow",
        "Resource": [
          "arn:aws:s3:::<cw-bucket>/*",
          "arn:aws:s3:::<cw-bucket>"
        ],
        "Principal": {
          "CW": "arn:aws:iam::wandb:static/wandb-integration"
        },
        "Condition": {
          "StringLike": {
            "wandb:OrgID": [
              "<wb-org-id>"
            ]
          }
        }
      },
      {
        "Sid": "AllowUsersInOrg",
        "Action": "s3:*",
        "Effect": "Allow",
        "Resource": [
          "arn:aws:s3:::<cw-bucket>",
          "arn:aws:s3:::<cw-bucket>/*"
        ],
        "Principal": {
          "CW": "arn:aws:iam::<cw-storage-org-id>:*"
        }
      }]
    }
    ```

"Sid": "AllowUsersInOrg" 절은 W&B 조직에 속한 사용자가 해당 버킷에 직접 접근하는 것을 허용합니다. 필요하지 않다면 이 절은 정책에서 생략할 수 있습니다.

{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
자세한 내용은 AWS 공식 문서의 [Create an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)을 참고하세요.
1. KMS Key 프로비저닝

    W&B는 S3 버킷의 데이터를 암호화/복호화하기 위해 KMS Key를 사전에 준비하도록 요구합니다. 키 유형은 `ENCRYPT_DECRYPT`여야 하며, 다음 정책을 키에 적용하세요:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid" : "Internal",
          "Effect" : "Allow",
          "Principal" : { "AWS" : "<Your_Account_Id>" },
          "Action" : "kms:*",
          "Resource" : "<aws_kms_key.key.arn>"
        },
        {
          "Sid" : "External",
          "Effect" : "Allow",
          "Principal" : { "AWS" : "<aws_principal_and_role_arn>" },
          "Action" : [
            "kms:Decrypt",
            "kms:Describe*",
            "kms:Encrypt",
            "kms:ReEncrypt*",
            "kms:GenerateDataKey*"
          ],
          "Resource" : "<aws_kms_key.key.arn>"
        }
      ]
    }
    ```

    `<Your_Account_Id>`와 `<aws_kms_key.key.arn>` 값을 알맞게 대체합니다.

    만약 [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})나 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용한다면, `<aws_principal_and_role_arn>` 값을 아래에 맞게 입력하십시오:

    * [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
    * [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `arn:aws:iam::830241207209:root`

    이 정책은 귀하의 AWS 계정에 전체 키 엑세스를 부여하며 W&B Platform이 호스팅된 AWS 계정에도 필요한 권한을 할당합니다. 반드시 KMS Key ARN을 기록해 두세요.

1. S3 버킷 생성

    AWS 계정에서 다음 단계를 따라 S3 버킷을 생성하세요:

    1. 원하는 이름으로 S3 버킷을 생성합니다. 필요한 경우, W&B 파일을 저장할 하위 폴더를 추가로 생성할 수 있습니다.
    1. 앞에서 만든 KMS 키로 서버-사이드 암호화(server side encryption)를 활성화합니다.
    1. 다음과 같이 CORS 정책을 적용합니다:

        ```json
        [
          {
              "AllowedHeaders": [
                  "*"
              ],
              "AllowedMethods": [
                  "GET",
                  "HEAD",
                  "PUT"
              ],
              "AllowedOrigins": [
                  "*"
              ],
              "ExposeHeaders": [
                  "ETag"
              ],
              "MaxAgeSeconds": 3000
          }
        ]
        ```
        {{% alert %}}버킷 내 데이터가 [오브젝트 수명 주기 정책](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html)으로 만료될 경우, 일부 run의 기록을 읽지 못할 수 있습니다.{{% /alert %}}
    1. W&B Platform이 호스팅된 AWS 계정에 필요한 S3 권한을 부여합니다. 이는 AI 워크로드나 사용자 브라우저에서 버킷에 엑세스할 때 필요한 [pre-signed URL]({{< relref path="./presigned-urls.md" lang="ko" >}})을 생성할 수 있도록 합니다.

        ```json
        {
          "Version": "2012-10-17",
          "Id": "WandBAccess",
          "Statement": [
            {
              "Sid": "WAndBAccountAccess",
              "Effect": "Allow",
              "Principal": { "AWS": "<aws_principal_and_role_arn>" },
                "Action" : [
                  "s3:GetObject*",
                  "s3:GetEncryptionConfiguration",
                  "s3:ListBucket",
                  "s3:ListBucketMultipartUploads",
                  "s3:ListBucketVersions",
                  "s3:AbortMultipartUpload",
                  "s3:DeleteObject",
                  "s3:PutObject",
                  "s3:GetBucketCORS",
                  "s3:GetBucketLocation",
                  "s3:GetBucketVersioning"
                ],
              "Resource": [
                "arn:aws:s3:::<wandb_bucket>",
                "arn:aws:s3:::<wandb_bucket>/*"
              ]
            }
          ]
        }
        ```

        `<wandb_bucket>`에 버킷 이름을 정확히 입력하고, 이 이름을 반드시 기록해 두세요. 다음으로는 [W&B 설정]({{< relref path="#configure-byob" lang="ko" >}})을 진행할 차례입니다.

        [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우, `<aws_principal_and_role_arn>` 값은 아래와 같습니다.

        * [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `arn:aws:iam::830241207209:root`
  
자세한 정보는 [AWS 셀프매니지드 호스팅 가이드]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ko" >}})에서 확인하세요.

{{% /tab %}}
{{% tab header="GCP" value="gcp"%}}
자세한 내용은 GCP 공식 문서의 [Create a bucket](https://cloud.google.com/storage/docs/creating-buckets)을 참고하세요.
1. GCS 버킷 준비.

    아래 단계에 따라 GCP 프로젝트에서 GCS 버킷을 만드세요:

    1. 원하는 이름으로 GCS 버킷을 생성하고, 필요하다면 모든 W&B 파일을 저장할 하위 폴더(서브 패스)를 추가로 생성합니다.
    1. 암호화 유형은 `Google-managed`로 설정합니다.
    1. `gsutil`을 이용해 CORS 정책을 적용하세요(UI에서는 불가).

       1. 로컬에 `cors-policy.json` 파일을 생성합니다.
       1. 아래의 CORS정책을 파일에 복사해 저장합니다.

           ```json
           [
             {
               "origin": ["*"],
               "responseHeader": ["Content-Type"],
               "exposeHeaders": ["ETag"],
               "method": ["GET", "HEAD", "PUT"],
               "maxAgeSeconds": 3000
             }
           ]
           ```

          {{% alert %}}버킷 내 데이터가 [오브젝트 수명 주기 정책](https://cloud.google.com/storage/docs/lifecycle)으로 만료될 경우, 일부 run의 기록을 읽지 못할 수 있습니다.{{% /alert %}}

      1. `<bucket_name>`을 버킷명으로 바꿔서 `gsutil`로 실행합니다.

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      1. 버킷 정책을 검증하세요. `<bucket_name>`을 알맞게 대체합니다.
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

1. [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 사용 시, GCP Project에서 W&B Platform과 연결된 GCP 서비스 계정에 `storage.admin` 역할을 부여해야 합니다. W&B는 이 역할을 통해 버킷의 CORS 설정, 오브젝트 버전관리 활성 여부 등 특성 확인을 수행합니다. 서비스 계정에 이 역할이 없으면 HTTP 403 에러가 발생할 수 있습니다.

    * [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우: `wandb-integration@wandb-production.iam.gserviceaccount.com`
    * [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})의 경우: `deploy@wandb-production.iam.gserviceaccount.com`

    버킷 이름을 기록해 두고, 이후 [W&B BYOB 설정]({{< relref path="#configure-byob" lang="ko" >}})으로 이동합니다.
{{% /tab %}}

{{% tab header="Azure" value="azure" %}}
자세한 내용은 Azure 공식 문서의 [Create a blob storage container](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal)를 참고하세요.
1. Azure Blob Storage 컨테이너 준비

    인스턴스 레벨 BYOB의 경우, [이 Terraform 모듈](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob)을 사용하지 않는다면, 아래 단계를 따라 직접 Azure Blob Storage 버킷을 Azure 구독 내에 준비하세요:

    1. 원하는 이름으로 버킷을 생성합니다. 필요시, 모든 W&B 파일을 저장할 하위 폴더를 만들 수 있습니다.
    1. 버킷에 CORS 정책을 적용합니다.

        UI에서 CORS를 설정하려면 Blob Storage로 이동하여 `Settings/Resource Sharing (CORS)`를 클릭하고 아래와 같이 입력하세요:

        | 파라미터   | 값 |
        | ---        | --- |
        | Allowed Origins | `*`  |
        | Allowed Methods | `GET`, `HEAD`, `PUT` |
        | Allowed Headers | `*` |
        | Exposed Headers | `*` |
        | Max Age         | `3000` |

        {{% alert %}}버킷 내 데이터가 [오브젝트 수명 주기 정책](https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-policy-configure?tabs=azure-portal)으로 만료될 경우, 일부 run의 기록을 읽지 못할 수 있습니다.{{% /alert %}}
1. 스토리지 계정 엑세스 키를 생성하고, 키 이름과 스토리지 계정명을 기록하세요. [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용할 경우, 안전한 방식으로 해당 정보(스토리지 계정명과 엑세스 키)를 W&B팀에 공유해야 합니다.

    팀 레벨 BYOB의 경우, Azure Blob Storage 버킷 및 필요한 엑세스 메커니즘, 권한까지 [Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)을 활용해 구성할 것을 권장합니다. Dedicated Cloud 사용 시, 인스턴스의 OIDC 발급자 URL을 제공하세요. 아래 정보를 메모해두십시오:

    * 스토리지 계정명
    * 스토리지 컨테이너명
    * 관리형 ID 클라이언트 ID
    * Azure 테넌트 ID

{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
S3-호환 버킷을 생성하고 아래 정보를 기록하세요:
- 엑세스 키
- 시크릿 엑세스 키
- URL 엔드포인트
- 버킷 이름
- (필요시) 폴더 경로
- 리전

{{% /tab %}}
{{< /tabpane >}}

다음으로 [스토리지 어드레스 결정하기]({{< relref path="#determine-the-storage-address" lang="ko" >}})로 이동하세요.

## 스토리지 어드레스 결정하기 {#determine-the-storage-address}
이 섹션에서는 W&B Team을 BYOB 스토리지 버킷에 연결할 때 사용하는 문법을 설명합니다. 예시의 `<>` 괄호 안 변수들은 실제 버킷 정보를 대체해 적용하세요.
상세 설명을 위해 탭을 선택해주세요.

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
이 섹션은 **Dedicated Cloud** 또는 **Self-Managed**에서 팀 레벨 BYOB에만 해당됩니다. 인스턴스 레벨 BYOB 또는 Multi-tenant Cloud의 경우, 바로 [W&B 설정]({{< relref path="#configure-byob" lang="ko" >}}) 단계로 진행하세요.

아래 포맷을 참고하여 전체 버킷 경로를 완성하십시오. 각 자리값은 해당 값으로 대체합니다.

**버킷 포맷**:
```none
cw://<accessKey>:<secretAccessKey>@cwobject.com/<bucketName>?tls=true
```

  `cwobject.com` HTTPS 엔드포인트만 지원됩니다. TLS 1.3 필요. 다른 CoreWeave 엔드포인트 지원에 관심 있다면 [support](mailto:support@wandb.com)로 문의 주세요.
{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
**버킷 포맷**:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>?region=<region>
```
주소에서 `region` 파라미터는, W&B 인스턴스와 스토리지 버킷 모두 AWS에 배포되어 있고 W&B 인스턴스의 `AWS_REGION`이 버킷의 S3 리전과 일치하는 경우를 제외하면 필수입니다.
{{% /tab %}}
{{% tab header="GCP" value="gcp" %}}
**버킷 포맷**:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```
{{% /tab %}}
{{% tab header="Azure" value="azure" %}}
**버킷 포맷**:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```
{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
**버킷 포맷**:
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>&tls=true
```
주소에서 `region` 파라미터는 필수입니다.

{{% alert %}}
이 섹션은 AWS S3가 아닌, 직접 호스팅되는 S3 호환 스토리지(Minio 등)에 해당합니다. AWS S3에 호스팅된 버킷의 경우 **AWS** 탭의 내용을 참고하세요.

S3 호환 모드를 지원하는 클라우드 네이티브 스토리지는, 가급적 해당 클라우드의 네이티브 프로토콜을 쓰는 것이 권장됩니다. 예: CoreWeave 버킷은 `s3://` 대신 `cw://` 사용.
{{% /alert %}}
{{% /tab %}}
{{< /tabpane >}}

스토리지 어드레스를 결정했다면, [팀 레벨 BYOB 설정]({{< relref path="#configure-team-level-byob" lang="ko" >}})로 이동하세요.

## W&B 설정 {#configure-byob}
[버킷 준비]({{< relref path="#provision-your-bucket" lang="ko" >}})와 [버킷 주소 결정](#determine-the-storage-address)을 각각 마쳤다면, 이제 [인스턴스 레벨]({{< relref path="#instance-level-byob" lang="ko" >}}) 또는 [팀 레벨]({{< relref path="#team-level-byob" lang="ko" >}}) BYOB를 설정할 수 있습니다.

{{% alert color="secondary" %}}
스토리지 버킷 배치를 신중히 계획하세요. W&B에 스토리지 버킷을 설정한 후에는, 데이터를 다른 버킷으로 마이그레이션하는 작업이 복잡하며 W&B의 지원이 필요합니다. 이 원칙은 Dedicated Cloud, Self-Managed, Multi-tenant Cloud의 팀 레벨 스토리지 모두에 적용됩니다. 문의사항은 [support](mailto:support@wandb.com)로 연락 바랍니다.
{{% /alert %}}

### 인스턴스 레벨 BYOB

{{% alert %}}
CoreWeave AI Object Storage의 인스턴스 레벨 BYOB는, 아래 지침 대신 [W&B support](mailto:support@wandb.com)로 별도 문의해 주세요. 셀프 서비스 설정이 아직 지원되지 않습니다.
{{% /alert %}}

**Dedicated Cloud**의 경우: 버킷 정보를 W&B 팀에 전달하면, 담당자가 Dedicated Cloud 인스턴스를 설정해줍니다.

**Self-Managed**의 경우, W&B App에서 인스턴스 레벨 BYOB를 직접 설정 가능합니다.
1. `admin` 역할의 사용자로 W&B에 로그인합니다.
1. 우측 상단 유저 아이콘을 클릭, **System Console**을 선택하세요.
1. **Settings** > **System Connections**로 이동합니다.
1. **Bucket Storage** 항목에서, **Identity** 필드의 계정이 신규 버킷에 엑세스 권한이 부여되어 있는지 확인합니다.
1. **Provider** 선택.
1. **Bucket Name**을 입력.
1. 필요시 **Path**(하위 경로) 입력.
1. **Save**를 클릭합니다.

{{% alert %}}
Self-Managed 환경에서는 W&B가 관리하는 Terraform 모듈을 사용하여 스토리지 버킷 및 필수 엑세스 구조(IAM 설정 등)까지 동시에 구성할 것을 권장합니다:

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [인스턴스 레벨 BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) 또는 [팀 레벨 BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

### 팀 레벨 BYOB

[버킷의 스토리지 위치](#determine-the-storage-address)를 확인한 후, 팀을 생성하면서 W&B App에서 팀 레벨 BYOB를 설정할 수 있습니다.

{{% alert %}}
- 팀이 생성된 후에는 스토리지를 변경할 수 없습니다.
- 인스턴스 레벨 BYOB는 [인스턴스 레벨 BYOB]({{< relref path="#instance-level-byob" lang="ko" >}})를 참고하세요.
- 팀의 스토리지로 CoreWeave를 사용할 계획이라면, 팀 생성 후 스토리지 설정 변경이 불가능하므로, 미리 CoreWeave에서 버킷 설정 및 팀 구성 검증에 대해 [support](mailto:support@wandb.com)로 문의 바랍니다.
{{% /alert %}}

배포 타입을 선택하여 안내를 따라주세요.

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Hosted" value="dedicated" %}}

1. **Dedicated Cloud**: 버킷 경로를 귀하의 담당 어카운트 팀에 전달하여, 해당 버킷이 인스턴스의 지원 파일 저장소에 등록된 후에 아래 단계를 진행해야 합니다.
1. **Self-Managed**: 버킷 경로를 `GORILLA_SUPPORTED_FILE_STORES` 환경변수에 등록한 후, W&B를 재시작해야 다음 단계를 진행할 수 있습니다.
1. `admin` 역할의 사용자로 W&B에 로그인하고, 좌상단 아이콘을 클릭하여 왼쪽 네비게이션을 연 다음 **Create a team to collaborate**를 클릭하세요.
1. 팀 이름을 입력합니다.
1. **Storage Type**을 **External storage**로 설정합니다.

    {{% alert %}}팀 스토리지로 인스턴스 레벨 스토리지를 사용(내부 또는 외부 상관없이)하고자 한다면, 인스턴스 레벨 버킷이 BYOB로 구성되어 있더라도 **Storage Type**을 **Internal**로 그대로 두세요. 별도의 외부 스토리지를 쓰고 싶다면 **External**로 설정 후 다음 단계에서 버킷 정보를 입력합니다.{{% /alert %}}

1. **Bucket location**을 클릭합니다.
1. 기존 버킷을 사용할 경우 목록에서 선택하고, 새 버킷을 추가하려면 **Add bucket**을 클릭하여 상세 정보를 입력합니다.

    **Cloud provider**를 클릭하고 **CoreWeave**, **AWS**, **GCP**, **Azure** 중 선택하세요.
    
    클라우드 제공업체가 보이지 않으면 1단계 절차 누락이 없는지, 인스턴스에 지원 파일 스토어로 등록되어 있는지 확인하세요. 그래도 목록에 없다면 [support](mailto:support@wandb.ai)로 문의하세요.
1. 버킷 정보를 입력합니다.
    - **CoreWeave**의 경우 버킷명만 입력.
    - Amazon S3, GCP, S3-compatible 스토리지는 [이전에 확인한 전체 버킷 경로](#determine-the-storage-address)를 입력.
    - Azure(W&B Dedicated, Self-Managed)는 **Account name**(계정명), **Container name**(컨테이너명) 입력.
    - 필요시:
      - 하위 경로(**Path**) 입력 가능.
      - **AWS**: **KMS key ARN**에 KMS 암호화 키 ARN 입력.
      - **Azure**: 필요시 **Tenant ID**, **Managed Identity Client ID** 입력.
1. **Create team** 클릭.

버킷 엑세스 문제나 설정 오류가 있으면 하단에 에러/경고가 나타납니다. 없으면 팀이 생성됩니다.

{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="multi-tenant" %}}

1. 새 팀 생성을 시작해둔 브라우저 창에서 W&B 조직 ID를 확인하거나, `admin` 권한 사용자로 W&B에 로그인하여 좌상단 아이콘 클릭, **Create a team to collaborate**로 진입합니다.
1. 팀 이름을 입력합니다.
1. **Storage Type**을 **External storage**로 설정합니다.
1. **Bucket location**을 클릭합니다.
1. 기존 버킷을 사용할 경우 목록에서 선택, 새 버킷을 추가하려면 **Add bucket** 클릭 후 상세 입력.

    **Cloud provider**에서 **CoreWeave**, **AWS**, **GCP**, **Azure** 중 선택합니다.
1. 버킷 정보를 입력합니다.
    - **CoreWeave**는 버킷명만 입력.
    - Amazon S3, GCP, S3-compatible은 [이전에 확인한 전체 버킷 경로](#determine-the-storage-address)를 입력.
    - Azure(W&B Dedicated, Self-Managed)는 **Account name**, **Container name** 입력.
    - 필요시:
      - **Path**에 하위 경로 입력 가능.
      - **AWS**: **KMS key ARN**에 KMS 암호화 키 ARN 입력.
      - **Azure**: 필요시 **Tenant ID**, **Managed Identity Client ID** 입력.
     - **Invite team members**에서 이메일 주소를 쉼표로 구분하여 입력하면 팀 멤버를 초대할 수 있습니다. 아니면 추후 팀 생성 후 추가해도 됩니다.
1. **Create team** 클릭.

버킷 엑세스 오류나 설정 오류가 있으면 하단에 에러/경고가 표시됩니다. 성공 시 바로 팀이 생성됩니다.

{{% /tab %}}
{{< /tabpane >}}

## 문제 해결
<details open>
<summary>CoreWeave AI Object Storage 연결</summary>

- **연결 오류**
  - W&B 인스턴스에서 CoreWeave 네트워크 엔드포인트로 연결할 수 있는지 확인하세요.
  - CoreWeave는 bucket 이름이 경로의 가장 앞에 위치하는 virtual-hosted style을 사용합니다.
    예: `cw://bucket-name.cwobject.com` (o), ~`cw://cwobject.com/bucket-name/`~ (x)
  - 버킷명에는 밑줄(`_`) 또는 DNS 규칙과 호환되지 않는 문자가 포함되면 안됩니다.
  - 버킷명은 CoreWeave 각 위치에서 전역적으로 유일해야 합니다.
  - 버킷명은 `cw-` 또는 `vip-`로 시작하면 안됩니다(예약어).
- **CORS 검증 실패**
  - CORS 정책은 필수입니다. CoreWeave는 S3 호환이므로, 자세한 CORS 정책 예시는 [AWS 공식 가이드](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html)를 참고하세요.
  - `AllowedMethods`에는 반드시 `GET`, `PUT`, `HEAD`가 모두 포함되어야 합니다.
  - `ExposeHeaders`에는 반드시 `ETag`가 포함되어야 합니다.
  - W&B 프론트엔드 도메인 전체가 CORS 정책의 `AllowedOrigins`에 포함되어야 합니다. 예시 정책은 `*`로 전체 도메인을 포함합니다.
- **LOTA endpoint 관련 이슈**
  - LOTA endpoint로의 직접 연결은 아직 지원되지 않습니다. 관심 있으시면 [support](mailto:support@wandb.com)로 문의해주세요.
- **엑세스키/권한 오류**
  - CoreWeave API 엑세스 키가 만료되었는지 확인하세요.
  - CoreWeave API 엑세스 키 및 시크릿 키에 `GetObject`, `PutObject`, `DeleteObject`, `ListBucket` 권한이 있는지 확인하세요(이 문서의 예제 정책들이 해당 조건을 만족함). 자세한 안내는 CoreWeave 공식문서 [Create and Manage Access Keys](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys)를 참고하세요.

</details>