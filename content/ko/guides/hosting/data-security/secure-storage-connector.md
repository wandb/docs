---
title: Bring your own bucket (BYOB)
menu:
  default:
    identifier: ko-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

Bring your own bucket (BYOB)을 사용하면 W&B 아티팩트 및 기타 관련 민감한 데이터를 자체 클라우드 또는 온프레미스 인프라에 저장할 수 있습니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 버킷에 저장하는 데이터는 W&B 관리형 인프라에 복사되지 않습니다.

{{% alert %}}
* W&B SDK / CLI / UI와 버킷 간의 통신은 [사전 서명된 URL]({{< relref path="./presigned-urls.md" lang="ko" >}})을 사용하여 이루어집니다.
* W&B는 가비지 수집 프로세스를 사용하여 W&B Artifacts를 삭제합니다. 자세한 내용은 [Artifacts 삭제]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ko" >}})를 참조하세요.
* 버킷을 구성할 때 하위 경로를 지정하여 W&B가 버킷의 루트에 있는 폴더에 파일을 저장하지 않도록 할 수 있습니다. 이는 조직의 버킷 거버넌스 정책을 준수하는 데 도움이 될 수 있습니다.
{{% /alert %}}

## 중앙 데이터베이스와 버킷에 저장되는 데이터

BYOB 기능을 사용할 때 특정 유형의 데이터는 W&B 중앙 데이터베이스에 저장되고 다른 유형의 데이터는 버킷에 저장됩니다.

### 데이터베이스

- 사용자, Teams, Artifacts, Experiments 및 Projects에 대한 메타데이터
- Reports
- Experiment 로그
- 시스템 메트릭

## 버킷

- Experiment 파일 및 메트릭
- Artifact 파일
- 미디어 파일
- Run 파일

## 설정 옵션
스토리지 버킷을 구성할 수 있는 범위는 두 가지입니다. *인스턴스 수준* 또는 *Team 수준*.

- 인스턴스 수준: 조직 내에서 관련 권한을 가진 모든 사용자는 인스턴스 수준 스토리지 버킷에 저장된 파일에 엑세스할 수 있습니다.
- Team 수준: W&B Team의 멤버는 Team 수준에서 구성된 버킷에 저장된 파일에 엑세스할 수 있습니다. Team 수준 스토리지 버킷은 매우 민감한 데이터 또는 엄격한 규정 준수 요구 사항이 있는 Teams에 대해 더 나은 데이터 엑세스 제어 및 데이터 격리를 제공합니다.

인스턴스 수준과 조직 내의 하나 이상의 Teams에 대해 별도로 버킷을 구성할 수 있습니다.

예를 들어 조직에 Kappa라는 Team이 있다고 가정합니다. 조직(및 Team Kappa)은 기본적으로 인스턴스 수준 스토리지 버킷을 사용합니다. 다음으로 Omega라는 Team을 만듭니다. Team Omega를 만들 때 해당 Team에 대한 Team 수준 스토리지 버킷을 구성합니다. Team Omega에서 생성된 파일은 Team Kappa에서 엑세스할 수 없습니다. 그러나 Team Kappa에서 생성된 파일은 Team Omega에서 엑세스할 수 있습니다. Team Kappa에 대한 데이터를 격리하려면 해당 Team에 대한 Team 수준 스토리지 버킷도 구성해야 합니다.

{{% alert %}}
Team 수준 스토리지 버킷은 특히 인프라 및 관리 리소스를 효율적으로 활용하기 위해 서로 다른 사업부 및 부서가 인스턴스를 공유할 때 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에 대해 동일한 이점을 제공합니다. 이는 별도의 고객 참여에 대한 AI 워크플로우를 관리하는 별도의 프로젝트 Teams가 있는 회사에도 적용됩니다.
{{% /alert %}}

## 가용성 매트릭스
다음 표는 다양한 W&B 서버 배포 유형에서 BYOB의 가용성을 보여줍니다. `X`는 특정 배포 유형에서 해당 기능을 사용할 수 있음을 의미합니다.

| W&B 서버 배포 유형 | 인스턴스 수준 | Team 수준 | 추가 정보 |
|----------------------------|--------------------|----------------|------------------------|
| 전용 클라우드 | X | X | 인스턴스 및 Team 수준 BYOB는 Amazon Web Services, Google Cloud Platform 및 Microsoft Azure에서 모두 사용할 수 있습니다. Team 수준 BYOB의 경우 동일하거나 다른 클라우드의 클라우드 네이티브 스토리지 버킷 또는 클라우드 또는 온프레미스 인프라에서 호스팅되는 [MinIO](https://github.com/minio/minio)와 같은 S3 호환 보안 스토리지에 연결할 수 있습니다. |
| SaaS Cloud | 해당 없음 | X | Team 수준 BYOB는 Amazon Web Services 및 Google Cloud Platform에서만 사용할 수 있습니다. W&B는 Microsoft Azure에 대한 기본 및 유일한 스토리지 버킷을 완전히 관리합니다. |
| 자체 관리 | X | X | 인스턴스가 사용자에 의해 완전히 관리되기 때문에 인스턴스 수준 BYOB가 기본값입니다. 자체 관리 인스턴스가 클라우드에 있는 경우 Team 수준 BYOB에 대해 동일하거나 다른 클라우드의 클라우드 네이티브 스토리지 버킷에 연결할 수 있습니다. 인스턴스 또는 Team 수준 BYOB에 대해 [MinIO](https://github.com/minio/minio)와 같은 S3 호환 보안 스토리지를 사용할 수도 있습니다. |

{{% alert color="secondary" %}}
전용 클라우드 또는 자체 관리 인스턴스에 대한 인스턴스 또는 Team 수준 스토리지 버킷 또는 SaaS Cloud 계정에 대한 Team 수준 스토리지 버킷을 구성한 후에는 해당 범위에 대한 스토리지 버킷을 변경하거나 재구성할 수 없습니다. 여기에는 데이터를 다른 버킷으로 마이그레이션하고 기본 제품 스토리지에서 관련 참조를 다시 매핑하는 기능이 포함됩니다. W&B는 인스턴스 또는 Team 수준 범위에 대해 구성하기 전에 스토리지 버킷 레이아웃을 신중하게 계획하는 것이 좋습니다. 질문이 있으시면 W&B 팀에 문의하십시오.
{{% /alert %}}

## Team 수준 BYOB에 대한 교차 클라우드 또는 S3 호환 스토리지

[전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 Team 수준 BYOB에 대해 다른 클라우드의 클라우드 네이티브 스토리지 버킷 또는 [MinIO](https://github.com/minio/minio)와 같은 S3 호환 스토리지 버킷에 연결할 수 있습니다.

교차 클라우드 또는 S3 호환 스토리지 사용을 활성화하려면 W&B 인스턴스에 대한 `GORILLA_SUPPORTED_FILE_STORES` 환경 변수를 사용하여 다음 형식 중 하나로 관련 엑세스 키를 포함하는 스토리지 버킷을 지정합니다.

<details>
<summary>전용 클라우드 또는 자체 관리 인스턴스에서 Team 수준 BYOB에 대한 S3 호환 스토리지 구성</summary>

다음 형식을 사용하여 경로를 지정합니다.
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>?tls=true
```
W&B 인스턴스가 AWS에 있고 W&B 인스턴스 노드에 구성된 `AWS_REGION`이 S3 호환 스토리지에 구성된 지역과 일치하는 경우를 제외하고 `region` 파라미터는 필수입니다.

</details>
<details>
<summary>전용 클라우드 또는 자체 관리 인스턴스에서 Team 수준 BYOB에 대한 교차 클라우드 네이티브 스토리지 구성</summary>

W&B 인스턴스 및 스토리지 버킷의 위치에 따라 특정 형식으로 경로를 지정합니다.

GCP 또는 Azure의 W&B 인스턴스에서 AWS의 버킷으로:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>
```

GCP 또는 AWS의 W&B 인스턴스에서 Azure의 버킷으로:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```

AWS 또는 Azure의 W&B 인스턴스에서 GCP의 버킷으로:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```

</details>

{{% alert %}}
Team 수준 BYOB에 대한 S3 호환 스토리지에 대한 연결은 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 사용할 수 없습니다. 또한 Team 수준 BYOB에 대한 AWS 버킷에 대한 연결은 해당 인스턴스가 GCP에 있기 때문에 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 교차 클라우드입니다. 해당 교차 클라우드 연결은 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 및 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에 대해 이전에 설명한 엑세스 키 및 환경 변수 기반 메커니즘을 사용하지 않습니다.
{{% /alert %}}

자세한 내용은 support@wandb.com으로 W&B 지원팀에 문의하십시오.

## W&B 플랫폼과 동일한 클라우드의 클라우드 스토리지

유스 케이스에 따라 Team 또는 인스턴스 수준에서 스토리지 버킷을 구성합니다. 스토리지 버킷을 프로비저닝하거나 구성하는 방법은 Azure의 엑세스 메커니즘을 제외하고 구성된 수준에 관계없이 동일합니다.

{{% alert %}}
W&B는 필요한 엑세스 메커니즘 및 관련 IAM 권한과 함께 스토리지 버킷을 프로비저닝하기 위해 W&B에서 관리하는 Terraform 모듈을 사용하는 것이 좋습니다.

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [인스턴스 수준 BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) 또는 [Team 수준 BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="AWS" value="aws" %}}
1. KMS 키 프로비저닝

    W&B는 S3 버킷에서 데이터를 암호화하고 해독하기 위해 KMS 키를 프로비저닝해야 합니다. 키 사용 유형은 `ENCRYPT_DECRYPT`여야 합니다. 다음 정책을 키에 할당합니다.

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

    `<Your_Account_Id>` 및 `<aws_kms_key.key.arn>`을 적절하게 바꿉니다.

    [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 `<aws_principal_and_role_arn>`을 해당 값으로 바꿉니다.

    * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
    * [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `arn:aws:iam::830241207209:root`

    이 정책은 AWS 계정에 키에 대한 모든 엑세스 권한을 부여하고 W&B 플랫폼을 호스팅하는 AWS 계정에 필요한 권한을 할당합니다. KMS 키 ARN을 기록해 둡니다.

2. S3 버킷 프로비저닝

    다음 단계에 따라 AWS 계정에서 S3 버킷을 프로비저닝합니다.

    1. 원하는 이름으로 S3 버킷을 만듭니다. 선택적으로 모든 W&B 파일을 저장하기 위해 하위 경로로 구성할 수 있는 폴더를 만듭니다.
    2. 버킷 버전 관리를 활성화합니다.
    3. 이전 단계에서 KMS 키를 사용하여 서버 측 암호화를 활성화합니다.
    4. 다음 정책으로 CORS를 구성합니다.

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
                "MaxAgeSeconds": 3600
            }
        ]
        ```

    5. 클라우드 인프라 또는 사용자 브라우저의 AI 워크로드가 버킷에 엑세스하는 데 사용하는 [사전 서명된 URL]({{< relref path="./presigned-urls.md" lang="ko" >}})을 생성하는 데 필요한 W&B 플랫폼을 호스팅하는 AWS 계정에 필요한 S3 권한을 부여합니다.

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

        `<wandb_bucket>`을 적절하게 바꾸고 버킷 이름을 기록해 둡니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 인스턴스 수준 BYOB의 경우 W&B 팀과 버킷 이름을 공유합니다. 모든 배포 유형에서 Team 수준 BYOB의 경우 [Team을 만드는 동안 버킷을 구성합니다]({{< relref path="#configure-byob-in-wb" lang="ko" >}}).

        [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 `<aws_principal_and_role_arn>`을 해당 값으로 바꿉니다.

        * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `arn:aws:iam::830241207209:root`
  
  자세한 내용은 [AWS 자체 관리 호스팅 가이드]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ko" >}})를 참조하세요.
{{% /tab %}}

{{% tab header="GCP" value="gcp"%}}
1. GCS 버킷 프로비저닝

    다음 단계에 따라 GCP 프로젝트에서 GCS 버킷을 프로비저닝합니다.

    1. 원하는 이름으로 GCS 버킷을 만듭니다. 선택적으로 모든 W&B 파일을 저장하기 위해 하위 경로로 구성할 수 있는 폴더를 만듭니다.
    2. 소프트 삭제를 활성화합니다.
    3. 오브젝트 버전 관리를 활성화합니다.
    4. 암호화 유형을 `Google-managed`로 설정합니다.
    5. `gsutil`로 CORS 정책을 설정합니다. 이는 UI에서 불가능합니다.

      1. `cors-policy.json` 파일을 로컬로 만듭니다.
      2. 다음 CORS 정책을 파일에 복사하여 저장합니다.

          ```json
          [
          {
            "origin": ["*"],
            "responseHeader": ["Content-Type"],
            "exposeHeaders": ["ETag"],
            "method": ["GET", "HEAD", "PUT"],
            "maxAgeSeconds": 3600
          }
          ]
          ```

      3. `<bucket_name>`을 올바른 버킷 이름으로 바꾸고 `gsutil`을 실행합니다.

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      4. 버킷의 정책을 확인합니다. `<bucket_name>`을 올바른 버킷 이름으로 바꿉니다.
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

2. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 W&B 플랫폼에 연결된 GCP 서비스 계정에 `Storage Admin` 역할을 부여합니다.

    * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 계정은 `wandb-integration@wandb-production.iam.gserviceaccount.com`입니다.
    * [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})의 경우 계정은 `deploy@wandb-production.iam.gserviceaccount.com`입니다.

    버킷 이름을 기록해 둡니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 인스턴스 수준 BYOB의 경우 W&B 팀과 버킷 이름을 공유합니다. 모든 배포 유형에서 Team 수준 BYOB의 경우 [Team을 만드는 동안 버킷을 구성합니다]({{< relref path="#configure-byob-in-wb" lang="ko" >}}).
{{% /tab %}}

{{% tab header="Azure" value="azure"%}}
1. Azure Blob Storage 프로비저닝

    인스턴스 수준 BYOB의 경우 [이 Terraform 모듈](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob)을 사용하지 않는 경우 다음 단계에 따라 Azure 구독에서 Azure Blob Storage 버킷을 프로비저닝합니다.

    * 원하는 이름으로 버킷을 만듭니다. 선택적으로 모든 W&B 파일을 저장하기 위해 하위 경로로 구성할 수 있는 폴더를 만듭니다.
    * Blob 및 컨테이너 소프트 삭제를 활성화합니다.
    * 버전 관리를 활성화합니다.
    * 버킷에서 CORS 정책을 구성합니다.

      UI를 통해 CORS 정책을 설정하려면 Blob Storage로 이동하여 `설정/리소스 공유(CORS)`로 스크롤한 다음 다음을 설정합니다.

      | 파라미터 | 값 |
      | --- | --- |
      | 허용된 출처 | `*` |
      | 허용된 메소드 | `GET`, `HEAD`, `PUT` |
      | 허용된 헤더 | `*` |
      | 노출된 헤더 | `*` |
      | 최대 수명 | `3600` |

2. 스토리지 계정 엑세스 키를 생성하고 스토리지 계정 이름과 함께 기록해 둡니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 보안 공유 메커니즘을 사용하여 스토리지 계정 이름과 엑세스 키를 W&B 팀과 공유합니다.

    Team 수준 BYOB의 경우 W&B는 필요한 엑세스 메커니즘 및 권한과 함께 Azure Blob Storage 버킷을 프로비저닝하기 위해 [Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)을 사용하는 것이 좋습니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})를 사용하는 경우 인스턴스에 대한 OIDC 발급자 URL을 제공합니다. [Team을 만드는 동안 버킷을 구성]({{< relref path="#configure-byob-in-wb" lang="ko" >}})하는 데 필요한 세부 정보를 기록해 둡니다.

    * 스토리지 계정 이름
    * 스토리지 컨테이너 이름
    * 관리 ID 클라이언트 ID
    * Azure 테넌트 ID
{{% /tab %}}
{{< /tabpane >}}

## W&B에서 BYOB 구성

{{< tabpane text=true >}}

{{% tab header="Team 수준" value="team" %}}
{{% alert %}}
[전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 Team 수준 BYOB에 대해 다른 클라우드의 클라우드 네이티브 스토리지 버킷 또는 [MinIO](https://github.com/minio/minio)와 같은 S3 호환 스토리지 버킷에 연결하는 경우 [Team 수준 BYOB에 대한 교차 클라우드 또는 S3 호환 스토리지]({{< relref path="#cross-cloud-or-s3-compatible-storage-for-team-level-byob" lang="ko" >}})를 참조하십시오. 이러한 경우 아래 지침에 따라 Team에 대해 구성하기 전에 W&B 인스턴스에 대한 `GORILLA_SUPPORTED_FILE_STORES` 환경 변수를 사용하여 스토리지 버킷을 지정해야 합니다.
{{% /alert %}}

W&B Team을 만들 때 Team 수준에서 스토리지 버킷을 구성하려면 다음을 수행합니다.

1. **Team 이름** 필드에 Team 이름을 제공합니다.
2. **스토리지 유형** 옵션에 대해 **외부 스토리지**를 선택합니다.
3. 드롭다운에서 **새 버킷**을 선택하거나 기존 버킷을 선택합니다.

    여러 W&B Teams가 동일한 클라우드 스토리지 버킷을 사용할 수 있습니다. 이를 활성화하려면 드롭다운에서 기존 클라우드 스토리지 버킷을 선택합니다.

4. **클라우드 공급자** 드롭다운에서 클라우드 공급자를 선택합니다.
5. **이름** 필드에 스토리지 버킷 이름을 제공합니다. Azure에서 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스가 있는 경우 **계정 이름** 및 **컨테이너 이름** 필드에 값을 제공합니다.
6. (선택 사항) 선택적 **경로** 필드에 버킷 하위 경로를 제공합니다. W&B가 버킷의 루트에 있는 폴더에 파일을 저장하지 않으려면 이 작업을 수행합니다.
7. (AWS 버킷을 사용하는 경우 선택 사항) **KMS 키 ARN** 필드에 KMS 암호화 키의 ARN을 제공합니다.
8. (Azure 버킷을 사용하는 경우 선택 사항) **테넌트 ID** 및 **관리 ID 클라이언트 ID** 필드에 값을 제공합니다.
9. ([SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 선택 사항) Team을 만들 때 Team 멤버를 선택적으로 초대합니다.
10. **Team 만들기** 버튼을 누릅니다.

{{< img src="/images/hosting/prod_setup_secure_storage.png" alt="" >}}

버킷에 엑세스하는 데 문제가 있거나 버킷에 잘못된 설정이 있는 경우 페이지 하단에 오류 또는 경고가 나타납니다.
{{% /tab %}}

{{% tab header="인스턴스 수준" value="instance"%}}
전용 클라우드 또는 자체 관리 인스턴스에 대한 인스턴스 수준 BYOB를 구성하려면 support@wandb.com으로 W&B 지원팀에 문의하십시오.
{{% /tab %}}
{{< /tabpane >}}
