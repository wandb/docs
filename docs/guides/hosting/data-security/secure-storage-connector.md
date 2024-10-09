---
title: Bring your own bucket (BYOB)
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

BYOB(Bucket 소유자 제공)는 W&B Artifacts 및 기타 관련 민감 데이터를 사용자의 클라우드나 온프레미스 인프라에 저장할 수 있도록 합니다. [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [SaaS 클라우드](../hosting-options/saas_cloud.md) 경우, 버킷에 저장된 데이터는 W&B가 관리하는 인프라로 복사되지 않습니다.

:::info
* W&B SDK / CLI / UI와 버킷 간의 통신은 [사전 서명된 URL](./presigned-urls.md)을 사용하여 이루어집니다.
* W&B는 가비지 컬렉션 프로세스를 사용하여 W&B Artifacts를 삭제합니다. 자세한 내용은 [Deleting Artifacts](../../artifacts/delete-artifacts.md)을 참조하세요.
* 버킷을 설정할 때 하위 경로를 지정하여 W&B가 버킷의 루트 폴더에 파일을 저장하지 않도록 할 수 있습니다. 이를 통해 조직의 버킷 거버넌스 정책에 더욱 잘 맞출 수 있습니다.
:::

## 설정 옵션
저장소 버킷을 설정할 두 가지 범위가 있습니다: *인스턴스 수준* 또는 *팀 수준*.

- 인스턴스 수준: 조직 내 적절한 권한을 가진 사용자는 인스턴스 수준 저장소 버킷에 저장된 파일에 엑세스할 수 있습니다.
- 팀 수준: W&B 팀의 멤버는 팀 수준에서 설정된 버킷에 저장된 파일에 엑세스할 수 있습니다. 팀 수준 저장소 버킷은 고도로 민감한 데이터나 엄격한 규정 준수를 요구하는 팀에 대해 더욱 높은 데이터 엑세스 제어와 데이터 격리를 제공합니다.

조직 내에서 인스턴스 수준으로 설정하거나, 하나 이상의 팀에 대해 별도로 설정할 수 있습니다.

예를 들어, 조직에 Kappa라는 팀이 있다고 가정합니다. 조직(및 Kappa 팀)은 기본적으로 인스턴스 수준 저장소 버킷을 사용합니다. 이후 Omega라는 팀을 만듭니다. Omega 팀을 만들 때, 팀 수준 저장소 버킷을 해당 팀에 대해 설정합니다. Omega 팀이 생성한 파일은 Kappa 팀에서 엑세스할 수 없습니다. 그러나 Kappa 팀이 생성한 파일은 Omega 팀에서 엑세스할 수 있습니다. Kappa 팀의 데이터를 격리하고자 한다면, 그들에게도 팀 수준 저장소 버킷을 설정해야 합니다.

:::tip
팀 수준 저장소 버킷은 [자가 관리](../hosting-options/self-managed.md) 인스턴스에서도 동일한 이점을 제공합니다. 특히, 서로 다른 사업 단위와 부서가 인스턴스를 공유하여 인프라와 관리 자원을 효율적으로 활용하는 경우에 유용합니다. 또한, 별개의 고객 참여를 위한 AI 워크플로우를 관리하는 별개 프로젝트 팀이 있는 기업에도 적용됩니다.
:::

## 가용성 매트릭스
다음 표는 다양한 W&B 서버 배포 유형에서 BYOB의 가용성을 보여줍니다. `X`는 해당 배포 유형에서 기능이 가용함을 의미합니다.

| W&B 서버 배포 유형 | 인스턴스 수준 | 팀 수준 | 추가 정보 |
|----------------------------|--------------------|----------------|------------------------|
| 전용 클라우드 | X | X | 인스턴스와 팀 수준 모두 AWS, 구글 클라우드 플랫폼, 마이크로소프트 애저에서 가용합니다. 팀 수준 BYOB의 경우, 동일하거나 다른 클라우드, 또는 온프레미스 인프라에 호스트된 S3 호환 보안 저장소인 [MinIO](https://github.com/minio/minio)와 같은 클라우드 네이티브 저장소 버킷에 연결할 수 있습니다. |
| SaaS 클라우드 | 적용 불가 | X | 팀 수준 BYOB는 아마존 웹 서비스와 구글 클라우드 플랫폼에서만 사용할 수 있습니다. W&B는 마이크로소프트 애저에 대한 기본 및 유일한 저장소 버킷을 완전히 관리합니다. |
| 자가 관리 | X | X | 인스턴스는 귀하가 완전 관리하므로 인스턴스 수준 BYOB가 기본 값입니다. 자가 관리 인스턴스가 클라우드에 있다면, 동일하거나 다른 클라우드의 클라우드 네이티브 저장소 버킷에 연결하여 팀 수준 BYOB를 사용할 수 있습니다. 또는 인스턴스나 팀 수준 BYOB에 대해 S3 호환 보안 저장소인 [MinIO](https://github.com/minio/minio)를 사용할 수 있습니다. |

## 팀 수준 BYOB를 위한 크로스 클라우드 또는 S3 호환 저장소

[전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [자가 관리된](../hosting-options/self-managed.md) 인스턴스의 팀 수준 BYOB에 대해 다른 클라우드의 클라우드 네이티브 저장소 버킷에 연결하거나 [MinIO](https://github.com/minio/minio)와 같은 S3 호환 저장소 버킷에 연결할 수 있습니다.

크로스 클라우드 또는 S3 호환 저장소의 사용을 활성화하려면, W&B 인스턴스의 `GORILLA_SUPPORTED_FILE_STORES` 환경 변수를 사용하여 다음 형식 중 하나로 엑세스 키를 포함하여 저장소 버킷을 지정합니다.

<details>
<summary>전용 클라우드 또는 자가 관리 인스턴스에서 팀 수준 BYOB를 위한 S3 호환 저장소 설정</summary>

다음 형식으로 경로를 지정합니다:
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>?tls=true
```
`region` 파라미터는 필수입니다. W&B 인스턴스가 AWS에 있으며, W&B 인스턴스 노드에 설정된 `AWS_REGION`이 S3 호환 저장소에 설정된 region과 일치하는 경우는 예외입니다.

</details>
<details>
<summary>전용 클라우드 또는 자가 관리 인스턴스에서 팀 수준 BYOB를 위한 크로스 클라우드 네이티브 저장소 설정</summary>

W&B 인스턴스와 저장소 버킷의 위치에 따라 각기 다른 형식으로 경로를 지정합니다:

GCP 또는 Azure에 있는 W&B 인스턴스에서 AWS에 있는 버킷으로:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>
```

GCP 또는 AWS에 있는 W&B 인스턴스에서 Azure에 있는 버킷으로:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```

AWS 또는 Azure에 있는 W&B 인스턴스에서 GCP에 있는 버킷으로:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```

</details>

:::info
팀 수준 BYOB를 위한 S3 호환 저장소 연결은 [SaaS 클라우드](../hosting-options/saas_cloud.md)에서 가용하지 않습니다. 또한, 팀 수준 BYOB를 위한 AWS 버킷 연결은 [SaaS 클라우드](../hosting-options/saas_cloud.md)에서 크로스 클라우드로 간주됩니다. 해당 인스턴스는 GCP에 호스트되어 있기 때문입니다. 이 크로스 클라우드 연결은 [전용 클라우드](../hosting-options/dedicated_cloud.md) 및 [자가 관리된](../hosting-options/self-managed.md) 인스턴스를 위한 엑세스 키와 환경 변수 기반 메커니즘을 위에서 언급한 것처럼 사용하지 않습니다.
:::

추가 정보가 필요하시면 W&B 지원팀(support@wandb.com)에 문의하세요.

## W&B 플랫폼과 동일한 클라우드에 있는 클라우드 저장소

사용 사례에 따라 팀 또는 인스턴스 수준에서 저장소 버킷을 설정합니다. 저장소 버킷이 제공되거나 구성되는 방식은 Azure의 엑세스 메커니즘을 제외하고는 설정된 수준과 상관없이 동일합니다.

:::tip
W&B는 Terraform 모듈을 사용하여 저장소 버킷을 필요 엑세스 메커니즘 및 관련 IAM 권한과 함께 프로비저닝 할 것을 권장합니다:

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [인스턴스 수준 BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) 또는 [팀 수준 BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
:::

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'GCP', value: 'gcp'},
    {label: 'Azure', value: 'azure'},
  ]}>
  <TabItem value="aws">

#### KMS 키 프로비저닝

W&B는 S3 버킷의 데이터를 암호화하고 복호화하는 데 필요한 KMS 키를 프로비저닝할 것을 요구합니다. 키 사용 유형은 `ENCRYPT_DECRYPT` 여야 합니다. 다음 정책을 키에 할당하세요:

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
`<Your_Account_Id>`와 `<aws_kms_key.key.arn>`을 적절히 대체합니다.

[SaaS 클라우드](../hosting-options/saas_cloud.md) 또는 [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우, `<aws_principal_and_role_arn>`을 해당 값으로 대체합니다:

* [SaaS 클라우드](../hosting-options/saas_cloud.md)의 경우: `arn:aws:iam::725579432336:role/WandbIntegration`
* [전용 클라우드](../hosting-options/dedicated_cloud.md)의 경우: `arn:aws:iam::830241207209:root`

이 정책은 AWS 계정에 키에 대한 전체 엑세스를 부여하며 또한 W&B 플랫폼을 호스팅하는 AWS 계정에 필요한 권한을 할당합니다. KMS 키 ARN을 기록해 두세요.

#### S3 버킷 프로비저닝

AWS 계정에서 S3 버킷을 프로비저닝하려면 다음 단계를 따르세요:

* 원하는 이름으로 S3 버킷을 생성하세요. 선택적으로 모든 W&B 파일을 저장할 하위 경로로 구성할 수 있는 폴더를 생성하세요.
* 버킷 버전 관리를 활성화하세요.
* 이전 단계의 KMS 키를 사용하여 서버 측 암호화를 활성화하세요.
* 다음 정책으로 CORS를 구성하세요:

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

* W&B 플랫폼을 호스팅하는 AWS 계정에 필수 S3 권한을 부여합니다. 이러한 권한은 사전 서명된 [pre-signed URLs](./presigned-urls.md)을 생성하는 데 사용되며, 사용자의 클라우드 인프라나 사용자 브라우저에서 버킷에 엑세스하는 데 활용됩니다.

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
`<wandb_bucket>`을 적절히 대체하고 버킷 이름을 기록해 두세요. [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우 인스턴스 수준 BYOB의 경우 W&B 팀과 버킷 이름을 공유하세요. 아무 배포 유형의 팀 수준 BYOB의 경우 [팀 생성 시 버킷을 구성](#configure-byob-in-wb)하세요.

[SaaS 클라우드](../hosting-options/saas_cloud.md) 또는 [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우, `<aws_principal_and_role_arn>`을 해당 값으로 대체하세요.

* [SaaS 클라우드](../hosting-options/saas_cloud.md)의 경우: `arn:aws:iam::725579432336:role/WandbIntegration`
* [전용 클라우드](../hosting-options/dedicated_cloud.md)의 경우: `arn:aws:iam::830241207209:root`
 
 자세한 내용은 [AWS 자체 관리 호스팅 가이드](../self-managed/aws-tf.md)를 참조하세요.
  </TabItem>
  <TabItem value="gcp">

#### GCS 버킷 프로비저닝

GCP 프로젝트에서 GCS 버킷을 프로비저닝하려면 다음 단계를 따르세요:

* 원하는 이름으로 GCS 버킷을 생성하세요. 선택적으로 모든 W&B 파일을 저장할 하위 경로로 구성할 수 있는 폴더를 생성하세요.
* 소프트 삭제를 활성화하세요.
* 오브젝트 버전 관리를 활성화하세요.
* 암호화 유형을 `Google-managed`로 설정하세요.
* `gsutil`을 사용하여 CORS 정책을 설정하세요. UI에서는 불가능합니다.

   1. `cors-policy.json`이라는 파일을 로컬에 만듭니다.
   2. 다음 CORS 정책을 파일에 복사하고 저장하세요.
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

   3. `<bucket_name>`을 올바른 버킷 이름으로 대체하고 `gsutil`을 실행하세요.
    ```bash
    gsutil cors set cors-policy.json gs://<bucket_name>
    ```

   4. 버킷에 정책이 첨부되었는지 확인하세요. `<bucket_name>`을 올바른 버킷 이름으로 대체하세요.
    ```bash
    gsutil cors get gs://<bucket_name>
    ```

[SaaS 클라우드](../hosting-options/saas_cloud.md) 또는 [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우, W&B 플랫폼에 연결된 GCP 서비스 계정에 `Storage Admin` 역할을 부여하세요:

  * [SaaS 클라우드](../hosting-options/saas_cloud.md)의 경우, 계정은: `wandb-integration@wandb-production.iam.gserviceaccount.com`
  * [전용 클라우드](../hosting-options/dedicated_cloud.md)의 경우, 계정은: `deploy@wandb-production.iam.gserviceaccount.com`

버킷 이름을 기록해 두세요. [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우 인스턴스 수준 BYOB의 경우 W&B 팀과 버킷 이름을 공유하세요. 아무 배포 유형의 팀 수준 BYOB의 경우 [팀 생성 시 버킷을 구성](#configure-byob-in-wb)하세요.

  </TabItem>
  <TabItem value="azure">

#### Azure Blob Storage 프로비저닝

인스턴스 수준 BYOB의 경우, [이 Terraform 모듈](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob)을 사용하지 않는 경우, Azure 구독에서 Azure Blob Storage 버킷을 프로비저닝하려면 다음 절차를 따르세요:

* 원하는 이름으로 버킷을 생성하세요. 선택적으로 모든 W&B 파일을 저장할 하위 경로로 구성할 수 있는 폴더를 생성하세요.
* 블롭 및 컨테이너 소프트 삭제를 활성화하세요.
* 버전 관리를 활성화하세요.
* 버킷에 CORS 정책을 구성하세요

   UI를 통해 CORS 정책을 설정하려면 Blob Storage로 이동하여 `Settings/Resource Sharing (CORS)`로 스크롤한 후 다음을 설정하세요:

   | 파라미터 | 값 |
   | --- | --- |
   | 허용된 출처 | `*`  |
   | 허용된 메소드 | `GET`, `HEAD`, `PUT` |
   | 허용된 헤더 | `*` |
   | 노출된 헤더 | `*` |
   | 최대 연령 | `3600` |

저장소 계정 엑세스 키를 생성하고, 저장소 계정 이름과 함께 기록해 두십시오. [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우 인스턴스 수준 BYOB 시 다른 엑세스 키와 저장소 계정 이름을 W&B 팀에게 안전하게 공유하세요.

팀 수준 BYOB의 경우, W&B는 [이 Terraform 모듈](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)을 사용하여 Azure Blob Storage 버킷과 필요한 엑세스 메커니즘 및 권한을 프로비저닝할 것을 적극 권장합니다. [전용 클라우드](../hosting-options/dedicated_cloud.md)를 사용하는 경우, 해당 Terraform 모듈을 사용할 때 인스턴스의 `OIDC 발급자 URL`이 필요합니다. 완료되면 [팀 생성 시 버킷을 구성](#configure-byob-in-wb)하기 위해 다음을 기록하십시오:

* 저장소 계정 이름
* 저장소 컨테이너 이름
* 관리되는 ID 클라이언트 ID
* Azure 테넌트 ID

</TabItem>
</Tabs>

## W&B에서 BYOB 구성하기

<Tabs
  defaultValue="team"
  values={[
    {label: '팀 수준', value: 'team' },
    {label: '인스턴스 수준', value: 'instance' },
  ]}>
  <TabItem value="team">

:::info
[전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [자가 관리된](../hosting-options/self-managed.md) 인스턴스의 팀 수준 BYOB에 대해 다른 클라우드의 클라우드 네이티브 저장소 버킷에 연결하거나 [MinIO](https://github.com/minio/minio) 같은 S3 호환 저장소 버킷에 연결하는 경우, [크로스 클라우드 또는 팀 수준 BYOB를 위한 S3 호환 저장소](#cross-cloud-or-s3-compatible-storage-for-team-level-byob)를 참조하십시오. 이 경우 W&B 인스턴스의 `GORILLA_SUPPORTED_FILE_STORES` 환경 변수를 사용하여 저장소 버킷을 지정한 후 아래 지침에 따라 팀을 구성해야 합니다.
:::

W&B 팀을 만들 때 팀 수준에서 저장소 버킷을 구성하세요:

1. **팀 이름** 필드에 팀의 이름을 입력하세요. 
2. **저장소 유형** 옵션에 **외부 저장소**를 선택하세요. 
3. 드롭다운에서 **새 버킷**을 선택하거나 기존 버킷을 선택하세요.

:::tip
여러 W&B 팀이 동일한 클라우드 저장소 버킷을 사용할 수 있습니다. 이를 활성화하려면 드롭다운에서 기존 클라우드 저장소 버킷을 선택하십시오.
:::

4. **클라우드 제공자** 드롭다운에서 클라우드 제공자를 선택하세요.
5. 저장소 버킷 이름을 **이름** 필드에 입력하세요. [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [자가 관리된](../hosting-options/self-managed.md) 인스턴스가 Azure에 있는 경우, **계정 이름** 및 **컨테이너 이름** 필드의 값을 입력하십시오.
6. (선택 사항) 선택적 **경로** 필드에 버킷 하위 경로를 입력하세요. W&B가 버킷의 루트 폴더에 파일을 저장하지 않도록 설정하려면 이렇게 하십시오.
7. (AWS 버킷 사용 시 선택 사항) **KMS 키 ARN** 필드에 KMS 암호화 키의 ARN을 입력하세요.
8. (Azure 버킷 사용 시 선택 사항) **테넌트 ID** 및 **관리되는 ID 클라이언트 ID** 필드의 값을 입력하세요.
9. ([SaaS 클라우드](../hosting-options/saas_cloud.md) 시 선택 사항) 팀 멤버를 팀 생성 시 초대할 수 있는 옵션입니다.
10. **팀 생성** 버튼을 누르세요.

![](/images/hosting/prod_setup_secure_storage.png)

버킷에 엑세스하는 데 문제가 있거나 버킷에 잘못된 설정이 있는 경우 페이지 하단에 오류나 경고가 나타납니다.

</TabItem>
<TabItem value="instance">

전용 클라우드 또는 자가 관리된 인스턴스를 위해 인스턴스 수준 BYOB를 구성하려면 W&B 지원팀(support@wandb.com)에 연락하십시오.

</TabItem>
</Tabs>