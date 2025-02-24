---
title: Access BYOB using pre-signed URLs
menu:
  default:
    identifier: ko-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B는 AI 워크로드 또는 사용자 브라우저에서 blob 저장소에 대한 엑세스를 간소화하기 위해 사전 서명된 URL을 사용합니다. 사전 서명된 URL에 대한 기본 정보는 [AWS S3용 사전 서명된 URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html), [Google Cloud Storage용 서명된 URL](https://cloud.google.com/storage/docs/access-control/signed-urls) 및 [Azure Blob Storage용 공유 엑세스 서명](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview)을 참조하세요.

필요한 경우 네트워크 내의 AI 워크로드 또는 사용자 브라우저 클라이언트는 W&B Platform에서 사전 서명된 URL을 요청합니다. 그러면 W&B Platform은 관련 blob 저장소에 엑세스하여 필요한 권한으로 사전 서명된 URL을 생성하고 클라이언트에 다시 반환합니다. 그런 다음 클라이언트는 사전 서명된 URL을 사용하여 오브젝트 업로드 또는 검색 작업을 위해 blob 저장소에 엑세스합니다. 오브젝트 다운로드를 위한 URL 만료 시간은 1시간이고, 일부 대형 오브젝트는 청크 단위로 업로드하는 데 시간이 더 오래 걸릴 수 있으므로 오브젝트 업로드는 24시간입니다.

## 팀 수준 엑세스 제어

각 사전 서명된 URL은 W&B platform의 [팀 수준 엑세스 제어]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})에 따라 특정 버킷으로 제한됩니다. 사용자가 [보안 저장소 커넥터]({{< relref path="./secure-storage-connector.md" lang="ko" >}})를 사용하여 blob 저장소 버킷에 매핑된 팀의 구성원이고, 해당 사용자만 해당 팀의 구성원인 경우 해당 요청에 대해 생성된 사전 서명된 URL은 다른 팀에 매핑된 blob 저장소 버킷에 엑세스할 수 있는 권한이 없습니다.

{{% alert %}}
W&B는 사용자를 해당 사용자가 속해야 하는 팀에만 추가하는 것을 권장합니다.
{{% /alert %}}

## 네트워크 제한

W&B는 버킷에 대한 IAM 정책 기반 제한을 사용하여 사전 서명된 URL을 사용하여 blob 저장소에 엑세스할 수 있는 네트워크를 제한하는 것이 좋습니다.

AWS의 경우 [VPC 또는 IP 어드레스 기반 네트워크 제한](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)을 사용할 수 있습니다. 이렇게 하면 사용자가 W&B UI를 사용하여 Artifacts에 엑세스하는 경우, W&B 특정 버킷에 AI 워크로드가 실행 중인 네트워크 또는 사용자 컴퓨터에 매핑되는 게이트웨이 IP 어드레스에서만 엑세스할 수 있습니다.

## 감사 로그

W&B는 blob 저장소 특정 감사 로그 외에도 [W&B 감사 로그]({{< relref path="../monitoring-usage/audit-logging.md" lang="ko" >}})를 사용하는 것이 좋습니다. 후자의 경우 [AWS S3 엑세스 로그](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html), [Google Cloud Storage 감사 로그](https://cloud.google.com/storage/docs/audit-logging) 및 [Azure blob 저장소 모니터링](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage)을 참조하십시오. 관리 및 보안 팀은 감사 로그를 사용하여 W&B 제품에서 어떤 사용자가 무엇을 하고 있는지 추적하고 특정 사용자에 대해 일부 작업을 제한해야 한다고 판단되는 경우 필요한 조치를 취할 수 있습니다.

{{% alert %}}
사전 서명된 URL은 W&B에서 지원되는 유일한 blob 저장소 엑세스 메커니즘입니다. W&B는 위험 감수 수준에 따라 위에 나열된 보안 제어 중 일부 또는 전부를 구성하는 것이 좋습니다.
{{% /alert %}}
