---
title: Access BYOB using pre-signed URLs
displayed_sidebar: default
---

W&B는 사전 서명된 URL을 사용하여 AI 워크로드나 사용자 브라우저에서 blob 스토리지로의 엑세스를 간소화합니다. 사전 서명된 URL에 대한 기본 정보는 [AWS S3의 사전 서명된 URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html), [Google Cloud Storage의 서명된 URL](https://cloud.google.com/storage/docs/access-control/signed-urls) 및 [Azure Blob Storage의 공유 엑세스 서명](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview)을 참조하십시오.

필요한 경우, 네트워크 내의 AI 워크로드나 사용자 브라우저 클라이언트는 W&B 플랫폼에서 사전 서명된 URL을 요청합니다. W&B 플랫폼은 관련 blob 스토리지에 엑세스하여 필요한 권한과 함께 사전 서명된 URL을 생성하고 이를 클라이언트에게 반환합니다. 클라이언트는 사전 서명된 URL을 사용하여 오브젝트 업로드 또는 검색 작업을 위해 blob 스토리지에 엑세스합니다. 오브젝트 다운로드에 대한 URL 만료 시간은 1시간이며, 대량의 오브젝트가 청크 단위로 업로드되는 데 시간이 더 필요할 수 있기 때문에 오브젝트 업로드의 경우 24시간입니다.

## 팀 수준 엑세스 제어

각 사전 서명된 URL은 W&B 플랫폼에서 [팀 수준 엑세스 제어](../iam/manage-users.md#manage-a-team)에 기반하여 특정 버킷(bucket)에 제한됩니다. 사용자가 [보안 스토리지 커넥터](./secure-storage-connector.md)를 사용하여 blob 스토리지 버킷에 매핑된 팀의 일부인 경우, 그리고 그 사용자가 오직 그 팀의 일부인 경우, 그들의 요청을 위해 생성된 사전 서명된 URL은 다른 팀에 매핑된 blob 스토리지 버킷에 엑세스할 권한을 갖지 않습니다.

:::info
W&B는 사용자를 그들이 속하도록 지정된 팀에만 추가할 것을 권장합니다.
:::

## 네트워크 제한

W&B는 버킷에 대한 IAM 정책 기반 제한을 사용하여 사전 서명된 URL을 통해 blob 스토리지에 엑세스할 수 있는 네트워크를 제한할 것을 권장합니다.

AWS의 경우, [VPC 또는 IP 어드레스 기반의 네트워크 제한](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)을 사용할 수 있습니다. 이를 통해 W&B 전용 버킷이 AI 워크로드가 실행 중인 네트워크에서만 엑세스되도록 하거나, W&B UI를 이용해 아티팩트에 엑세스하는 경우 사용자 머신에 매핑되는 게이트웨이 IP 어드레스에서만 엑세스되도록 보장합니다.

## 감사 로그

W&B는 blob 스토리지 전용 감사 로그 이외에도 [W&B 감사 로그](../monitoring-usage/audit-logging.md)를 사용할 것을 권장합니다. 후자의 경우, [AWS S3 엑세스 로그](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html), [Google Cloud Storage 감사 로그](https://cloud.google.com/storage/docs/audit-logging) 및 [Azure blob storage 모니터링](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage)을 참조하십시오. 감사 로그를 통해 관리자 및 보안팀은 W&B 제품 내에서 어떤 사용자가 무엇을 하는지 추적할 수 있으며, 특정 사용자에 대해 일부 작업을 제한해야 할 필요가 있다고 판단되면 필요한 조치를 취할 수 있습니다.

:::note
사전 서명된 URL은 W&B에서 지원하는 유일한 blob 스토리지 엑세스 메커니즘입니다. W&B는 위험 감수 수준에 따라 위의 보안 통제 목록 중 일부 또는 모두를 구성할 것을 권장합니다.
:::