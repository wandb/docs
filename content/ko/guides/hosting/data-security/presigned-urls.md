---
title: 미리 서명된 URL을 사용하여 BYOB 엑세스하기
menu:
  default:
    identifier: ko-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B는 AI 워크로드나 사용자 브라우저가 blob 스토리지에 쉽게 엑세스할 수 있도록 pre-signed URL을 사용합니다. pre-signed URL에 대한 기본적인 정보는 아래 클라우드 제공업체의 문서를 참고해 주세요.

- [AWS S3의 Pre-signed URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html) — S3-compatible 스토리지(예: [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage))에도 적용됩니다.
- [Google Cloud Storage의 Signed URLs](https://cloud.google.com/storage/docs/access-control/signed-urls)
- [Azure Blob Storage의 Shared Access Signature](https://learn.microsoft.com/azure/storage/common/storage-sas-overview)

작동 방식:
1. 필요할 때, 네트워크 내의 AI 워크로드 또는 사용자 브라우저 클라이언트가 W&B에 pre-signed URL을 요청합니다.
1. W&B는 해당 요청을 받아 blob 스토리지에 엑세스하여, 필요한 권한이 설정된 pre-signed URL을 생성합니다.
1. W&B는 클라이언트에게 pre-signed URL을 반환합니다.
1. 클라이언트는 이 pre-signed URL을 사용해 blob 스토리지에 읽기 또는 쓰기 작업을 수행합니다.

pre-signed URL의 만료 시간은 다음과 같습니다:
- **읽기**: 1시간
- **쓰기**: 24시간 (대용량 오브젝트를 여러 번에 나눠 업로드할 수 있도록 더 긴 시간이 부여됨)

## 팀 레벨 엑세스 제어

각 pre-signed URL은 W&B 플랫폼의 [팀 레벨 엑세스 제어]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})를 기반으로 특정 버킷에만 제한됩니다. 사용자가 [secure storage connector]({{< relref path="./secure-storage-connector.md" lang="ko" >}})를 통해 스토리지 버킷과 매핑된 팀의 일부이고, 해당 사용자가 그 팀에만 속해 있다면, 해당 사용자의 요청으로 생성된 pre-signed URL에는 다른 팀에 매핑된 스토리지 버킷에 접근할 권한이 부여되지 않습니다.

{{% alert %}}
W&B에서는 사용자를 꼭 필요한 팀에만 추가하는 것을 권장합니다.
{{% /alert %}}

## 네트워크 제한
W&B는 IAM 정책을 활용해, pre-signed URL을 통한 외부 스토리지 접근이 네트워크별로 제한되도록 설정할 것을 권장합니다. 이를 통해 AI 워크로드가 실행되는 네트워크나, 사용자 기기를 대표하는 게이트웨이 IP 주소에서만 W&B 전용 버킷에 접근하도록 할 수 있습니다.

- CoreWeave AI Object Storage의 정책은 CoreWeave 문서 내 [Bucket 정책 레퍼런스](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy#condition)를 참고하세요.
- AWS S3, 또는 자체 구축한 MiniIO 등 S3-compatible 스토리지의 경우 [S3 사용자 가이드](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities), [MinIO 문서](https://github.com/minio/minio) 또는 기타 S3-compatible 스토리지 공급업체 문서를 참고하세요.

## 감사 로그

W&B에서는 [W&B 감사 로그]({{< relref path="../monitoring-usage/audit-logging.md" lang="ko" >}})와 blob 스토리지 전용 감사 로그의 병행 사용을 권장합니다. 각 클라우드 제공업체별 blob 스토리지 감사 로그는 아래를 참고하세요.
- [CoreWeave 감사 로그](https://docs.coreweave.com/docs/products/storage/object-storage/concepts/audit-logging#audit-logging-policies)
- [AWS S3 엑세스 로그](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)
- [Google Cloud Storage 감사 로그](https://cloud.google.com/storage/docs/audit-logging)
- [Azure blob storage 모니터링](https://learn.microsoft.com/azure/storage/blobs/monitor-blob-storage)

관리자 및 보안 팀은 감사 로그를 활용하여, W&B에서 어떤 사용자가 어떤 활동을 했는지 추적할 수 있습니다. 필요할 경우 특정 사용자의 작업을 제한하는 조치도 취할 수 있습니다.

{{% alert %}}
Pre-signed URL은 W&B에서 지원하는 유일한 blob 스토리지 엑세스 방식입니다. 조직의 정책에 따라 위에서 안내한 보안 제어 항목 중 일부 또는 전부를 적용하기를 권장합니다.
{{% /alert %}}

## pre-signed URL을 요청한 사용자 확인 방법
W&B가 pre-signed URL을 반환할 때, URL 내 쿼리 파라미터로 요청자의 사용자명이 포함됩니다.

| 스토리지 제공업체                 | Signed URL 쿼리 파라미터 |
|-----------------------------|------------------|
| CoreWeave AI Object Storage | `X-User`         |
| AWS S3 storage              | `X-User`         |
| Google Cloud storage        | `X-User`         |
| Azure blob storage          | `scid`           |
