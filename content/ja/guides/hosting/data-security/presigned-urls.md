---
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
title: Access BYOB using pre-signed URLs
weight: 2
---

W&B uses pre-signed URLs to simplify access to blob storage from your AI workloads or user browsers. For basic information on pre-signed URLs, refer to the cloud provider's documentation:

- [Pre-signed URLs for AWS S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html), which also applies to S3-compatible storage like [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage).
- [Signed URLs for Google Cloud Storage](https://cloud.google.com/storage/docs/access-control/signed-urls)
- [Shared Access Signature for Azure Blob Storage](https://learn.microsoft.com/azure/storage/common/storage-sas-overview)

How it works:
1. When needed, AI workloads or user browser clients within your network request pre-signed URLs from W&B.
1. W&B responds to the request by accessing the blob storage to generate the pre-signed URL with the required permissions.
1. W&B returns the pre-signed URL to the client.
1. The client uses the pre-signed URL to read or write to the blob storage.

A pre-signed URL expires after:
- **Reading**: 1 hour
- **Writing**: 24 hours, to allow more time to upload large objects in chunks.

## Team-level access control

Each pre-signed URL is restricted to specific buckets based on [team level access control]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) in the W&B platform. If a user is part of a team which is mapped to a storage bucket using [secure storage connector]({{< relref path="./secure-storage-connector.md" lang="ja" >}}), and if that user is part of only that team, then the pre-signed URLs generated for their requests would not have permissions to access storage buckets mapped to other teams. 

{{% alert %}}
W&B recommends adding users to only the teams that they are supposed to be a part of.
{{% /alert %}}

## Network restriction
W&B recommends using IAM policies to restrict the networks that can use pre-signed URLs to access external storage using pre-signed URLs. This helps to ensure that your W&B specific buckets are accessed only from networks where your AI workloads are running, or from gateway IP addresses that map to your user machines. 

- For CoreWeave AI Object Storage, refer to [Bucket policy reference](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy#condition) in the CoreWeave documentation.
- For AWS S3 or S3-compatible storage like MiniIO hosted on your premises, refer to the [S3 userguide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities), the [MinIO documentation](https://github.com/minio/minio), or the documentation for your S3-compatible storage provider.

## Audit logs

W&B recommends using [W&B audit logs]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}}) together with blob storage specific audit logs. For blob storage audit logs, refer to the documentation for each cloud provider:
- [CoreWeave audit logs](https://docs.coreweave.com/docs/products/storage/object-storage/concepts/audit-logging#audit-logging-policies)
- [AWS S3 access logs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)
- [Google Cloud Storage audit logs](https://cloud.google.com/storage/docs/audit-logging)
- [Monitor Azure blob storage](https://learn.microsoft.com/azure/storage/blobs/monitor-blob-storage).

Admin and security teams can use audit logs to keep track of which user is doing what in the W&B product and take necessary action if they determine that some operations need to be limited for certain users.

{{% alert %}}
Pre-signed URLs are the only supported blob storage access mechanism in W&B. W&B recommends configuring some or all of the above list of security controls according to your organization's needs.
{{% /alert %}}

## Determine the user that requested a pre-signed URL
When W&B returns a pre-signed URL, a query parameter in the URL contains the requester's username:

| Storage provider            | Signed URL query parameter |
|-----------------------------|-----------|
| CoreWeave AI Object Storage | `X-User`  |
| AWS S3 storage              | `X-User`  |
| Google Cloud storage        | `X-User`  |
| Azure  blob storage         | `scid`    |