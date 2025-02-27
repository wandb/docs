---
menu:
  default:
    identifier: audit-logging
    parent: monitoring-and-usage
title: Track user activity with audit logs
weight: 1
---

Use W&B audit logs to track user activity within your organization and to conform to your enterprise governance requirements. Audit logs are available in JSON format. How to access audit logs depends on your W&B platform deployment type:

| W&B Platform Deployment type | Audit logs access mechanism |
|----------------------------|--------------------------------|
| [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) | Synced to instance-level bucket every 10 minutes. Also available using [the API]({{< relref "#fetch-audit-logs-using-api" >}}). |
| [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) with [secure storage connector (BYOB)]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) | Synced to instance-level bucket (BYOB) every 10 minutes. Also available using [the API]({{< relref "#fetch-audit-logs-using-api" >}}). |
| [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) with W&B managed storage (without BYOB) | Available only by using [the API]({{< relref "#fetch-audit-logs-using-api" >}}). |
| [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) | Available for Enterprise plans only. Available only by using [the API]({{< relref "#fetch-audit-logs-using-api" >}}).

Once you've access to your audit logs, analyze those using your preferred tools, such as [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric), and more. You may need to transform the JSON-formatted audit logs into a format relevant to the tool before analysis. Information on how to transform your audit logs for specific tools is outside the scope of W&B documentation.

{{% alert %}}
**Audit Log Retention:** If a compliance, security or risk team in your organization requires audit logs to be retained for a specific period of time, W&B recommends to periodically transfer the logs from your instance-level bucket to a long-term retention storage. If you're instead using the API to access the audit logs, you can implement a simple script that runs periodically (like daily or every few days) to fetch any logs that may have been generated since the time of the last script run, and store those in a short-term storage for analysis or directly transfer to a long-term retention storage.
{{% /alert %}}

HIPAA compliance requires that you retain audit logs for a minimum of 6 years. For HIPAA-compliant [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) instances with [BYOB]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}), you must configure guardrails for your managed storage including any long-term retention storage, to ensure that no internal or external user can delete audit logs before the end of the mandatory retention period.

## Audit log schema
The following table lists all the different keys that might be present in your audit logs. Each log contains only the assets relevant to the corresponding action, and others are omitted from the log.

| Key | Definition |
|---------| -------|
|`timestamp`               | Time stamp in [RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339). For example: `2023-01-23T12:34:56Z`, represents `12:34:56 UTC` time on Jan 23, 2023.
|`action`                  | What [action]({{< relref "#actions" >}}) did the user take.
|`actor_user_id`           | If present, ID of the logged-in user who performed the action.
|`response_code`          | Http response code for the action.
|`artifact_asset`          | If present, action was taken on this artifact id
|`artifact_sequence_asset` | If present, action was taken on this artifact sequence id
|`entity_asset`            | If present, action was taken on this entity or team id.
|`project_asset`           | If present, action was taken on this project id.
|`report_asset`            | If present, action was taken on this report id.
|`user_asset`              | If present, action was taken on this user asset.
|`cli_version`             | If the action is taken via python SDK, this will contain the version
|`actor_ip`                | IP address of the logged-in user.
|`actor_email`             | if present, action was taken on this actor email.
|`artifact_digest`         | if present, action was taken on this artifact digest.
|`artifact_qualified_name` | if present, action was taken on this artifact.
|`entity_name`             | if present, action was taken on this entity or team name.
|`project_name`            | if present, action was taken on this project name.
|`report_name`             | if present, action was taken on this report name.
|`user_email`              | if present, action was taken on this user email.

### Personally identifiable information (PII)

Personally identifiable information (PII), such as email ids and the names of projects, teams, and reports, is available only using the API endpoint option.
- For [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) and 
  [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}), an organization admin can [exclude PII]({{< relref "#exclude-pii" >}}) when fetching audit logs.
- For [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), the API endpoint always returns relevant fields for audit logs, including PII. This is not configurable.

## Fetch audit logs
An organization or instance admin can fetch the audit logs for a W&B instance using the `audit_logs` API.

{{% alert %}}
If you are an admin of multiple Enterprise [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) organizations, you must configure the organization where audit logging API requests are sent. Click your profile image, then click **User Settings**. The setting is named **Default API organization**.
{{% /alert %}}

1. Determine the correct API endpoint for your instance:

  - [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}): `<wandb-platform-url>/admin/audit_logs`
  - [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}): `<wandb-platform-url>/admin/audit_logs`
  - [SaaS Cloud (Enterprise required)]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}): `https://api.wandb.ai/audit_logs`

  In proceeding steps, replace `<API-endpoint>` with your API endpoint.
1. Construct the full API endpoint from the base endpoint, and optionally include URL parameters:
    - `anonymize`: if set to `true`, remove any PII; defaults to `false`. Refer to [Exclude PII when fetching audit logs]({{< relref "#exclude-pii" >}}). Not supported for SaaS Cloud.
    - `numDays`: logs will be fetched starting from `today - numdays` to most recent; defaults to `0`, which returns logs only for `today`. For SaaS Cloud, you can fetch audit logs from a maximum of 7 days in the past.
    - `startDate`: an optional date with format `YYYY-MM-DD`.

    {{% alert %}}
    - If you set both `startDate` and `numDays`, logs are returned from `startDate` to `startDate` + `numDays`.
    - If you omit `startDate` butr include `numDays`, logs are returned from `today` to `numDays`.fetch for `today` only.
    - If you set neither `startDate` nor `numDays`, logs are returned for `today` only.
    {{% /alert %}}


1. Execute an HTTP GET request on the constructed fully qualified API endpoint using a web browser or a tool like [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), or cURL.

The API response contains new-line separated JSON objects. Objects will include the fields described in the schema. It's the same format which is used when syncing audit log files to an instance-level bucket (wherever applicable as mentioned earlier). In those cases, the audit logs are located at the `/wandb-audit-logs` directory in your bucket.

{{% alert %}}
Only an W&B [instance or organization admin]({{< relref "/guides/hosting/iam/access-management/" >}}) can fetch audit logs using the API. Otherwise, the error `HTTP 403 Forbidden` occurs.
{{% /alert %}}

### Use basic authentication
To use basic authentication with your API key to access the audit logs API, set the HTTP request's `Authorization` header to the string `Basic` followed by a space, then the base-64 encoded string in the format `username:API-KEY`. In other words, replace the username and API key with your values separated with a `:` character, then base-64-encode the result. For example, to authorize as `demo:p@55w0rd`, the header should be `Authorization: Basic ZGVtbzpwQDU1dzByZA==`.

### Exclude PII when fetching audit logs {#exclude-pii}
For [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) and [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}), a W&B organization or instance admin can exclude PII when fetching audit logs. For [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), the API endpoint always returns relevant fields for audit logs, including PII. This is not configurable.


To exclude PII, pass the `anonymize=true` URL parameter. For example, if your W&B instance URL is `https://mycompany.wandb.io` and you would like to get audit logs for user activity within the last week and exclude PII, use an API endpoint like:

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true`.
```

## Actions
The following table describes possible actions that can be recorded by W&B.

{{% alert %}}
On [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), audit logs are not collected for:
- Open or Public projects.
- The `report:read` action.
- `User` actions which are not tried to a specific organization.
{{% /alert %}}

|Action | Definition |
|-----|-----|
| `artifact:create`             | Artifact is created.
| `artifact:delete   `          | Artifact is deleted.
| `artifact:read`               | Artifact is read.
| `project:delete`              | Project is deleted.
| `project:read`                | Project is read.
| `report:read`                 | Report is read. Not collected on [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
| `run:delete`                  | Run is deleted.
| `run:delete_many`             | Runs are deleted in batch.
| `run:update_many`             | Runs are updated in batch.
| `run:stop`                    | Run is stopped.
| `run:undelete_many`           | Runs are brought back from trash in batch.
| `run:update`                  | Run is updated.
| `sweep:create_agent`          | Sweep agent is created.
| `team:invite_use`r            | User is invited to team.
| `team:create_service_account` | Service account is created for the team.
| `team:create`                 | Team is created.
| `team:uninvite`               | User or service account is uninvited from team.
| `team:delete`                 | Team is deleted.
| `user:create`                 | User is created.
| `user:delete_api_key`         | API key for the user is deleted.
| `user:deactivate`             | User is deactivated.
| `user:create_api_key`         | API key for the user is created.
| `user:permanently_delete`     | User is permanently deleted.
| `user:reactivate`             | User is reactivated.
| `user:update`                 | User is updated.
| `user:read`                   | User profile is read.
| `user:login`                  | User logs in.
| `user:initiate_login`         | User initiates log in.
| `user:logout`                 | User logs out.
