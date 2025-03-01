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
| [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) with W&B managed storage (without BYOB) | Only available using [the API]({{< relref "#fetch-audit-logs-using-api" >}}). |

{{% alert %}}
Audit logs are not available for [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
{{% /alert %}}

Once you've access to your audit logs, analyze those using your preferred tools, such as [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric), and more. You may need to transform the JSON-formatted audit logs into a format relevant to the tool before analysis. Information on how to transform your audit logs for specific tools is outside the scope of W&B documentation.

{{% alert %}}
**Audit Log Retention:** If a compliance, security or risk team in your organization requires audit logs to be retained for a specific period of time, W&B recommends to periodically transfer the logs from your instance-level bucket to a long-term retention storage. If you're instead using the API to access the audit logs, you can implement a simple script that runs periodically (like daily or every few days) to fetch any logs that may have been generated since the time of the last script run, and store those in a short-term storage for analysis or directly transfer to a long-term retention storage.
{{% /alert %}}

HIPAA compliance requires that you retain audit logs for a minimum of 6 years. For HIPAA-compliant [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) instances with [BYOB]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}), you must configure guardrails for your managed storage including any long-term retention storage, to ensure that no internal or external user can delete audit logs before the end of the mandatory retention period.

## Audit log schema
The following table lists all the different keys that might be present in your audit logs. Each log contains only the assets relevant to the corresponding action, and others are omitted from the log.

| Key | Definition |
|---------| -------|
|timestamp               | Time stamp in [RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339). For example: `2023-01-23T12:34:56Z`, represents `12:34:56 UTC` time on Jan 23, 2023.
|action                  | What [action]({{< relref "#actions" >}}) did the user take.
|actor_user_id           | If present, ID of the logged-in user who performed the action.
|response_code           | Http response code for the action.
|artifact_asset          | If present, action was taken on this artifact id
|artifact_sequence_asset | If present, action was taken on this artifact sequence id
|entity_asset            | If present, action was taken on this entity or team id.
|project_asset           | If present, action was taken on this project id.
|report_asset            | If present, action was taken on this report id.
|user_asset              | If present, action was taken on this user asset.
|cli_version             | If the action is taken via python SDK, this will contain the version
|actor_ip                | IP address of the logged-in user.
|actor_email             | if present, action was taken on this actor email.
|artifact_digest         | if present, action was taken on this artifact digest.
|artifact_qualified_name | if present, action was taken on this artifact.
|entity_name             | if present, action was taken on this entity or team name.
|project_name            | if present, action was taken on this project name.
|report_name             | if present, action was taken on this report name.
|user_email              | if present, action was taken on this user email.

Personally identifiable information (PII), such as email ids and the names of projects, teams, and reports, is available only using the API endpoint option, and can be turned off as [described below]({{< relref "#fetch-audit-logs-using-api" >}}).

## Fetch audit logs using API
An instance admin can fetch the audit logs for your W&B instance using the following API:
1. Construct the full API endpoint using a combination of the base endpoint `<wandb-platform-url>/admin/audit_logs` and the following URL parameters:
    - `numDays`: logs will be fetched starting from `today - numdays` to most recent; defaults to `0`, which returns logs only for `today`.
    - `anonymize`: if set to `true`, remove any PII; defaults to `false`
2. Execute HTTP GET request on the constructed full API endpoint, either by directly running it within a modern browser, or by using a tool like [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL command or more.

An organization or instance admin can use basic authentication with their API key to access the audit logs API. Set the HTTP request's `Authorization` header to the string `Basic` followed by a space, then the base-64 encoded string in the format `username:API-KEY`. In other words, replace the username and API key with your values separated with a `:` character, then base-64-encode the result. For example, to authorize as `demo:p@55w0rd`, the header should be `Authorization: Basic ZGVtbzpwQDU1dzByZA==`.

If your W&B instance URL is `https://mycompany.wandb.io` and you would like to get audit logs without PII for user activity within the last week, you must use the API endpoint `https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true`.

{{% alert %}}
Only W&B [instance admins]({{< relref "/guides/hosting/iam/access-management/" >}}) can fetch audit logs using the API. If you are not an instance admin or not logged into your organization, you get a `HTTP 403 Forbidden` error.
{{% /alert %}}

The API response contains new-line separated JSON objects. Objects will include the fields described in the schema. It's the same format which is used when syncing audit log files to an instance-level bucket (wherever applicable as mentioned earlier). In those cases, the audit logs are located at the `/wandb-audit-logs` directory in your bucket.

## Actions
The following table describes possible actions that can be recorded by W&B:

|Action | Definition |
|-----|-----|
| artifact:create             | Artifact is created.
| artifact:delete             | Artifact is deleted.
| artifact:read               | Artifact is read.
| project:delete              | Project is deleted.
| project:read                | Project is read.
| report:read                 | Report is read.
| run:delete                  | Run is deleted.
| run:delete_many             | Runs are deleted in batch.
| run:update_many             | Runs are updated in batch.
| run:stop                    | Run is stopped.
| run:undelete_many           | Runs are brought back from trash in batch.
| run:update                  | Run is updated.
| sweep:create_agent          | Sweep agent is created.
| team:invite_user            | User is invited to team.
| team:create_service_account | Service account is created for the team.
| team:create                 | Team is created.
| team:uninvite               | User or service account is uninvited from team.
| team:delete                 | Team is deleted.
| user:create                 | User is created.
| user:delete_api_key         | API key for the user is deleted.
| user:deactivate             | User is deactivated.
| user:create_api_key         | API key for the user is created.
| user:permanently_delete     | User is permanently deleted.
| user:reactivate             | User is reactivated.
| user:update                 | User is updated.
| user:read                   | User profile is read.
| user:login                  | User logs in.
| user:initiate_login         | User initiates log in.
| user:logout                 | User logs out.