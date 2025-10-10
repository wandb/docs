---
menu:
  default:
    identifier: audit-logging
    parent: monitoring-and-usage
title: Track user activity with audit logs
weight: 1
---

Use W&B audit logs to track user activity within your organization and to conform to your enterprise governance requirements. Audit logs are available in JSON format. Refer to [Audit log schema]({{< relref "#audit-log-schema" >}}).

How to access audit logs depends on your W&B platform deployment type:

| W&B Platform Deployment type | Audit logs access mechanism |
|----------------------------|--------------------------------|
| [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) | <ul><li>[Instance-level BYOB]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}): Synced to instance-level bucket (BYOB) every 10 minutes. Also available using [the API]({{< relref "#fetch-audit-logs-using-api" >}}).</li><li>Default instance-level storage: Available only by using [the API]({{< relref "#fetch-audit-logs-using-api" >}}).</li></ul> |
| [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) | Available for Enterprise plans only. Available only by using [the API]({{< relref "#fetch-audit-logs-using-api" >}}). |
| [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) | Synced to instance-level bucket every 10 minutes. Also available using [the API]({{< relref "#fetch-audit-logs-using-api" >}}). |

After fetching audit logs, you can analyze them using tools like [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), or [Microsoft Fabric](https://www.microsoft.com/microsoft-fabric). Some audit log analysis tools do not support JSON; refer to the documentation for your analysis tool for guidelines and requirements for transforming the JSON-formatted audit logs before analysis.

For more details about the format of the logs, see [Audit log schema](#audit-log-schema) and [Actions](#actions).

## Audit log retention
- If you require audit logs to be retained for a specific period of time, W&B recommends periodically transferring logs to long-term storage, either using storage buckets or the Audit Logging API.
- If you are subject to the [Health Insurance Portability and Accountability Act of 1996 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html), audit logs must be retained for a minimum of 6 years in an environment where they cannot be deleted or modified by any internal or exterrnal actor before the end of the mandatory retention period. For HIPAA-compliant [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) instances with [BYOB]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}), you must configure guardrails for your managed storage, including any long-term retention storage.

## Before you begin
1. Organization level admins can fetch audit logs. If you receive a `403` error, ensure that you or your service account has adequate permission.
1. **Multi-tenant Cloud**: If you are a member of multiple Multi-tenant Cloud organizations, you _must_ configure the **Default API organization**, which determines where audit logging API calls are routed. Otherwise, you will receive the following error:
     ```text
     user is associated with multiple organizations but no valid org ID found in user info
     ```
     To specify your default API organization:
     1. Click your profile image, then click **User Settings**.
     1. For **Default API organization**, select an organization.

     This is not applicable to a service account, which can be a member of only one Multi-tenant Cloud organization.

## Fetch audit logs
To fetch audit logs:

1. Determine the correct API endpoint for your instance:

    - [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}): `<instance-name>.wandb.io/admin/audit_logs`
    - [Multi-tenant Cloud (Enterprise required)]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}): `https://api.wandb.ai/audit_logs`

    In the following steps, replace `<API-endpoint>` with your API endpoint.
1. (Optional) Construct query parameters to append to the endpoint. In the following steps, replace `<parameters> with the resulting string.
    - `anonymize`: if the URL includes the parameter `anonymize=true`, remove any PII. Otherwise, PII is included. Refer to [Exclude PII when fetching audit logs]({{< relref "#exclude-pii" >}}). Not supported for Multi-tenant Cloud, where all fields are included, including PII.
    - Configure the date window of logs to fetch using a combination of `numdays` and `startDate`. Each parameter is optional, and they interact.
      - If neither parameter is included, only today's logs are fetched.
      - `numDays`: An integer indicating the number of days backward from `startDate` to fetch logs. If it is omitted or set to `0`, logs are fetched for `startDate` only. **Multi-tenant Cloud** organizations can fetch up to 7 days of audit logs. In other words, if you set `numDays=9`, the effective parameter is `numDays=7`.
      - `startDate`: Controls the newest logs to fetch, in the format `startDate=YYYY-MM-DD`. If it is omitted or explicitly set to today's date, logs are fetched from today to `numDays` (up to `7` for Multi-tenant Cloud).
1. Construct the fully qualified endpoint URL in the format `<API-endpoint>?<parameters>`.
1. Execute an HTTP `GET` request on the fully qualified API endpoint using a web browser or a tool like [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), or cURL.

The API response contains new-line separated JSON objects. Objects will include the fields described in the [schema]({{< relref "#audit-log-schema" >}}), just like when audit logs are synced to an instance-level bucket. In those cases, the audit logs are located in the `/wandb-audit-logs` directory in your bucket.

### Use basic authentication
To use basic authentication with your API key to access the audit logs API, set the HTTP request's `Authorization` header to the string `Basic` followed by a space, then the base-64 encoded string in the format `username:API-KEY`. In other words, replace the username and API key with your values separated with a `:` character, then base-64-encode the result. For example, to authorize as `demo:p@55w0rd`, the header should be `Authorization: Basic ZGVtbzpwQDU1dzByZA==`.

### Exclude PII when fetching audit logs {#exclude-pii}
For [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) and [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}), a W&B organization or instance admin can exclude PII when fetching audit logs. For [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), the API endpoint always returns relevant fields for audit logs, including PII. This is not configurable.


To exclude PII, pass the `anonymize=true` URL parameter. For example, if your W&B instance URL is `https://mycompany.wandb.io` and you would like to get audit logs for user activity within the last week and exclude PII, use an API endpoint like:

```text
https://mycompany.wandb.io/admin/audit_logs?anonymize=true&<additional-parameters>.
```

## Audit log schema
This table shows all keys which may appear in an audit log entry, ordered alphabetically. Depending on the action and the circumstances, a specific log entry may include only a subset of the possible fields.

| Key | Definition |
|---------| -------|
|`action`                  | The [action]({{< relref "#actions" >}}) of the event.
|`actor_email`             | The email address of the user that initiated the action, if applicable.
|`actor_ip`                | The IP address of the user that initiated the action.
|`actor_user_id`           | The ID of the logged-in user who performed the action, if applicable.
|`artifact_asset`          | The artifact ID associated with the action, if applicable.
|`artifact_digest`         | The artifact digest associated with the action, if applicable.
|`artifact_qualified_name` | The full name of the artifact associated with the action, if applicable.
|`artifact_sequence_asset` | The artifact sequence ID associated with the action, if applicable.
|`cli_version`             | The version of the Python SDK that initiated the action, if applicable.
|`entity_asset`            | The entity or team ID associated with the action, if applicable.
|`entity_name`             | The entity or team name associated with the action, if applicable.
|`project_asset`           | The project associated with the action, if applicable.
|`project_name`            | The name of the project associated with the action, if applicable.
|`report_asset`            | The report ID associated with the action, if applicable.
|`report_name`             | The name of the report associated with the action, if applicable.
|`response_code`           | The HTTP response code for the action, if applicable.
|`timestamp`               | The time of the event in [RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339). For example, `2023-01-23T12:34:56Z` represents January 23, 2023 at 12:34:56 UTC.
|`user_asset`              | The user asset the action impacts (rather than the user performing the action), if applicable.
|`user_email`              | The email address of the user the action impacts (rather than the email address of the user performing the action), if applicable.

### Personally identifiable information (PII)

Personally identifiable information (PII), such as email addresses and the names of projects, teams, and reports, is available only using the API endpoint option.
- For [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) and 
  [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}), an organization admin can [exclude PII]({{< relref "#exclude-pii" >}}) when fetching audit logs.
- For [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), the API endpoint always returns relevant fields for audit logs, including PII. This is not configurable.

## Actions
This table describes possible actions that can be recorded by W&B, sorted alphabetically.

|Action | Definition |
|-----|-----|
| `artifact:create`             | Artifact is created.
| `artifact:delete   `          | Artifact is deleted.
| `artifact:read`               | Artifact is read.
| `project:delete`              | Project is deleted.
| `project:read`                | Project is read.
| `report:read`                 | Report is read. <sup><a href="#1">1</a></sup>
| `run:delete_many`             | Batch of runs is deleted.
| `run:delete`                  | Run is deleted.
| `run:stop`                    | Run is stopped.
| `run:undelete_many`           | Batch of runs is restored from trash.
| `run:update_many`             | Batch of runs is updated.
| `run:update`                  | Run is updated.
| `sweep:create_agent`          | Sweep agent is created.
| `team:create_service_account` | Service account is created for the team.
| `team:create`                 | Team is created.
| `team:delete`                 | Team is deleted.
| `team:invite_user`            | User is invited to team.
| `team:uninvite`               | User or service account is uninvited from team.
| `user:create_api_key`         | API key for the user is created. <sup><a href="#1">1</a></sup>
| `user:create`                 | User is created. <sup><a href="#1">1</a></sup>
| `user:deactivate`             | User is deactivated. <sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | API key for the user is deleted. <sup><a href="#1">1</a></sup>
| `user:initiate_login`         | User initiates log in. <sup><a href="#1">1</a></sup>
| `user:login`                  | User logs in. <sup><a href="#1">1</a></sup>
| `user:logout`                 | User logs out. <sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | User is permanently deleted. <sup><a href="#1">1</a></sup>
| `user:reactivate`             | User is reactivated. <sup><a href="#1">1</a></sup>
| `user:read`                   | User profile is read. <sup><a href="#1">1</a></sup>
| `user:update`                 | User is updated. <sup><a href="#1">1</a></sup>

<a id="1">1</a>: On [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), audit logs are not collected for:
- Open or Public projects.
- The `report:read` action.
- `User` actions which are not tied to a specific organization.
