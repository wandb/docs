---
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
title: Track user activity with audit logs
weight: 1
---

Use W&B audit logs to track user activity within your organization and to conform to your enterprise governance requirements. Audit logs are available in JSON format. Refer to [Audit log schema]({{< relref path="#audit-log-schema" lang="ko" >}}).

How to access audit logs depends on your W&B platform deployment type:

| W&B Platform Deployment type | Audit logs access mechanism |
|----------------------------|--------------------------------|
| [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) | Synced to instance-level bucket every 10 minutes. Also available using [the API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}}). |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) with [secure storage connector (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) | Synced to instance-level bucket (BYOB) every 10 minutes. Also available using [the API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}}). |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) with W&B managed storage (without BYOB) | Available only by using [the API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}}). |
| [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) | Available for Enterprise plans only. Available only by using [the API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}}).

After fetching audit logs, you can analyze them using tools like [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), or [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric). Some audit log analysis tools do not support JSON; refer to the documentation for your analysis tool for guidelines and requirements for transforming the JSON-formatted audit logs before analysis.

{{% alert title="Audit log retention" %}}
If you require audit logs to be retained for a specific period of time, W&B recommends periodically transferring logs to long-term storage, either using storage buckets or the Audit Logging API.

If you are subject to the [Health Insurance Portability and Accountability Act of 1996 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html), audit logs must be retained for a minimum of 6 years in an environment where they cannot be deleted or modified by any internal or exterrnal actor before the end of the mandatory retention period. For HIPAA-compliant [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) instances with [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}), you must configure guardrails for your managed storage, including any long-term retention storage.
{{% /alert %}}

## Audit log schema
This table shows all keys which may appear in an audit log entry, ordered alphabetically. Depending on the action and the circumstances, a specific log entry may include only a subset of the possible fields.

| Key | Definition |
|---------| -------|
|`action`                  | The [action]({{< relref path="#actions" lang="ko" >}}) of the event.
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
- For [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) and 
  [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), an organization admin can [exclude PII]({{< relref path="#exclude-pii" lang="ko" >}}) when fetching audit logs.
- For [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}), the API endpoint always returns relevant fields for audit logs, including PII. This is not configurable.

## Fetch audit logs
An organization or instance admin can fetch the audit logs for a W&B instance using the Audit Logging API, at the endpoint `audit_logs/`.

{{% alert %}}
- If a user other than an admin attempts to fetch audit logs, a HTTP `403` error occurs, indicating that access is denied.

- If you are an admin of multiple Enterprise [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organizations, you must configure the organization where audit logging API requests are sent. Click your profile image, then click **User Settings**. The setting is named **Default API organization**.
{{% /alert %}}

1. Determine the correct API endpoint for your instance:

    - [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [SaaS Cloud (Enterprise required)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `https://api.wandb.ai/audit_logs`

    In proceeding steps, replace `<API-endpoint>` with your API endpoint.
1. Construct the full API endpoint from the base endpoint, and optionally include URL parameters:
    - `anonymize`: if set to `true`, remove any PII; defaults to `false`. Refer to [Exclude PII when fetching audit logs]({{< relref path="#exclude-pii" lang="ko" >}}). Not supported for SaaS Cloud.
    - `numDays`: logs will be fetched starting from `today - numdays` to most recent; defaults to `0`, which returns logs only for `today`. For SaaS Cloud, you can fetch audit logs from a maximum of 7 days in the past.
    - `startDate`: an optional date with format `YYYY-MM-DD`. Supported only on [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}).

      `startDate` and `numDays` interact:
        - If you set both `startDate` and `numDays`, logs are returned from `startDate` to `startDate` + `numDays`.
        - If you omit `startDate` but include `numDays`, logs are returned from `today` to `numDays`.
        - If you set neither `startDate` nor `numDays`, logs are returned for `today` only.

1. Execute an HTTP `GET` request on the constructed fully qualified API endpoint using a web browser or a tool like [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), or cURL.

The API response contains new-line separated JSON objects. Objects will include the fields described in the [schema]({{< relref path="#audit-log-schemag" lang="ko" >}}), just like when audit logs are synced to an instance-level bucket. In those cases, the audit logs are located in the `/wandb-audit-logs` directory in your bucket.

### Use basic authentication
To use basic authentication with your API key to access the audit logs API, set the HTTP request's `Authorization` header to the string `Basic` followed by a space, then the base-64 encoded string in the format `username:API-KEY`. In other words, replace the username and API key with your values separated with a `:` character, then base-64-encode the result. For example, to authorize as `demo:p@55w0rd`, the header should be `Authorization: Basic ZGVtbzpwQDU1dzByZA==`.

### Exclude PII when fetching audit logs {#exclude-pii}
For [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) and [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), a W&B organization or instance admin can exclude PII when fetching audit logs. For [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}), the API endpoint always returns relevant fields for audit logs, including PII. This is not configurable.


To exclude PII, pass the `anonymize=true` URL parameter. For example, if your W&B instance URL is `https://mycompany.wandb.io` and you would like to get audit logs for user activity within the last week and exclude PII, use an API endpoint like:

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

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

<a id="1">1</a>: On [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}), audit logs are not collected for:
- Open or Public projects.
- The `report:read` action.
- `User` actions which are not tied to a specific organization.