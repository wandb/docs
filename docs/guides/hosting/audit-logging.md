# Audit logs
Use audit logs to track and understand activity within your team. Audit logs sync to your bucket store every 10 seconds. Optionally, download your audit logs and view them with your preferred tool, such as [Pandas](https://pandas.pydata.org/docs/index.html), [BigQuery](https://cloud.google.com/bigquery), and more.

:::info
This feature is currently in Private Preview.
:::

## Audit log schema
The following table lists all the different keys that might be present in your audit logs. Each log contains only the assets relevant to the corresponding action, and others are omitted from the log.

| Key | Definition |
|---------| -------|
|timestamp               | Time stamp in [RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339). For example: `2023-01-23T12:34:56Z`, represents `12:34:56 UTC` time on Jan 23, 2023.
|action                  | What [action](#actions) did the user take.
|actor_user_id           | If present, ID of the logged in user who performed the action.
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
|artifact_sequence_name  | if present, action was taken on this artifact sequence name.
|entity_name             | if present, action was taken on this entity or team name.
|project_name            | if present, action was taken on this project name.
|report_name             | if present, action was taken on this report name.
|user_email              | if present, action was taken on this user email.

Personally identifiable information (PII) like email ids, project, team and report names are returned only by the endpoint, and can be turned off as [described below](#view-audit-logs).

## View audit logs
To view the audit logs for your W&B server instance, follow these steps:
1. Admin users can go to `<wandb-server-url>/admin/audit_logs`
2. Pass in the following URL parameters:
    - `numDays` : logs will be fetch starting from `today - numdays` to most recent; defaults to `0`
    - `anonymize` : if set to `true`, remove any PII; defaults to `false`

Note that only W&B server admins are allowed to request this information. If you are not an admin you will an authentication error. The response contains new-line separated JSON objects. Objects will have fields described in the schema.

All historical audit logs are stored in the storage bucket that backs your W&B Server installation. One file is uploaded per day. The files contain new-line separated JSON objects. These logs have the same format as the ones returned by the end points, except that they do not contain any PII for security reasons.

To view your historical audit logs, complete the following steps:

1. Navigate to your `/wandb-audit-logs` directory in your bucket.
2. Download the files for the period you are interested in.


## Actions
The following table describes possible actions that can be recorded by W&B:

|Action | Definition |
|-----|-----|
| artifact:create             | Artifact is created.
| artifact:delete             | Artifact is deleted.
| artifact:read               | Aritfact is read.
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

