# Audit logs
Use audit logs to track and understand activity within your team. Audit logs are synced to your bucket store every 10 seconds. Optionally download your audit logs and view them with your tool of choice such as [Pandas](https://pandas.pydata.org/docs/index.html), [BigQuery](https://cloud.google.com/bigquery), and more. 

:::info
This feature is currently in Private Preview.
:::

## Audit log schema

| Key | Definition |
|---------| -------|
|timestamp | Time stamp in [RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339). For example: `2023-01-23T12:34:56Z`, represents `12:34:56 UTC` time on Jan 23, 2023.|
|action | What [action](#actions) did the user take. |
|actor_user_id| If present, ID of the logged in user who performed the action. |
|response_code |Http response code for the action. |
|user_asset | If present, action returned this user asset. |
|project_asset | If present, action returned this project asset. |
|run_asset|If present, action returned this run asset. |
|artifact_asset|If present, action returned this artifact asset.|


## View audit logs
View audit logs in the bucket that backs your W&B Server installation.

1. Navigate to your `/wandb-audit-logs` directory in your bucket.
2. Download the files for the period you are interested in.


One file is uploaded per day. The files contain new-line separated JSON objects. Objects written to the file will have fields described in the schema.


## Actions
The following table describes possible actions that can be recorded by W&B:

|Action | Definition |
|-----|-----|
|||
|"artifact:create" | |
|"artifact:read" | |
|"artifact:read" | |
|"project:delete"  | |
|"project:read" | |
|"project:read"  | |
|"report:read" | |
|"run:delete" | |
|"run:delete_many" | |
|"run:update_many" | |
|"run:stop" | |
|"run:undelete_many" | |
|"run:update" | |
|"sweep:create_agent" | |
|"team:invite_user" | |
|"team:create_service_account" | |
|"team:create" | |
|"team:uninvite" | |
|"team:delete" | |
|"user:create" | |
|"user:delete_api_key" | |
|"user:deactivate" | |
|"user:create_api_key" | |
|"user:permanently_delete" | |
|"user:reactivate" | |
|"user:update" | |
|"user:read" | |
|"user:login" | |
|"user:initiate_login" | |
|"user:logout" | |

