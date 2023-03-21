# Audit logs
Use audit logs to track and understand activity within your team. Audit logs are synced to your bucket store every 10 seconds. Optionally download your audit logs and view them with your tool of choice such as [Pandas](https://pandas.pydata.org/docs/index.html), [BigQuery](https://cloud.google.com/bigquery), and more. 

:::info
This feature is currently in Private Preview.
:::

## Audit log schema

| Key | Definition |
|---------| -------|
|timestamp | |
|action | |
|actor_user_id| |
|response_code | |
|user_asset | |
|project_asset | |
|run_asset||
|artifact_asset||


## View audit logs
View audit logs in the bucket that backs your W&B Server installation.

1. Navigate to your `/wandb-audit-logs` directory in your bucket.
2. Download the files for the period you are interested in.


One file is uploaded per day. The files contain new-line separated JSON objects. Objects written to the file will have fields described in the schema.

## Audit log schema


