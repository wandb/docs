---
description: How to configure the W&B Server installation
menu:
  default:
    identifier: env-vars
    parent: w-b-platform
title: Configure environment variables
weight: 7
---

In addition to configuring instance level settings via the System Settings admin UI, W&B also provides a way to configure these values via code using Environment Variables. Also, refer to [advanced configuration for IAM](./iam/advanced_env_vars.md).

## Environment variable reference

| Environment Variable             | Description                                                                                                                                                                              |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | Your wandb/local license                                                                                                                                                                 |
| MYSQL                            | The MySQL connection string                                                                                                                                                              |
| BUCKET                           | The S3 / GCS bucket for storing data                                                                                                                                                     |
| BUCKET_QUEUE                     | The SQS / Google PubSub queue for object creation events                                                                                                                                 |
| NOTIFICATIONS_QUEUE              | The SQS queue on which to publish run events                                                                                                                                             |
| AWS_REGION                       | The AWS Region where your bucket lives                                                                                                                                                   |
| HOST                             | The FQD of your instance, that is `https://my.domain.net`                                                                                                       |
| OIDC_ISSUER                      | A URL to your Open ID Connect identity provider, that is `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd` |
| OIDC_CLIENT_ID                   | The Client ID of application in your identity provider                                                                                                                                   |
| OIDC_AUTH_METHOD                 | Implicit (default) or pkce, see below for more context                                                                                                                                   |
| SLACK_CLIENT_ID                  | The client ID of the Slack application you want to use for alerts                                                                                                                        |
| SLACK_SECRET                     | The secret of the Slack application you want to use for alerts                                                                                                                           |
| LOCAL_RESTORE                    | You can temporarily set this to true if you're unable to access your instance. Check the logs from the container for temporary credentials.                                              |
| REDIS                            | Can be used to setup an external REDIS instance with W&B.                                                                                                                                |
| LOGGING_ENABLED                  | When set to true, access logs are streamed to stdout. You can also mount a sidecar container and tail `/var/log/gorilla.log` without setting this variable.                              |
| GORILLA_ALLOW_USER_TEAM_CREATION | When set to true, allows non-admin users to create a new team. False by default.                                                                                                         |
| GORILLA_DATA_RETENTION_PERIOD | How long to retain deleted data from runs in hours. Deleted run data is unrecoverable. Append an `h` to the input value. For example, `"24h"`. |
| ENABLE_REGISTRY_UI               |  When set to true, enables the new W&B Registry UI.            |

{{% alert %}}
Use the GORILLA_DATA_RETENTION_PERIOD environment variable cautiously. Data is removed immediately once the environment variable is set. We also recommend that you backup both the database and the storage bucket before you enable this flag.
{{% /alert %}}

## Advanced Reliability Settings

### Redis

Configuring an external Redis server is optional but recommended for production systems. Redis helps improve the reliability of the service and enable caching to decrease load times, especially in large projects. Use a managed Redis service such ElastiCache with high availability (HA) and the following specifications:

- Minimum 4GB of memory, suggested 8GB
- Redis version 6.x
- In transit encryption
- Authentication enabled

To configure the Redis instance with W&B, you can navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. Enable the "Use an external Redis instance" option, and fill in the Redis connection string in the following format:

{{< img src="/images/hosting/configure_redis.png" alt="Configuring REDIS in W&B" >}}

You can also configure Redis using the environment variable `REDIS` on the container or in your Kubernetes deployment. Alternatively, you could also setup `REDIS` as a Kubernetes secret.

This page assumes the Redis instance is running at the default port of `6379`. If you configure a different port, setup authentication and also want to have TLS enabled on the `redis` instance the connection string format would look something like: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`