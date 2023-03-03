---
description: How to configure the W&B Local Server installation
---

# Advanced Configuration

W&B server starts ready-to-use on boot using `wandb server start`. However, several advanced configuration options are available using the `/system-admin` page on your server once it's up and running. You can email [contact@wandb.com](mailto:contact@wandb.com) to request a trial license to enable more users and teams.

The following is detailed information about the advanced configuration of a local server. When possible we suggest you use our [existing Terraform](https://github.com/wandb/local) to configure your instance.

## Configuration as code

All configuration settings can be set via the UI however if you would like to manage these configuration options via code you can set the following environment variables:

| Environment Variable | Description                                                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LICENSE              | Your wandb/local license                                                                                                                                                                   |
| MYSQL                | The MySQL connection string                                                                                                                                                                |
| BUCKET               | The S3 / GCS bucket for storing data                                                                                                                                                       |
| BUCKET\_QUEUE        | The SQS / Google PubSub queue for object creation events                                                                                                                                   |
| NOTIFICATIONS\_QUEUE | The SQS queue on which to publish run events                                                                                                                                               |
| AWS\_REGION          | The AWS Region where your bucket lives                                                                                                                                                     |
| HOST                 | The FQD of your instance, i.e. [https://my.domain.net](https://my.domain.net)                                                                                                              |
| OIDC\_ISSUER         | A url to your Open ID Connect identity provider, i.e. [https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd](https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd) |
| OIDC\_CLIENT\_ID     | The Client ID of application in your identity provider                                                                                                                                     |
| OIDC\_AUTH\_METHOD   | Implicit (default) or pkce, see below for more context                                                                                                                                     |
| SLACK\_CLIENT\_ID    | The client ID of the Slack application you want to use for alerts                                                                                                                          |
| SLACK\_SECRET        | The secret of the Slack application you want to use for alerts                                                                                                                             |
| LOCAL\_RESTORE       | You can temporarily set this to true if you're unable to access your instance. Check the logs from the container for temporary credentials.                                                |
| REDIS                | Can be used to setup an external REDIS instance with W&B.                                                                                                                                 |
| LOGGING\_ENABLED     | When set to true, access logs are streamed to stdout. You can also mount a sidecar container and tail `/var/log/gorilla.log` without setting this variable.                                |

### Host Configuration

To change the host and port that you want to deploy your `wandb server` instance then you can run the command

`wandb server -e HOST=http://<HOST>:<PORT>`

You can connect to this instance by then explicitly defining the HOST for our authentication method for `wandb` client. Here are various ways to perform this action.

1. `wandb login --host=<HOST>:<PORT>`
2. `wandb.login(host="<HOST>:<PORT>")`
3. `export WANDB_BASE_URL=<HOST>:<PORT>`\
   `export WANDB_API_KEY=<API-KEY>`

## SSO & Authentication

By default, a W&B server runs with manual user management. Licensed versions of _wandb/local_ also unlock SSO. Email [contact@wandb.com](mailto:contact@wandb.com) to schedule a time with us to configure an [Auth0](https://auth0.com) tenant for you with any Identity provider they support such as SAML, Ping Federate, Active Directory, etc.

If you already use Auth0 or have an Open ID Connect compatible server, you can follow the instructions below.

### Open ID Connect

_wandb/local_ uses Open ID Connect for authentication. When creating an application client in your IDP you should choose Single Page Application or Public Client.

#### Setting up with AWS Cognito

![Because we're only using OIDC for authentication and not authorization, public clients simplify setup](/images/hosting/setup_aws_cognito.png)

To configure an application client in your identity provider you'll need to provide an allowed callback url:

* Add the following allowed Callback URL `http(s)://YOUR-W&B-HOST/oidc/callback`
* If your IDP supports universal logout, set Logout URL to `http(s)://YOUR-W&B-HOST`

For example, in [AWS Cognito](https://aws.amazon.com/cognito/) if your application was running at `https://wandb.mycompany.com`:

![If your instance is accessible from multiple hosts, be sure to include all of them here.](/images/hosting/setup_aws_cognito_ui_settings.png)

_wandb/local_ will use the ["implicit" grant with the "form\_post" response type](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) by default. You can also configure _wandb/local_ to perform an "authorization\_code" grant using the [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) flow. We request the following scopes for the grant: "openid", "profile", and "email". Your identity provider will need to allow these scopes. For example in AWS Cognito the application should look like:

![openid, profile, and email are required](/images/hosting/setup_aws_required_fields.png)

To tell _wandb/local_ which grant to use you can select the Auth Method in the settings page or set the OIDC\_AUTH\_METHOD environment variable.

:::info
For AWS Cognito providers you must set the Auth Method to "pkce"
:::

You'll need a Client ID and the url of your OIDC issuer. The OpenID discovery document must be available at `$OIDC_ISSUER/.well-known/openid-configuration` For example, when using AWS Cognito you can generate your issuer url by appending your User Pool ID to the Cognito IDP url from the _User Pools > App Integration_ tab:

![The issuer URL would be https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
Do not use the "Cognito domain" for the IDP url. Cognito provides it's discovery document at `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID`
:::

Once you have everything configured you can provide the Issuer, Client ID, and Auth method to _wandb/local_ via `/system-admin` or the environment variables and SSO will be configured.

![](/images/hosting/enable_sso.png)

#### Setting up with Okta

First set up a new application by navigating in your provider's UI, Click on Add apps

![](/images/hosting/okta.png)

Name your App Integration (ex: Weights & Biases) and select grant type `implicit (hybrid)`

W&B also supports the Authorization Code grant type with PKCE

![](/images/hosting/pkce.png)

To configure an application client in your identity provider you'll need to provide an allowed callback url:

* Add the following allowed Callback URL `http(s)://YOUR-W&B-HOST/oidc/callback`
* If your IDP supports universal logout, set Logout URL to `http(s)://YOUR-W&B-HOST`

For example, if your application was running at `https://localhost:8080`,
the redirect URI would look like `https://localhost:8080/oidc/callback`![](/images/hosting/redirect_uri.png)

Set the sign-out redirect to `http(s)://YOUR-W&B-HOST/logout`

![](/images/hosting/signout_redirect.png)

Once you have everything configured you can provide the Issuer, Client ID, and Auth method to `wandb/local` via `/system-admin` or the environment variables and SSO will be configured.

Sign in to your Weights and Biases server and navigate to the `System Settings` page. Navigate to upper-right, dropdown menu:

![](/images/hosting/system_settings.png)

Next, select **System Settings**

![](/images/hosting/system_settings_select_settings.png)

Enter your Issuer, Client ID, and Authentication Method. Select **Update settings**.

![](/images/hosting/system_settings_select_update.png)

:::info
If you're unable to login to your instance after configuring SSO, you can restart the instance with the `LOCAL_RESTORE=true` environment variable set. This will output a temporary password to the containers logs and disable SSO. Once you've resolved any issues with SSO, you must remove that environment variable to enable SSO again.
:::

## File Storage

By default, a W&BEnterprise Server saves files to a local data disk with a capacity that you set when you provision your instance. To support limitless file storage, you may configure your server to use an external cloud file storage bucket with an S3-compatible API.

:::info
You should always specify the bucket you're using with the BUCKET environment variable. This removes the need for a persistent volume as all settings can then be persisted to your bucket.
:::

### Amazon Web Services

To use an AWS S3 bucket as the file storage backend for W&B, you'll need to create a bucket, along with an SQS queue configured to receive object creation notifications from that bucket. Your instance will need permissions to read from this queue.

**Create an S3 Bucket and Bucket Notifications**

Then, create an S3 bucket. Under the bucket properties page in the console, in the "Events" section of "Advanced Settings", click "Add notification", and configure all object creation events to be sent to the SQS Queue you configured earlier.

![Enterprise file storage settings](@site/static/images/hosting/s3-notification.png)

Enable CORS access: your CORS configuration should look like the following:

```markup
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

**Create an SQS Queue**

First, create an SQS Standard Queue. Add a permission for all principals for the `SendMessage`, `ReceiveMessage`, `ChangeMessageVisibility`, `DeleteMessage`, and `GetQueueUrl` actions. If you'd like you can further lock this down using an advanced policy document. For instance, the policy for accessing SQS with a statement is as follows:

```json
{
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : "*",
        "Action" : ["sqs:SendMessage"],
        "Resource" : "<sqs-queue-arn>",
        "Condition" : {
          "ArnEquals" : { "aws:SourceArn" : "<s3-bucket-arn>" }
        }
      }
    ]
}
```

**Grant Permissions to Node Running W&B**

The node on which W&B server is running must be configured to permit access to S3 and SQS. Depending on the type of server deployment you've opted for, you may need to add the following policy statements to your node role:

```
{
   "Statement":[
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":"s3:*",
         "Resource":"arn:aws:s3:::<WANDB_BUCKET>"
      },
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":[
            "sqs:*"
         ],
         "Resource":"arn:aws:sqs:<REGION>:<ACCOUNT>:<WANDB_QUEUE>"
      }
   ]
}
```

**Configure W&B server**

Finally, navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. Enable the "Use an external file storage backend" option, and fill in the s3 bucket, region, and SQS queue in the following format:

* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS only)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

![](/images/hosting/configure_file_store.png)

Press "Update settings" to apply the new settings.

### Google Cloud Platform

To use a GCP Storage bucket as a file storage backend for W&B, you'll need to create a bucket, along with a PubSub topic and subscription configured to receive object creation messages from that bucket.

**Create PubSub Topic and Subscription**

Navigate to Pub/Sub > Topics in the GCP Console, and click "Create topic". Choose a name and create a topic.

Then click "Create subscription" in the subscriptions table at the bottom of the page. Choose a name, and make sure Delivery Type is set to "Pull". Click "Create".

Make sure the service account or account that your instance is running as has access to this subscription.

**Create Storage Bucket**

Navigate to Storage > Browser in the GCP Console, and click "Create bucket". Make sure to choose "Standard" storage class.

Make sure the service account or account that your instance is running as has access to this bucket.

**Create PubSub Notification**

Creating a notification stream from the Storage Bucket to the PubSub Topic can unfortunately only be done in the console. Make sure you have `gsutil` installed, and logged into the correct GCP Project, then run the following:

```bash
gcloud pubsub topics list  # list names of topics for reference
gsutil ls                  # list names of buckets for reference

# create bucket notification
gsutil notification create -t <TOPIC-NAME> -f json gs://<BUCKET-NAME>
```

[Further reference is available on the Cloud Storage website.](https://cloud.google.com/storage/docs/reporting-changes)

**Add Signing Permissions**

To create signed file URLs, your W&B server also needs the `iam.serviceAccounts.signBlob` permission in GCP. You can add it by adding the `Service Account Token Creator` role to the service account or IAM member that your instance is running as.

**Grant Permissions to Node Running W&B server**

The node on which W&B server is running must be configured to permit access to S3 and SQS. Depending on the type of server deployment you've opted for, you may need to add the following policy statements to your node role:

```
{
   "Statement":[
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":"s3:*",
         "Resource":"arn:aws:s3:::<WANDB_BUCKET>"
      },
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":[
            "sqs:*"
         ],
         "Resource":"arn:aws:sqs:<REGION>:<ACCOUNT>:<WANDB_QUEUE>"
      }
   ]
}
```

**Configure W&B server**

Finally, navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. Enable the "Use an external file storage backend" option, and fill in the s3 bucket, region, and SQS queue in the following format:

* **File Storage Bucket**: `gs://<bucket-name>`
* **File Storage Region**: blank
* **Notification Subscription**: `pubsub:/<project-name>/<topic-name>/<subscription-name>`

![](/images/hosting/configure_file_store.png)

Press "update settings" to apply the new settings.

### Azure

To use an Azure blob container as the file storage for W&B, you'll need to create a storage account (if you don't already have one you want to use), create a blob container and a queue within that storage account, and then create an event subscription that sends "blob created" notifications to the queue from the blob container.

#### Create a Storage Account

If you have a storage account you want to use already, you can skip this step.

Navigate to [Storage Accounts > Add ](https://portal.azure.com/#create/Microsoft.StorageAccount)in the Azure portal. Select an Azure subscription, and select any resource group or create a new one. Enter a name for your storage account.

![Azure storage account setup](/images/hosting/azure_create_storage_account.png)

Click Review and Create, and then, on the summary screen, click Create:

![Azure storage account details review](/images/hosting/azure_create_storage_account_click_create.png)

#### Creating the blob container

Go to [Storage Accounts](https://portal.azure.com/#blade/HubsExtension/BrowseResource/resourceType/Microsoft.Storage%2FStorageAccounts) in the Azure portal, and click on your new storage account. In the storage account dashboard, click on Blob service > Containers in the menu:

![](/images/hosting/create_blob_container_1.png)

Create a new container, and set it to Private:

![](/images/hosting/create_blob_container_2.png)

Go to Settings > CORS > Blob service, and enter the IP of your wandb server as an allowed origin, with allowed methods `GET` and `PUT`, and all headers allowed and exposed, then save your CORS settings.

![](/images/hosting/create_blob_container_3.png)

#### Creating the Queue

Go to Queue service > Queues in your storage account, and create a new Queue:

![](/images/hosting/create_blob_container_4.png)

Go to Events in your storage account, and create an event subscription:

![](/images/hosting/create_blob_container_5.png)

Give the event subscription the Event Schema "Event Grid Schema", filter to only the "Blob Created" event type, set the Endpoint Type to Storage Queues, and then select the storage account/queue as the endpoint.

![](/images/hosting/create_blob_container_6.png)

In the Filters tab, enable subject filtering for subjects beginning with `/blobServices/default/containers/your-blob-container-name/blobs/`

![](/images/hosting/create_blob_container_7.png)

#### Configure W&B server

Go to Settings > Access keys in your storage account, click "Show keys", and then copy either key1 > Key or key2 > Key. Set this key on your W&B server as the environment variable `AZURE_STORAGE_KEY`.

![](/images/hosting/create_blob_container_8.png)

Finally, navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. Enable the "Use an external file storage backend" option, and fill in the s3 bucket, region, and SQS queue in the following format:

* **File Storage Bucket**: `az://<storage-account-name>/<blob-container-name>`
* **Notification Subscription**: `az://<storage-account-name>/<queue-name>`

![](/images/hosting/create_blob_container_9.png)

Press "Update settings" to apply the new settings.

### Secure Storage Connector
The team-level secure storage connector allows teams within W&B to utilize a separate cloud file storage bucket from the rest of the W&B instance. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements.

:::info
This feature is currently only available for Google Cloud Platform and Amazon Web Services. To request a license to enable this feature, email contact@wandb.com.
:::

A cloud storage bucket can be configured only once for a team at the time of team creation. Select **External Storage** when you create a team tp configure a cloud storage bucket. You can configure a cloud storage bucket once the bucket is provisioned. Select your provider and fill out your bucket name and storage encryption key ID, if applicable, and select **Create team bucket**.

An error or warning will appear at the bottom of the page if there are issues accessing the bucket or the bucket has invalid settings.

![](/images/hosting/prod_setup_secure_storage.png)

Only system administrators have the permissions to configure the secure storage connector. The same cloud storage bucket can be used amongst multiple teams by selecting an existing cloud storage bucket from the dropdown.

## Advanced Reliability Settings

#### Redis

Configuring an external redis server will improve the reliability of the service and enable caching which will decrease load times especially in large projects. We recommend using a managed redis service (ex: ElastiCache) with HA and the following specs:

* Minimum 4GB of memory, suggested 8GB
* Redis version 6.x
* In transit encryption
* Authentication enabled

#### Configuring REDIS in the W&B server

To configure the redis instance with W&B, you can navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. Enable the "Use an external Redis instance" option, and fill in the `redis` connection string in the following format:

![Configuring REDIS in W&B](/images/hosting/configure_redis.png)

You can also configure `redis` using the environment variable `REDIS` on the container or in your Kubernetes deployment. Alternatively, you could also setup `REDIS` as a Kubernetes secret.

The above assumes the `redis` instance is running at the default port of `6379`. If you configure a different port, setup authentication and also want to have TLS enabled on the `redis` instance the connection string format would look something like: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`

## Slack

In order to integrate your W&B server installation with Slack, you'll need to create a suitable Slack application.

#### Creating the Slack application

Visit [https://api.slack.com/apps](https://api.slack.com/apps) and select **Create New App** in the top right.

![](/images/hosting/create_slack_ap_1.png)

You can name it whatever you like, but what's important is to select the same Slack workspace as the one you intend to use for alerts.

![](/images/hosting/create_slack_ap_2.png)

#### Configuring the Slack application

Now that we have a Slack application ready, we need to authorize for use as an OAuth bot. Select **OAuth & Permissions** in the sidebar to the left.

![](/images/hosting/create_slack_ap_3.png)

Under **Scopes**, supply the bot with the **incoming\_webhook** scope.

![](/images/hosting/create_slack_ap_4.png)

Finally, configure the **Redirect URL** to point to your W&B installation. You should use the same value as what you set **Frontend Host** to in your local system settings. You can specify multiple URLs if you have different DNS mappings to your instance.

![](/images/hosting/create_slack_ap_5.png)

Hit **Save URLs** once finished.

To further secure your Slack application and prevent abuse, you can specify an IP range under **Restrict API Token Usage**, whitelisting the IP or IP range of your W&B instance(s).

#### Register your Slack application with W&B

Navigate to the **System Settings** page of your W&B instance. Check the box to enable a custom Slack application:

![](/images/hosting/create_slack_ap_6.png)

You'll need to supply your Slack application's client ID and secret, which you can find in the **Basic Information** tab.

![](/images/hosting/create_slack_ap_7.png)

That's it! You can now verify that everything is working by setting up a Slack integration in the W&B app. Visit [this page](https://docs.wandb.ai/guides/track/alert) for more detailed information.
