---
title: Configuring W&B Enterprise Server
sidebar_label: Configuration
---

Your W&B Enterprise Server comes up ready-to-use on boot. However several advanced configuration options are available, and accessible from the `/vm-settings` page on your server once it's up and running.

## Authentication

By default, a W&B Enterprise Server runs in "single-user mode": right from booting up, you can log in and start sending data to your server. However, to unlock the full sharing functionality of W&B, you should configure authentication.

Your server supports any authentication provider supported by [Auth0](https://auth0.com). You should set up your own Auth0 domain and application that will be under your teams' control.

After creating an Auth0 app, you'll need to configure your Auth0 callbacks to the host of your W&B Server. By default, the server supports http from the public or private IP address provided by the host. You can also configure a DNS hostname and SSL certificate if you choose.

* Set the Callback URL to `http(s)://YOUR-W&B-SERVER-HOST`
* Set the Allowed Web Origin to `http(s)://YOUR-W&B-SERVER-HOST`
* Set the Logout URL to `http(s)://YOUR-W&B-SERVER-HOST/logout`

![Auth0 Settings](/img/auth0-1.png)

Save the Client ID and domain from your Auth0 app.

![Auth0 Settings](/img/auth0-2.png)

![Enterprise authentication settings](/img/enterprise-auth.png)

Then, navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/vm-settings`. Enable the "Customize Authentication with Auth0" option, and fill in the Client ID and domain from your Auth0 app. The press "Update settings and restart W&B".

## File Storage

By default, a W&B Enterprise Server saves files to a local data disk with a capacity that you set when you provision your instance.

To support limitless file storage, you may configure your server to use an external cloud file storage bucket with an S3-compatible API.

### Amazon Web Services

#### Set up SQS Queue, S3 Bucket, and Bucket Notifications

First, create an SQS Standard Queue. Make sure to add a permission for Amazon S3 to post to that queue, and EC2 to read from the queue.

![Enterprise file storage settings](/img/sqs-perms.png)

Then, create an S3 bucket. Under the bucket properties page in the console, in the "Events" section of "Advanced Settings", click "Add notification", and configure all object creation events to be sent to the SQS Queue you configured earlier.

![Enterprise file storage settings](/img/s3-notification.png)

Then, navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/vm-settings`. Enable the "Use an external file storage backend" option, and fill in the s3 bucket, region, and SQS queue in the following format:

* **File Storage URL**: `s3:///<bucket-name>`
* **File Storage Region**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

![AWS file storage settings](/img/aws-filestore.png)

Press "update settings and restart W&B" to apply the new settings.

### Google Cloud Platform

#### Set up Pubsub Topic and Subscription, Storage Bucket, and Bucket Notifications

For AWS, your standard endpoint will be `http://s3.YOUR-REGION.amazonaws.com`. For GCP, it will be `https://storage.googleapis.com`. You can configure S3-compatible keys for Google Cloud Storage by [following the instructions here](https://cloud.google.com/storage/docs/migrating#keys).
