---
description: Run Weights and Biases on your own machines using Docker
displayed_sidebar: default
---

# Getting started

Follow this "Hello, world!" example to learn the general workflow to install W&B Server workflow for Dedicated Cloud and Self Managed hosting options. By the end of this demo, you will know how to host W&B Server on your local machine. 

For demonstration purposes, this demo uses a local development server on port 8080 (localhost:8080).

:::tip
**Trial Mode vs. Production Setup**

In Trial Mode of W&B Local, you run the Docker container on a single machine. This setup is ideal for testing the product, but it is not scalable.

For production work, set up a scalable file system to avoid data loss. We suggest you:
* allocate extra space in advance, 
* resize the file system proactively as you log more data
* configure external metadata and object stores for backup.

The instance will stop working if you run out of space. In this case, any additional data will be lost.
:::

## Prerequisites
Before you get started, ensure your local machine satisfies the following requirements: 

1. [Python](https://www.python.org) is installed
2. [Docker](https://www.docker.com) is running 
3. Install or upgrade the latest version of W&B:
   ```bash
   pip install --upgrade wandb
   ```
##  1. Pull the W&B Docker image

Run the following in your terminal:

```bash
wandb server start
```

The command pulls the latest W&B Docker image [`wandb/local`](https://hub.docker.com/r/wandb/local).


## 2. Create a W&B account
Navigate to `http://localhost:8080/signup` and create an initial user account.  Provide a name, email address, a username, and a password: 

![](/images/hosting/signup_localhost.png)

Click on the **Sign Up** button to create a W&B account. 

### Copy API key
After you create an account, navigate to `http://localhost:8080/authorize`.  

Copy the W&B API key that appears on the screen. You will need this key at a later step to verify your login credentials.

![](/images/hosting/copy_api_key.png)

## 3. Generate a license
Navigate to the W&B Deploy Manager at https://deploy.wandb.ai/deploy to generate a W&B license.

1. Select Docker as your provider
![](/images/hosting/deploy_manager_platform.png)
2. Click **Next**.
3. Select a license owner from the **Owner of license** dropdown.
![](/images/hosting/deploy_manager_info.png)
4. Click **Next**.
5. Provide a name for your license in the **Name of Instance** field.
6. (Optional) Provide a description about your license in the **Description** field. 
7. Click on the **Generate License Key** button.
![](/images/hosting/deploy_manager_generate.png)

When you generate a license key, you will be redirected to your Deployment License page. The URL will consist of https://deploy.wandb.ai/org/ and the unique Deployment ID that is generated for you when you create a license. 

## 4. Add license to Local Host
1. Within your Deployment License page, click on the **Copy License** button.
![](/images/hosting/deploy_manager_get_license.png)
2. Navigate to http://localhost:8080/system-admin/
3. Paste your license into to **License field**.
![](/images/hosting/License.gif)
4. Click on **Update settings**.

## 5. Check browser is running W&B App UI
Check that W&B is running on your local machine. Navigate to `http://localhost:8080/home`. You should see the W&B App UI in your browser.

![](/images/hosting/check_local_host.png)

## 6. Add programmatic access to your local W&B instance

1. Navigate to `http://localhost:8080/authorize` to obtain your API key.
2. Within your terminal, execute the following:
   ```bash
   wandb login --host=http://localhost:8080/
   ```
   If you are already logged into W&B with a different count, add the `relogin` flag:
   ```bash
   wandb login --relogin --host=http://localhost:8080
   ```
3. Paste your API key when prompted.

W&B appends a `localhost` profile and your API key to your .netrc profile at `/Users/username/.netrc` for future automatic logins.


<!-- 
## OLD
## 2. Log in to W&B

If this is your first time logging in then you will need to create your local W&B server account and authorize your API key. There are several ways to control where your runs are logged to. This is particularly useful if you run `wandb` on multiple machines or you switch between a private instance and W&B cloud. 

Follow the procedure below to send metrics to the shared private instance. Ensure you have already set up DNS:

1. Set the host flag to the address of the private instance whenever you login:

```bash
wandb login --host=http://wandb.your-shared-local-host.com
```

2. Set the environment variable `WANDB_BASE_URL` to the address of the local instance:

```bash
export WANDB_BASE_URL="http://wandb.your-shared-local-host.com"
```

In an automated environment, you can set the `WANDB_API_KEY`. Find your key at [wandb.your-shared-local-host.com/authorize](http://wandb.your-shared-local-host.com/authorize).

Set the host to `api.wandb.ai` to log to the public cloud instance of W&B:

```bash
wandb login --cloud
```

or

```bash
export WANDB_BASE_URL="https://api.wandb.ai"
```

You can also switch to your cloud API key, available at [https://wandb.ai/settings](https://wandb.ai/settings) when you are logged in to your cloud-hosted wandb account in your browser.

## 3. Generate a free license

You need a license to complete your configuration of a W&B server. [**Open the Deploy Manager** ](https://deploy.wandb.ai/deploy)to generate a free license. If you do not already have a cloud account then you will need to create one to generate your free license. You can generate either a personal or team or free license:

1. [**Personal licenses**](https://deploy.wandb.ai/deploy) are free forever for personal work: ![](/images/hosting/personal_license.png)
2. [**Team trial licenses**](https://deploy.wandb.ai/deploy) are free and last 30 days, allowing you to set up a team and connect a scalable backend: ![](/images/hosting/team_trial_license.png)

## 4. Add a license to your Local host

1. Copy your license from your Deployment and navigate back to your W&B server's localhost: ![](/images/hosting/add_license_local_host.png)
2. Add it to your local settings by pasting it into the `/system-admin` page of your localhost:
   ![](@site/static/images/hosting/License.gif)

## 5. Check for W&B Server updates

New versions of _wandb/local_ are pushed to DockerHub regularly. We suggest you keep your version up to date. To upgrade, copy and paste the following command into your terminal:

```shell
$ wandb server start --upgrade
```

Alternatively, you can upgrade your instance manually. Copy and paste the following code snippets into your terminal:

```shell
$ docker pull wandb/local
$ docker stop wandb-local
$ docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
```

## 6. (Optional) Mount a persistence volume 

- All metadata and files sent to W&B server are stored in the `/vol` directory. If you do not mount a persistent volume at this location all data will be lost when the docker process dies.
- This solution is not meant for [production](../hosting-options/intro.md) workloads.
- You can store metadata in an external MySQL database and files in an external storage bucket.
- The underlying file store should be resizable. Alerts should be put in place to let you know once minimum storage thresholds are crossed to resize the underlying file system.
- For enterprise trials, we recommend at least 100GB free space in the underlying volume for non-image/video/audio heavy workloads.


## 7. (Optional) Create and scale a shared instance

To enjoy the powerful collaborative features of W&B, you will need a shared instance on a central server, which you can [set up on AWS, GCP, Azure, Kubernetes, or Docker](../hosting-options/intro.md).

[Contact sales](https://wandb.ai/site/contact) to learn more about Enterprise options for W&B server.


## FAQ: How does wandb persist user account data?

When a Kubernetes instance is stopped, the W&B application bundles all the user account data into a tarball and uploads it to the Amazon S3 object store. W&B pulls previously uploaded tarball files when you restart an instance and provide the `BUCKET` environment variable. W&B will also load your user account information into the newly started Kubernetes deployment.

When an external object store is enabled, strong access controls should be enforced as it will contain all users data.
W&B persists instance settings in the external bucket when it is configured. W&B also persist certificates, and secrets in the bucket. -->