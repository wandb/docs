---
description: Run Weights and Biases on your own machines using Docker
---

# Basic Setup

Run Weights and Biases on your own machines using Docker.

### Installation

- On any machine with [Docker](https://www.docker.com) and [Python](https://www.python.org) installed, run:

```
pip install wandb
wandb server start
```

### Login

If this is your first time logging in then you will need to create your local W&B server account and authorize your API key.

If you're running `wandb` on multiple machines or switching between a private instance and the wandb cloud, there are several ways to control where your runs are logged. If you want to send metrics to the shared private instance and you've configured DNS, you can

- set the host flag to the address of the private instance whenever you login:

```
 wandb login --host=http://wandb.your-shared-local-host.com
```

- set the environment variable `WANDB_BASE_URL` to the address of the local instance:

```python
export WANDB_BASE_URL = "http://wandb.your-shared-local-host.com"
```

In an automated environment, you can set the `WANDB_API_KEY` which is accessible at [wandb.your-shared-local-host.com/authorize](http://wandb.your-shared-local-host.com/authorize).

To switch to logging to the public **cloud** instance of wandb, set the host to `api.wandb.ai`:

```
wandb login --cloud
```

or

```python
export WANDB_BASE_URL = "https://api.wandb.ai"
```

You can also switch to your cloud API key, available at [https://wandb.ai/settings](https://wandb.ai/settings) when you're logged in to your cloud-hosted wandb account in your browser.

### Generate a free license

You need a license to complete your configuration of a W&B server. [**Open the Deploy Manager** ](https://deploy.wandb.ai/deploy)to generate a free license. If you do not already have a cloud account then you will need to create one to generate your free license. We offer two options:

1. [**Personal licenses ->**](https://deploy.wandb.ai/deploy) are free forever for personal work: ![](/images/hosting/personal_license.png)
2. [**Team trial licenses ->**](https://deploy.wandb.ai/deploy) are free and last 30 days, allowing you to set up a team and connect a scalable backend: ![](/images/hosting/team_trial_license.png)

### Add a license to your Local host

1. Copy your license from your Deployment and navigate back to your W&B server's localhost: ![](/images/hosting/add_license_local_host.png)
2. Add it to your local settings by pasting it into the `/system-admin` page of your localhost:\
   ![](@site/static/images/hosting/License.gif)

### Upgrades

We are pushing new versions of _wandb/local_ to DockerHub regularly. To upgrade you can run:

```shell
$ wandb server start --upgrade
```

To upgrade your instance manually you can run the following

```shell
$ docker pull wandb/local
$ docker stop wandb-local
$ docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
```

### Persistence and Scalability

- All metadata and files sent to W&B server are stored in the `/vol` directory. If you do not mount a persistent volume at this location all data will be lost when the docker process dies.
- This solution is not meant for [production](/guides/hosting/hosting-options) workloads.
- You can store metadata in an external MySQL database and files in an external storage bucket.
- The underlying file store should be resizable. Alerts should be put in place to let you know once minimum storage thresholds are crossed to resize the underlying file system.
- For enterprise trials, we recommend at least 100GB free space in the underlying volume for non-image/video/audio heavy workloads.

#### How does wandb persist user account data?

When a Kubernetes instance is stopped, the wandb application bundles all the user account data into a tarball and uploads it to the S3 object store. On restarting the instance and providing the `BUCKET` environment variable, wandb pulls that previously uploaded tarball and loads the user account info into the newly started Kubernetes deployment.

Wandb persists instance settings in the external bucket when it's configured. We also persist certificates, and secrets in the bucket but should be moving those into proper secret stores or at least adding a layer of encryption. When an external object store is enabled, strong access controls should be enforced as it will contain all users data

#### Create and scale a shared instance

To enjoy the powerful collaborative features of W&B, you will need a shared instance on a central server, which you can [set up on AWS, GCP, Azure, Kubernetes, or Docker](/guides/hosting/hosting-options).

:::warning
**Trial Mode vs. Production Setup**

In Trial Mode of W&B Local, you're running the Docker container on a single machine. This setup is quick and painless, and it's great for testing the product, but it isn't scalable.

Once you're ready to move from test projects to real production work, it is crucial that you set up a scalable file system to avoid data loss: allocate extra space in advance, resize the file system proactively as you log more data, and configure external metadata and object stores for backup. If you run out of disk space, the instance will stop working, and additional data will be lost.
:::

[**Contact sales -**](https://wandb.ai/site/local-contact)**>** to learn more about Enterprise options for W&B server.
