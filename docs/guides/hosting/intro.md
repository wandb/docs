---
slug: /guides/hosting
---

# Private Hosting

## W&BHosting Options

:::info
We recommend that you consider using the wandb.ai cloud before privately hosting a W&B server on your infrastructure. The cloud is simple and secure, with no configuration required. Click [here](https://docs.wandb.ai/quickstart) to learn more.
:::

There are three main ways to set up a W&B server in a production environment:

1. [Production Cloud](setup/private-cloud.md): Set up a production deployment on a private cloud in just a few steps using terraform scripts provided by W&B.
2. [Dedicated Cloud](setup/dedicated-cloud.md): A managed, dedicated deployment on W&B's single-tenant infrastructure in your choice of cloud region.
3. [On-Prem / Bare Metal](setup/on-premise-baremetal.md): W&B supports setting up a production server on most bare metal servers in your on-premise data centers. Quickly get started by running `wandb server` to easily start hosting W&Bon your local infrastructure.

## W&B server Quickstart

1.  On any machine with [Docker](https://www.docker.com) and [Python](https://www.python.org) installed, run:

    ```
    pip install wandb
    wandb server start 
    ```
2. Generate a free license from the [Deployer](https://deploy.wandb.ai/).
3. Add it to your local settings.

Paste the license in the `/system-admin` page on your localhost

![Copy your license from Deployer and paste it into your Local settings](@site/static/images/hosting/License.gif)
