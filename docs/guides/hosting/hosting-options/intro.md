---
slug: /guides/hosting/hosting-options
description: Deploying W&B in production
displayed_sidebar: default
---

# W&B Server Hosting Options

There are three ways to deploy W&B Server:

- **W&B managed Dedicated Cloud**: A fully managed solution hosted and maintained by W&B on W&B Cloud
- **Customer managed OnPrem Private Cloud**: A self managed solution hosted and maintained by customer on their Private Cloud
- **Customer managed OnPrem Bare Metal**: A self managed solution hosted and maintained by customer on their Bare Metal infrastructure

## Shared Responsibility Matrix

The following shared responsibility matrix outlines the respective responsibilities of W&B and the customer for each of the hosting options specified above.

![](/images/hosting/shared_responsibility_matrix.png)

## Obtain your license

You need a W&B license to complete your configuration of a W&B server. Open the [Deploy Manager](https://deploy.wandb.ai/deploy) to generate a free license. 

:::note
If you do not already have a cloud W&B account then you will need to create one to generate your free license.
:::

The URL will redirect you to a **Get a License for W&B Local** form. Provide the following information:

1. Choose a deployment type from the **Choose Platform** step.
2. Select the owner of the license or add a new organization in the **Basic Information** step.
3. Provide a name for the instance in the **Name of Instance** field and optionally provide a description in the **Description** field in the **Get a License** step.
4. Select the **Generate License Key** button.

A page with an overview of your deployment along with licenses associated to the instance will be displayed.

For information on how to set up your deployment type, see [our How-to guides](../how-to-guides/intro.md) section.