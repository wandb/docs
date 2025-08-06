---
description: Use model registry role based access controls (RBAC) to control who can
  update protected aliases.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-access_controls
    parent: model-registry
title: Manage data governance and access control
weight: 10
---

Use *protected aliases* to represent key stages of your model development pipeline. Only *Model Registry Administrators* can add, modify, or remove protected aliases. Model registry admins can define and use protected aliases. W&B blocks non admin users from adding or removing protected aliases from model versions.

{{% alert %}}
Only Team admins or current registry admins can manage the list of registry admins.
{{% /alert %}}

For example, suppose you set `staging` and `production` as protected aliases. Any member of your team can add new model versions. However, only admins can add a `staging` or `production` alias.


## Set up access control
The following steps describe how to set up access controls for your team’s model registry.

1. Navigate to the [W&B Model Registry app](https://wandb.ai/registry/model).
2. Select the gear button on the top right of the page.
{{< img src="/images/models/rbac_gear_button.png" alt="Registry settings gear" >}}
3. Select the **Manage registry admins** button. 
4. Within the **Members** tab, select the users you want to grant access to add and remove protected aliases from model versions.
{{< img src="/images/models/access_controls_admins.gif" alt="Managing registry admins" >}}


## Add protected aliases
1. Navigate to the [W&B Model Registry app](https://wandb.ai/registry/model).
2. Select the gear button on the top right of the page.
{{< img src="/images/models/rbac_gear_button.png" alt="Registry settings gear button" >}}
3. Scroll down to the **Protected Aliases** section.
4. Click on the plus icon (**+**) icon to add new a new alias.
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="Adding protected aliases" >}}