---
description: Use model registry role based access controls (RBAC) to control who can update protected aliases.
displayed_sidebar: default
---

# Access control

Use model registry role based access controls (RBAC) to control who can update *protected aliases*. Protected aliases are the aliases you use to represent key stages of your model development pipeline. With RBAC you can:

1. Assign *Model Registry Admins*: Give specific members of your team admin permissions for the model registry. Model registry admins can define and use protected aliases, and non-admins are blocked from adding and removing protected aliases from model versions.
2. Create *Protected Aliases*: Define a list of aliases that only admins can manage. For example, you might use the `production` alias to indicate a model is deployed.



For example, suppose you set `staging` and `production` as protected aliases. Any member of your team can add new model versions. These newly added models are assigned a `latest` alias.  However, only admins can add a `staging` or `production` alias.




## Set up access control
The following steps describe how to set up access controls for your team’s model registry.

:::note
Only Team admins or current registry admins can manage the list of registry admins.
:::


1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select the gear button on the top right of the page.
![](/images/models/rbac_gear_button.png)
3. Select the **Manage registry** admins button. Add a set of people who should be allowed to add and remove protected aliases from model versions.
![](/images/models/access_controls_admins.gif)


## Add protected aliases
1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select the gear button on the top right of the page.
![](/images/models/rbac_gear_button.png)
3. Scroll down to the **Protected Aliases** section.
4. Click on the plus icon (**+**) icon to add new a new alias.
![](/images/models/access_controls_add_protected_aliases.gif)
