---
description: Use model registry role based access controls (RBAC) to control who can update protected aliases.
displayed_sidebar: default
---

# Access control

Use model registry role based access controls (RBAC) to control who can update *protected aliases*. Protected aliases are the aliases you use to represent key stages of your model development pipeline.

## Example use case
For example, you could set the aliases `staging` and `production` as Protected Aliases. In this scenario, any member of your team could add a new model version. This would automatically create an alias marked `latest`. However, only admins could then move that model into `staging` or `production`.

## Features
The access controls add two new features:
1. **Model Registry Admins**: Give specific members of your team admin permissions for the model registry. These admins can define and use protected aliases, and non-admins will be blocked from adding and removing protected aliases from model versions (both in the UI and programatically).
2. **Protected Aliases**: Define a list of aliases that only admins can manage. For example, you might use the **production** alias to indicate a model being deployed, and only a few members of your team should be able to make that decision.
 


## How to set up access control
The following steps describe how to set up access controls for your team’s model registry.

1. [Open Access Controls](https://wandb.ai/registry/model/access-control) from your Models page.
![](/images/models/access_controls_button.gif)
2. **Manage registry admins**: Add a set of people who should be allowed to add and remove protected aliases from model versions.
![](/images/models/access_controls_admins.gif)
3. **Add protected aliases**: Set the list of aliases that only admins can update. For example, you might use the `production` alias to represent a special state only admins can use.
![](/images/models/access_controls_add_protected_aliases.gif)
