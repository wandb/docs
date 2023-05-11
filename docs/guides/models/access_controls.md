---
description: Use model registry role based access controls (RBAC) to control who can update protected aliases.
---

# Access control

Use model registry role based access controls (RBAC) to control who can update protected aliases — these are the aliases you use to represent key stages of your model development pipeline.

## Example use case
For example, you could set the aliases `staging` and `production` as Protected Aliases. In this scenario, any member of your team could add a new model version, which would automatically be marked `latest`, but only admins could then move that model into `staging` or `production`.

## Features
The access controls add two new features:
1. **Model Registry Admins**: Give members of your team admin permissions for the model registry. These admins can define and use protected aliases.
2. **Protected Aliases**: A protected alias such as 


## How to set up access control
The following steps describe how to set up access controls for your team’s model registry.

1. [Open Access Controls](https://wandb.ai/registry/model/access-control) from your Models page.
![](/images/models/access_controls_button.gif)
2. **Manage registry admins**: Add a set of people who should be allowed to add and remove protected aliases from model versions.
![](/images/models/access_controls_admins.gif)
3. **Add protected aliases**: Set the list of aliases that only admins can update. For example, you might use the `production` alias to represent a special state only admins can use.
![](/images/models/access_controls_add_protected_aliases.gif)