---
menu:
  default:
    identifier: model_registry_eol
    parent: registry
title: Migrate from legacy Model Registry
weight: 8
---

W&B will transition assets from the legacy [W&B Model Registry](../model_registry/intro.md) to the new [W&B Registry](./intro.md). This migration will be fully managed and triggered by W&B, requiring no intervention from users. The process is designed to be as seamless as possible, with minimal disruption to existing workflows.

The transition will take place once the new W&B Registry includes all the functionalities currently available in the Model Registry. W&B will attempt to preserve current workflows, codebases, and references.

This guide is a living document and will be updated regularly as more information becomes available. For any questions or support, contact support@wandb.com.

## How W&B Registry differs from the legacy Model Registry

W&B Registry introduces a range of new features and enhancements designed to provide a more robust and flexible environment for managing models, datasets, and other artifacts.

{{% alert %}}
To view the legacy Model Registry, navigate to the Model Registry in the W&B App. A banner appears at the top of the page that enables you to use the legacy Model Registry App UI.

{{< img src="/images/registry/nav_to_old_model_reg.gif" >}}
{{% /alert %}}

### Organizational visibility
Artifacts linked to the legacy Model Registry have team level visibility. This means that only members of your team can view your artifacts in the legacy W&B Model Registry. W&B Registry has organization level visibility. This means that members across an organization, with correct permissions, can view artifacts linked to a registry.


### Restrict visibility to a registry
Restrict who can view and access a custom registry. You can restrict visibility to a registry when you create a custom registry or after you create a custom registry. In a Restricted registry, only selected members can access the content, maintaining privacy and control. For more information about registry visibility, see [Registry visibility types](./configure_registry.md#registry-visibility-types).

### Create custom registries
Unlike the legacy Model Registry, W&B Registry is not limited to models or dataset registries. You can create custom registries tailored to specific workflows or project needs, capable of holding any arbitrary object type. This flexibility allows teams to organize and manage artifacts according to their unique requirements. For more information on how to create a custom registry, see [Create a custom registry](./create_registry.md).  

{{< img src="/images/registry/mode_reg_eol.png" alt="" >}}

### Custom access control
Each registry supports detailed access control, where members can be assigned specific roles such as Admin, Member, or Viewer. Admins can manage registry settings, including adding or removing members, setting roles, and configuring visibility. This ensures that teams have the necessary control over who can view, manage, and interact with the artifacts in their registries.

{{< img src="/images/registry/registry_access_control.png" alt="" >}}

### Terminology update
Registered models are now referred to as *collections*. 


### Summary of changes

|               | Legacy W&B Model Registry | W&B Registry |
| -----         | ----- | ----- |
| Artifact visibility| Only members of team can view or access artifacts | Members in your organization, with correct permissions, can view or access artifacts linked to a registry  |
| Custom access control | Not available | Available |
| Custom registry | Not available | Available |
| Terminology update | A set of pointers (links) to model versions are called *registered models*. | A set of pointers (links) to artifact versions are called *collections*. | 
| `wandb.init.link_model` | Model Registry specific API | Currently only compatible with legacy model registry |


## Preparing for the migration

W&B will migrate registered models (now called collections) and associated artifact versions from the legacy Model Registry to the W&B Registry. This process will be conducted automatically, with no action required from users.

### Team visibility to organization visibility

After the migration, your model registry will have organization level visibility. You can restrict who has access to a registry by [assigning roles](./configure_registry.md). This helps ensure that only specific members have access to specific registries.

The migration will preserve existing permission boundaries of your current team-level registered models (soon to be called collections) in the legacy W&B Model Registry. Permissions currently defined in the legacy Model Registry will be preserved in the new Registry.  This means that collections currently restricted to specific team members will remain protected during and after the migration. 

### Artifact path continuity

No action is currently required.

## During the migration

W&B will initiate the migration process. The migration will occur during a time window that minimizes disruption to W&B services. The legacy Model Registry will transition to a read-only state once the migration begins and will remain accessible for reference.

## After the migration

Post-migration, collections, artifact versions, and associated attributes will be fully accessible within the new W&B Registry. The focus is on ensuring that current workflows remain intact, with ongoing support available to help navigate any changes.

### Using the new registry

Users are encouraged to explore the new features and capabilities available in the W&B Registry. The Registry will not only support the functionalities currently relied upon but also introduces enhancements such as custom registries, improved visibility, and flexible access controls.

Support is available if you are interested in trying the W&B Registry early, or for new users that prefer to start with Registry and not the legacy W&B Model Registry. Contact support@wandb.com or your Sales MLE to enable this functionality. Note that any early migration will be into a BETA version. The BETA version of W&B Registry might not have all the functionality or features of the legacy Model Registry.

For more details and to learn about the full range of features in the W&B Registry, visit the [W&B Registry Guide](./intro.md).

## FAQs

#### Why is W&B migrating assets from Model Registry to W&B Registry?

W&B is evolving its platform to offer more advanced features and capabilities with the new Registry. This migration is a step towards providing a more integrated and powerful toolset for managing models, datasets, and other artifacts.

#### What needs to be done before the migration?

No action is required from users before the migration. W&B will handle the transition, ensuring that workflows and references are preserved.

#### Will access to model artifacts be lost?

No, access to model artifacts will be retained after the migration. The legacy Model Registry will remain in a read-only state, and all relevant data will be migrated to the new Registry.

#### Will metadata related to artifacts be preserved?

Yes, important metadata related to artifact creation, lineage, and other attributes will be preserved during the migration. Users will continue to have access to all relevant metadata after the migration, ensuring that the integrity and traceability of their artifacts remain intact.

#### Who do I contact if I need help?

Support is available for any questions or concerns.  Reach out to support@wandb.com for assistance.