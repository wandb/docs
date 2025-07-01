---
menu:
  default:
    identifier: model_registry_eol
    parent: registry
title: Migrate from legacy Model Registry
weight: 8
---

W&B will transition assets from the legacy [W&B Model Registry]({{< relref "/guides/core/registry/model_registry/" >}}) to the new [W&B Registry]({{< relref "./" >}}). This migration will be fully managed and triggered by W&B, requiring no intervention from users. The process is designed to be as seamless as possible, with minimal disruption to existing workflows.

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
Restrict who can view and access a custom registry. You can restrict visibility to a registry when you create a custom registry or after you create a custom registry. In a Restricted registry, only selected members can access the content, maintaining privacy and control. For more information about registry visibility, see [Registry visibility types]({{< relref "./configure_registry.md#registry-visibility-types" >}}).

### Create custom registries
Unlike the legacy Model Registry, W&B Registry is not limited to models or dataset registries. You can create custom registries tailored to specific workflows or project needs, capable of holding any arbitrary object type. This flexibility allows teams to organize and manage artifacts according to their unique requirements. For more information on how to create a custom registry, see [Create a custom registry]({{< relref "./create_registry.md" >}}).  

{{< img src="/images/registry/mode_reg_eol.png" alt="" >}}

### Custom access control
Each registry supports detailed access control, where members can be assigned specific roles such as Admin, Member, or Viewer. Admins can manage registry settings, including adding or removing members, setting roles, and configuring visibility. This ensures that teams have the necessary control over who can view, manage, and interact with the artifacts in their registries.

{{< img src="/images/registry/registry_access_control.png" alt="" >}}

### Terminology update
Registered models are now referred to as *collections*. 

### Existing data
Existing model registry data such as collections, linked artifact versions, version history, aliases, tags, descriptions, automations and permissions will be migrated to the new W&B Registry. 


### Summary of changes

|               | Legacy W&B Model Registry | W&B Registry |
| -----         | ----- | ----- |
| Artifact visibility| Only members of team can view or access artifacts | Members in your organization, with correct permissions, can view or access artifacts linked to a registry  |
| Custom access control | Not available | Available |
| Custom registry | Not available | Available |
| Terminology update | A set of pointers (links) to model versions are called *registered models*. | A set of pointers (links) to artifact versions are called *collections*. | 
| `wandb.init.link_model` | Model Registry specific API | Currently only compatible with legacy model registry |
| Data | Collections, linked artifact versions, version history, aliases, tags, descriptions, automations and permissions | Copied from legacy Model Registry to W&B Registry |


## Preparing for the migration

No action is required from users before the migration. W&B will handle the transition, ensuring that workflows and references are preserved.


## During the migration

W&B will initiate the migration process. The migration will occur during a time window that minimizes disruption to W&B services. There will be a brief pause (estimated around 2-3 hours) on write operations for a specific team's assets. This means that while your team's legacy Model Registry data is being copied to the new W&B Registry, you won't be able to save new versions or make changes in either that specific legacy Model Registry or its new Registry counterpart. Once the copy is complete for your team, write operations will resume normally in the new W&B Registry. The legacy Model Registry will then become read-only. 

The following will happen during the migration:

* W&B will migrate registered models (now called *collections*) and associated artifact versions [from the legacy Model Registry to the W&B Registry]({{< relref "#artifact-paths" >}}). 
* API calls that reference the legacy Model Registry will be [redirected to the new W&B Registry]({{< relref "#legacy-model-registry-apis" >}}). This means that any existing references to model artifacts in your code or workflows will continue to function without requiring updates.
* Team level visibility of the legacy Model Registry will be [transitioned to organization level visibility]({{< relref "#team-visibility-to-organization-visibility" >}}) in the W&B Registry.
* The legacy Model Registry will transition to a [read-only state]({{< relref "#legacy-model-registry-read-only-state" >}}) once the migration begins and will remain accessible for reference.



## After the migration

Post-migration, collections, artifact versions, and associated attributes will be fully accessible within the new W&B Registry. The focus is on ensuring that current workflows remain intact, with ongoing support available to help navigate any changes.


### Artifact paths

Legacy model registry path will be redirected to the new W&B Registry path. This means that any existing references to model artifacts in your code or workflows will continue to function without requiring updates.

For example, if you have a model artifact path in your code like:

```python
model_reg_path = "team-name/model-registry/collection-name:version"
```

W&B will automatically redirect to its new W&B Registry path:

```python
registry_path = "org-name/wandb-registry-team-name/collection-name:version"
```

### Legacy Model Registry APIs

The following API calls will be supporting automatic redirects to the new W&B Registry:

* `wandb.Api().artifact()`
* `wandb.Api().use_artifact()`
* `wandb.run.link_artifact()`
* `wandb.Artifact.link()`

{{% alert title="Python SDK version"%}}
Based on your W&B Python SDK version, the redirection behavior will vary:

* Users with Python SDK versions v0.22 and greater will receive a non-breaking warning in your SDK logs about this redirection, encouraging users to update paths in new code over time.
* Users with Python SDK versions below v0.22 will not receive a warning, but the redirection will still occur.

{{% /alert %}}

<!-- Add note about degraded experience? If yes, get more concrete details from the engineering team. -->

### Team visibility to organization visibility

After the migration, your model registry will have organization-level visibility. You can restrict who has access to a registry by [assigning roles]({{< relref "./configure_registry.md" >}}). This helps ensure that only specific members have access to specific registries.

W&B will attempt to preserve existing permission boundaries of your current team-level registered models in the legacy W&B Model Registry. W&B suggests that admins to review access to the new migrated organization-level registries to ensure they align with your expectations.

### Legacy Model Registry read-only state
The legacy Model Registry will transition to a read-only state once the migration begins. This means that while you can still view and reference your existing registered models, you will not be able to make any changes or updates to them in the legacy Model Registry. All new model registrations and updates will need to be done in the new W&B Registry.

### Using the new registry

Users are encouraged to explore the new features and capabilities available in the W&B Registry. The Registry will not only support the functionalities currently relied upon but also introduces enhancements such as custom registries, improved visibility, and flexible access controls.

Support is available if you are interested in trying the W&B Registry early, or for new users that prefer to start with Registry and not the legacy W&B Model Registry. Contact support@wandb.com or your Sales MLE to enable this functionality. Note that any early migration will be into a BETA version. The BETA version of W&B Registry might not have all the functionality or features of the legacy Model Registry.

For more details and to learn about the full range of features in the W&B Registry, visit the [W&B Registry Guide]({{< relref "./" >}}).

## FAQs

### How much notice will we receive before our organization is scheduled for migration?

W&B is committed to providing sufficient notice. Expect to receive direct communication and see an advisory banner in the W&B UI approximately 4-6 weeks before your organization's scheduled migration window. This will give your team time to prepare and ask any questions. 

### How long will copying data from the legacy Model Registry to the W&B Registry take?
This depends on the volume of data in your legacy Model Registry. During this brief copy period for your specific team, write access to that registry (both the old Model Registry and the new W&B Registry) will be temporarily paused to ensure data integrity. W&B anticipates this specific write-pause to be around an hour or so based on current testing. W&B will provide more specific estimates during targeted communications. Other W&B services will remain fully operational.

### What if we use features like Protected Aliases or Team Service Accounts heavily?
The migration is designed to handle these. Protected Aliases will have an equivalent in the W&B Registry, and your existing ones will be mapped. Team Service Account permissions will be preserved within the context of the migrated registries.

### How long will we be able to access our legacy Model Registry (MR) after our organization is migrated?
After your organization's data is successfully copied to the new W&B Registry (GR), the legacy MR will immediately become read-only. You will be able to view your data in the legacy MR in this read-only state for approximately 4 weeks. After this period, the UI views for the legacy MR will be hidden.

### Will our data in the legacy Model Registry be deleted?
No, your data in the legacy MR will not be deleted as part of this migration. W&B will copy your data to the new W&B Registry, not moving it. This ensures data safety and provides a fallback in case things don’t work out as we expect it to.

### Why is W&B migrating assets from Model Registry to W&B Registry?

W&B is evolving its platform to offer more advanced features and capabilities with the new Registry. This migration is a step towards providing a more integrated and powerful toolset for managing models, datasets, and other artifacts.

### What needs to be done before the migration?

No action is required from users before the migration. W&B will handle the transition, ensuring that workflows and references are preserved.

### Will access to model artifacts be lost?

No, access to model artifacts will be retained after the migration. The legacy Model Registry will remain in a read-only state, and all relevant data will be migrated to the new Registry.

### Will metadata related to artifacts be preserved?

Yes, important metadata related to artifact creation, lineage, and other attributes will be preserved during the migration. Users will continue to have access to all relevant metadata after the migration, ensuring that the integrity and traceability of their artifacts remain intact.


### Who do I contact if I need help?

Support is available for any questions or concerns. Reach out to support@wandb.com for assistance.