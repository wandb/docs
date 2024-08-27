---
displayed_sidebar: default
---

# Migrating from legacy Model Registry

W&B will be transitioning assets from the legacy [W&B Model Registry](../model_registry/intro.md) to the new [W&B Registry](./intro.md). This migration will be fully managed and triggered by W&B, requiring no intervention from users. The process is designed to be as seamless as possible, with minimal disruption to existing workflows.

The transition will take place once the new W&B Registry includes all the functionalities currently available in the Model Registry, ensuring that users have access to all the tools and features they rely on. Ample notice will be provided ahead of the migration to ensure there are no surprises. The goal is to preserve current workflows, codebases, and references to the greatest extent possible during this transition.

This guide is a living document and will be updated regularly as more information becomes available. For any questions or support, contact support@wandb.com.

## What is changing?

W&B Registry introduces a range of new features and enhancements designed to provide a more robust and flexible environment for managing models, datasets, and other artifacts.

Key changes include:

- **Organization-Wide Visibility:** The Registry enhances visibility at the organization level, ensuring that models, datasets, and other artifacts are accessible and manageable across the entire organization. However, registries can also be configured with fine-grained control, allowing them to be either visible to the entire organization or set as "Restricted." In a Restricted registry, only selected members can access the content, maintaining privacy and control.
- **Custom Registries:** Unlike the legacy Model Registry, W&B Registry is not limited to models or datasets. Users can create custom registries tailored to specific workflows or project needs, capable of holding any arbitrary object type. This flexibility allows teams to organize and manage artifacts according to their unique requirements.

![](/images/registry/mode_reg_eol.png)

- **Fine-Grained Access Control:** Each registry supports detailed access control, where members can be assigned specific roles such as Admin, Member, or Viewer. Admins can manage registry settings, including adding or removing members, setting roles, and configuring visibility. This ensures that teams have the necessary control over who can view, manage, and interact with the artifacts in their registries.

![](/images/registry/registry_access_control.png)

- **Terminology Updates:** Registered models are now referred to as collections, aligning with the new terminology introduced in W&B Registry.

## Looking ahead

While the W&B Registry already brings significant enhancements, we are continuously working on adding even more powerful features.  In the coming months we plan to add: 

- **Org-Level Service Accounts:** Upcoming org-level service account support enabling enhanced control and management at the organizational level, streamlining how service accounts interact with the Registry.
- **Programmatic Access:** An API that allows for creating, reading, updating, and deleting (CRUD) operations programmatically within the Registry.
- **Advanced Artifact Management:** Fetch artifacts and utilize advanced filtering capabilities, both through the UI and the API, to streamline your workflows.
- **Comprehensive Search:** Sophisticated search functionality in the UI and API that spans all registries, collections, and artifacts, making it easier than ever to find what you need.
- **Enhanced Access Control:** Further refinement of access control, offering even more granularity in managing member roles and permissions within each registry.
- **Customize Collection and Version Displays.**  Choose relevant metrics and properties to show in your collection and version lists, making it faster to find and act on Model
- **Job Registry**
- **And more!** Stay tuned as we continue to innovate and expand the capabilities of the W&B Registry to meet the evolving needs of our users.

## Preparing for the migration

W&B will handle the migration of registered models (now called collections) and associated artifact versions from the legacy Model Registry to the W&B Registry. This process will be conducted automatically, with no action required from users.

## Team visibility to organization visibility

After the migration, registries will have an organization-level scope, allowing for seamless and transparent sharing across the entire organization. However, each registry will retain the ability to control permissions, just as team-level collections did in the Model Registry. Registries can be made "Restricted," ensuring that only specific members have access, and that access can be finely managed with roles such as Member, Admin, and Viewer.

Importantly, the migration will preserve the existing permission boundaries of team-level collections. This means that collections currently restricted to specific team members will remain protected, with permissions accurately mimicked in the new Registry. There will be no unintentional exposure of team-level collections to the broader organization.

## Artifact path continuity

Efforts are being made to ensure that existing links to artifact versions within collections continue to function post-migration. The goal is to preserve current workflows and code references, providing a seamless transition to the new Registry.

## During the migration

The migration process will be initiated by W&B and will be conducted in a way to minimze any distruption to the services. The legacy Model Registry will transition to a read-only state once the migration begins and will remain accessible for reference.

## After the migration

Post-migration, collections, artifact versions, and associated attributes will be fully accessible within the new W&B Registry. The focus is on ensuring that current workflows remain intact, with ongoing support available to help navigate any changes.

## Using the new registry

Users are encouraged to explore the new features and capabilities available in the W&B Registry. The Registry will not only support the functionalities currently relied upon but also introduces enhancements such as custom registries, improved visibility, and flexible access controls.

For those interested in trying the W&B Registry early, or for new users preferring to start directly with Registry, support is available. Contact support@wandb.com or your Sales MLE to enable this functionality. Note that any early migration will be into a beta version, which may not yet be feature complete compared to the legacy Model Registry.

For more details and to learn about the full range of features in the W&B Registry, visit the [W&B Registry Guide](./intro.md).

## FAQs

**Why is W&B migrating assets from Model Registry to W&B Registry?**

W&B is evolving its platform to offer more advanced features and capabilities with the new Registry. This migration is a step towards providing a more integrated and powerful toolset for managing models, datasets, and other artifacts.

**What needs to be done before the migration?**

No action is required from users before the migration. W&B will handle the transition, ensuring that workflows and references are preserved.

**Will access to model artifacts be lost?**

No, access to model artifacts will be retained after the migration. The legacy Model Registry will remain in a read-only state, and all relevant data will be migrated to the new Registry.

**Will metadata related to artifacts be preserved?**

Yes, important metadata related to artifact creation, lineage, and other attributes will be preserved during the migration. Users will continue to have access to all relevant metadata after the migration, ensuring that the integrity and traceability of their artifacts remain intact.

**How can assistance be obtained if issues arise?**

Support is available for any questions or concerns. Please reach out to support@wandb.com for assistance.