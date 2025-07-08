---
menu:
  default:
    identifier: model_registry_eol
    parent: registry
title: Migrate from legacy Model Registry
weight: 9
---

W&B is migrating from the legacy **Model Registry** to the enhanced **W&B Registry**. This transition is designed to be seamless and fully managed by W&B. The migration process will preserve your workflows while unlocking powerful new features. For any questions or support, contact [support@wandb.com](mailto:support@wandb.com).

## Reasons for the migration

W&B Registry offers major improvements over the legacy Model Registry:

- **Unified, organization-level experience**: Share and manage curated artifacts across your organization, regardless of teams.
- **Improved governance**: Use access control, restricted registries, and visibility settings to manage user access.
- **Enhanced functionality**: New features such as custom registries, better search, audit trails, and automation support help modernize your ML infrastructure.
<!-- - **Future-ready**: All ongoing and future W&B enhancements for artifact lifecycle management will be built on the W&B Registry. -->

The following table summarizes the key differences between the legacy Model Registry and the new W&B Registry:

| Feature | Legacy W&B Model Registry | W&B Registry |
| ----- | ----- | ----- |
| Artifact Visibility | Team-level only - access restricted to team members | Org-level visibility with fine-grained permission controls | 
| Custom Registries |Not supported | Fully supported — create registries for any artifact type |
| Access Control | Not available | Role-based access (Admin, Member, Viewer) at the registry level |
| Terminology |“Registered models”: pointers to model versions | “Collections”: pointers to any artifact versions |
| Registry Scope |Only supports model versioning  | Supports models, datasets, custom artifacts, and more |
| Automations | Registry-level automations | Registry- and collection-level automations supported and copied during migration |
| Search & Discoverability | Limited search and discoverability | Central search within W&B Registry across all registries in the organization |
| API Compatibility | Uses `wandb.init.link_model()` and MR-specific patterns | Modern SDK APIs (`link_artifact()`, `use_artifact()`) with auto-redirection |
| Migration | End-of-life | Automatically migrated and enhanced — data is copied, not deleted |

## Preparing for the migration

- **No action required**: The migration is fully automated and managed by W&B. You do not need to run scripts, update configurations, or move data manually.
- **Stay informed**: You will receive communications (banners in the W&B App UI) 2 weeks prior to your scheduled migration.
- **Review permissions**: After the migration, admins should check registry access to ensure alignment with your team’s needs.
- **Use new paths in future work**: Old code continues to work, W&B recommends using the new W&B Registry paths for new projects.


## Migration process

### Temporary write operation pause
During migration, write operations for your team’s Model Registry will be paused to ensure data consistency for up to one hour. Write operations to the newly created migrated W&B Registry will also be paused during the migration.

### Data migration
W&B will migrate the following data from the legacy Model Registry to the new W&B Registry:

- Collections
- Linked artifact versions
- Version history
- Aliases, tags, and descriptions
- Automations (both collection and registry-level)
- Permissions, including service account roles and protected aliases

Within the W&B App UI, the legacy Model Registry will be replaced with the new W&B Registry. Migrated registries will have the name of your team followed by `mr-migrated`:

```text
<team-name>-mr-migrated
```

These registries default to **Restricted** visibility, preserving your existing privacy boundaries. Only the original members of the `<team-name>` will have access to their respective registries. 


## After the migration

After the migration completes:

- The legacy Model Registry becomes **read-only**. You can still view and access your data, but no new writes will be allowed.
- Data in the legacy Model Registry is **copied** to the new W&B Registry, not moved. No data is deleted.
- Access all your data from the new W&B Registry.
- Use the new Registry UI for versioning, governance, audit trails, and automation.
- Continue using your old code.
   - [Existing paths and API calls will automatically redirect to the new W&B Registry.]({{< relref "#code-will-continue-to-work" >}})
   - [Artifact version paths are redirected.]({{< relref "#legacy-paths-will-redirect-to-new-wb-registry-paths" >}})
- The legacy Model Registry will temporarily remain visible in the UI. W&B will eventually hide the legacy Model Registry.
- Explore enhanced functionality in the Registry such as:
    - [Organization-level access]({{< relref "/guides/core/registry/create_registry/#visibility-types" >}})
    - [Role-based access control]({{< relref "/guides/core/registry/configure_registry/" >}})
    - [Registry-level lineage tracking]({{< relref "/guides/core/registry/lineage/" >}})
    - Automations

### Code will continue to work

Existing API calls in your code that refer to the legacy Model Registry will automatically redirect to the new W&B Registry. The following API calls will continue to work without any changes:

- `wandb.Api().artifact()`
- `wandb.run.use_artifact()`
- `wandb.run.link_artifact()`
- `wandb.Artifact().link()`

### Legacy paths will redirect to new W&B Registry paths

W&B will automatically redirect legacy Model Registry paths to the new W&B Registry format. This means you can continue using your existing code without needing to refactor paths immediately.

<!-- - Existing training and deployment workflows remain intact.
- CI/CD pipelines built on model promotion or artifact linking will not break.
- Teams can adopt the new W&B Registry gradually, without needing to refactor old codebases immediately. -->

In the legacy Model Registry, paths consisted of a team name, a `"model-registry"` string, collection name, and version:

```python
f"{team-name}/model-registry/{collection-name}:{version}"
```

W&B will automatically redirect these paths to the new W&B Registry format, which includes the organization name, a `"wandb-registry"` string, the team name, collection name, and version:

```python
# Redirects to new path
f"{org-name}/wandb-registry-{team-name}/{collection-name}:{version}"
```


{{% alert title="Python SDK warnings" %}}

A warning error may appear if you continue to use legacy Model Registry paths in your code. The warning will not break your code, but it indicates that you should update your paths to the new W&B Registry format. 

Whether a warning appears depends on the version of the W&B Python SDK you are using:

* Users on the latest W&B SDK (`v0.21.0` and above) will see a non-breaking warning in their logs indicating that a redirect has occurred.
* For older SDK versions, the redirect will still work silently without emitting a warning. Some metadata such as entity or project names may reflect legacy values.

{{% /alert %}} 


## Frequently asked questions

### How will I know when my org is being migrated?

You will receive advance notice with an in-app banner or direct communication from W&B.

### Will there be downtime?

Write operations to the legacy Model Registry and the new W&B Registry will be paused for a approximately one hour during the migration. All other W&B services will remain available.

### Will this break my code?

No. All legacy Model Registry paths and Python SDK calls will automatically redirect to the new Registry.

### Will my data be deleted?

No. Your data will be copied to the new W&B Registry. The legacy Model Registry becomes read-only and later hidden. No data is removed or lost.

### What if I’m using an older SDK?

Redirects will still work, but you will not see warnings about them. For the best experience, upgrade to the latest version of the W&B SDK.

### Can I rename/modify my migrated registry?

Yes, renaming and other operations such as adding or removing members from a migrated registry are allowed. These registries are simply custom registries underneath, and the redirection will continue working even after migration. 

## Questions?

For support or to discuss your migration, contact [support@wandb.com](mailto:support@wandb.com). W&B is committed to helping you transition smoothly to the new W&B Registry.