---
menu:
  default:
    identifier: model_registry_eol
    parent: registry
title: Migrate from legacy Model Registry
weight: 9
---

W&B is upgrading your experience by migrating from the legacy **Model Registry** to the enhanced **W&B Registry**. This transition is designed to be seamless, fully managed by W&B, and will preserve your workflows while unlocking powerful new features. For any questions or support, contact [support@wandb.com](mailto:support@wandb.com).

## Why We’re Migrating

The new W&B Registry offers major improvements over the legacy Model Registry:

- **Unified, Organization-Level Experience**: Share and manage curated artifacts across your organization, regardless of teams.
- **Improved Governance**: Use access control, restricted registries, and visibility settings to better manage who sees what.
- **Enhanced Functionality**: Features like custom registries, better search, audit trails, and automation support help modernize your ML infrastructure.
- **Future-Ready**: All ongoing and future W&B enhancements for artifact lifecycle management will be built on the W&B Registry.

| Feature | Legacy W&B Model Registry | W&B Registry |
| ----- | ----- | ----- |
| Artifact Visibility | Team-level only - access restricted to team members | Org-level visibility with fine-grained permission controls | 
| Custom Registries |Not supported | Fully supported — create registries for any artifact type |
| Access Control | Not available | Role-based access (Admin, Member, Viewer) at the registry level |
| Terminology |“Registered models” — pointers to model versions | “Collections” — pointers to any artifact versions |
| Registry Scope |Only supports model versioning  | Supports models, datasets, custom artifacts, and more |
| Automations | | Registry- and collection-level automations supported and copied during migration |
| Search & Discoverability | Limited search and discoverability | Central search within W&B Registry across all registries in the organization |
| API Compatibility | Uses `wandb.init.link_model()` and MR-specific patterns | Modern SDK APIs (`link_artifact()`, `use_artifact()`) with auto-redirection |
| Migration | End-of-life | Automatically migrated and enhanced — data is copied, not deleted |

### No Manual Work Required

The migration is **fully automated and managed by W&B**. You do not need to run scripts, update configurations, or move data manually.

### Your Code Will Still Work

All existing API calls using the legacy Model Registry will **automatically redirect** to the new W&B Registry:

- `api.artifact()`
- `run.use_artifact()`
- `run.link_artifact()`
- `artifact.link()`

You don’t need to update paths immediately. For example:

```python
# Old path
"team-name/model-registry/collection-name:version"

# Redirects to new path
"org-name/wandb-registry-team-name-mr-migrated/collection-name:version"
```

This ensures that:

- **Existing training and deployment workflows** remain intact.
- **CI/CD pipelines** built on model promotion or artifact linking will **not break**.
- Teams can adopt the new W&B Registry **gradually**, without needing to refactor old codebases immediately.

{{% alert %}}
Note: If you're using the latest W&B SDK (v0.21.0 and above), you’ll see a helpful non-breaking warning in your logs indicating that a redirect has occurred. This encourages you to update paths in new code over time.

For older SDK versions, the redirect will still work silently without emitting a warning. Some metadata like entity or project names may reflect legacy values. We recommend upgrading to avoid inconsistencies in how artifact information is displayed or accessed via SDK. 

W&B strongly recommend upgrading to the latest SDK for full visibility into redirection behavior and to benefit from the latest fixes and features.
{{% /alert %}}

### Minimal Downtime, Preserved Access

During migration, write operations for your team’s Model Registry will be **briefly paused (up to 1 hour)** to ensure data consistency. Similarly, write operations to the newly created migrated W&B Registry will also be paused to ensure data integrity. Afterwards:

- The new W&B Registry becomes the primary location for your assets.
- The legacy Model Registry becomes **read-only**.
- No data is deleted. All data is **copied**, not moved.
- Eventually, legacy Model Registry UI views will be hidden.

## What’s Being Migrated?

We’ll copy your:

- Collections
- Linked artifact versions
- Version history
- Aliases, tags, and descriptions
- Automations (collection- and registry-level)
- Permissions, including service account roles and protected aliases

The new registries are created and will show up on the UI with name:

```text
<team-name>-mr-migrated
```

These registries default to **Restricted** visibility, preserving your existing privacy boundaries. Only the original members of the `<team-name>` will have access to their respective registries. 

## After the Migration

You’ll be able to:

- Access all your data from the new W&B Registry.
- Use the new Registry UI for versioning, governance, audit trails, and automation.
- Continue using your old code - paths are redirected.
- Explore enhanced functionality like:
    - Organization-level access
    - Role-based access control
    - Registry-level lineage tracking
    - Automations

## What You Should Do

- **No action is required.** W&B will handle everything.
- **Stay informed.** You’ll receive communications (banners in the UI) 2 weeks prior your migration is scheduled.
- **Use new paths in future work.** While old code continues to work, we recommend using the new W&B Registry paths for new projects.
- **Review permissions.** After migration, admins should double-check registry access to ensure alignment with your team’s needs.

## FAQs

### How will I know when my org is being migrated?

You’ll receive advance notice via in-app banners or direct communication from W&B.

### Will there be downtime?

Only for **write operations** to the legacy Model Registry and the new W&B Registry for your team — and only for a few hours. All other W&B services will remain available.

### Will this break my code?

No. All legacy Model Registry paths and SDK calls will automatically redirect to the new Registry.

### Will my data be deleted?

No. Your data will be **copied** to the new W&B Registry. The legacy Model Registry becomes read-only and later hidden — but nothing is removed or lost.

### What if I’m using an older SDK?

Redirects will still work, but you won’t see warnings about them. For the best experience, upgrade to the latest version of the W&B SDK.

### Can I rename/modify my migrated registry?

Yes, renaming and other operations like adding/removing members from a migrated registry are perfectly fine. These registries are simply custom registries underneath, and the redirection will continue working even after migration. 

## Questions?

For support or to discuss your migration, contact [support@wandb.com](mailto:support@wandb.com). We’re here to ensure your transition is smooth and successful.