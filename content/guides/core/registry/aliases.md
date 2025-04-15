---
title: Use aliases to point to production ready versions
---


Use aliases to reference a specific artifact version. Common aliases include `production`, `staging`, and `latest`.

{{% alert title="When to use a tag versus using an alias" %}}
Use aliases reference a specific artifact version.

Use tags to organize and group artifact versions or collections based on a common theme. Multiple artifact versions or collections can share the same tag.
{{% /alert %}}


## Add an alias


## Add a protected alias

Add a protected alias to an artifact version to prevent future modification or deletion of that artifact version. Common scenarios include:

- **Production**: The artifact version is ready for production use.
- **Staging**: The artifact version is ready for testing.

Each registry has its own set of protected aliases. Protected aliases are not shared across registries. Only registry admins can add, modify, or remove protected aliases from a registry. See [Configure registry access]({{< relref "/guides/core/registry/configure_registry.md" >}}) for information on how to manage users and assign roles in a registry.

The following procedure describes how to add a protected alias to a registry:

1. Navigate to the Registry App
2. Click on a registry
3. Select the gear button on the top right of the page to view the registry's settings.
4. Within the **Protected Aliases** section, click on the plus icon (**+**) to add one or more protected aliases.

