---
displayed_sidebar: default
---

# Organizations

## Custom role and role assignment SCIM API
Use [SCIM API](./scim.md) to manage users, and the teams they belong to, in an efficient and repeatable manner. You can also use the SCIM API to manage custom roles or assign roles to users in your W&B organization. Role endpoints are not part of the official SCIM schema. W&B adds role endpoints to support automated management of custom roles and to assign roles to users in W&B organizations.

[Custom role and role assignment SCIM API](./scim.md#role-resource) allows for managing custom roles, including creating, listing, or updating custom roles in an organization. This API also supports assigning predefined or custom roles to users in an organization.

:::caution
Delete a custom role with caution.

Delete a custom role within a W&B organization with the `DELETE Role` endpoint. The predefined role that the custom role inherits is assigned to all users that are assigned the custom role before the operation.

Update the inherited role for a custom role with the `PUT Role` endpoint. This operation doesn't affect any of the existing, that is, non-inherited custom permissions in the custom role.
:::

:::caution
The request type and path for the role assignment APIs are same as for the update custom role permissions API. Both types of APIs implement the `PATCH Role` endpoint. Difference is that the URI for role assignment APIs expects a `:userId` parameter, while the URI for custom role API expects a `:roleId`. Expected request bodies for both types of APIs are also different. 

Be careful with the parameter value in the URI and the request body such that those map to the intended operation.
:::