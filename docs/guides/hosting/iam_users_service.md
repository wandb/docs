---
displayed_sidebar: default
---
# Users and service accounts




## User SCIM API
Use [SCIM API](./scim.md) to manage users, and the teams they belong to, in an efficient and repeatable manner. You can also use the SCIM API to manage custom roles or assign roles to users in your W&B organization. Role endpoints are not part of the official SCIM schema. W&B adds role endpoints to support automated management of custom roles and to assign roles to users in W&B organizations.

[User SCIM API](./scim.md#user-resource) allows for creating, deactivating, getting the details of a user, or listing all users in a W&B organization.

:::info
Deactivate a user within a W&B organization with the `DELETE User` endpoint. Deactivated users can no longer sign in. However, deactivated users still appears in the organization's user list.

To fully remove a deactivated user from the user list, you must [remove the user from the organization](#remove-a-user).

It is possible to re-enable a deactivated user, if needed.
:::