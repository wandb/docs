---
menu:
  default:
    identifier: scim
    parent: identity-and-access-management-iam
title: Manage users, groups, and roles with SCIM
weight: 4
---

{{% alert %}}
Watch a [video demonstrating SCIM in action](https://www.youtube.com/watch?v=Nw3QBqV0I-o) (12 min)
{{% /alert %}}

The System for Cross-domain Identity Management (SCIM) API allows instance or organization admins to manage users, groups, and custom roles in their W&B organization. SCIM groups map to W&B teams. 

The SCIM API is accessible at `<host-url>/scim/` and supports the `/Users` and `/Groups` endpoints with a subset of the fields found in the [RC7643 protocol](https://www.rfc-editor.org/rfc/rfc7643). It additionally includes the `/Roles` endpoints which are not part of the official SCIM schema. W&B adds the `/Roles` endpoints to support automated management of custom roles in W&B organizations.

{{% alert %}}
If you are an admin of multiple Enterprise [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) organizations, you must configure the organization where SCIM API requests are sent. Click your profile image, then click **User Settings**. The setting is named **Default API organization**. This is required for all hosting options, including [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}), [Self-managed instances]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}), and [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}). In SaaS Cloud, the organization admin must configure the default organization in user settings to ensure that the SCIM API requests go to the right organization.

The chosen hosting option determines the value for the `<host-url>` placeholder used in the examples in this page.

In addition, examples use user IDs such as `abc` and `def`. Real requests and responses have hashed values for user IDs.
{{% /alert %}}

## Authentication

Access to the SCIM API can be authenticated in two ways:

### Users

An organization or instance admin can use basic authentication with their API key to access the SCIM API. Set the HTTP request's `Authorization` header to the string `Basic` followed by a space, then the base-64 encoded string in the format `username:API-KEY`. In other words, replace the username and API key with your values separated with a `:` character, then base-64-encode the result. For example, to authorize as `demo:p@55w0rd`, the header should be `Authorization: Basic ZGVtbzpwQDU1dzByZA==`.

### Service accounts

An organization service account with the `admin` role can access the SCIM API. The username is left blank and only the API key is used. Find the API key for service accounts in the **Service account** tab in the organization dashboard. Refer to [Organization-scoped service accounts]({{< relref "/guides/hosting/iam/authentication/service-accounts.md/#organization-scoped-service-accounts" >}}).

Set the HTTP request's `Authorization` header to the string `Basic` followed by a space, then the base-64 encoded string in the format `:API-KEY` (notice the colon at the beginning with no username). For example, to authorize with only an API key such as `sa-p@55w0rd`, set the header to: `Authorization: Basic OnNhLXBANTV3MHJk`.

## User resource

The SCIM user resource maps to W&B users.

### Get user

- **Endpoint:** **`<host-url>/scim/Users/{id}`**
- **Method**: GET
- **Description**: Retrieve the information for a specific user in your [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) organization or your [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) instance by providing the user's unique ID.
- **Request Example**:

```bash
GET /scim/Users/abc
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```

### List users

- **Endpoint:** **`<host-url>/scim/Users`**
- **Method**: GET
- **Description**: Retrieve the list of all users in your [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) organization or your [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) instance.
- **Request Example**:

```bash
GET /scim/Users
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "Resources": [
        {
            "active": true,
            "displayName": "Dev User 1",
            "emails": {
                "Value": "dev-user1@test.com",
                "Display": "",
                "Type": "",
                "Primary": true
            },
            "id": "abc",
            "meta": {
                "resourceType": "User",
                "created": "2023-10-01T00:00:00Z",
                "lastModified": "2023-10-01T00:00:00Z",
                "location": "Users/abc"
            },
            "schemas": [
                "urn:ietf:params:scim:schemas:core:2.0:User"
            ],
            "userName": "dev-user1"
        }
    ],
    "itemsPerPage": 9999,
    "schemas": [
        "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    ],
    "startIndex": 1,
    "totalResults": 1
}
```

### Create user

Select your deployment type to view an example of creating a user using SCIM. The request and response schemas differ between **Dedicated Cloud or Self-Managed** and **Multi-tenant Cloud**.

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud or Self-managed" %}}
- **Endpoint**: **`<host-url>/scim/Users`**
- **Method**: POST
- **Description**: Create a new user resource.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| emails | Multi-Valued Array | Yes (Make sure `primary` email is set) |
| userName | String | Yes |
- **Request Example**:

    ```bash
    POST /scim/Users
    ```
    ```json
    {
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "emails": [
        {
        "primary": true,
        "value": "dev-user2@test.com"
        }
    ],
    "userName": "dev-user2"
    }
    ```

- **Response Example**:

    ```bash
    (Status 201)
    ```
    ```json
    {
        "active": true,
        "displayName": "Dev User 2",
        "emails": {
            "Value": "dev-user2@test.com",
            "Display": "",
            "Type": "",
            "Primary": true
        },
        "id": "def",
        "meta": {
            "resourceType": "User",
            "created": "2023-10-01T00:00:00Z",
            "location": "Users/def"
        },
        "schemas": [
            "urn:ietf:params:scim:schemas:core:2.0:User"
        ],
        "userName": "dev-user2"
    }
    ```
{{% /tab %}}

{{% tab header="Multi-tenant Cloud" %}}
In **Multi-tenant Cloud**, a user can belong to more than one organization. To "create" a user in your organization, you add them to one or more teams.
- **Endpoint**: **`<host-url>/scim/Users`**
- **Method**: POST
- **Description**: Create a new user resource.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| emails | Multi-Valued Array | Yes (Make sure `primary` email is set) |
| userName | String | Yes |
| teams | Multi-Valued Array | Yes (the user must belong to at minimum one team) |

- **Request Example**:

    ```bash
    POST /scim/Users
    ```

    ```json
    {
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User",
        "urn:ietf:params:scim:schemas:extension:teams:2.0:User"
    ],
    "emails": [
        {
        "primary": true,
        "value": "dev-user2@test.com"
        }
    ],
    "userName": "dev-user2",
    "urn:ietf:params:scim:schemas:extension:teams:2.0:User": {
        "teams": ["my-team"]
        }
    }
    ```
- **Response Example**:
    ```bash
    (Status 201)
    ```
    ```json
    {
    "active": true,
    "displayName": "Dev User 2",
    "emails": {
        "Value": "dev-user2@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "def",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "location": "Users/def"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User",
        "urn:ietf:params:scim:schemas:extension:teams:2.0:User"
    ],
    "userName": "dev-user2",
    "organizationRole": "member",
    "teamRoles": [
        {
        "teamName": "my-team",
        "roleName": "member"
        }
    ],
    "groups": [
        {
        "value": "my-team-id"
        }
    ]
    }
    ```
{{% /tab %}}
{{< /tabpane >}}


### Delete user

{{% alert color="warning" title="Maintain admin access" %}}
You must ensure that at least one admin user exists in your instance or organization at all times. Otherwise, no user will be able to configure or maintain your organization's W&B account. If an organization uses SCIM or another automated process to deprovision users from W&B, a deprovisioning operation could inadvertently remove the last remaining admin from the instance or organization.

For assistance with developing operational procedures, or to restore admin access, contact [support](mailto:support@wandb.com).
{{% /alert %}}

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: DELETE
- **Description**: Fully delete a user from your [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) organization or your [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) instance by providing the user's unique ID. Use the [Create user]({{< relref "#create-user" >}}) API to add the user again to the organization or instance if needed.
- **Request Example**:

{{% alert %}}
To temporarily deactivate the user, refer to [Deactivate user]({{< relref "#deactivate-user" >}}) API which uses the `PATCH` endpoint.
{{% /alert %}}

```bash
DELETE /scim/Users/abc
```

- **Response Example**:

```json
(Status 204)
```

### Deactivate user

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: Temporarily deactivate a user in your [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) instance by providing the user's unique ID. Use the [Reactivate user]({{< relref "#reactivate-user" >}}) API to reactivate the user when needed.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| value | Object | Object `{"active": false}` indicating that the user should be deactivated. |

{{% alert %}}
User deactivation and reactivation operations are not supported in [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
{{% /alert %}}

- **Request Example**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "value": {"active": false}
        }
    ]
}
```

- **Response Example**:
This returns the User object.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```

### Reactivate user

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: Reactivate a deactivated user in your [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) instance by providing the user's unique ID.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| value | Object | Object `{"active": true}` indicating that the user should be reactivated. |

{{% alert %}}
User deactivation and reactivation operations are not supported in [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
{{% /alert %}}

- **Request Example**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "value": {"active": true}
        }
    ]
}
```

- **Response Example**:
This returns the User object.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```

### Assign organization-level role to user

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: Assign an organization-level role to a user. The role can be one of `admin` or `member` as described [here]({{< relref "access-management/manage-organization.md#invite-a-user" >}}).

  {{% alert %}}
  The `viewer` role is deprecated and can no longer be set in the UI. W&B assigns the `member` role to a user if you attempt to assign the `viewer` role using SCIM. The user is automatically provisioned with Models and Weave seats if possible. Otherwise, a `Seat limit reached` error is logged. For organizations that use **Registry**, the user is automatically assigned the `viewer` role in registries that are visible at the organization level.
  {{% /alert %}}
    
  For [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), ensure that you have configured the correct organization for SCIM API in user settings.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| path | String | The scope at which role assignment operation takes effect. The only allowed value is `organizationRole`. |
| value | String | The predefined organization-level role to assign to the user. It can be one of `admin` or `member`. This field is case insensitive for predefined roles. |
- **Request Example**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "organizationRole",
            "value": "admin" // will set the user's organization-scoped role to admin
        }
    ]
}
```

- **Response Example**:
This returns the User object.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1",
    "teamRoles": [  // Returns the user's roles in all the teams that they are a part of
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // Returns the user's role at the organization scope
}
```

### Assign team-level role to user

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: Assign a team-level role to a user. The role can be one of `admin`, `viewer`, `member` or a custom role as described [here]({{< relref "access-management/manage-organization.md#assign-or-update-a-team-members-role" >}}). For [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}), ensure that you have configured the correct organization for SCIM API in user settings.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| path | String | The scope at which role assignment operation takes effect. The only allowed value is `teamRoles`. |
| value | Object array | A one-object array where the object consists of `teamName` and `roleName` attributes. The `teamName` is the name of the team where the user holds the role, and `roleName` can be one of `admin`, `viewer`, `member` or a custom role. This field is case insensitive for predefined roles and case sensitive for custom roles. |
- **Request Example**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "teamRoles",
            "value": [
                {
                    "roleName": "admin", // role name is case insensitive for predefined roles and case sensitive for custom roles
                    "teamName": "team1" // will set the user's role in the team team1 to admin
                }
            ]
        }
    ]
}
```

- **Response Example**:
This returns the User object.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1",
    "teamRoles": [  // Returns the user's roles in all the teams that they are a part of
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // Returns the user's role at the organization scope
}
```

## Group resource

The SCIM group resource maps to W&B teams, that is, when you create a SCIM group in a W&B deployment, it creates a W&B team. Same applies to other group endpoints.

### Get team

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: GET
- **Description**: Retrieve team information by providing the team’s unique ID.
- **Request Example**:

```bash
GET /scim/Groups/ghi
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "abc",
            "Ref": "",
            "Type": "",
            "Display": "dev-user1"
        }
    ],
    "meta": {
        "resourceType": "Group",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Groups/ghi"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```

### List teams

- **Endpoint**: **`<host-url>/scim/Groups`**
- **Method**: GET
- **Description**: Retrieve a list of teams.
- **Request Example**:

```bash
GET /scim/Groups
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "Resources": [
        {
            "displayName": "wandb-devs",
            "id": "ghi",
            "members": [
                {
                    "Value": "abc",
                    "Ref": "",
                    "Type": "",
                    "Display": "dev-user1"
                }
            ],
            "meta": {
                "resourceType": "Group",
                "created": "2023-10-01T00:00:00Z",
                "lastModified": "2023-10-01T00:00:00Z",
                "location": "Groups/ghi"
            },
            "schemas": [
                "urn:ietf:params:scim:schemas:core:2.0:Group"
            ]
        }
    ],
    "itemsPerPage": 9999,
    "schemas": [
        "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    ],
    "startIndex": 1,
    "totalResults": 1
}
```

### Create team

- **Endpoint**: **`<host-url>/scim/Groups`**
- **Method**: POST
- **Description**: Create a new team resource.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| displayName | String | Yes |
| members | Multi-Valued Array | Yes (`value` sub-field is required and maps to a user ID) |
- **Request Example**:

Creating a team called `wandb-support` with `dev-user2` as its member.

```bash
POST /scim/Groups
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
    "displayName": "wandb-support",
    "members": [
        {
            "value": "def"
        }
    ]
}
```

- **Response Example**:

```bash
(Status 201)
```

```json
{
    "displayName": "wandb-support",
    "id": "jkl",
    "members": [
        {
            "Value": "def",
            "Ref": "",
            "Type": "",
            "Display": "dev-user2"
        }
    ],
    "meta": {
        "resourceType": "Group",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Groups/jkl"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```

### Update team

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: PATCH
- **Description**: Update an existing team's membership list.
- **Supported Operations**: `add` member, `remove` member
- **Request Example**:

Adding `dev-user2` to `wandb-devs`

```bash
PATCH /scim/Groups/ghi
```

```json
{
	"schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
	"Operations": [
		{
			"op": "add",
			"path": "members",
			"value": [
	      {
					"value": "def",
				}
	    ]
		}
	]
}
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "abc",
            "Ref": "",
            "Type": "",
            "Display": "dev-user1"
        },
        {
            "Value": "def",
            "Ref": "",
            "Type": "",
            "Display": "dev-user2"
        }
    ],
    "meta": {
        "resourceType": "Group",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:01:00Z",
        "location": "Groups/ghi"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```

### Delete team

- Deleting teams is currently unsupported by the SCIM API since there is additional data linked to teams. Delete teams from the app to confirm you want everything deleted.

## Role resource

The SCIM role resource maps to W&B custom roles. As mentioned earlier, the `/Roles` endpoints are not part of the official SCIM schema, W&B adds `/Roles` endpoints to support automated management of custom roles in W&B organizations.

### Get custom role

- **Endpoint:** **`<host-url>/scim/Roles/{id}`**
- **Method**: GET
- **Description**: Retrieve information for a custom role by providing the role's unique ID.
- **Request Example**:

```bash
GET /scim/Roles/abc
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // indicates the predefined role
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // inherited from member predefined role
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // custom permission added by admin
        }
    ],
    "schemas": [
        ""
    ]
}
```

### List custom roles

- **Endpoint:** **`<host-url>/scim/Roles`**
- **Method**: GET
- **Description**: Retrieve information for all custom roles in the W&B organization
- **Request Example**:

```bash
GET /scim/Roles
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
   "Resources": [
        {
            "description": "A sample custom role for example",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // indicates the predefined role that the custom role inherits from
            "meta": {
                "resourceType": "Role",
                "created": "2023-11-20T23:10:14Z",
                "lastModified": "2023-11-20T23:31:23Z",
                "location": "Roles/Um9sZTo3"
            },
            "name": "Sample custom role",
            "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
            "permissions": [
                {
                    "name": "artifact:read",
                    "isInherited": true // inherited from member predefined role
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // custom permission added by admin
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // indicates the predefined role that the custom role inherits from
            "meta": {
                "resourceType": "Role",
                "created": "2023-11-21T01:07:50Z",
                "location": "Roles/Um9sZToxMg=="
            },
            "name": "Sample custom role 2",
            "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
            "permissions": [
                {
                    "name": "launchagent:read",
                    "isInherited": true // inherited from viewer predefined role
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // custom permission added by admin
                }
            ],
            "schemas": [
                ""
            ]
        }
    ],
    "itemsPerPage": 9999,
    "schemas": [
        "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    ],
    "startIndex": 1,
    "totalResults": 2
}
```

### Create custom role

- **Endpoint**: **`<host-url>/scim/Roles`**
- **Method**: POST
- **Description**: Create a new custom role in the W&B organization.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| name | String | Name of the custom role |
| description | String | Description of the custom role |
| permissions | Object array | Array of permission objects where each object includes a `name` string field that has value of the form `w&bobject:operation`. For example, a permission object for delete operation on W&B runs would have `name` as `run:delete`. |
| inheritedFrom | String | The predefined role which the custom role would inherit from. It can either be `member` or `viewer`. |
- **Request Example**:

```bash
POST /scim/Roles
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Sample custom role",
    "description": "A sample custom role for example",
    "permissions": [
        {
            "name": "project:update"
        }
    ],
    "inheritedFrom": "member"
}
```

- **Response Example**:

```bash
(Status 201)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // indicates the predefined role
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // inherited from member predefined role
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // custom permission added by admin
        }
    ],
    "schemas": [
        ""
    ]
}
```

### Delete custom role

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: DELETE
- **Description**: Delete a custom role in the W&B organization. **Use it with caution**. The predefined role from which the custom role inherited is now assigned to all users that were assigned the custom role before the operation.
- **Request Example**:

```bash
DELETE /scim/Roles/abc
```

- **Response Example**:

```bash
(Status 204)
```

### Update custom role permissions

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PATCH
- **Description**: Add or remove custom permissions in a custom role in the W&B organization.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| operations | Object array | Array of operation objects |
| op | String | Type of operation within the operation object. It can either be `add` or `remove`. |
| path | String | Static field in the operation object. Only value allowed is `permissions`. |
| value | Object array | Array of permission objects where each object includes a `name` string field that has value of the form `w&bobject:operation`. For example, a permission object for delete operation on W&B runs would have `name` as `run:delete`. |
- **Request Example**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // indicates the type of operation, other possible value being `remove`
            "path": "permissions",
            "value": [
                {
                    "name": "project:delete"
                }
            ]
        }
    ]
}
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // indicates the predefined role
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // inherited from member predefined role
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // existing custom permission added by admin before the update
        },
        {
            "name": "project:delete",
            "isInherited": false // new custom permission added by admin as part of the update
        }
    ],
    "schemas": [
        ""
    ]
}
```

### Update custom role metadata

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PUT
- **Description**: Update the name, description or inherited role for a custom role in the W&B organization. This operation doesn't affect any of the existing, that is, non-inherited custom permissions in the custom role.
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| name | String | Name of the custom role |
| description | String | Description of the custom role |
| inheritedFrom | String | The predefined role which the custom role inherits from. It can either be `member` or `viewer`. |
- **Request Example**:

```bash
PUT /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Sample custom role",
    "description": "A sample custom role for example but now based on viewer",
    "inheritedFrom": "viewer"
}
```

- **Response Example**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example but now based on viewer", // changed the descripton per the request
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // indicates the predefined role which is changed per the request
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // inherited from viewer predefined role
        },
        ... // Any permissions that are in member predefined role but not in viewer will not be inherited post the update
        {
            "name": "project:update",
            "isInherited": false // custom permission added by admin
        },
        {
            "name": "project:delete",
            "isInherited": false // custom permission added by admin
        }
    ],
    "schemas": [
        ""
    ]
}
```
