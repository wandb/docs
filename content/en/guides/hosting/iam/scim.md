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

## Overview

The System for Cross-domain Identity Management (SCIM) API allows instance or organization admins to manage users, groups, and custom roles in their W&B organization. SCIM groups map to W&B teams. 

The SCIM API is accessible at `<host-url>/scim/` and supports the `/Users` and `/Groups` endpoints with a subset of the fields found in the [RFC7643 protocol](https://www.rfc-editor.org/rfc/rfc7643) and [RFC7644 protocol](https://www.rfc-editor.org/rfc/rfc7644) for SCIM 2.0. It additionally includes the `/Roles` endpoints which are not part of the official SCIM schema. W&B adds the `/Roles` endpoints to support automated management of custom roles in W&B organizations.

### Supported Features
- **Filtering**: The API supports filtering for `/Users` and `/Groups` endpoints
- **PATCH Operations**: Supports PATCH for partial resource updates
- **ETag Support**: Conditional updates using ETags for conflict detection  
- **Service Account Authentication**: Organization service accounts can access the API

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

## User Management

The SCIM user resource maps to W&B users. Use these endpoints to manage users in your organization.
```
### Get User

Retrieves information for a specific user in your organization.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: GET

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user |

#### Example

{{< tabpane text=true >}}
{{% tab header="Get User Request" %}}
```bash
GET /scim/Users/abc
```
{{% /tab %}}
{{% tab header="Get User Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### List Users

Retrieves a list of all users in your organization.

#### Filtering Users

The `/Users` endpoint supports filtering to search for specific users:

##### Supported Filters
- `userName eq "value"` - Filter by username
- `emails.value eq "value"` - Filter by email address

##### Example
```bash
GET /scim/Users?filter=userName eq "john.doe"
GET /scim/Users?filter=emails.value eq "john@example.com"

#### Endpoint
- **URL**: `<host-url>/scim/Users`
- **Method**: GET

#### Example

{{< tabpane text=true >}}
{{% tab header="List Users Request" %}}
```bash
GET /scim/Users
```
{{% /tab %}}
{{% tab header="List Users Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### Create User

Creates a new user in your organization.

#### Endpoint
- **URL**: `<host-url>/scim/Users`
- **Method**: POST

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| emails | array | Yes | Array of email objects. Must include a primary email |
| userName | string | Yes | The username for the new user |

#### Example

{{< tabpane text=true >}}
{{% tab header="Create User Request (Dedicated/Self-managed)" %}}
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
{{% /tab %}}
{{% tab header="Create User Request (Multi-tenant)" %}}
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
{{% /tab %}}
{{< /tabpane >}}

#### Response

{{< tabpane text=true >}}
{{% tab header="Create User Response (Dedicated/Self-managed)" %}}
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
{{% tab header="Create User Response (Multi-tenant)" %}}
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

### Delete User

{{% alert color="warning" title="Maintain admin access" %}}
You must ensure that at least one admin user exists in your instance or organization at all times. Otherwise, no user will be able to configure or maintain your organization's W&B account. If an organization uses SCIM or another automated process to deprovision users from W&B, a deprovisioning operation could inadvertently remove the last remaining admin from the instance or organization.

For assistance with developing operational procedures, or to restore admin access, contact [support](mailto:support@wandb.com).
{{% /alert %}}

Fully deletes a user from your organization.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: DELETE

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user to delete |

#### Example

{{< tabpane text=true >}}
{{% tab header="Delete User Request" %}}
```bash
DELETE /scim/Users/abc
```
{{% /tab %}}
{{% tab header="Delete User Response" %}}
```bash
(Status 204)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
To temporarily deactivate the user, refer to [Deactivate user](#deactivate-user) API which uses the `PATCH` endpoint.
{{% /alert %}}

### Update User Email (Single Tennant instances only)

Updates a user's primary email address. This operation is only available in self-managed single-tenant deployments.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user |
| op | string | Yes | Must be "replace" |
| path | string | Yes | Must be "emails" |
| value | array | Yes | Array with new email object |

#### Example

{{< tabpane text=true >}}
{{% tab header="Update Email Request" %}}
```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "emails",
            "value": [
                {
                    "value": "newemail@example.com",
                    "primary": true
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Update Email Response" %}}
```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "newemail@example.com",
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
{{% /tab %}}
{{< /tabpane >}}

### Update User Display Name (Self-managed instances only)

Updates a user's display name. This operation is only available in self-managed single-tenant deployments.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user |
| op | string | Yes | Must be "replace" |
| path | string | Yes | Must be "displayName" |
| value | string | Yes | New display name |

#### Example

{{< tabpane text=true >}}
{{% tab header="Update Display Name Request" %}}
```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "displayName",
            "value": "John Doe"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Update Display Name Response" %}}
```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "John Doe",
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
{{% /tab %}}
{{< /tabpane >}}

### Deactivate User

Temporarily deactivates a user in your organization.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user to deactivate |
| op | string | Yes | Must be "replace" |
| value | object | Yes | Object with `{"active": false}` |

{{% alert %}}
User deactivation and reactivation operations are not supported in [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
{{% /alert %}}

#### Example

{{< tabpane text=true >}}
{{% tab header="Deactivate User Request" %}}
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
{{% /tab %}}
{{% tab header="Deactivate User Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### Reactivate User

Reactivates a previously deactivated user in your organization.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user to reactivate |
| op | string | Yes | Must be "replace" |
| value | object | Yes | Object with `{"active": true}` |

{{% alert %}}
User deactivation and reactivation operations are not supported in [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}).
{{% /alert %}}

#### Example

{{< tabpane text=true >}}
{{% tab header="Reactivate User Request" %}}
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
{{% /tab %}}
{{% tab header="Reactivate User Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### Assign Organization Role

Assigns an organization-level role to a user.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user |
| op | string | Yes | Must be "replace" |
| path | string | Yes | Must be "organizationRole" |
| value | string | Yes | Role name ("admin" or "member") |

{{% alert %}}
The `viewer` role is deprecated and can no longer be set in the UI. W&B assigns the `member` role to a user if you attempt to assign the `viewer` role using SCIM. The user is automatically provisioned with Models and Weave seats if possible. Otherwise, a `Seat limit reached` error is logged. For organizations that use **Registry**, the user is automatically assigned the `viewer` role in registries that are visible at the organization level.
{{% /alert %}}

#### Example

{{< tabpane text=true >}}
{{% tab header="Assign Org Role Request" %}}
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
            "value": "admin"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Assign Org Role Response" %}}
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
    "teamRoles": [
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin"
}
```
{{% /tab %}}
{{< /tabpane >}}

### Assign Team Role

Assigns a team-level role to a user.

#### Endpoint
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | The unique ID of the user |
| op | string | Yes | Must be "replace" |
| path | string | Yes | Must be "teamRoles" |
| value | array | Yes | Array of objects with `teamName` and `roleName` |

#### Example

{{< tabpane text=true >}}
{{% tab header="Assign Team Role Request" %}}
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
                    "roleName": "admin",
                    "teamName": "team1"
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Assign Team Role Response" %}}
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
    "teamRoles": [
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin"
}
```
{{% /tab %}}
{{< /tabpane >}}

## Group resource

The SCIM group resource maps to W&B teams, that is, when you create a SCIM group in a W&B deployment, it creates a W&B team. Same applies to other group endpoints.

### Service Accounts

When a team is created via SCIM, all organization-level service accounts are automatically added as members of the team. This ensures service accounts maintain access to team resources.

### Filtering Groups

The `/Groups` endpoint supports filtering to search for specific teams:

#### Supported Filters
- `displayName eq "value"` - Filter by team display name

#### Example
```bash
GET /scim/Groups?filter=displayName eq "engineering-team"
```

### Get team

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: GET
- **Description**: Retrieve team information by providing the team's unique ID.
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
- **Supported Operations**: `add` member, `remove` member, `replace` members

{{% alert %}}
The remove operations follow RFC 7644 SCIM protocol specifications. Use the filter syntax `members[value eq "{user_id}"]` to remove a specific user, or `members` to remove all users from the team.

**User Identification**: The `{user_id}` in member operations can be either:
- A W&B user ID
- An email address (e.g., "user@example.com")
{{% /alert %}}


{{% alert color="info" %}}
Replace `{team_id}` with the actual team ID and `{user_id}` with the actual user ID or email address in your requests.
{{% /alert %}}

### Replace team members

Replaces all members of a team with a new list.

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: PUT
- **Description**: Replace the entire team membership list.

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PUT /scim/Groups/{team_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
    "displayName": "wandb-devs",
    "members": [
        {
            "value": "{user_id_1}"
        },
        {
            "value": "{user_id_2}"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "user_id_1",
            "Ref": "",
            "Type": "",
            "Display": "user1"
        },
        {
            "Value": "user_id_2",
            "Ref": "",
            "Type": "",
            "Display": "user2"
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
{{% /tab %}}
{{< /tabpane >}}

**Adding a user to a team**

Adding `dev-user2` to `wandb-devs`:

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Groups/{team_id}
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
                    "value": "{user_id}"
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

**Removing a specific user from a team**

Removing `dev-user2` from `wandb-devs`:

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Groups/{team_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "remove",
            "path": "members[value eq \"{user_id}\"]"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
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
        "lastModified": "2023-10-01T00:01:00Z",
        "location": "Groups/ghi"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```
{{% /tab %}}
{{< /tabpane >}}

**Removing all users from a team**

Removing all users from `wandb-devs`:

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Groups/{team_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "remove",
            "path": "members"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": null,
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
{{% /tab %}}
{{< /tabpane >}}

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

### Update custom role

#### Add permissions to role

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PATCH
- **Description**: Add permissions to an existing custom role.

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Roles/{role_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add",
            "path": "permissions",
            "value": [
                {
                    "name": "project:delete"
                },
                {
                    "name": "run:stop"
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

Returns the updated role with new permissions added.
{{% /tab %}}
{{< /tabpane >}}

#### Remove permissions from role

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PATCH
- **Description**: Remove permissions from an existing custom role.

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Roles/{role_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "remove",
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
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

Returns the updated role with specified permissions removed.
{{% /tab %}}
{{< /tabpane >}}

### Replace custom role

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PUT
- **Description**: Replace an entire custom role definition.

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PUT /scim/Roles/{role_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Updated custom role",
    "description": "Updated description for the custom role",
    "permissions": [
        {
            "name": "project:read"
        },
        {
            "name": "run:read"
        },
        {
            "name": "artifact:read"
        }
    ],
    "inheritedFrom": "viewer"
}
```
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

Returns the completely replaced role definition.
{{% /tab %}}
{{< /tabpane >}}

### Delete custom role

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: DELETE
- **Description**: Delete a custom role in the W&B organization. **Use it with caution**. The predefined role from which the custom role inherited is now assigned to all users that were assigned the custom role before the operation.
- **Request Example**:

```bash
DELETE /scim/Roles/abc
```

## Advanced Features

### ETag Support

The SCIM API supports ETags for conditional updates to prevent concurrent modification conflicts. ETags are returned in the `ETag` response header and the `meta.version` field.

#### Using ETags

1. **Get current ETag**: When you GET a resource, note the ETag header in the response
2. **Conditional update**: Include the ETag in the `If-Match` header when updating

#### Example with ETag

```bash
# Get user and note ETag
GET /scim/Users/abc
# Response includes: ETag: W/"xyz123"

# Update with ETag
PATCH /scim/Users/abc
If-Match: W/"xyz123"

{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "organizationRole",
            "value": "admin"
        }
    ]
}
```

If the resource has been modified since you retrieved it, the server will return a `412 Precondition Failed` error.

### Multi-tenant SCIM Extensions

For SaaS Cloud deployments, the API supports additional schema extensions for team management during user creation:

#### Schema Extension
- **Schema ID**: `urn:ietf:params:scim:schemas:extension:teams:2.0:User`
- **Purpose**: Assign users to teams during creation

#### Example: Create user with team assignment (SaaS multi-tenant only)

```json
{
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User",
        "urn:ietf:params:scim:schemas:extension:teams:2.0:User"
    ],
    "emails": [
        {
            "primary": true,
            "value": "newuser@example.com"
        }
    ],
    "userName": "newuser",
    "urn:ietf:params:scim:schemas:extension:teams:2.0:User": {
        "teams": ["engineering", "ml-team"]
    }
}
```

### Error Handling

The SCIM API returns standard SCIM error responses:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (successful deletion) |
| 400 | Bad Request - Invalid parameters or request body |
| 401 | Unauthorized - Authentication failed |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource does not exist |
| 409 | Conflict - Resource already exists |
| 412 | Precondition Failed - ETag mismatch |
| 500 | Internal Server Error |

### Implementation Differences

W&B maintains two separate SCIM API implementations, and the features differ between them:

| Feature | SaaS Multi-tenant | Self-managed Single-tenant |
|---------|-------------------|---------------------------|
| Update user email | ❌ Not supported | ✅ Supported |
| Update user display name | ❌ Not supported | ✅ Supported |
| User deactivation/reactivation | ❌ Not supported | ✅ Supported |
| Multiple emails per user | ✅ Supported | ❌ Single email only |

### Limitations

- **Maximum results**: 9999 items per request
- **Single-tenant environments**: Only support one email per user
- **Team deletion**: Not supported via SCIM (use the W&B web interface)
- **User deactivation/reactivation**: Not supported in SaaS Cloud environments
- **Seat limits**: Operations may fail if organization seat limits are reached