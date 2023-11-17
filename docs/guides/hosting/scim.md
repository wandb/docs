---
displayed_sidebar: default
---

# SCIM

The W&B Server System for Cross-domain Identity Management (SCIM) API lets instance admins manage users and groups (W&B teams) in a server deployment. The SCIM API can be accessed at `<host-url>/scim/` and supports the `/Users` and `/Groups` endpoint with a subset of the fields found in the [RC7643 protocol](https://www.rfc-editor.org/rfc/rfc7643).

## Authentication

The SCIM API can be accessed by instance admins using their API token.

## User Resource

The SCIM user resource maps to W&B users.

### Get User

- **Endpoint:** **`<host-url>/scim/Users/{id}`**
- **Method**: GET
- **Description**: Retrieve user information by providing the user's unique ID.
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

### List Users

- **Endpoint:** **`<host-url>/scim/Users`**
- **Method**: GET
- **Description**: Retrieve a list of users.
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

### Create User

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
      "value": "admin-user2@test.com"
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

### Deactivate User

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: DELETE
- **Description**: Deactivate a user by providing the user's unique ID.
- **Request Example**:

```bash
DELETE /scim/Users/abc
```

- **Response Example**:

```json
(Status 204)
```

### Reactivate User

- Reactivating a previously deactivated user is currently unsupported in the SCIM API.

## Group Resource

The SCIM group resource maps to W&B teams i.e. when you create a SCIM group in a W&B deployment, it creates a W&B team. And so on.

### Get Team

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

### List Teams

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

### Create Team

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

### Update Team

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

### Delete Team

- Deleting teams is currently unsupported by the SCIM API since there is additional data linked to teams. Please delete teams from the application to confirm you want everything deleted.