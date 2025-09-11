---
title: Users
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/users.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for managing users and API keys. 

This module provides classes for managing W&B users and their API keys. 



**Note:**

> This module is part of the W&B Public API and provides methods to manage users and their authentication. Some operations require admin privileges. 



---

## <kbd>class</kbd> `User`
A class representing a W&B user with authentication and management capabilities. 

This class provides methods to manage W&B users, including creating users, managing API keys, and accessing team memberships. It inherits from Attrs to handle user attributes.

### <kbd>method</kbd> `User.__init__`

```python
__init__(client, attrs)
```

**Args:**
 
 - `client`:  (`wandb.apis.internal.Api`) The client instance to use 
 - `attrs`:  (dict) The user attributes 



**Note:**

> Some operations require admin privileges 







---

### <kbd>property</kbd> User.api_keys

List of API key names associated with the user. 



**Returns:**
 
 - `list[str]`:  Names of API keys associated with the user. Empty list if user  has no API keys or if API key data hasn't been loaded. 

---

### <kbd>property</kbd> User.teams

List of team names that the user is a member of. 



**Returns:**
 
 - `list` (list):  Names of teams the user belongs to. Empty list if user has no  team memberships or if teams data hasn't been loaded. 

---

### <kbd>property</kbd> User.user_api

An instance of the api using credentials from the user. 



---

### <kbd>classmethod</kbd> `User.create`

```python
create(api, email, admin=False)
```

Create a new user. 



**Args:**
 
 - `api` (`Api`):  The api instance to use 
 - `email` (str):  The name of the team 
 - `admin` (bool):  Whether this user should be a global instance admin 



**Returns:**
 A `User` object 

---

### <kbd>method</kbd> `User.delete_api_key`

```python
delete_api_key(api_key)
```

Delete a user's api key. 



**Args:**
 
 - `api_key` (str):  The name of the API key to delete. This should be  one of the names returned by the `api_keys` property. 



**Returns:**
 Boolean indicating success 



**Raises:**
 ValueError if the api_key couldn't be found 

---

### <kbd>method</kbd> `User.generate_api_key`

```python
generate_api_key(description=None)
```

Generate a new api key. 



**Args:**
 
 - `description` (str, optional):  A description for the new API key. This can be  used to identify the purpose of the API key. 



**Returns:**
 The new api key, or None on failure 


