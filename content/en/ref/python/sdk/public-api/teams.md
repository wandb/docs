---
title: teams
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for managing teams and team members. 

This module provides classes for managing W&B teams and their members. 



**Note:**

> This module is part of the W&B Public API and provides methods to manage teams and their members. Team management operations require appropriate permissions. 



---

## <kbd>class</kbd> `Member`
A member of a team.

### <kbd>method</kbd> `Member.__init__`

```python
__init__(client, team, attrs)
```

**Args:**
 
 - `client` (`wandb.apis.internal.Api`):  The client instance to use 
 - `team` (str):  The name of the team this member belongs to 
 - `attrs` (dict):  The member attributes 









---

### <kbd>method</kbd> `Member.delete`

```python
delete()
```

Remove a member from a team. 



**Returns:**
  Boolean indicating success 


---

## <kbd>class</kbd> `Team`
A class that represents a W&B team. 

This class provides methods to manage W&B teams, including creating teams, inviting members, and managing service accounts. It inherits from Attrs to handle team attributes.

### <kbd>method</kbd> `Team.__init__`

```python
__init__(client, name, attrs=None)
```

**Args:**
 
 - `client` (`wandb.apis.public.Api`):  The api instance to use 
 - `name` (str):  The name of the team 
 - `attrs` (dict):  Optional dictionary of team attributes 



**Note:**

> Team management requires appropriate permissions. 









---

### <kbd>classmethod</kbd> `Team.create`

```python
create(api, team, admin_username=None)
```

Create a new team. 



**Args:**
 
 - `api`:  (`Api`) The api instance to use 
 - `team`:  (str) The name of the team 
 - `admin_username`:  (str) optional username of the admin user of the team, defaults to the current user. 



**Returns:**
 A `Team` object 

---

### <kbd>method</kbd> `Team.create_service_account`

```python
create_service_account(description)
```

Create a service account for the team. 



**Args:**
 
 - `description`:  (str) A description for this service account 



**Returns:**
 The service account `Member` object, or None on failure 

---

### <kbd>method</kbd> `Team.invite`

```python
invite(username_or_email, admin=False)
```

Invite a user to a team. 



**Args:**
 
 - `username_or_email`:  (str) The username or email address of the user  you want to invite. 
 - `admin`:  (bool) Whether to make this user a team admin.  Defaults to `False`. 



**Returns:**
 `True` on success, `False` if user was already invited or didn't exist. 

---

