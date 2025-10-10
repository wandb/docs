---
title: Team
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




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

