---
title: teams
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/teams.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: teams. 



---

## <kbd>class</kbd> `Member`




### <kbd>method</kbd> `Member.__init__`

```python
__init__(client, team, attrs)
```








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




### <kbd>method</kbd> `Team.__init__`

```python
__init__(client, name, attrs=None)
```








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
 
 - `username_or_email`:  (str) The username or email address of the user you want to invite 
 - `admin`:  (bool) Whether to make this user a team admin, defaults to False 



**Returns:**
 True on success, False if user was already invited or didn't exist 

---

### <kbd>method</kbd> `Team.load`

```python
load(force=False)
```






