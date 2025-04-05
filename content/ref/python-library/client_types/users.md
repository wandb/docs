---
title: users
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/users.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: users. 



---

## <kbd>class</kbd> `User`




### <kbd>method</kbd> `User.__init__`

```python
__init__(client, attrs)
```






---

### <kbd>property</kbd> User.api_keys





---

### <kbd>property</kbd> User.teams





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
 
 - `api`:  (`Api`) The api instance to use 
 - `email`:  (str) The name of the team 
 - `admin`:  (bool) Whether this user should be a global instance admin 



**Returns:**
 A `User` object 

---

### <kbd>method</kbd> `User.delete_api_key`

```python
delete_api_key(api_key)
```

Delete a user's api key. 



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



**Returns:**
  The new api key, or None on failure 


