---
title: Member
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




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

