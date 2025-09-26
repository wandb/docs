---
title: Reports
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




## <kbd>class</kbd> `Reports`
Reports is a lazy iterator of `BetaReport` objects.

### <kbd>method</kbd> `Reports.__init__`

```python
__init__(client, project, name=None, entity=None, per_page=50)
```

**Args:**
 
 - `client` (`wandb.apis.internal.Api`):  The API client instance to use. 
 - `project` (`wandb.sdk.internal.Project`):  The project to fetch reports from. 
 - `name` (str, optional):  The name of the report to filter by. If `None`,  fetches all reports. 
 - `entity` (str, optional):  The entity name for the project. Defaults to  the project entity. 
 - `per_page` (int):  Number of reports to fetch per page (default is 50). 







---


### <kbd>property</kbd> Reports.length





---


### <kbd>method</kbd> `Reports.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to File objects. 

---

### <kbd>method</kbd> `Reports.update_variables`

```python
update_variables()
```

Updates the GraphQL query variables for pagination. 

