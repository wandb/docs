---
title: Reports
object_type: public_apis_namespace
data_type_classification: module
---
> Training and fine-tuning models is done elsewhere in [the W&B Python SDK]({{< relref "/ref/python/sdk" >}}), not the Public API.

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Report objects. 

This module provides classes for interacting with W&B reports and managing report-related data. 



---

## <kbd>class</kbd> `Reports`
Reports is a lazy iterator of `BetaReport` objects. 



**Args:**
 
 - `client` (`wandb.apis.internal.Api`):  The API client instance to use. 
 - `project` (`wandb.sdk.internal.Project`):  The project to fetch reports from. 
 - `name` (str, optional):  The name of the report to filter by. If `None`,  fetches all reports. 
 - `entity` (str, optional):  The entity name for the project. Defaults to  the project entity. 
 - `per_page` (int):  Number of reports to fetch per page (default is 50). 

### <kbd>method</kbd> `Reports.__init__`

```python
__init__(client, project, name=None, entity=None, per_page=50)
```






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


---

## <kbd>class</kbd> `BetaReport`
BetaReport is a class associated with reports created in W&B. 

Provides access to report attributes (name, description, user, spec, timestamps) and methods for retrieving associated runs, sections, and for rendering the report as HTML. 



**Attributes:**
 
 - `id` (string):  Unique identifier of the report. 
 - `display_name` (string):  Human-readable display name of the report. 
 - `name` (string):  The name of the report. Use `display_name` for a more user-friendly name. 
 - `description` (string):  Description of the report. 
 - `user` (User):  Dictionary containing user info (username, email) who  created the report. 
 - `spec` (dict):  The spec of the report. 
 - `url` (string):  The URL of the report. 
 - `updated_at` (string):  Timestamp of last update. 
 - `created_at` (string):  Timestamp when the report was created. 

### <kbd>method</kbd> `BetaReport.__init__`

```python
__init__(client, attrs, entity=None, project=None)
```






---

### <kbd>property</kbd> BetaReport.created_at





---

### <kbd>property</kbd> BetaReport.description





---

### <kbd>property</kbd> BetaReport.display_name





---

### <kbd>property</kbd> BetaReport.id





---

### <kbd>property</kbd> BetaReport.name





---

### <kbd>property</kbd> BetaReport.sections

Get the panel sections (groups) from the report. 

---

### <kbd>property</kbd> BetaReport.spec





---

### <kbd>property</kbd> BetaReport.updated_at





---

### <kbd>property</kbd> BetaReport.url





---

### <kbd>property</kbd> BetaReport.user







---

### <kbd>method</kbd> `BetaReport.runs`

```python
runs(section, per_page=50, only_selected=True)
```

Get runs associated with a section of the report. 

---

### <kbd>method</kbd> `BetaReport.to_html`

```python
to_html(height=1024, hidden=False)
```

Generate HTML containing an iframe displaying this report. 


---



