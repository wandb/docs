---
title: reports
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Report objects. 

This module provides classes for interacting with W&B reports and managing report-related data. 



---

## <kbd>class</kbd> `Reports`
Reports is an iterable collection of `BetaReport` objects. 



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

The number of reports in the project. 

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

WARNING: this API will likely change in a future release 



**Attributes:**
 
 - `name` (string):  The name of the report. 
 - `description` (string):  Report description. 
 - `user` (User):  The user that created the report. 
 - `spec` (dict):  The spec off the report. 
 - `updated_at` (string):  timestamp of last update. 

### <kbd>method</kbd> `BetaReport.__init__`

```python
__init__(client, attrs, entity=None, project=None)
```






---

### <kbd>property</kbd> BetaReport.sections

Get the panel sections (groups) from the report. 

---

### <kbd>property</kbd> BetaReport.updated_at

Timestamp of last update 

---

### <kbd>property</kbd> BetaReport.url

URL of the report. 

Contains the entity, project, display name, and id. 



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



