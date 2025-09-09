---
title: Report
object_type: python_sdk_reports_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `Report`
An object that represents a W&B Report. Use the returned object's `blocks` attribute to customize your report. Report objects do not automatically save. Use the `save()` method to persists changes. 



**Attributes:**
 
 - `project` (str):  The name of the W&B project you want to load in.  The project specified appears in the report's URL. 
 - `entity` (str):  The W&B entity that owns the report.  The entity appears in the report's URL. 
 - `title` (str):  The title of the report. The title  appears at the top of the report as an H1 heading. 
 - `description` (str):  A description of the report.  The description appears underneath the report's title. 
 - `blocks` (LList[BlockTypes]):  A list of one or more HTML tags,  plots, grids, runsets, and more. 
 - `width` (Literal['readable', 'fixed', 'fluid']):  The width of the report. Options include 'readable', 'fixed', 'fluid'. 


---

### <kbd>property</kbd> Report.url

The URL where the report is hosted. The report URL consists of `https://wandb.ai/{entity}/{project_name}/reports/`. Where `{entity}` and `{project_name}` consists of the entity that the report belongs to and the name of the project, respectively. 



---

### <kbd>method</kbd> `Report.delete`

```python
delete() → bool
```

Delete this report from W&B. 

This will also delete any draft views that reference this report. 



**Returns:**
 
 - `bool`:  ``True`` if the delete operation was acknowledged as successful by the backend, ``False`` otherwise. 

---

### <kbd>classmethod</kbd> `Report.from_url`

```python
from_url(url: str, as_model: bool = False)
```

Load in the report into current environment. Pass in the URL where the report is hosted. 



**Arguments:**
 
 - `url` (str):  The URL where the report is hosted. 
 - `as_model` (bool):  If True, return the model object instead of the Report object.  By default, set to `False`. 

---

### <kbd>method</kbd> `Report.save`

```python
save(draft: bool = False, clone: bool = False)
```

Persists changes made to a report object. 

---

### <kbd>method</kbd> `Report.to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

Generate HTML containing an iframe displaying this report. Commonly used to within a Python notebook. 



**Arguments:**
 
 - `height` (int):  Height of the iframe. 
 - `hidden` (bool):  If True, hide the iframe. Default set to `False`. 

