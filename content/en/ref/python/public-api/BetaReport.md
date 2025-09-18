---
title: BetaReport
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




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

