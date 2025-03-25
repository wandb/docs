---
title: reports
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: reports. 



---

## <kbd>class</kbd> `Reports`
Reports is an iterable collection of `BetaReport` objects. 

### <kbd>method</kbd> `Reports.__init__`

```python
__init__(client, project, name=None, entity=None, per_page=50)
```






---

### <kbd>property</kbd> Reports.cursor





---

### <kbd>property</kbd> Reports.length





---

### <kbd>property</kbd> Reports.more







---

### <kbd>method</kbd> `Reports.convert_objects`

```python
convert_objects()
```





---

### <kbd>method</kbd> `Reports.update_variables`

```python
update_variables()
```






---

## <kbd>class</kbd> `BetaReport`
BetaReport is a class associated with reports created in wandb. 

WARNING: this API will likely change in a future release 



**Attributes:**
 
 - `name` (string):  report name 
 - `description` (string):  report description; 
 - `user` (User):  the user that created the report 
 - `spec` (dict):  the spec off the report; 
 - `updated_at` (string):  timestamp of last update 

### <kbd>method</kbd> `BetaReport.__init__`

```python
__init__(client, attrs, entity=None, project=None)
```






---

### <kbd>property</kbd> BetaReport.sections





---

### <kbd>property</kbd> BetaReport.updated_at





---

### <kbd>property</kbd> BetaReport.url







---

### <kbd>method</kbd> `BetaReport.runs`

```python
runs(section, per_page=50, only_selected=True)
```





---

### <kbd>method</kbd> `BetaReport.to_html`

```python
to_html(height=1024, hidden=False)
```

Generate HTML containing an iframe displaying this report. 


---

## <kbd>class</kbd> `PythonMongoishQueryGenerator`




### <kbd>method</kbd> `PythonMongoishQueryGenerator.__init__`

```python
__init__(run_set)
```








---

### <kbd>method</kbd> `PythonMongoishQueryGenerator.back_to_front`

```python
back_to_front(name)
```





---

### <kbd>method</kbd> `PythonMongoishQueryGenerator.front_to_back`

```python
front_to_back(name)
```





---

### <kbd>method</kbd> `PythonMongoishQueryGenerator.pc_back_to_front`

```python
pc_back_to_front(name)
```





---

### <kbd>method</kbd> `PythonMongoishQueryGenerator.pc_front_to_back`

```python
pc_front_to_back(name)
```





---

### <kbd>method</kbd> `PythonMongoishQueryGenerator.python_to_mongo`

```python
python_to_mongo(filterstr)
```






---

## <kbd>class</kbd> `PanelMetricsHelper`







---

### <kbd>method</kbd> `PanelMetricsHelper.back_to_front`

```python
back_to_front(name)
```





---

### <kbd>method</kbd> `PanelMetricsHelper.front_to_back`

```python
front_to_back(name)
```





---

### <kbd>method</kbd> `PanelMetricsHelper.special_back_to_front`

```python
special_back_to_front(name)
```





---

### <kbd>method</kbd> `PanelMetricsHelper.special_front_to_back`

```python
special_front_to_back(name)
```






