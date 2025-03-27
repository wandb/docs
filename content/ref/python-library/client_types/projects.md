---
title: projects
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: projects. 



## <kbd>class</kbd> `Projects`
An iterable collection of `Project` objects. 

### <kbd>method</kbd> `Projects.__init__`

```python
__init__(client, entity, per_page=50)
```






---

### <kbd>property</kbd> Projects.cursor





---

### <kbd>property</kbd> Projects.length





---

### <kbd>property</kbd> Projects.more







---

### <kbd>method</kbd> `Projects.convert_objects`

```python
convert_objects()
```






---

## <kbd>class</kbd> `Project`
A project is a namespace for runs. 

### <kbd>method</kbd> `Project.__init__`

```python
__init__(client, entity, project, attrs)
```






---

### <kbd>property</kbd> Project.path





---

### <kbd>property</kbd> Project.url







---

### <kbd>method</kbd> `Project.artifacts_types`

```python
artifacts_types(per_page=50)
```





---

### <kbd>method</kbd> `Project.sweeps`

```python
sweeps()
```





---

### <kbd>method</kbd> `Project.to_html`

```python
to_html(height=420, hidden=False)
```

Generate HTML containing an iframe displaying this project. 


