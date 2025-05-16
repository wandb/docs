---
title: automations
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/automations.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Automation objects. 



## <kbd>class</kbd> `Automations`
An iterable collection of `Automation` objects. 

### <kbd>method</kbd> `Automations.__init__`

```python
__init__(
    client: '_Client',
    variables: 'Mapping[str, Any]',
    per_page: 'int' = 50,
    _query: 'Document | None' = None
)
```






---

### <kbd>property</kbd> Automations.cursor

The start cursor to use for the next page. 

---

### <kbd>property</kbd> Automations.more

Whether there are more items to fetch. 



---

### <kbd>method</kbd> `Automations.convert_objects`

```python
convert_objects() → Iterable[Automation]
```

Parse the page data into a list of objects. 


