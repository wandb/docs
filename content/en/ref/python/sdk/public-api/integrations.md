---
title: Integrations
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/integrations.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for integrations. 

This module provides classes for interacting with W&B integrations. 

## <kbd>class</kbd> `Integrations`
An lazy iterator of `Integration` objects. 

### <kbd>method</kbd> `Integrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```






---



### <kbd>method</kbd> `Integrations.convert_objects`

```python
convert_objects() → Iterable[Integration]
```

Parse the page data into a list of integrations. 


---



