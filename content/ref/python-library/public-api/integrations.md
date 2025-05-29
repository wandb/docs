---
title: integrations
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/integrations.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for integrations. 

This module provides classes for interacting with W&B integrations. 

## <kbd>class</kbd> `Integrations`




### <kbd>method</kbd> `Integrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```






---

### <kbd>property</kbd> Integrations.cursor

The start cursor to use for the next page. 

---

### <kbd>property</kbd> Integrations.more

Whether there are more Integrations to fetch. 



---

### <kbd>method</kbd> `Integrations.convert_objects`

```python
convert_objects() → Iterable[Integration]
```

Parse the page data into a list of integrations. 


---

## <kbd>class</kbd> `WebhookIntegrations`




### <kbd>method</kbd> `WebhookIntegrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```






---

### <kbd>property</kbd> WebhookIntegrations.cursor

The start cursor to use for the next page. 

---

### <kbd>property</kbd> WebhookIntegrations.more

Whether there are more webhook integrations to fetch. 



---

### <kbd>method</kbd> `WebhookIntegrations.convert_objects`

```python
convert_objects() → Iterable[WebhookIntegration]
```

Parse the page data into a list of webhook integrations. 


---

## <kbd>class</kbd> `SlackIntegrations`




### <kbd>method</kbd> `SlackIntegrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```






---

### <kbd>property</kbd> SlackIntegrations.cursor

The start cursor to use for the next page. 

---

### <kbd>property</kbd> SlackIntegrations.more

Whether there are more Slack integrations to fetch. 



---

### <kbd>method</kbd> `SlackIntegrations.convert_objects`

```python
convert_objects() → Iterable[SlackIntegration]
```

Parse the page data into a list of Slack integrations. 


