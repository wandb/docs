---
title: query_generator
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/query_generator.py >}}




# <kbd>module</kbd> `wandb.apis.public`






---

## <kbd>class</kbd> `QueryGenerator`
QueryGenerator is a helper object to write filters for runs. 

### <kbd>method</kbd> `QueryGenerator.__init__`

```python
__init__()
```








---

### <kbd>method</kbd> `QueryGenerator.filter_to_mongo`

```python
filter_to_mongo(filter)
```





---

### <kbd>classmethod</kbd> `QueryGenerator.format_order_key`

```python
format_order_key(key: str)
```





---

### <kbd>method</kbd> `QueryGenerator.key_to_server_path`

```python
key_to_server_path(key)
```





---

### <kbd>method</kbd> `QueryGenerator.keys_to_order`

```python
keys_to_order(keys)
```





---

### <kbd>method</kbd> `QueryGenerator.mongo_to_filter`

```python
mongo_to_filter(filter)
```





---

### <kbd>method</kbd> `QueryGenerator.order_to_keys`

```python
order_to_keys(order)
```





---

### <kbd>method</kbd> `QueryGenerator.server_path_to_key`

```python
server_path_to_key(path)
```






