---
title: query_generator
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/query_generator.py >}}




# <kbd>module</kbd> `wandb.apis.public`






---


### <kbd>method</kbd> `QueryGenerator.filter_to_mongo`

```python
filter_to_mongo(filter)
```

Returns dictionary with filter format converted to MongoDB filter. 

---

### <kbd>classmethod</kbd> `QueryGenerator.format_order_key`

```python
format_order_key(key: str)
```

Format a key for sorting. 

---

### <kbd>method</kbd> `QueryGenerator.key_to_server_path`

```python
key_to_server_path(key)
```

Convert a key dictionary to the corresponding server path string. 

---

### <kbd>method</kbd> `QueryGenerator.keys_to_order`

```python
keys_to_order(keys)
```

Convert a list of key dictionaries to an order string. 

---

### <kbd>method</kbd> `QueryGenerator.mongo_to_filter`

```python
mongo_to_filter(filter)
```

Returns dictionary with MongoDB filter converted to filter format. 

---

### <kbd>method</kbd> `QueryGenerator.order_to_keys`

```python
order_to_keys(order)
```

Convert an order string to a list of key dictionaries. 

---

### <kbd>method</kbd> `QueryGenerator.server_path_to_key`

```python
server_path_to_key(path)
```

Convert a server path string to the corresponding key dictionary. 


