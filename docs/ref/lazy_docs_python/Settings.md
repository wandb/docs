import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Settings

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py'/>




## <kbd>class</kbd> `Settings`
Settings for the W&B SDK. 

### <kbd>method</kbd> `Settings.__init__`

```python
__init__(**kwargs: Any) → None
```








---

### <kbd>method</kbd> `Settings.copy`

```python
copy() → Settings
```





---

### <kbd>method</kbd> `Settings.freeze`

```python
freeze() → None
```





---

### <kbd>method</kbd> `Settings.get`

```python
get(key: str, default: Optional[Any] = None) → Any
```





---

### <kbd>method</kbd> `Settings.is_frozen`

```python
is_frozen() → bool
```





---

### <kbd>method</kbd> `Settings.items`

```python
items() → ItemsView[str, Any]
```





---

### <kbd>method</kbd> `Settings.keys`

```python
keys() → Iterable[str]
```





---

### <kbd>method</kbd> `Settings.to_dict`

```python
to_dict() → Dict[str, Any]
```

Return a dict representation of the settings. 

---

### <kbd>method</kbd> `Settings.to_proto`

```python
to_proto() → Settings
```

Generate a protobuf representation of the settings. 

---

### <kbd>method</kbd> `Settings.unfreeze`

```python
unfreeze() → None
```





---

### <kbd>method</kbd> `Settings.update`

```python
update(
    settings: Optional[Dict[str, Any], ForwardRef('Settings')] = None,
    source: int = <Source.OVERRIDE: 0>,
    **kwargs: Any
) → None
```

Update individual settings.