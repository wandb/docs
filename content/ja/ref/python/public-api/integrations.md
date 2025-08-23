---
title: インテグレーション
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-integrations
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/integrations.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B パブリック API のインテグレーション用モジュールです。

このモジュールは、W&B インテグレーションにアクセスするためのクラスを提供します。

## <kbd>class</kbd> `Integrations`




### <kbd>method</kbd> `Integrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```






---



### <kbd>method</kbd> `Integrations.convert_objects`

```python
convert_objects() → Iterable[Integration]
```

ページ内のデータを解析し、インテグレーションのリストに変換します。


---