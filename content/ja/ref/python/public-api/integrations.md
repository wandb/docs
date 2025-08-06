---
title: インテグレーション
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/integrations.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for インテグレーション。

このモジュールは、W&B インテグレーションとのやり取りのためのクラスを提供します。

## <kbd>class</kbd> `Integrations`




### <kbd>method</kbd> `Integrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```
コンストラクタ。client、変数、およびページごとの件数を指定して初期化します。





---



### <kbd>method</kbd> `Integrations.convert_objects`

```python
convert_objects() → Iterable[Integration]
```

ページ内のデータを解析し、インテグレーションのリストに変換します。


---