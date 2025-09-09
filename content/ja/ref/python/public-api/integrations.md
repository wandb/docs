---
title: インテグレーション
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-integrations
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/integrations.py >}}




# <kbd>モジュール</kbd> `wandb.apis.public`
W&B の インテグレーション向けの 公開 API です。 

このモジュールは W&B の インテグレーション とやり取りするためのクラスを提供します。 

## <kbd>クラス</kbd> `Integrations`
`Integration` オブジェクトの遅延イテレータです。 

### <kbd>メソッド</kbd> `Integrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```






---



### <kbd>メソッド</kbd> `Integrations.convert_objects`

```python
convert_objects() → Iterable[Integration]
```

ページのデータを解析して、インテグレーションのリストに変換します。 


---