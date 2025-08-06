---
title: 인테그레이션
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-integrations
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/integrations.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B 인테그레이션을 위한 Public API입니다.

이 모듈은 W&B 인테그레이션과 상호작용할 수 있는 클래스들을 제공합니다.

## <kbd>class</kbd> `Integrations`




### <kbd>method</kbd> `Integrations.__init__`

```python
__init__(client: '_Client', variables: 'dict[str, Any]', per_page: 'int' = 50)
```
Integrations 클래스의 인스턴스를 초기화합니다.




---



### <kbd>method</kbd> `Integrations.convert_objects`

```python
convert_objects() → Iterable[Integration]
```

페이지 데이터를 인테그레이션 목록으로 파싱합니다.


---