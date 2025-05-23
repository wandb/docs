---
title: Html
menu:
  reference:
    identifier: ko-ref-python-data-types-html
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/html.py#L18-L107 >}}

임의의 HTML을 위한 Wandb 클래스입니다.

```python
Html(
    data: Union[str, 'TextIO'],
    inject: bool = (True)
) -> None
```

| ARG |  |
| :--- | :--- |
|  `data` |  (문자열 또는 IO 오브젝트) wandb에 표시할 HTML |
|  `inject` |  (부울) HTML 오브젝트에 스타일시트를 추가합니다. False로 설정하면 HTML이 변경되지 않고 전달됩니다. |

## 메소드

### `inject_head`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/html.py#L59-L74)

```python
inject_head() -> None
```
