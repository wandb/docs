# Html

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/html.py#L18-L107' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

임의의 HTML을 위한 Wandb 클래스입니다.

```python
Html(
    data: Union[str, 'TextIO'],
    inject: bool = (True)
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `data` |  (문자열 또는 io 오브젝트) wandb에 표시할 HTML |
|  `inject` |  (불리언) HTML 오브젝트에 스타일시트를 추가합니다. False로 설정하면 HTML은 수정 없이 그대로 전달됩니다. |

## 메소드

### `inject_head`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/html.py#L59-L74)

```python
inject_head() -> None
```