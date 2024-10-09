# WBTraceTree

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/trace_tree.py#L80-L119' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

트레이스 트리 데이터를 위한 미디어 오브젝트입니다.

```python
WBTraceTree(
    root_span: Span,
    model_dict: typing.Optional[dict] = None
)
```

| 인수 |  |
| :--- | :--- |
|  root_span (Span): 트레이스 트리의 루트 스팬입니다. model_dict (dict, optional): 모델 덤프를 포함하는 사전입니다. 참고: model_dict는 전적으로 사용자가 정의한 사전입니다. UI는 이 사전에 대해 JSON 뷰어를 렌더링하며, `_kind` 키를 가진 사전은 특별히 처리합니다. 이는 모델 공급업체의 직렬화 형식이 매우 다르기 때문에 유연성이 필요하기 때문입니다. |