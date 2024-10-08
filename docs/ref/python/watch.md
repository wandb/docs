# watch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_watch.py#L20-L106' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

torch 모델에 훅을 걸어 그레이디언트와 토폴로지를 수집합니다.

```python
watch(
    models,
    criterion=None,
    log: Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
    log_freq: int = 1000,
    idx: Optional[int] = None,
    log_graph: bool = (False)
)
```

임의의 머신러닝 모델을 허용하도록 확장되어야 합니다.

| Args |  |
| :--- | :--- |
|  `models` |  (torch.Module) 훅을 걸 모델로, 튜플일 수 있습니다 |
|  `criterion` |  (torch.F) 최적화되는 선택적 손실 값 |
|  `log` |  (str) "gradients", "parameters", "all" 중 하나이거나 None |
|  `log_freq` |  (int) 매 N 배치마다 그레이디언트와 파라미터를 로그 |
|  `idx` |  (int) 여러 모델에서 wandb.watch를 호출할 때 사용될 인덱스 |
|  `log_graph` |  (boolean) 그래프 토폴로지를 로그 |

| Returns |  |
| :--- | :--- |
|  `wandb.Graph`: 첫 번째 역전파 후 채워질 그래프 오브젝트 |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init`보다 먼저 호출되거나, models 중 어느 하나라도 torch.nn.Module이 아닐 경우 |
