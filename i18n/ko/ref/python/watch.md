
# watch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_watch.py#L20-L106' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


그레이디언트와 토폴로지를 수집하기 위해 토치 모델에 후크합니다.

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

임의의 ML 모델을 받아들일 수 있도록 확장되어야 합니다.

| 인수 |  |
| :--- | :--- |
|  `models` |  (torch.Module) 후크할 모델, 튜플일 수 있음 |
|  `criterion` |  (torch.F) 최적화되고 있는 선택적인 손실 값 |
|  `log` |  (str) "gradients", "parameters", "all", 또는 None 중 하나 |
|  `log_freq` |  (int) N 배치마다 그레이디언트와 파라미터를 로그 |
|  `idx` |  (int) 여러 모델에 wandb.watch를 호출할 때 사용될 인덱스 |
|  `log_graph` |  (boolean) 그래프 토폴로지를 로그 |

| 반환값 |  |
| :--- | :--- |
|  `wandb.Graph`: 첫 번째 역전파 후 채워질 그래프 개체 |

| 예외 |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init` 전에 호출되거나 모델 중 하나가 torch.nn.Module이 아닌 경우 발생합니다. |