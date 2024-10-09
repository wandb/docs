# 컨트롤러

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_sweep.py#L95-L119' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 코드 보기</a></button></p>

공용 스윕 컨트롤러 생성자.

```python
controller(
    sweep_id_or_config: Optional[Union[str, Dict]] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> "_WandbController"
```

#### 사용법:

```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)  # 스윕 설정 출력
print(tuner.sweep_id)      # 스윕 ID 출력
tuner.configure_search(...) # 탐색 구성
tuner.configure_stopping(...) # 중지 구성
```