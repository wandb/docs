# Audio

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L983-L1127' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

오디오 클립을 위한 wandb 클래스입니다.

```python
Audio(
    data_or_path, sample_rate=None, caption=None
)
```

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (string 또는 numpy array) 오디오 파일 경로 또는 오디오 데이터의 numpy array. |
|  `sample_rate` |  (int) 오디오 데이터의 원시 numpy array를 전달할 때 필요한 샘플 레이트. |
|  `caption` |  (string) 오디오와 함께 표시할 캡션. |

## 메소드

### `durations`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L1085-L1087)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L1101-L1113)

```python
resolve_ref()
```

### `sample_rates`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L1089-L1091)

```python
@classmethod
sample_rates(
    audio_list
)
```