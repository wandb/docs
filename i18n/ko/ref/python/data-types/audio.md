
# 오디오

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L979-L1123' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

오디오 클립을 위한 Wandb 클래스.

```python
Audio(
    data_or_path, sample_rate=None, caption=None
)
```

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (문자열 또는 numpy 배열) 오디오 파일의 경로 또는 오디오 데이터의 numpy 배열입니다. |
|  `sample_rate` |  (정수) 오디오 데이터의 raw numpy 배열을 전달할 때 필요한 샘플 레이트입니다. |
|  `caption` |  (문자열) 오디오와 함께 표시할 캡션입니다. |

## 메소드

### `durations`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1081-L1083)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1097-L1109)

```python
resolve_ref()
```

### `sample_rates`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L1085-L1087)

```python
@classmethod
sample_rates(
    audio_list
)
```