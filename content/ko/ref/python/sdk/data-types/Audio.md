---
title: 오디오
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Audio
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/audio.py >}}




## <kbd>class</kbd> `Audio`
오디오 클립을 위한 W&B 클래스입니다.

### <kbd>method</kbd> `Audio.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, list, ForwardRef('np.ndarray')],
    sample_rate: Optional[int] = None,
    caption: Optional[str] = None
)
```

오디오 파일 경로나 오디오 데이터의 넘파이 배열을 입력받습니다.



**ARG:**
 
 - `data_or_path`:  오디오 파일의 경로나 오디오 데이터의 NumPy 배열입니다.
 - `sample_rate`:  원시 NumPy 배열을 입력할 때 필요한 샘플링 레이트입니다.
 - `caption`:  오디오와 함께 표시할 캡션입니다.




---



### <kbd>classmethod</kbd> `Audio.durations`

```python
durations(audio_list)
```

오디오 파일들의 재생 시간을 계산합니다.

---



### <kbd>classmethod</kbd> `Audio.sample_rates`

```python
sample_rates(audio_list)
```

오디오 파일의 샘플링 레이트를 가져옵니다.

---