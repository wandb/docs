
# 히스토그램

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/data_types/histogram.py#L18-L96' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


히스토그램을 위한 wandb 클래스입니다.

```python
Histogram(
    sequence: Optional[Sequence] = None,
    np_histogram: Optional['NumpyHistogram'] = None,
    num_bins: int = 64
) -> None
```

이 개체는 numpy의 histogram 함수와 마찬가지로 작동합니다.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

#### 예시:

시퀀스에서 히스토그램 생성

```python
wandb.Histogram([1, 2, 3])
```

np.histogram에서 효율적으로 초기화하기.

```python
hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
```

| 인수 |  |
| :--- | :--- |
|  `sequence` |  (array_like) 히스토그램을 위한 입력 데이터 |
|  `np_histogram` |  (numpy 히스토그램) 사전에 계산된 히스토그램의 대안적 입력 |
|  `num_bins` |  (int) 히스토그램의 구간 수. 기본 구간 수는 64입니다. 최대 구간 수는 512입니다 |

| 속성 |  |
| :--- | :--- |
|  `bins` |  ([float]) 구간의 경계 |
|  `histogram` |  ([int]) 각 구간에 속하는 요소의 수 |

| 클래스 변수 |  |
| :--- | :--- |
|  `MAX_LENGTH`<a id="MAX_LENGTH"></a> |  `512` |