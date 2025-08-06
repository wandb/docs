---
title: pr_curve()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-pr_curve
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/pr_curve.py >}}




### <kbd>function</kbd> `pr_curve`

```python
pr_curve(
    y_true: 'Iterable[T] | None' = None,
    y_probas: 'Iterable[numbers.Number] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[T] | None' = None,
    interp_size: 'int' = 21,
    title: 'str' = 'Precision-Recall Curve',
    split_table: 'bool' = False
) → CustomChart
```

PR(Precision-Recall) 곡선을 생성합니다.

PR 곡선은 불균형 데이터셋에서 분류기를 평가할 때 특히 유용합니다. PR 곡선 아래 면적이 높다는 것은 높은 정밀도(거짓 양성 비율이 낮음)와 높은 재현율(거짓 음성 비율이 낮음)을 모두 의미합니다. 이 곡선은 다양한 임계값에서 거짓 양성과 거짓 음성의 균형을 시각적으로 보여주어 모델의 성능을 평가하는 데 도움을 줍니다.



**인자(Args):**
 
 - `y_true`:  실제 이진 레이블. 형태는 (`num_samples`,)이어야 합니다. 
 - `y_probas`:  각 클래스에 대한 예측 점수 또는 확률.  확률 추정치, 신뢰 점수 또는 임계값이 없는 결정 값이 들어갈 수 있습니다. 형태는 (`num_samples`, `num_classes`)입니다. 
 - `labels`:  숫자 레이블을 시각적으로 더 보기 쉽게 변경해주는  선택적 클래스명 리스트입니다.  예를 들어, `labels = ['dog', 'cat', 'owl']`로 설정하면 0은 'dog', 1은 'cat', 2는 'owl'로 그림에서 표시됩니다. 만약 제공하지 않으면, `y_true`의 숫자 값이 그대로 사용됩니다. 
 - `classes_to_plot`:  y_true에서 PR 곡선에 포함할  고유 클래스 값들의 선택적 리스트입니다. 지정하지 않으면 y_true에 있는 모든 고유 클래스가 그려집니다. 
 - `interp_size`:  recall 값을 보간할 지점의 개수.  recall 값이 [0, 1] 구간에 균일하게 분포된 `interp_size`개로 고정되며,  이에 따라 precision 값이 보간됩니다. 
 - `title`:  플롯의 제목. 기본값은 "Precision-Recall Curve"입니다. 
 - `split_table`:  W&B UI에서 테이블을 별도의 섹션에 분리해서  보여줄지 여부입니다. `True`로 하면 "Custom Chart Tables"라는 이름의  별도 섹션에 테이블이 표시됩니다. 기본값은 `False`입니다. 



**반환(Returns):**
 
 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다.  이 차트를 로그하려면 `wandb.log()`에 전달해 주세요.



**예외(Raises):**
 
 - `wandb.Error`:  NumPy, pandas, 또는 scikit-learn이 설치되어 있지 않을 때 발생합니다.





**예시(Example):**
 

```python
import wandb

# 스팸 탐지를 위한 이진 분류 예시
y_true = [0, 1, 1, 0, 1]  # 0 = 스팸 아님, 1 = 스팸
y_probas = [
    [0.9, 0.1],  # 첫 번째 샘플에 대한 예측 확률 (스팸 아님)
    [0.2, 0.8],  # 두 번째 샘플 (스팸), 이하 동일
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7],
]

labels = ["not spam", "spam"]  # 가독성을 위한 선택적 클래스명

with wandb.init(project="spam-detection") as run:
    pr_curve = wandb.plot.pr_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=labels,
         title="Precision-Recall Curve for Spam Detection",
    )
    run.log({"pr-curve": pr_curve})
```