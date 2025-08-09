---
title: confusion_matrix()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-confusion_matrix
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/confusion_matrix.py >}}




### <kbd>function</kbd> `confusion_matrix`

```python
confusion_matrix(
    probs: 'Sequence[Sequence[float]] | None' = None,
    y_true: 'Sequence[T] | None' = None,
    preds: 'Sequence[T] | None' = None,
    class_names: 'Sequence[str] | None' = None,
    title: 'str' = 'Confusion Matrix Curve',
    split_table: 'bool' = False
) → CustomChart
```

확률 또는 예측값 시퀀스로부터 혼동 행렬(confusion matrix)을 생성합니다. 



**인자:**
 
 - `probs`:  각 클래스별 예측 확률의 시퀀스입니다. 시퀀스의 형태는 (N, K)여야 하며, N은 샘플 개수, K는 클래스 개수입니다. 이 인자를 제공하는 경우, `preds`는 제공하지 않아야 합니다.
 - `y_true`:  실제 라벨의 시퀀스입니다.
 - `preds`:  예측된 클래스 라벨의 시퀀스입니다. 이 인자를 제공하는 경우, `probs`는 제공하지 않아야 합니다.
 - `class_names`:  클래스 이름의 시퀀스입니다. 제공하지 않으면 "Class_1", "Class_2" 등으로 자동 지정됩니다.
 - `title`:  혼동 행렬 차트의 제목입니다.
 - `split_table`:  테이블을 W&B UI 내 별도의 섹션에 분리하여 표시할지 여부입니다. `True`로 설정하면 "Custom Chart Tables"라는 섹션에 테이블이 표시됩니다. 기본값은 `False`입니다.



**반환값:**
 
 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()`에 전달하세요.



**예외 발생:**
 
 - `ValueError`:  `probs`와 `preds`를 동시에 제공하거나, 예측값과 실제 라벨의 개수가 다를 경우 발생합니다. 또는, 고유 예측 클래스 개수, 고유 실제 라벨 개수가 클래스 이름 개수를 초과할 경우에도 발생합니다.
 - `wandb.Error`:  numpy가 설치되지 않은 경우 발생합니다.



**예시:**
 야생동물 분류를 위한 랜덤 확률로 혼동 행렬을 로그하는 방법: 

```python
import numpy as np
import wandb

# 야생동물 클래스 이름 정의
wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]

# 10개 샘플에 대해 0~3의 랜덤 실제 라벨 생성
wildlife_y_true = np.random.randint(0, 4, size=10)

# 각 클래스에 대한 랜덤 확률 생성 (10샘플 x 4클래스)
wildlife_probs = np.random.rand(10, 4)
wildlife_probs = np.exp(wildlife_probs) / np.sum(
    np.exp(wildlife_probs),
    axis=1,
    keepdims=True,
)

# W&B run을 초기화하고 혼동 행렬을 로그합니다
with wandb.init(project="wildlife_classification") as run:
    confusion_matrix = wandb.plot.confusion_matrix(
         probs=wildlife_probs,
         y_true=wildlife_y_true,
         class_names=wildlife_class_names,
         title="Wildlife Classification Confusion Matrix",
    )
    run.log({"wildlife_confusion_matrix": confusion_matrix})
```

이 예시에서는 랜덤 확률을 사용해 혼동 행렬을 생성합니다. 

85% 정확도의 모델 예측으로 혼동 행렬을 로그하는 예시: 

```python
import numpy as np
import wandb

# 야생동물 클래스 이름 정의
wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]

# 200장 동물 이미지에 대한 실제 라벨 시뮬레이션 (불균형 분포)
wildlife_y_true = np.random.choice(
    [0, 1, 2, 3],
    size=200,
    p=[0.2, 0.3, 0.25, 0.25],
)

# 85% 정확도의 모델 예측값 시뮬레이션
wildlife_preds = [
    y_t
    if np.random.rand() < 0.85
    else np.random.choice([x for x in range(4) if x != y_t])
    for y_t in wildlife_y_true
]

# W&B run을 초기화하고 혼동 행렬을 로그합니다
with wandb.init(project="wildlife_classification") as run:
    confusion_matrix = wandb.plot.confusion_matrix(
         preds=wildlife_preds,
         y_true=wildlife_y_true,
         class_names=wildlife_class_names,
         title="Simulated Wildlife Classification Confusion Matrix",
    )
    run.log({"wildlife_confusion_matrix": confusion_matrix})
```

이 예시에서는 85% 정확도의 예측값을 시뮬레이션하여 혼동 행렬을 만듭니다.