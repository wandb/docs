---
title: roc_curve()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-custom-charts-roc_curve
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/roc_curve.py >}}




### <kbd>function</kbd> `roc_curve`

```python
roc_curve(
    y_true: 'Sequence[numbers.Number]',
    y_probas: 'Sequence[Sequence[float]] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[numbers.Number] | None' = None,
    title: 'str' = 'ROC Curve',
    split_table: 'bool' = False
) → CustomChart
```

수신자 조작 특성(ROC) 커브 차트를 생성합니다.



**인자:**
 
 - `y_true`:  타겟 변수의 실제 클래스 라벨(그라운드 트루스)입니다. 형태는 (num_samples,)여야 합니다.
 - `y_probas`:  각 클래스에 대한 예측 확률 또는 결정 점수입니다. 형태는 (num_samples, num_classes)여야 합니다.
 - `labels`:  `y_true` 내 클래스 인덱스에 해당하는 사람이 읽을 수 있는 라벨입니다. 예를 들어, `labels=['dog', 'cat']`인 경우 플롯에서 클래스 0은 'dog', 클래스 1은 'cat'으로 표시됩니다. None이라면, `y_true`의 원래 클래스 인덱스가 사용됩니다. 기본값은 None입니다.
 - `classes_to_plot`:  ROC 커브에 포함할 고유 클래스 라벨의 서브셋입니다. None이면, `y_true`에 있는 모든 클래스가 플롯됩니다. 기본값은 None입니다.
 - `title`:  ROC 커브 플롯의 제목입니다. 기본값은 "ROC Curve"입니다.
 - `split_table`:  테이블을 W&B UI 내에서 별도의 섹션으로 분리해 표시할지 여부입니다. `True`로 설정하면, 테이블이 "Custom Chart Tables"라는 이름의 섹션에 표시됩니다. 기본값은 `False`입니다.



**반환값:**
 
 - `CustomChart`:  W&B에 로그할 수 있는 커스텀 차트 오브젝트입니다. 차트를 로그하려면 `wandb.log()`에 전달하면 됩니다.



**예외:**
 
 - `wandb.Error`:  numpy, pandas 또는 scikit-learn이 설치되어 있지 않을 경우 발생합니다.



**예시:**
 ```python
import numpy as np
import wandb

# 세 가지 질병에 대한 의료 진단 분류 문제를 시뮬레이션합니다.
n_samples = 200
n_classes = 3

# 실제 라벨: 각 샘플마다 "Diabetes", "Hypertension", "Heart Disease" 중 하나를 지정합니다.
disease_labels = ["Diabetes", "Hypertension", "Heart Disease"]
# 0: Diabetes, 1: Hypertension, 2: Heart Disease
y_true = np.random.choice([0, 1, 2], size=n_samples)

# 예측 확률: 각 샘플별로 확률의 총합이 1이 되도록 예측값을 시뮬레이션합니다.
y_probas = np.random.dirichlet(np.ones(n_classes), size=n_samples)

# 플롯할 클래스 지정 (세 질병 모두 플롯)
classes_to_plot = [0, 1, 2]

# W&B run을 초기화하고 질병 분류에 대한 ROC 커브 플롯을 로그합니다.
with wandb.init(project="medical_diagnosis") as run:
    roc_plot = wandb.plot.roc_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=disease_labels,
         classes_to_plot=classes_to_plot,
         title="ROC Curve for Disease Classification",
    )
    run.log({"roc-curve": roc_plot})
```