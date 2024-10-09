---
title: Embed objects
description: W&B의 Embedding Projector는 사용자가 PCA, UMAP, t-SNE와 같은 일반적인 차원 축소 알고리즘을 사용하여 다차원 임베딩을 2D 평면에 플로팅할 수 있도록 해줍니다.
displayed_sidebar: default
---

![](/images/weave/embedding_projector.png)

[임베딩](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)은 오브젝트(사람, 이미지, 게시물, 단어 등)를 숫자 목록으로 표현하는 데 사용됩니다 - 때때로 이를 _벡터_라고 합니다. 기계학습과 데이터 과학 유스 케이스에서, 임베딩은 다양한 애플리케이션 전반에 걸쳐 다양한 접근 방식을 통해 생성될 수 있습니다. 이 페이지는 독자가 임베딩에 익숙하며 W&B 내에서 이를 시각적으로 분석하는 데 관심이 있다고 가정합니다.

## 임베딩 예시

- [라이브 인터랙티브 데모 리포트](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq) 
- [Example Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_).

### Hello World

W&B는 `wandb.Table` 클래스를 사용하여 임베딩을 로그할 수 있습니다. 다음은 5차원으로 구성된 3개의 임베딩 예시입니다:

```python
import wandb

wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5],  # embedding 1
    [0.3, 0.1, 0.9, 0.2, 0.7],  # embedding 2
    [0.4, 0.5, 0.2, 0.2, 0.1],  # embedding 3
]
wandb.log(
    {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
)
wandb.finish()
```

위의 코드를 실행한 후, W&B 대시보드에는 새로운 테이블이 데이터와 함께 생성됩니다. 오른쪽 패널 선택기에서 `2D Projection`을 선택하여 2차원으로 임베딩을 플롯 할 수 있습니다. 스마트 기본값이 자동으로 선택되며, 이는 기어 아이콘을 클릭하여 액세스할 수 있는 설정 메뉴에서 쉽게 변경할 수 있습니다. 이 예시에서는 자동으로 제공된 모든 5개의 숫자 차원을 사용합니다.

![](/images/app_ui/weave_hello_world.png)

### Digits MNIST

위의 예가 임베딩을 로그하는 기본 메커니즘을 보여주지만, 일반적으로는 훨씬 더 많은 차원과 샘플을 다루게 됩니다. [SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)을 통해 제공되는 MNIST Digits 데이터셋을 고려해 보겠습니다. 이 데이터셋은 64차원을 가진 1797개의 레코드를 포함하고 있으며, 문제는 10클래스 분류 유스 케이스입니다. 우리는 입력 데이터를 이미지로 변환하여 시각화할 수도 있습니다.

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# 데이터셋 불러오기
ds = load_digits(as_frame=True)
df = ds.data

# "target" 열 생성
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# "image" 열 생성
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```

위의 코드를 실행한 후, 다시 UI에서 테이블이 생성됩니다. `2D Projection`을 선택하면 임베딩의 정의, 색상, 알고리즘(PCA, UMAP, t-SNE), 알고리즘 파라미터, 심지어 오버레이까지도 정의할 수 있습니다 (이 경우 포인트에 마우스를 올리면 이미지를 보여줍니다). 이 특정 사례에서는 모두 "스마트 기본값"이며, `2D Projection`을 단 한 번의 클릭으로 매우 유사한 결과를 볼 수 있습니다. ([이 예시와 상호작용하려면 여기를 클릭하세요](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney)).

![](/images/weave/embedding_projector.png)

## 로그 옵션

임베딩은 여러 가지 형식으로 로그할 수 있습니다:

1. **단일 임베딩 열:** 종종 데이터는 "매트릭스" 형식으로 이미 존재합니다. 이 경우, 셀 값의 데이터 유형이 `list[int]`, `list[float]`, 또는 `np.ndarray`인 단일 임베딩 열을 만들 수 있습니다.
2. **여러 숫자 열:** 위의 두 예시에서 우리가 사용한 접근 방식으로 각각의 차원에 대해 열을 만듭니다. 현재 우리는 셀에 대해 Python의 `int` 또는 `float`를 지원합니다.

![Single Embedding Column](/images/weave/logging_options.png)
![Many Numeric Columns](/images/weave/logging_option_image_right.png)

게다가, 모든 테이블과 마찬가지로, 테이블을 구성하는 방법에 대한 많은 옵션이 있습니다:

1. **데이터프레임**에서 직접 `wandb.Table(dataframe=df)` 사용
2. **데이터 목록**에서 직접 `wandb.Table(data=[...], columns=[...])` 사용
3. 테이블을 **루프에서 한 행씩** 점진적으로 빌드할 수 있음. `table.add_data(...)`를 사용하여 테이블에 행 추가
4. 테이블에 **임베딩 열** 추가 (임베딩 형태로 예측값 목록이 있는 경우에 적합): `table.add_col("col_name", ...)`
5. **계산된 열** 추가 (테이블에 매핑할 함수나 모델이 있는 경우에 적합): `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## 플롯 옵션

`2D Projection`을 선택한 후, 기어 아이콘을 클릭하여 렌더링 설정을 편집할 수 있습니다. 의도한 열을 선택할 수 있는 것 외에도 (위 참조), 관심 있는 알고리즘과 원하는 파라미터도 선택할 수 있습니다. 아래에는 각각 UMAP과 t-SNE의 파라미터가 나와 있습니다.

![](/images/weave/plotting_options_left.png) 
![](/images/weave/plotting_options_right.png)

:::info
참고: 현재 우리는 모든 세 알고리즘에 대해 1000행과 50차원의 무작위 서브셋으로 다운샘플합니다.
:::