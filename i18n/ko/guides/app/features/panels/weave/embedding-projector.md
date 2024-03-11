---
description: W&B's Embedding Projector allows users to plot multi-dimensional embeddings
  on a 2D plane using common dimension reduction algorithms like PCA, UMAP, and t-SNE.
displayed_sidebar: default
---

# 임베딩 프로젝터

![](/images/weave/embedding_projector.png)

[임베딩](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)은 오브젝트(사람, 이미지, 게시물, 단어 등)를 숫자 목록으로 나타내는 데 사용되며, 때때로 _벡터_라고도 합니다. 기계학습 및 데이터 과학 유스 케이스에서는 다양한 접근 방식을 사용하여 다양한 애플리케이션에서 임베딩을 생성할 수 있습니다. 이 페이지는 독자가 임베딩에 익숙하며 W&B 내부에서 시각적으로 분석하는 데 관심이 있다고 가정합니다.

## 임베딩 예시

[라이브 인터랙티브 데모 리포트](https://wandb.ai/timssweeney/toy\_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq)로 바로 점프하거나 이 리포트의 코드를 [예제 코랩](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm\_)에서 실행할 수 있습니다.

### 안녕 세상

W&B는 `wandb.Table` 클래스를 사용하여 임베딩을 로그할 수 있습니다. 다음은 각각 5차원으로 구성된 3개의 임베딩에 대한 예입니다:

```python
import wandb

wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5],  # 임베딩 1
    [0.3, 0.1, 0.9, 0.2, 0.7],  # 임베딩 2
    [0.4, 0.5, 0.2, 0.2, 0.1],  # 임베딩 3
]
wandb.log(
    {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
)
wandb.finish()
```

위 코드를 실행한 후, W&B 대시보드에 데이터가 포함된 새 테이블이 표시됩니다. 우측 상단 패널 선택기에서 `2D 프로젝션`을 선택하여 임베딩을 2차원으로 플롯할 수 있습니다. 자동으로 선택되는 스마트 기본값을 쉽게 재정의할 수 있으며, 톱니바퀴 아이콘을 클릭하여 설정 메뉴에 접근하여 변경할 수 있습니다. 이 예에서는 사용 가능한 5개의 숫자 차원을 모두 자동으로 사용합니다.

![](/images/app_ui/weave_hello_world.png)

### 숫자 MNIST

위의 예시는 임베딩 로깅의 기본 메커니즘을 보여줍니다만, 일반적으로 훨씬 더 많은 차원과 샘플로 작업하게 됩니다. MNIST 숫자 데이터셋([UCI ML 필기 숫자 데이터셋](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits))을 고려해 보겠습니다. 이 데이터셋은 1797개의 레코드를 가지고 있으며, 각각 64차원입니다. 문제는 10 클래스 분류 유스 케이스입니다. 입력 데이터를 시각화를 위해 이미지로 변환할 수도 있습니다.

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# 데이터셋 로드
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

위 코드를 다시 실행한 후, UI에 테이블이 표시됩니다. `2D 프로젝션`을 선택하면 임베딩의 정의, 색상, 알고리즘(PCA, UMAP, t-SNE), 알고리즘 파라미터를 구성할 수 있으며, 심지어 오버레이(이 경우에는 포인트 위에 마우스를 올렸을 때 이미지를 보여줍니다)까지 설정할 수 있습니다. 이 특정 경우에는 모두 "스마트 기본값"이며, `2D 프로젝션`을 한 번 클릭하면 매우 유사한 것을 볼 수 있습니다. ([여기를 클릭하여](https://wandb.ai/timssweeney/embedding\_tutorial/runs/k6guxhum?workspace=user-timssweeney) 이 예시와 상호작용해 보세요).

![](/images/weave/embedding_projector.png)

## 로깅 옵션

다양한 형식으로 임베딩을 로그할 수 있습니다:

1. **단일 임베딩 열:** 종종 데이터가 이미 "행렬"-같은 형식으로 되어 있습니다. 이 경우, 셀 값의 데이터 유형이 `list[int]`, `list[float]`, 또는 `np.ndarray`일 수 있는 단일 임베딩 열을 생성할 수 있습니다.
2. **다수의 숫자 열:** 위의 두 예에서는 이 접근 방식을 사용하고 각 차원에 대한 열을 생성합니다. 현재 셀에 대해 파이썬 `int` 또는 `float`을 허용합니다.

![단일 임베딩 열](/images/weave/logging_options.png)
![다수의 숫자 열](/images/weave/logging_option_image_right.png)

또한 모든 테이블처럼 테이블을 구성하는 데 많은 옵션이 있습니다:

1. `wandb.Table(dataframe=df)`를 사용하여 **데이터프레임**에서 직접
2. `wandb.Table(data=[...], columns=[...])`를 사용하여 **데이터 목록**에서 직접
3. **행별로 점진적으로 테이블 구축** (코드에 루프가 있는 경우 유용함). `table.add_data(...)`를 사용하여 테이블에 행 추가
4. **임베딩 열**을 테이블에 추가 (임베딩 형태의 예측값 목록이 있는 경우 유용함): `table.add_col("col_name", ...)`
5. **계산된 열 추가** (테이블에 매핑하고자 하는 함수 또는 모델이 있는 경우 유용함): `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## 플로팅 옵션

`2D 프로젝션`을 선택한 후, 기어 아이콘을 클릭하여 렌더링 설정을 편집할 수 있습니다. 위에서 언급한 것처럼 의도한 열을 선택할 수 있을 뿐만 아니라, 관심 있는 알고리즘(및 원하는 파라미터)을 선택할 수 있습니다. 아래에서는 각각 UMAP과 t-SNE의 파라미터를 볼 수 있습니다.

![](/images/weave/plotting_options_left.png) 
![](/images/weave/plotting_options_right.png)

:::안내
참고: 현재 모든 세 알고리즘에 대해 1000 행과 50 차원까지의 무작위 서브셋으로 다운샘플링합니다.
:::