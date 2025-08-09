---
title: 오브젝트 임베드하기
description: W&B의 Embedding Projector를 사용하면 사용자들이 PCA, UMAP, t-SNE와 같은 일반적인 차원 축소
  알고리즘을 이용하여 다차원 임베딩을 2D 평면에 시각화할 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-app-features-panels-query-panels-embedding-projector
    parent: query-panels
---

{{< img src="/images/weave/embedding_projector.png" alt="Embedding projector" >}}

[임베딩(Embeddings)](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)은 오브젝트(사람, 이미지, 게시글, 단어 등)를 일련의 숫자(벡터라고도 함)로 표현하는 방법입니다. 기계학습 및 데이터 사이언스 유스 케이스에서는 다양한 방식과 다양한 애플리케이션 전반에 걸쳐 임베딩을 생성할 수 있습니다. 이 페이지는 독자가 임베딩에 익숙하며, W&B에서 임베딩을 시각적으로 분석하고자 한다고 가정합니다.

## 임베딩 예시

- [라이브 인터랙티브 데모 Reports](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq) 
- [예제 Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_)

### Hello World

W&B에서는 `wandb.Table` 클래스를 이용해 임베딩을 로그할 수 있습니다. 다음은 5차원으로 구성된 3개의 임베딩 예시입니다.

```python
import wandb

with wandb.init(project="embedding_tutorial") as run:
  embeddings = [
      # D1   D2   D3   D4   D5
      [0.2, 0.4, 0.1, 0.7, 0.5],  # 임베딩 1
      [0.3, 0.1, 0.9, 0.2, 0.7],  # 임베딩 2
      [0.4, 0.5, 0.2, 0.2, 0.1],  # 임베딩 3
  ]
  run.log(
      {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
  )
  run.finish()
```

위의 코드를 실행하면, W&B 대시보드에 새로운 Table이 생성되어 데이터가 저장됩니다. 오른쪽 상단 패널 셀렉터에서 `2D Projection`을 선택하면 임베딩을 2차원으로 시각화할 수 있습니다. 스마트 디폴트 설정이 자동으로 적용되지만, 설정 메뉴(톱니바퀴 아이콘 클릭)에서 쉽게 변경할 수 있습니다. 이 예시에서는 5개의 수치형 차원이 모두 자동으로 사용됩니다.

{{< img src="/images/app_ui/weave_hello_world.png" alt="2D projection example" >}}

### Digits MNIST

앞선 예시는 임베딩을 로그하는 가장 기본적인 방식을 보여줍니다. 실제 업무에서는 훨씬 더 많은 차원과 샘플을 다루게 됩니다. MNIST Digits 데이터셋([UCI ML 필기 숫자 데이터셋](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)), [SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)을 통해 사용할 수 있습니다. 이 데이터셋은 64차원의 특징이 있는 1797개의 레코드를 제공하며, 문제는 10개 클래스를 분류하는 유스 케이스입니다. 입력 데이터를 이미지로 변환해 시각화할 수도 있습니다.

```python
import wandb
from sklearn.datasets import load_digits

with wandb.init(project="embedding_tutorial") as run:

  # 데이터셋 로드
  ds = load_digits(as_frame=True)
  df = ds.data

  # "target" 컬럼 생성
  df["target"] = ds.target.astype(str)
  cols = df.columns.tolist()
  df = df[cols[-1:] + cols[:-1]]

  # "image" 컬럼 생성
  df["image"] = df.apply(
      lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
  )
  cols = df.columns.tolist()
  df = df[cols[-1:] + cols[:-1]]

  run.log({"digits": df})
```

위 코드를 실행하면 UI에서 Table을 볼 수 있습니다. `2D Projection`을 선택하면 임베딩 정의, 색상, 사용 알고리즘(PCA, UMAP, t-SNE), 알고리즘 파라미터, 오버레이 설정(포인트 위에 마우스를 올리면 이미지 표시) 등 다양한 설정이 가능합니다. 실제로 대부분의 설정이 스마트 디폴트로 제공되어, `2D Projection`을 한 번만 클릭해도 유사한 결과를 확인할 수 있습니다. ([이 임베딩 튜토리얼 예제를 직접 활용해보세요](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney)).

{{< img src="/images/weave/embedding_projector.png" alt="MNIST digits projection" >}}

## 로그 옵션

임베딩을 다양한 포맷으로 로그할 수 있습니다.

1. **단일 임베딩 컬럼:** 데이터가 이미 "행렬" 형태라면 한 개의 임베딩 컬럼을 만들 수 있습니다. 이 때 cell 값의 타입은 `list[int]`, `list[float]`, 또는 `np.ndarray`가 가능합니다.
2. **다수의 수치형 컬럼:** 위의 두 예시처럼 각 차원별로 컬럼을 만드는 방법입니다. 현재 cell에는 python `int` 또는 `float` 타입만 허용됩니다.

{{< img src="/images/weave/logging_options.png" alt="Single embedding column" >}}
{{< img src="/images/weave/logging_option_image_right.png" alt="Multiple numeric columns" >}}

또한, 모든 Table과 마찬가지로 Table을 생성할 때 다양한 방법을 선택할 수 있습니다.

1. **dataframe으로부터 직접 생성:** `wandb.Table(dataframe=df)` 사용
2. **리스트 데이터로부터 직접 생성:** `wandb.Table(data=[...], columns=[...])` 사용
3. **row-by-row로 점진적으로 Table 빌드:** (코드에 루프가 있는 경우 유용) `table.add_data(...)`로 row 추가
4. **임베딩 컬럼 추가:** (임베딩 형태 예측값 리스트가 있을 때) `table.add_col("col_name", ...)` 사용
5. **computed 컬럼 추가:** (함수 또는 모델을 Table 전체에 적용하고 싶을 때) `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})` 사용

## 시각화(플로팅) 옵션

`2D Projection`을 선택한 후, 톱니바퀴 아이콘을 클릭해 렌더링 설정을 편집할 수 있습니다. 위에서 설명한 컬럼 선택 외에도, 원하는 알고리즘(및 해당 파라미터)을 선택할 수 있습니다. 아래는 각각 UMAP 및 t-SNE에 대한 파라미터 설정 예시입니다.

{{< img src="/images/weave/plotting_options_left.png" alt="UMAP parameters" >}} 
{{< img src="/images/weave/plotting_options_right.png" alt="t-SNE parameters" >}}

{{% alert %}}
참고: 현재 세 가지 알고리즘 모두에 대해 무작위 서브셋 1000개 row와 50개 차원으로 다운샘플링합니다.
{{% /alert %}}