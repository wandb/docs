---
title: Tables
description: 데이터셋을 반복하고 모델 예측값을 이해합니다
cascade:
- url: guides/models/tables/:filename
menu:
  default:
    identifier: ko-guides-models-tables-_index
    parent: models
url: guides/models/tables
weight: 2
---

{{< cta-button productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb" >}}

W&B Tables를 사용하여 표 형식 데이터를 시각화하고 쿼리합니다. 예를 들면 다음과 같습니다.

* 동일한 테스트 세트에서 다양한 모델의 성능을 비교합니다.
* 데이터에서 패턴을 식별합니다.
* 샘플 모델 예측값을 시각적으로 살펴봅니다.
* 일반적으로 잘못 분류된 예제를 찾기 위해 쿼리합니다.

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="" >}}
위 이미지는 시멘틱 세분화 및 사용자 정의 메트릭이 있는 테이블을 보여줍니다. [W&B ML Course의 샘플 프로젝트](https://wandb.ai/av-team/mlops-course-001)에서 이 테이블을 볼 수 있습니다.

## 작동 방식

Table은 각 열에 단일 유형의 데이터가 있는 2차원 데이터 그리드입니다. Tables는 기본 및 숫자 유형은 물론 중첩 목록, 딕셔너리 및 다양한 미디어 유형을 지원합니다.

## Table 기록

몇 줄의 코드로 table을 기록합니다.

- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}): 결과를 추적하기 위해 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성합니다.
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ko" >}}): 새 table 오브젝트를 생성합니다.
  - `columns`: 열 이름을 설정합니다.
  - `data`: table의 내용을 설정합니다.
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ko" >}}): table을 기록하여 W&B에 저장합니다.

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 시작 방법
* [퀵스타트]({{< relref path="./tables-walkthrough.md" lang="ko" >}}): 데이터 table 기록, 데이터 시각화 및 데이터 쿼리에 대해 알아봅니다.
* [Tables Gallery]({{< relref path="./tables-gallery.md" lang="ko" >}}): Tables의 유스 케이스 예시를 확인하세요.
