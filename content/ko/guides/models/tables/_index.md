---
title: 테이블
description: 데이터셋을 반복적으로 개선하고 모델 예측값을 이해하세요
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

W&B Tables를 사용하여 표 형태의 데이터를 시각화하고 쿼리할 수 있습니다. 예를 들면 다음과 같습니다:

* 서로 다른 모델이 동일한 테스트 세트에서 어떻게 성능을 내는지 비교하기
* 데이터에 존재하는 패턴 찾기
* 모델 예측값을 시각적으로 살펴보기
* 오분류가 많이 발생한 예시를 쿼리로 찾아내기

{{< img src="/images/data_vis/tables_sample_predictions.png" alt="시멘틱 세그멘테이션 예측값 테이블" >}}
위 이미지는 시멘틱 세그멘테이션과 커스텀 메트릭이 포함된 테이블입니다. [W&B ML 코스의 샘플 프로젝트](https://wandb.ai/av-team/mlops-course-001)에서 이 테이블을 실제로 확인할 수 있습니다.

## 작동 방식

Table은 각 열이 하나의 데이터 타입을 가지는 2차원 그리드 형태의 데이터입니다. Tables는 원시 및 숫자 타입뿐만 아니라, 중첩된 리스트, 딕셔너리, 그리고 다양한 미디어 타입도 지원합니다.

## Table 로그하기

몇 줄의 코드만으로 테이블을 로그할 수 있습니다:

- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}): 결과를 추적할 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 생성
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ko" >}}): 새로운 table 오브젝트 생성
  - `columns`: 열 이름 설정
  - `data`: 테이블에 들어갈 내용 설정
- [`run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ko" >}}): Table을 로그해서 W&B에 저장

```python
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 시작 방법
* [퀵스타트]({{< relref path="./tables-walkthrough.md" lang="ko" >}}): 데이터 테이블을 로그하고, 시각화 및 쿼리하는 방법을 배워보세요.
* [Tables Gallery]({{< relref path="./tables-gallery.md" lang="ko" >}}): Tables의 다양한 유스 케이스를 예시로 살펴보세요.