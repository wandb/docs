---
title: Visualize your data with tables
description: 데이터셋을 반복하여 모델 예측값을 이해하기
slug: /guides/tables
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

W&B Tables를 사용하여 표 형식의 데이터를 시각화하고 쿼리하세요. 예를 들어:

* 동일한 테스트 세트에서 서로 다른 모델의 성능 비교
* 데이터에서 패턴 식별
* 샘플 모델 예측값을 시각적으로 확인
* 잘못 분류된 예제를 찾기 위한 쿼리 실행

![](/images/data_vis/tables_sample_predictions.png) 위 이미지는 시멘틱 세그멘테이션과 커스텀 메트릭을 포함하는 테이블을 보여줍니다. 이 테이블은 [W&B ML 코스의 샘플 프로젝트](https://wandb.ai/av-team/mlops-course-001)에서 확인할 수 있습니다.

## 작동 방식

Table은 각 열에 단일 데이터 유형이 있는 2차원 데이터 그리드입니다. 테이블은 기본 및 숫자 유형뿐만 아니라 중첩된 리스트, 딕셔너리, 리치 미디어 유형도 지원합니다.

## 테이블 로그

몇 줄의 코드로 테이블을 로그하세요:

- [`wandb.init()`](../../ref/python/init.md): 결과를 추적하기 위한 [run](../runs/intro.md)을 생성합니다.
- [`wandb.Table()`](../../ref/python/data-types/table.md): 새로운 테이블 오브젝트를 생성합니다.
  - `columns`: 열 이름을 설정합니다.
  - `data`: 테이블의 내용을 설정합니다.
- [`run.log()`](../../ref/python/log.md): 테이블을 로그하여 W&B에 저장합니다.

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 시작 방법
* [퀵스타트](./tables-walkthrough.md): 데이터 테이블을 로그하고, 데이터를 시각화하고, 데이터를 쿼리하는 방법을 배웁니다.
* [Tables 갤러리](./tables-gallery.md): Tables의 예제 유스 케이스를 확인합니다.