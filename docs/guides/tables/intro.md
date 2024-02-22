---
description: Iterate on datasets and understand model predictions
slug: /guides/tables
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 데이터 시각화하기

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

W&B Tables을 사용하여 테이블 형태의 데이터를 시각화하고 쿼리하세요. 예를 들면:

* 동일한 테스트 세트에서 다양한 모델의 성능 비교
* 데이터에서 패턴 식별
* 샘플 모델 예측값을 시각적으로 확인
* 자주 잘못 분류된 예제 찾기 위해 쿼리


![](/images/data_vis/tables_sample_predictions.png)
위 이미지는 semantic segmentation과 사용자 정의 메트릭이 있는 테이블을 보여줍니다. 이 테이블을 여기서 확인하세요 [W&B ML 코스의 샘플 프로젝트](https://wandb.ai/av-team/mlops-course-001).

## 작동 방식

테이블은 각 열이 하나의 데이터 유형을 가진 데이터의 이차원 그리드입니다. 테이블은 기본 및 숫자형 유형뿐만 아니라 중첩된 리스트, 사전, 그리고 리치 미디어 유형도 지원합니다.

## 테이블 로그하기

몇 줄의 코드로 테이블을 로그하세요:

- [`wandb.init()`](../../ref/python/init.md): 결과를 추적하기 위한 [실행](../runs/intro.md) 생성.
- [`wandb.Table()`](../../ref/python/data-types/table.md): 새로운 테이블 개체 생성.
  - `columns`: 열 이름 설정.
  - `data`: 테이블의 내용 설정.
- [`run.log()`](../../ref/python/log.md): W&B에 테이블을 로그하여 저장.

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 시작 방법
* [퀵스타트](./tables-walkthrough.md): 데이터 테이블 로깅, 데이터 시각화, 데이터 쿼리 방법을 배웁니다.
* [테이블 갤러리](./tables-gallery.md): 테이블의 예제 사용 사례를 확인하세요.