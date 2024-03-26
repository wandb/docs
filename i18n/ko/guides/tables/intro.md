---
description: Iterate on datasets and understand model predictions
slug: /guides/tables
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 데이터 시각화하기

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

W&B Tables을 사용하여 테이블 형식의 데이터를 시각화하고 쿼리하세요. 예를 들면 다음과 같은 작업을 수행할 수 있습니다:

* 동일한 테스트 세트로 서로 다른 모델의 성능 비교
* 데이터에서의 패턴 식별
* 샘플 모델 예측값의 시각적 확인
* 일반적으로 잘못 분류된 예제를 찾기 위한 쿼리


![](/images/data_vis/tables_sample_predictions.png)
위의 이미지는 시멘틱 세그멘테이션과 사용자 정의 메트릭이 있는 테이블을 보여줍니다. 이 테이블은 [W&B ML 코스의 샘플 프로젝트](https://wandb.ai/av-team/mlops-course-001)에서 확인할 수 있습니다.

## 작동 방식

Table은 각 열에 하나의 데이터 유형이 있는 2차원 데이터 그리드입니다. Tables는 원시와 숫자 유형은 물론 중첩된 리스트, 딕셔너리 및 리치 미디어 유형도  지원합니다.

## Table 로그하기

몇 줄의 코드로 테이블을 로깅하세요:

- [`wandb.init()`](../../ref/python/init.md): 결과를 추적할 [run](../runs/intro.md)을 생성합니다.
- [`wandb.Table()`](../../ref/python/data-types/table.md): 새로운 테이블 오브젝트를 생성합니다.
  - `columns`: 열 이름을 설정합니다.
  - `data`: 테이블의 내용을 설정합니다.
- [`run.log()`](../../ref/python/log.md): 테이블을 로깅하여 W&B에 저장합니다.

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 시작 방법
* [퀵스타트](./tables-walkthrough.md): 데이터 테이블 로깅, 데이터 시각화, 데이터 쿼리 방법을 배울 수 있습니다.
* [테이블 갤러리](./tables-gallery.md): Tables의 유스 케이스 예시를 참조하세요.