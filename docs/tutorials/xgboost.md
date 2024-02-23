
# XGBoost

[**여기서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

이 노트북에서는 제출된 대출 신청이 채무 불이행될지 여부를 분류하기 위해 XGBoost 모델을 학습할 것입니다. XGBoost와 같은 부스팅 알고리즘을 사용하면 대출 평가의 성능을 향상시키면서도 내부 위험 관리 기능과 외부 규제 기관을 위한 해석 가능성을 유지할 수 있습니다.

이 노트북은 스코샤은행의 Paul Edwards가 Nvidia GTC21에서 진행한 발표를 기반으로 합니다. 그는 XGBoost를 사용하여 성능이 더 우수하면서도 해석 가능한 신용 점수표를 구성하는 방법을 [소개했습니다](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/). 또한, 스코샤은행의 Stephen Denton(stephen.denton@scotiabank.com)이 이 코드를 공개적으로 공유했습니다.

### [이 노트북으로 구축된 실시간 W&B 대시보드 보기 및 상호 작용하려면 여기를 클릭하세요](https://wandb.ai/morgan/credit_scorecard)

# 이 노트북에서 다룰 내용

이 Colab에서는 Weights and Biases가 규제된 기관에 어떻게 도움이 되는지 다룰 것입니다:
- **데이터 ETL 파이프라인 추적 및 버전 관리** (로컬 또는 S3 및 GCS와 같은 클라우드 서비스에서)
- **실험 결과 추적** 및 학습된 모델 저장
- **여러 평가 메트릭 시각적 검사**
- **하이퍼파라미터 스윕으로 성능 최적화**

**실험 및 결과 추적**

모든 학습 하이퍼파라미터와 출력 메트릭을 추적하여 실험 대시보드를 생성할 것입니다:

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**최적의 하이퍼파라미터를 찾기 위한 하이퍼파라미터 스윕 실행**

Weights and Biases는 [Sweeps 기능](https://docs.wandb.ai/guides/sweeps)이나 [Ray Tune 통합](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)을 사용하여 하이퍼파라미터 스윕을 수행할 수 있도록 지원합니다. 보다 고급 하이퍼파라미터 스윕 옵션 사용 방법에 대해서는 문서를 참조하세요.

![credit_scorecard_2](/images/tutorials/credit_scorecard/credit_scorecard_2.png)

# 설정


```bash
!pip install -qq wandb>=0.13.10 dill
!pip install -qq xgboost>=1.7.4 scikit-learn>=1.2.1
```


```python
import ast
import sys
import json
from pathlib import Path
from dill.source import getsource
from dill import detect

import pandas as pd
import numpy as np
import plotly
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb

pd.set_option("display.max_columns", None)
```

# 데이터

## AWS S3, Google Cloud Storage 및 W&B 아티팩트

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights and Biases **아티팩트**는 항상 실험을 재현 가능하도록 엔드 투 엔드 학습 파이프라인을 로그할 수 있게 해줍니다.

데이터 프라이버시는 Weights & Biases에 매우 중요하므로 AWS S3 또는 Google Cloud Storage와 같은 자체 프라이빗 클라우드에서 참조 위치로부터 아티팩트를 생성하는 것을 지원합니다. 로컬, 온-프레미스의 W&B도 요청 시 이용 가능합니다.

기본적으로, W&B는 미국에 위치한 프라이빗 Google Cloud Storage 버킷에 아티팩트 파일을 저장합니다. 모든 파일은 저장 및 전송 시 암호화됩니다. 민감한 파일의 경우, 프라이빗 W&B 설치나 참조 아티팩트 사용을 권장합니다.

## 아티팩트 참조 예시
**S3/GCS 메타데이터로 아티팩트 생성**

아티팩트는 S3/GCS 객체에 대한 메타데이터(ETag, 크기, 버킷의 객체 버전 관리가 활성화된 경우 버전 ID 포함)로만 구성됩니다.

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**필요할 때 아티팩트를 로컬로 다운로드**

W&B는 아티팩트가 로그될 때 기록된 메타데이터를 사용하여 기본 버킷에서 파일을 검색합니다.

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

아티팩트 참조 사용 방법, 자격 증명 설정 등에 대한 자세한 내용은 [아티팩트 참조](https://docs.wandb.ai/guides/artifacts/references)를 참조하세요.

## W&B에 로그인
Weights and Biases에 로그인합니다.


```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## 차량 대출 데이터세트

W&B 아티팩트에 저장된 [차량 대출 기본 예측 데이터세트](https://www.kaggle.com/sneharshinde/ltfs-av-data)의 간략화된 버전을 사용할 것입니다.


```python
# 데이터를 저장할 폴더를 지정합니다. 존재하지 않는 경우 새 폴더가 생성됩니다.
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

함수를 피클링하는 함수 생성


```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B 아티팩트에서 데이터 다운로드

W&B 아티팩트에서 데이터세트를 다운로드할 것입니다. 먼저 W&B 실행 객체를 생성해야 합니다. 데이터가 다운로드되면 원-핫 인코딩됩니다. 이 처리된 데이터는 동일한 W&B에 새로운 아티팩트로 로그됩니다. 데이터를 다운로드한 W&B에 로깅함으로써, 이 새로운 아티팩트를 원본 데이터세트 아티팩트에 연결합니다.


```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&B에서 `train.csv` 및 `val.csv` 파일뿐만 아니라 몇 가지 유틸리티 파일이 포함된 차량 대출 기본 데이터의 부분 집합을 다운로드합니다.


```python
ARTIFACT_PATH = "morgan/credit_scorecard/vehicle_loan_defaults:latest"
dataset_art = run.use_artifact(ARTIFACT_PATH, type="dataset")
dataset_dir = dataset_art.download(data_dir)
```


```python
from data_utils import (
    describe_data_g_targ,
    one_hot_encode_data,
    load_training_data,
)
```

#### 데이터 원-핫 인코딩


```python
# 데이터를 데이터프레임으로 로드
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# 데이터 원-핫 인코딩
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 처리된 데이터 저장
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 처리된 데이터를 W&B 아티팩트에 로그


```python
# 처리된 데이터를 포함한 새로운 아티팩트를 아티팩트에 생성합니다. 이 아티팩트는 생성한 함수도 포함합니다.
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="One-hot 인코딩된 데이터세트",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 처리된 데이터를 아티팩트에 첨부합니다.
processed_ds_art.add_file(processed_data_path)

# 이 아티팩트를 현재 wandb 실행에 로그합니다.
run.log_artifact(processed_ds_art)

run.finish()
```

## 학습/검증 분할 가져오기

아래 셀에서는 wandb 실행 객체를 생성하는 대안적인 패턴을 보여줍니다. `wandb.init() as run`으로 데이터세트 분할 코드를 래핑합니다.

여기서 우리는:

- wandb 실행 시작
- Artifacts에서 원-핫 인코딩된 데이터세트 다운로드
- 학습/검증 분할 수행 및 분할에 사용된 파라미터 로그
- 새로운 `trndat` 및 `valdat` 데이터세트를 Artifacts에 로그
- wandb 실행을 자동으로 종료


```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # 여기서 config는 선택 사항입니다.
    # W&B에서 차량 대출 기본 데이터의 부분 집합을 다운로드합니다.
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # 분할 파라미터 설정
    test_size = 0.25
    random_state = 42

    # 분할 파라미터 로그
    run.config.update({"test_size": test_size, "random_state": random_state})

    # 학습/검증 분할 수행
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"학습 데이터셋 크기: {trndat[targ_var].value_counts()} \n")
    print(f"검증 데이터셋 크기: {valdat[targ_var].value_counts()}")

    # 분할 데이터셋 저장
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # 처리된 데이터를 포함한 새로운 아티팩트를 아티팩트에 생성합니다.
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="처리된 데이터셋을 학습 및 검증으로 분할",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 처리된 데이터를 아티팩트에 첨부합니다.
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # 아티팩트 로그
    run.log_artifact(split_ds_art)
```

#### 학습 데이터세트 조사
학습 데이터세트 개요 가져오기


```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B 테이블로 데이터세트 로그

W&B 테이블을 사용하면 이미지, 비디오, 오디오 등과 같은 리치 미디어가 포함된 테이블 데이터를 로그, 쿼리 및 분석할 수 있습니다. 이를 통해 데이터세트를 이해하고, 모델 예측을 시각화하며, 인사이트를 공유할 수 있습니다. 자세한 내용은 [W&B 테이블 가이드](https://docs.wandb.ai/guides/tables)를 참조하세요.


```python
# "log-dataset" 작업 유형을 사용하여 wandb 실행을 생성합니다. 이 옵션은 tidy를 유지하기 위한 것입니다.
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # 여기서 config는 선택 사항입니다.

# W&B 테이블을 생성하고 데이터세트의 1000개 랜덤 행을 탐색하기 위해 로그합니다.
table = wandb.Table(dataframe=trndat.sample(1000))

# 테이블을 W&B 워크스페이스에 로그합니다.
wandb.log({"processed_dataset": table})

# wandb 실행을 종료합니다.
wandb.finish()
```

# 모델링

## XGBoost 모델 학습

이제 차량 대출 신청이 채무 불이행으로 이어질지 여부를 분류하기 위해 XGBoost 모델을 학습할 것입니다.

### GPU에서 학습
GPU에서 XGBoost 모델을 학습하고 싶다면, XGBoost에 전달하는 파라미터에서 다음을 변경하면 됩니다:

```python
"tree_method": "gpu_hist"
```

#### 1) W&B 실행 초기화


```python
run = wandb.init(project=WANDB_PROJECT, job_type="train-model")
```

#### 2) 모델 파라미터 설정 및 로그


```python
base_rate = round(trndict["base_rate"], 6)
early_stopping_rounds = 40
```


```python
bst_params = {
    "objective": "binary:logistic",
    "base_score": base_rate,
    "gamma": 1,  ## 기본값: 0
    "learning_rate": 0.1,  ## 기본값: 0.1
    "max_depth": 3,
    "min_child_weight": 100,  ## 기본값: 1
    "n_estimators": 25,
    "nthread": 24,
    "random_state": 42,
    "reg_alpha": 0,
    "reg_lambda": 0,  ## 기본값: 1
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",  # GPU에서 학습하려면 `gpu_hist` 사용
}
```

W&B 실행 config에 xgboost 학습 파라미터 로그


```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B 아티팩트에서 학습 데이터 로드


```python
# Artifacts에서 학습 데이터를 로드합니다.
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## 타겟 열을 시리즈로 추출합니다.
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) 모델 학습, 결과를 W&B에 로그하고 모델을 W&B 아티팩트에 저장

`WandbCallback`을 사용해 xgboost 모델 파라미터를 모두 로그했습니다. 이를 통해 . [W&B 문서](https://docs.wand

#### 학습 함수 정의하기

그런 다음 이 하이퍼파라미터를 사용하여 모델을 학습시킬 함수를 정의합니다. 주목할 점은, 실행을 초기화할 때 `job_type='sweep'`로 설정하여, 필요한 경우 이러한 실행을 우리의 주 워크스페이스에서 쉽게 필터링할 수 있습니다.


```python
def train():
    with wandb.init(job_type="sweep") as run:
        bst_params = {
            "objective": "binary:logistic",
            "base_score": base_rate,
            "gamma": run.config["gamma"],
            "learning_rate": run.config["learning_rate"],
            "max_depth": 3,
            "min_child_weight": run.config["min_child_weight"],
            "n_estimators": 25,
            "nthread": 24,
            "random_state": 42,
            "reg_alpha": 0,
            "reg_lambda": 0,  ## 기본값: 1
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
        }

        # WandbCallback을 사용하여 XGBoostClassifier 초기화
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback()],
            early_stopping_rounds=run.config["early_stopping_rounds"]
        )

        # 모델 학습
        xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])

        bstr = xgbmodel.get_booster()

        # 부스터 메트릭 로그
        run.summary["best_ntree_limit"] = bstr.best_ntree_limit

        # 학습 및 검증 예측값 가져오기
        trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
        valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

        # 추가 학습 메트릭 로그
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
            y_trn, trnYpreds
        )
        run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
        run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
        run.summary["train_log_loss"] = -(
            y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
        ).sum() / len(y_trn)

        # 추가 검증 메트릭 로그
        ks_stat, ks_pval = ks_2samp(valYpreds[y_val == 1], valYpreds[y_val == 0])
        run.summary["val_ks_2samp"] = ks_stat
        run.summary["val_ks_pval"] = ks_pval
        run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
        run.summary["val_acc_0.5"] = metrics.accuracy_score(
            y_val, np.where(valYpreds >= 0.5, 1, 0)
        )
        run.summary["val_log_loss"] = -(
            y_val * np.log(valYpreds) + (1 - y_val) * np.log(1 - valYpreds)
        ).sum() / len(y_val)
```

#### 스윕 에이전트 실행하기


```python
count = 10  # 실행할 횟수
wandb.agent(sweep_id, function=train, count=count)
```

## W&B는 이미 여러분이 선호하는 ML 라이브러리에 통합되어 있습니다

Weights and Biases는 다음과 같은 여러분이 선호하는 ML 및 딥 러닝 라이브러리에 통합되어 있습니다:

- Pytorch Lightning
- Keras
- Hugging Face
- JAX
- Fastai
- XGBoost
- Sci-Kit Learn
- LightGBM 

**자세한 내용은 [W&B 통합 가이드](https://docs.wandb.ai/guides/integrations)를 참조하세요**