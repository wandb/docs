---
title: XGBoost
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb"></CTAButtons>

이 노트북에서는 XGBoost 모델을 트레이닝하여 제출된 대출 신청서가 기본이 될지 아닐지를 분류할 것입니다. XGBoost와 같은 부스팅 알고리즘을 사용하면 대출 평가의 성능을 향상시키면서 내부 위험 관리 기능과 외부 규제 기관에 대한 해석 가능성을 유지할 수 있습니다.

이 노트북은 ScotiaBank의 Paul Edwards가 Nvidia GTC21에서 [발표한](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/) 내용을 바탕으로 하고 있으며, XGBoost를 사용하여 성능이 뛰어나면서도 해석 가능한 신용 점수를 작성하는 방법에 대해 설명했습니다. 또한 Stephen Denton (stephen.denton@scotiabank.com) 덕분에 [샘플 코드](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)를 공유해 주셨으며, 이를 이 노트북 전반에 걸쳐 사용하겠습니다.

### 이 노트북으로 구축한 실시간 W&B 대시보드 보기 및 상호작용은 [여기를 클릭](https://wandb.ai/morgan/credit_scorecard)하세요

# 이 노트북에 대해

이 colab에서는 규제된 엔터프라이즈가 Weights & Biases를 사용하는 방법을 다룹니다. 
- **ETL 파이프라인**을 추적하고 버전 관리하기 (로컬 또는 S3 및 GCS와 같은 클라우드 서비스에서)
- **실험 결과 추적** 및 훈련된 모델 저장 
- **여러 평가 메트릭**을 시각적으로 검사 
- **하이퍼파라미터 탐색**으로 성능 최적화 

**실험 및 결과 추적**

모든 트레이닝 하이퍼파라미터와 출력 메트릭을 추적하여 실험 대시보드를 생성할 것입니다:

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**하이퍼파라미터 탐색 실행하여 최적의 하이퍼파라미터 찾기**

Weights & Biases는 또한 자체 [Sweeps 기능](/guides/sweeps) 또는 [Ray Tune integration](/guides/integrations/ray-tune)을 사용하여 하이퍼파라미터 탐색을 수행할 수 있게 합니다. 고급 하이퍼파라미터 탐색 옵션을 사용하는 방법에 대한 자세한 가이드는 우리의 문서를 참조하십시오.

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

## AWS S3, Google Cloud Storage 및 W&B Artifacts

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights and Biases의 **Artifacts**는 실험이 항상 재현 가능하도록 엔드 투 엔드 트레이닝 파이프라인을 로그할 수 있게 합니다.

데이터 개인정보보호는 Weights & Biases에게 중요하며, 따라서 AWS S3 또는 Google Cloud Storage와 같은 자체 프라이빗 클라우드와 같은 참조 위치에서 아티팩트를 생성하는 것을 지원합니다. 로컬, 온프레미스의 W&B도 요청에 의해 제공됩니다. 

기본적으로 W&B는 아티팩트 파일을 미국에 위치한 프라이빗 Google Cloud Storage 버킷에 저장합니다. 모든 파일은 유지 및 전송 중 암호화됩니다. 민감한 파일의 경우, 프라이빗 W&B 설치 또는 참조 아티팩트 사용을 권장합니다.

## 아티팩트 참조 예제
**S3/GCS 메타데이터로 아티팩트 생성**

아티팩트는 S3/GCS 오브젝트에 대한 ETag, 크기 및 버전 ID(버킷에서 오브젝트 버전 관리가 활성화된 경우)와 같은 메타데이터만으로 구성됩니다.

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**필요할 때 로컬에 아티팩트 다운로드**

W&B는 아티팩트가 로그될 때 기록된 메타데이터를 사용하여 기본 버킷에서 파일을 검색합니다.

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

참조에 의한 아티팩트 사용, 인증 설정 등에 대한 자세한 내용은 [reference artifacts](/guides/artifacts/track-external-files)를 참조하십시오.

## W&B 로그인
Weights and Biases에 로그인


```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## Vehicle Loan Dataset

L&T의 [Vehicle Loan Default Prediction 데이터셋](https://www.kaggle.com/sneharshinde/ltfs-av-data)의 단순화된 버전을 W&B Artifacts에 저장하였습니다. 

```python
# 데이터를 저장할 폴더 지정, 존재하지 않으면 새 폴더가 생성됩니다
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

함수를 피클(Pickle)로 만드는 함수 생성


```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifacts에서 데이터 다운로드

W&B Artifacts에서 데이터셋을 다운로드할 것입니다. 먼저 W&B run 오브젝트를 생성하여 데이터를 다운로드할 것입니다. 데이터가 다운로드되면 원-핫 인코딩됩니다. 이 처리된 데이터는 같은 W&B에 새로운 Artifact로 로그됩니다. 데이터 다운로드에 사용된 W&B에 로그함으로써 새로운 Artifact를 원시 데이터셋 Artifact에 연결합니다.

```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&B에서 vehicle loan default 데이터 서브셋을 다운로드합니다. 여기에는 `train.csv` 및 `val.csv` 파일이 포함되어 있으며 일부 utils 파일도 포함됩니다.

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
# 데이터프레임에 데이터 로드
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# 데이터 원-핫 인코딩
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 전처리된 데이터 저장
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 전처리된 데이터 W&B Artifacts에 로그

```python
# Artifacts에 전처리된 데이터, 이를 생성한 함수 포함하여 새로운 아티팩트 생성
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="One-hot encoded dataset",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 전처리된 데이터를 아티팩트에 첨부
processed_ds_art.add_file(processed_data_path)

# 이 Artifact를 현재 W&B run에 로그
run.log_artifact(processed_ds_art)

run.finish()
```

## Train/Validation 분할 가져오기

여기서는 wandb run 오브젝트를 생성하는 대안 패턴을 보여줍니다. 아래의 셀에서는 데이터셋을 분할하는 코드가 `wandb.init() as run` 호출에 감싸져 있습니다. 

여기서 우리는:

- wandb run 시작
- Artifacts에서 원-핫 인코딩된 데이터셋 다운로드
- Train/Val 분할을 수행하고 분할에 사용된 파라미터 로그
- 새로운 `trndat` 및 `valdat` 데이터셋을 Artifacts에 로그
- 자동으로 wandb run 종료

```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # config는 여기서 선택적임
    # W&B에서 vehicle loan default 데이터 서브셋 다운로드
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

    # Train/Val 분할 수행
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"Train dataset size: {trndat[targ_var].value_counts()} \n")
    print(f"Validation dataset sizeL {valdat[targ_var].value_counts()}")

    # 분할된 데이터셋 저장
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # Artifacts에 새로 처리된 데이터와 이를 생성한 함수 포함하여 새로운 아티팩트 생성
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="Processed dataset split into train and valiation",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 처리된 데이터를 아티팩트에 첨부
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # 아티팩트 로그
    run.log_artifact(split_ds_art)
```

#### 트레이닝 데이터셋 검사
트레이닝 데이터셋 개요 얻기

```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B 테이블로 데이터셋 로그

W&B 테이블을 사용하면 이미지, 비디오, 오디오 등과 같은 풍부한 미디어를 포함하는 표 형식의 데이터를 로그, 쿼리 및 분석할 수 있습니다. 이를 통해 데이터셋을 이해하고, 모델 예측을 시각화하며, 인사이트를 공유할 수 있습니다. 자세한 내용은 [W&B 테이블 가이드](/guides/tables)를 참조하세요.

```python
# 정리를 위해 선택적으로 "log-dataset" job 유형으로 wandb run 생성
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # config는 여기서 선택적임 

# W&B 테이블 생성 및 탐색을 위해 데이터셋의 1000개의 랜덤 행 로그
table = wandb.Table(dataframe=trndat.sample(1000))

# W&B 워크스페이스에 테이블 로그
wandb.log({"processed_dataset": table})

# wandb run 종료
wandb.finish()
```

# 모델링

## XGBoost 모델 학습

이제 XGBoost 모델을 학습시켜 차량 대출 신청서가 기본이 될지 아닐지를 분류할 것입니다.

### GPU에서 학습
XGBoost 모델을 GPU에서 학습하고 싶다면, XGBoost에 전달하는 파라미터에서 다음을 설정하세요:

```python
"tree_method": "gpu_hist"
```

#### 1) W&B Run 초기화

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
    "gamma": 1,  ## def: 0
    "learning_rate": 0.1,  ## def: 0.1
    "max_depth": 3,
    "min_child_weight": 100,  ## def: 1
    "n_estimators": 25,
    "nthread": 24,
    "random_state": 42,
    "reg_alpha": 0,
    "reg_lambda": 0,  ## def: 1
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",  # GPU에서 훈련하려면 'gpu_hist' 사용
}
```

xgboost 트레이닝 파라미터를 W&B run config에 로그

```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B Artifacts에서 트레이닝 데이터 로드

```python
# Artifacts에서 트레이닝 데이터 로드
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## 타겟 컬럼을 시리즈로 추출
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) 모델 학습, 결과 W&B에 로그 및 모델 W&B Artifacts에 저장

xgboost 모델의 모든 파라미터를 로그하려면 `WandbCallback`를 사용합니다. [W&B 문서](/guides/integrations)를 참조하면 LighGBM을 포함한 다른 라이브러리도 W&B에 통합된 문서를 볼 수 있습니다.

```python
from wandb.integration.xgboost import WandbCallback

# WandbCallback으로 XGBoostClassifier 초기화
xgbmodel = xgb.XGBClassifier(
    **bst_params,
    callbacks=[WandbCallback(log_model=True)],
    early_stopping_rounds=run.config["early_stopping_rounds"]
)

# 모델 훈련
xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])
```

#### 5) 추가적인 트레이닝 및 평가 메트릭 W&B에 로그

```python
bstr = xgbmodel.get_booster()

# 트레이닝 및 검증 예측 가져오기
trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

# 추가적인 트레이닝 메트릭 로그
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    y_trn, trnYpreds
)
run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
run.summary["train_log_loss"] = -(
    y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
).sum() / len(y_trn)

# 추가적인 검증 메트릭 로그
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

#### 6) ROC 곡선 W&B에 로그

```python
# ROC 곡선을 W&B에 로그
valYpreds_2d = np.array([1 - valYpreds, valYpreds])  # W&B는 2d 배열을 기대합니다
y_val_arr = y_val.values
d = 0
while len(valYpreds_2d.T) > 10000:
    d += 1
    valYpreds_2d = valYpreds_2d[::1, ::d]
    y_val_arr = y_val_arr[::d]

run.log(
    {
        "ROC_Curve": wandb.plot.roc_curve(
            y_val_arr,
            valYpreds_2d.T,
            labels=["no_default", "loan_default"],
            classes_to_plot=[1],
        )
    }
)
```

#### W&B Run 종료

```python
run.finish()
```

이제 단일 모델을 트레이닝했으니 하이퍼파라미터 탐색을 실행하여 성능을 최적화해 보겠습니다.

# 하이퍼파라미터 탐색

Weights and Biases는 또한 자체 [Sweeps 기능](/guides/sweeps) 또는 [Ray Tune integration](/guides/integrations/ray-tune)을 사용하여 하이퍼파라미터 탐색을 수행할 수 있게 합니다.

**[여기를 클릭](https://wandb.ai/morgan/credit_score_sweeps/sweeps/iuppbs45)**하여 이 노트북을 사용하여 생성된 1000개의 run sweep 결과를 확인하세요.

#### 탐색 구성 정의
먼저 탐색할 하이퍼파라미터와 사용할 탐색 방법을 정의합니다. 우리는 학습률, 감마, min_child_weights 및 early_stopping_rounds에 대해 랜덤 검색을 수행할 것입니다.

```python
sweep_config = {
    "method": "random",
    "parameters": {
        "learning_rate": {"min": 0.001, "max": 1.0},
        "gamma": {"min": 0.001, "max": 1.0},
        "min_child_weight": {"min": 1, "max": 150},
        "early_stopping_rounds": {"values": [10, 20, 30, 40]},
    },
}

sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
```

#### 트레이닝 함수 정의

그런 다음 이러한 하이퍼파라미터를 사용하여 모델을 학습시킬 함수를 정의합니다. run을 초기화할 때 `job_type='sweep'`이라는 점을 주의하여 필요하다면 우리의 주 워크스페이스에서 이러한 run을 쉽게 필터링할 수 있게 합니다.

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
            "reg_lambda": 0,  ## def: 1
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
        }

        # WandbCallback으로 XGBoostClassifier 초기화
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback()],
            early_stopping_rounds=run.config["early_stopping_rounds"]
        )

        # 모델 학습
        xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])

        bstr = xgbmodel.get_booster()

        # booster 메트릭 로그
        run.summary["best_ntree_limit"] = bstr.best_ntree_limit

        # 트레이닝 및 검증 예측 가져오기
        trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
        valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

        # 추가적인 트레이닝 메트릭 로그
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
            y_trn, trnYpreds
        )
        run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
        run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
        run.summary["train_log_loss"] = -(
            y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
        ).sum() / len(y_trn)

        # 추가적인 검증 메트릭 로그
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

#### Sweeps 에이전트 실행

```python
count = 10  # 실행할 runs 수
wandb.agent(sweep_id, function=train, count=count)
```

## 당신의 좋아하는 ML 라이브러리에 이미 포함된 W&B

Weights and Biases는 Pytorch Lightning, Keras, Hugging Face, JAX, Fastai, XGBoost, Sci-Kit Learn, LightGBM와 같은 좋아하는 ML 및 딥러닝 라이브러리에 대한 통합을 제공합니다.

**자세한 내용은 [W&B integrations](/guides/integrations)를 참조하십시오**
