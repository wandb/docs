
# XGBoost

[**Colab 노트북에서 시도해보기 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

이 노트북에서는 제출된 대출 신청이 채무불이행인지 아닌지를 분류하기 위해 XGBoost 모델을 훈련할 것입니다. XGBoost와 같은 부스팅 알고리즘을 사용하면 대출 평가의 성능을 높이면서도 내부 위험 관리 기능과 외부 규제 기관에 대한 해석 가능성을 유지할 수 있습니다.

이 노트북은 Nvidia GTC21에서 ScotiaBank의 Paul Edwards가 XGBoost를 사용하여 해석 가능성을 유지하면서 더 나은 성능의 신용 점수표를 구성할 수 있는 방법을 [소개한](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/) 강연을 기반으로 합니다. 그들은 친절하게도 이 노트북 전체에서 사용할 [샘플 코드](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)를 공개적으로 공유했습니다. Scotiabank의 Stephen Denton (stephen.denton@scotiabank.com)에게 이 코드를 공유해 주셔서 감사합니다.

### [이곳을 클릭하여](https://wandb.ai/morgan/credit_scorecard) 이 노트북으로 구축된 실시간 Weights & Biases 대시보드를 보고 상호 작용하세요

# 이 노트북에서 다룰 내용들

이 Colab에서는 조정된 법인이 어떻게 Weights and Biases를 사용하여
- **데이터 ETL 파이프라인을 추적하고 버전 관리**하기(로컬 또는 S3 및 GCS와 같은 클라우드 서비스에서)
- **실험 결과를 추적**하고 훈련된 모델 저장하기
- **여러 평가 메트릭을 시각적으로 검사**하기
- **하이퍼파라미터 탐색으로 성능 최적화**하기

**실험 및 결과 추적**

모든 훈련 하이퍼파라미터와 출력 메트릭을 추적하여 실험 대시보드를 생성할 것입니다:

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**최적의 하이퍼파라미터 찾기 위해 하이퍼파라미터 스윕 실행**

Weights and Biases는 우리 자체의 [Sweeps 기능](https://docs.wandb.ai/guides/sweeps)이나 [Ray Tune 인테그레이션](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)을 통해 하이퍼파라미터 스윕을 할 수 있도록 지원합니다. 보다 고급 하이퍼파라미터 스윕 옵션 사용 방법에 대한 전체 가이드는 문서를 참조하세요.

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

Weights and Biases의 **Artifacts**는 항상 재현 가능한 실험을 보장하기 위해 엔드투엔드 훈련 파이프라인을 로그할 수 있도록 해줍니다.

데이터 개인 정보 보호는 Weights & Biases에 매우 중요하므로 AWS S3 또는 Google Cloud Storage와 같은 자체 프라이빗 클라우드에서 참조 위치로부터 Artifacts를 생성하는 것을 지원합니다. 로컬, 온프레미스의 W&B도 요청 시 사용 가능합니다.

기본적으로, W&B는 미국에 위치한 프라이빗 Google Cloud Storage 버킷에 아티팩트 파일을 저장합니다. 모든 파일은 저장 시 및 전송 중에 암호화됩니다. 민감한 파일의 경우, 프라이빗 W&B 설치 또는 참조 아티팩트 사용을 권장합니다.

## 아티팩트 참조 예시
**S3/GCS 메타데이터로 아티팩트 생성하기**

아티팩트는 S3/GCS 오브젝트에 대한 메타데이터로만 구성됩니다(예: ETag, 크기, 버킷의 오브젝트 버전 관리가 활성화된 경우 버전 ID).

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**필요할 때 아티팩트를 로컬로 다운로드하기**

W&B는 아티팩트가 로그될 때 기록된 메타데이터를 사용하여 기본 버킷에서 파일을 검색합니다.

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

[아티팩트 참조](https://docs.wandb.ai/guides/artifacts/references)에서 참조를 통한 아티팩트 사용, 자격 증명 설정 등에 대한 자세한 내용을 확인하세요.

## W&B에 로그인하기
Weights and Biases에 로그인하기


```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## 차량 대출 데이터셋

L&T에서 제공하는 [차량 대출 기본 예측 데이터셋](https://www.kaggle.com/sneharshinde/ltfs-av-data)의 간소화된 버전을 사용할 것입니다. 이 데이터는 W&B Artifacts에 저장되어 있습니다.


```python
# 데이터를 저장할 폴더를 지정합니다. 존재하지 않는 경우 새 폴더가 생성됩니다.
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

함수를 피클로 저장하는 함수 생성하기


```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifacts에서 데이터 다운로드하기

W&B Artifacts에서 데이터셋을 다운로드할 것입니다. 먼저 W&B run 객체를 생성해야 합니다. 데이터가 다운로드되면 원-핫 인코딩됩니다. 이 처리된 데이터는 동일한 W&B에 새로운 아티팩트로 로그됩니다. 데이터를 다운로드한 W&B에 로깅함으로써 이 새로운 아티팩트를 원본 데이터셋 아티팩트와 연결합니다.


```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&B에서 차량 대출 기본 데이터의 서브셋을 다운로드합니다. 여기에는 `train.csv` 및 `val.csv` 파일과 몇 가지 유틸리티 파일이 포함되어 있습니다.


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

#### 데이터 원-핫 인코딩하기


```python
# 데이터를 Dataframe으로 로드하기
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# 데이터 원-핫 인코딩하기
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 처리된 데이터 저장하기
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### W&B Artifacts에 처리된 데이터 로그하기


```python
# 처리된 데이터를 포함하는 새로운 아티팩트 생성하기, Artifacts에 이를 생성하는 함수 포함
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="원-핫 인코딩된 데이터셋",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 처리된 데이터를 아티팩트에 첨부하기
processed_ds_art.add_file(processed_data_path)

# 현재 wandb run에 이 아티팩트를 로그하기
run.log_artifact(processed_ds_art)

run.finish()
```

## 훈련/검증 분할 가져오기

wandb run 객체를 생성하는 대안적인 패턴을 아래에서 보여드립니다. 아래 셀에서, 데이터셋을 분할하는 코드는 `wandb.init() as run` 호출로 래핑됩니다.

여기서 우리는:

- wandb run을 시작합니다
- Artifacts에서 원-핫 인코딩된 데이터셋을 다운로드합니다
- 훈련/검증 분할을 수행하고 분할에 사용된 파라미터를 로그합니다
- 새로운 `trndat` 및 `valdat` 데이터셋을 Artifacts에 로그합니다
- wandb run을 자동으로 종료합니다


```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # config is optional here
    # W&B에서 차량 대출 기본 데이터의 서브셋을 다운로드합니다
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # 분할 파라미터 설정하기
    test_size = 0.25
    random_state = 42

    # 분할 파라미터를 로그하기
    run.config.update({"test_size": test_size, "random_state": random_state})

    # 훈련/검증 분할 수행하기
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"훈련 데이터셋 크기: {trndat[targ_var].value_counts()} \n")
    print(f"검증 데이터셋 크기: {valdat[targ_var].value_counts()}")

    # 분할된 데이터셋 저장하기
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # 처리된 데이터를 포함하는 새로운 아티팩트 생성하기, Artifacts에 이를 생성하는 함수 포함
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="처리된 데이터셋을 훈련 및 검증으로 분할",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 처리된 데이터를 아티팩트에 첨부하기
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # 아티팩트를 로그하기
    run.log_artifact(split_ds_art)
```

#### 훈련 데이터셋 검사하기
훈련 데이터셋의 개요를 확인하기


```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B Tables로 데이터셋 로그하기

W&B Tables을 사용하면 이미지, 비디오, 오디오 등과 같은 리치 미디어를 포함하는 테이블 형태의 데이터를 로그, 쿼리, 분석할 수 있습니다. 이를 사용하면 데이터셋을 이해하고, 모델 예측을 시각화하고, 인사이트를 공유할 수 있습니다. 자세한 내용은 [W&B Tables 가이드](https://docs.wandb.ai/guides/tables)에서 확인하세요.


```python
# "log-dataset" job type으로 wandb run을 생성합니다. 이는 선택 사항입니다.
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # config is optional here

# W&B 테이블을 생성하고 데이터셋의 1000개 무작위 행을 로그하여 탐색합니다.
table = wandb.Table(dataframe=trndat.sample(1000))

# W&B 워크스페이스에 테이블을 로그합니다.
wandb.log({"processed_dataset": table})

# wandb run을 종료합니다.
wandb.finish()
```

# 모델링

## XGBoost 모델 피팅

이제 차량 대출 신청이 채무불이행으로 이어질지 여부를 분류하기 위해 XGBoost 모델을 피팅할 것입니다.

### GPU에서 훈련하기
GPU에서 XGBoost 모델을 훈련하려면, XGBoost에 전달하는 파라미터에서 다음을 변경하기만 하면 됩니다:

```python
"tree_method": "gpu_hist"
```

#### 1) W&B Run 초기화하기


```python
run = wandb.init(project=WANDB_PROJECT, job_type="train-model")
```

#### 2) 모델 파라미터 설정 및 로그하기


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
    "tree_method": "hist",  # GPU에서 훈련하려면 `gpu_hist` 사용
}
```

W&B run config에 xgboost 훈련 파라미터를 로그하기


```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B Artifacts에서 훈련 데이터 로드하기


```python
# Artifacts에서 훈련 데이터 로드하기
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## 타겟 컬럼을 시리즈로 추출하기
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) 모델 학습, W&B에