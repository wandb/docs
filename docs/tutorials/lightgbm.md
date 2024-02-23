
# LightGBM

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb)

Weights & Biases를 사용하여 머신 러닝 실험 추적, 데이터세트 버전 관리 및 프로젝트 협업을 진행하세요.

그래디언트 부스팅 결정 트리는 구조화된 데이터에 대한 예측 모델을 구축할 때 최첨단 기술입니다.

[LightGBM](https://github.com/microsoft/LightGBM), Microsoft의 그래디언트 부스팅 프레임워크는 xgboost를 제치고 가장 선호되는 GBDT 알고리즘이 되었습니다(그리고 catboost와 함께). LightGBM은 학습 시 연속적인 특징을 이산적인 버킷으로 분류하는 히스토그램 기반 알고리즘을 사용함으로써 학습 속도, 메모리 사용량 및 처리할 수 있는 데이터세트의 크기에서 xgboost를 능가합니다.

**[W&B + LightGBM 문서는 여기에서 확인할 수 있습니다](https://docs.wandb.ai/guides/integrations/boosting)**

## 이 노트북에서 다루는 내용
* LightGBM과 Weights and Biases의 쉬운 통합.
* 메트릭 로깅을 위한 `wandb_callback()` 콜백
* 특징 중요도 그래프 로깅 및 모델 저장을 W&B에 활성화하는 `log_summary()` 함수

우리는 사람들이 자신의 모델을 살펴보는 것을 더 쉽게 만들고자 하여, 단 한 줄의 코드로 LightGBM의 성능을 시각화할 수 있도록 도와주는 콜백을 개발했습니다.

**참고**: _Step_으로 시작하는 섹션은 W&B를 통합하는 데 필요한 모든 것입니다.

# 설치, 가져오기 및 로그인

## 평소처럼


```ipython
!pip install -Uq 'lightgbm>=3.3.1'
```


```python
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
```

## Step 0: W&B 설치


```ipython
!pip install -qU wandb
```

## Step 1: W&B 가져오기 및 로그인


```python
import wandb
from wandb.lightgbm import wandb_callback, log_summary

wandb.login()
```

# 데이터세트 다운로드 및 준비



```ipython
!wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train -qq
!wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test -qq
```


```python
# 데이터세트를 불러오거나 생성합니다
df_train = pd.read_csv("regression.train", header=None, sep="\t")
df_test = pd.read_csv("regression.test", header=None, sep="\t")

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# lightgbm을 위한 데이터세트 생성
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
```

# 학습

### Step 2: wandb 실행 초기화

`wandb.init()`를 사용하여 W&B 실행을 초기화합니다. 구성의 사전을 전달할 수도 있습니다. [공식 문서를 확인하세요 $\rightarrow$](https://docs.wandb.com/library/init)

ML/DL 워크플로에서 구성의 중요성을 부인할 수 없습니다. W&B는 모델을 재현할 수 있는 올바른 구성에 액세스할 수 있도록 보장합니다.

[이 Colab 노트북에서 구성에 대해 자세히 알아보세요 $\rightarrow$](http://wandb.me/config-colab)


```python
# 사전으로 구성을 지정합니다
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["rmse", "l2", "l1", "huber"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": 0,
}

wandb.init(project="my-lightgbm-project", config=params)
```

> 모델을 학습한 후에는 **프로젝트 페이지**를 클릭하여 돌아갑니다.

### Step 3: `wandb_callback`으로 학습


```python
# 학습
# lightgbm 콜백 추가
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=30,
    valid_sets=lgb_eval,
    valid_names=("validation"),
    callbacks=[wandb_callback()],
    early_stopping_rounds=5,
)
```

### Step 4: `log_summary`로 특징 중요도 로깅 및 모델 업로드
`log_summary`는 특징 중요도를 계산 및 업로드하고 (선택적으로) 훈련된 모델을 W&B 아티팩트에 업로드하여 나중에 사용할 수 있습니다


```python
log_summary(gbm, save_model_checkpoint=True)
```

# 평가


```python
# 예측
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 평가
print("예측의 rmse는:", mean_squared_error(y_test, y_pred) ** 0.5)
wandb.log({"rmse_prediction": mean_squared_error(y_test, y_pred) ** 0.5})
```

특정 W&B 실행에 대한 로깅을 마쳤다면 `wandb.finish()`를 호출하여 wandb 프로세스를 정리하는 것이 좋습니다(노트북/Colabs 사용 시에만 필요).


```python
wandb.finish()
```

# 결과 시각화

위의 **프로젝트 페이지** 링크를 클릭하여 결과를 자동으로 시각화된 것을 확인하세요.

<img src="https://imgur.com/S6lwSig.png" alt="Viz" />

# Sweep 101

Weights & Biases 스윕을 사용하여 하이퍼파라미터 최적화를 자동화하고 가능한 모델의 공간을 탐색하세요.

## [W&B 스윕을 사용한 XGBoost 하이퍼파라미터 최적화 확인하기 $\rightarrow$](http://wandb.me/xgb-colab)

Weights & Biases와 함께 하이퍼파라미터 스윕을 실행하는 것은 매우 쉽습니다. 단 3단계만 있습니다:

1. **스윕 정의:** 검색할 매개변수, 검색 전략, 최적화 메트릭 등을 지정하는 사전이나 [YAML 파일](https://docs.wandb.com/library/sweeps/configuration)을 생성함으로써 이를 수행합니다.

2. **스윕 초기화:** 
`sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행:** 
`wandb.agent(sweep_id, function=train)`

그게 다입니다! 하이퍼파라미터 스윕을 실행하는 것이 그만큼 쉽습니다!

<img src="https://imgur.com/SVtMfa2.png" alt="Sweep Result" />

# 예제 갤러리

W&B에서 추적 및 시각화된 프로젝트의 예시는 [갤러리에서 확인하세요 →](https://app.wandb.ai/gallery)

# 기본 설정
1. **프로젝트**: 여러 실행을 프로젝트에 로그하여 비교합니다. `wandb.init(project="project-name")`
2. **그룹**: 여러 프로세스 또는 교차 검증 폴드의 경우, 각 프로세스를 실행으로 로깅하고 함께 그룹화합니다. `wandb.init(group='experiment-1')`
3. **태그**: 현재 기준선이나 프로덕션 모델을 추적하기 위해 태그를 추가합니다.
4. **노트**: 실행 간 변경 사항을 추적하기 위해 테이블에 노트를 입력합니다.
5. **리포트**: 동료와 진행 상황에 대한 간단한 메모를 작성하고, ML 프로젝트의 대시보드와 스냅샷을 만듭니다.

# 고급 설정
1. [환경 변수](https://docs.wandb.com/library/environment-variables): 관리 클러스터에서 학습을 실행할 수 있도록 환경 변수에 API 키를 설정합니다.
2. [오프라인 모드](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun` 모드를 사용하여 오프라인으로 학습하고 나중에 결과를 동기화합니다.
3. [온-프레미스](https://docs.wandb.com/self-hosted): W&B를 자체 인프라의 프라이빗 클라우드 또는 에어갭 서버에 설치합니다. 우리는 학계부터 기업 팀까지 모두를 위한 로컬 설치를 제공합니다.
4. [스윕](https://docs.wandb.com/sweeps): 튜닝을 위한 가벼운 도구로 하이퍼파라미터 검색을 빠르게 설정합니다.