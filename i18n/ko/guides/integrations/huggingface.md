---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Transformers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

[Hugging Face Transformers](https://huggingface.co/transformers/) 라이브러리는 BERT와 같은 최첨단 NLP 모델과 혼합 정밀도 및 그레이디언트 체크포인팅과 같은 학습 기술을 쉽게 사용할 수 있게 해줍니다. [W&B 통합](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)은 사용의 용이성을 손상시키지 않으면서 인터랙티브 중앙 대시보드에 풍부하고 유연한 실험 추적 및 모델 버전 관리를 추가합니다.

## 🤗 몇 줄로 차원이 다른 로깅

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B 프로젝트 이름 지정
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 모든 모델 체크포인트 로깅

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B 로깅 활성화
trainer = Trainer(..., args=args)
```
![W&B 인터랙티브 대시보드에서 실험 결과 탐색](@site/static/images/integrations/huggingface_gif.gif)

:::info
작동하는 코드에 바로 뛰어들고 싶다면 이 [Google Colab](https://wandb.me/hf)을 확인하세요.
:::

## 시작하기: 실험 추적

### 1) 가입하고, `wandb` 라이브러리 설치하고 로그인하기

a) [**가입**](https://wandb.ai/site)하여 무료 계정 생성

b) `wandb` 라이브러리 pip 설치

c) 학습 스크립트에서 로그인하려면 www.wandb.ai에 로그인한 후 [**인증 페이지**](https://wandb.ai/authorize)에서 **API 키를 찾으세요.**

Weights and Biases를 처음 사용하는 경우 [**퀵스타트**](../../quickstart.md)를 확인할 수 있습니다.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Python', value: 'python'},
    {label: '명령줄', value: 'cli'},
  ]}>
  <TabItem value="cli">

```shell
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="python">

```notebook
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) 프로젝트 이름 지정하기

[프로젝트](../app/pages/project-page.md)는 관련 실행에서 로그된 모든 차트, 데이터 및 모델이 저장되는 곳입니다. 프로젝트 이름을 지정하면 작업을 구성하고 단일 프로젝트에 대한 모든 정보를 한 곳에 유지하는 데 도움이 됩니다.

`WANDB_PROJECT` 환경 변수를 프로젝트 이름으로 설정하여 프로젝트에 실행을 추가하기만 하면 됩니다. `WandbCallback`은 이 프로젝트 이름 환경 변수를 인식하여 실행을 설정할 때 사용합니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: '명령줄', value: 'cli'},
    {label: '노트북', value: 'notebook'}
  ]}>
  <TabItem value="cli">

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="python">

```notebook
import os
os.environ["WANDB_PROJECT"]="amazon_sentiment_analysis"
```

  </TabItem>
</Tabs>


:::info
`Trainer`를 초기화하기 _전에_ 프로젝트 이름을 설정하세요.
:::

프로젝트 이름이 지정되지 않은 경우 기본적으로 프로젝트 이름은 "huggingface"로 설정됩니다.

### 3) 학습 실행을 W&B에 로깅하기

**가장 중요한 단계입니다:** 코드 내부나 명령줄에서 `Trainer` 학습 인수를 정의할 때, Weights & Biases와 로깅을 활성화하기 위해 `report_to`를 `"wandb"`로 설정하는 것입니다.

`TrainingArguments`의 `logging_steps` 인수는 학습 중에 학습 메트릭이 W&B로 얼마나 자주 전송되는지를 제어합니다. 또한 `run_name` 인수를 사용하여 W&B에서 학습 실행에 이름을 지정할 수 있습니다.

이제 모델은 학습하는 동안 손실, 평가 메트릭, 모델 아키텍처 및 그레이디언트를 Weights & Biases에 로깅하게 됩니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: '명령줄', value: 'cli'},
  ]}>
  <TabItem value="cli">

```bash
python run_glue.py \     # Python 스크립트 실행
  --report_to wandb \    # W&B에 로깅 활성화
  --run_name bert-base-high-lr \   # W&B 실행 이름 (선택사항)
  # 여기에 다른 명령줄 인수
```

  </TabItem>
  <TabItem value="python">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 여기에 다른 args 및 kwargs
    report_to="wandb",  # W&B에 로깅 활성화
    run_name="bert-base-high-lr",  # W&B 실행 이름 (선택사항)
    logging_steps=1,  # W&B에 얼마나 자주 로깅할지
)

trainer = Trainer(
    # 여기에 다른 args 및 kwargs
    args=args,  # 학습 인수
)

trainer.train()  # 학습 및 W&B에 로깅 시작
```

  </TabItem>
</Tabs>


:::info
TensorFlow를 사용하나요? PyTorch `Trainer`를 TensorFlow `TFTrainer`로 교체하기만 하면 됩니다.
:::

### 4) 모델 체크포인팅 활성화하기 


Weights & Biases의 [아티팩트](../artifacts)를 사용하면 최대 100GB의 모델 및 데이터세트를 무료로 저장할 수 있으며, 이후에 Weights & Biases [모델 레지스트리](../model_registry)를 사용하여 모델을 스테이징 또는 프로덕션 환경에 배포할 준비를 할 수 있습니다.

Hugging Face 모델 체크포인트를 아티팩트에 로깅하려면 `WANDB_LOG_MODEL` 환경 변수를 `end` 또는 `checkpoint` 또는 `false` 중 하나로 설정하면 됩니다:

-  **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)에서 `args.save_steps`마다 체크포인트가 업로드됩니다.
- **`end`**:  학습이 끝날 때 모델이 업로드됩니다.

`WANDB_LOG_MODEL`과 `load_best_model_at_end`를 함께 사용하여 학습이 끝난 후에 최상의 모델을 업로드하세요.


<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: '명령줄', value: 'cli'},
    {label: '노트북', value: 'notebook'},
  ]}>

  <TabItem value="python">

```python
import os

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
```

  </TabItem>
  <TabItem value="cli">

```bash
WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
</Tabs>


이제부터 초기화하는 모든 Transformers `Trainer`는 모델을 W&B 프로젝트에 업로드합니다. 로깅한 모델 체크포인트는 [아티팩트](../artifacts) UI를 통해 확인할 수 있으며, 전체 모델 계보를 포함합니다 (UI에서 예시 모델 체크포인트를 [여기](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)에서 확인할 수 있습니다). 


:::info
기본적으로 `WANDB_LOG_MODEL`이 `end` 또는 `checkpoint`로 설정되면 모델은 W&B 아티팩트에 `model-{run_id}` 또는 `checkpoint-{run_id}`로 저장됩니다.
그러나 `TrainingArguments`에서 [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)을 전달하면 모델은 `model-{run_name}` 또는 `checkpoint-{run_name}`으로 저장됩니다.
:::

#### W&B 모델 레지스트리
체크포인트를 아티팩트에 로깅했다면, Weights & Biases **[모델 레지스트리](../model_registry)**를 사용하여 최상의 모델 체크포인트를 팀 전체에서 중앙 집중화하여 관리할 수 있습니다. 여기서는 ML 작업별로 최고의 모델을 정리하고, 프로덕션을 위해 모델을 스테이징하고, 추가 평가를 위해 북마크하거나, 모델 CI/CD 프로세스를 시작할 수 있습니다.

모델 아티팩트를 모델 레지스트리에 연결하는 방법은 [모델 레지스트리](../model_registry) 문서를 참조하세요.

### 5) 학습 중 평가 출력 시각화하기

학습 또는 평가 중에 모델 출력을 시각화하는 것은 모델이 어떻게 학습되고 있는지 정말로 이해하는 데 종종 필수적입니다.

Transformers Trainer의 콜백 시스템을 사용하면 모델의 텍스트 생성 출력이나 W&B 테이블과 같은 다른 예측을 W&B에 로깅할 수 있습니다.

**[사용자 정의 로깅 섹션](#custom-logging-log-and-view-evaluation-samples-during-training)**을 참조하여 학습 중 평가 출력을 W&B 테이블에 로깅하는 전체 가이드를 확인하십시오:


![W&B 테이블에 평가 출력이 표시됩니다](/images/integrations/huggingface_eval_tables.png)

### 6) W&B 실행 완료하기 (노트북 전용) 

학습이 Python 스크립트에 포함되어 있으면 스크립트가 끝나면 W&B 실행이 종료됩니다.

Jupyter 또는 Google Colab 노트북을 사용하는 경우, `wandb.finish()`를 호출하여 학습이 끝났음을 알려야 합니다.

```python
trainer.train()  # 학습 및 W&B에 로깅 시작

# 학습 후 분석, 테스트, 기타 로깅된 코드

wandb.finish()
```

### 7) 결과 시각화하기

학습 결과를 로깅했다면 [W&B 대시보드](../track/app.md)에서 결과를 동적으로 탐색할 수 있습니다. 한 번에 수십 개의 실행을 쉽게 비교하고, 흥미로운 발견에 초점을 맞추고, 복잡한 데이터에서 통찰력을 이끌어내기 위해 유연한 인터랙티브 시각화를 사용할 수 있습니다.

## 고급 기능 및 자주 묻는 질문

### 최고의 모델을 어떻게 저장하나요?
`Trainer`에 전달된 `TrainingArguments`에 `load_best_model_at_end=True`가 설정되어 있으면 W&B는 최고 성능 모델 체크포인트를 아티팩트에 저장합니다.

ML 작업별로 팀 전체에서 최고의 모델 버전을 중앙 집중화하여 정리하고, 프로덕션을 위해 스테이징하고, 추가 평가를 위해 북마크하거나, 모델 CI/CD 프로세스를 시작하려면 모델 체크포인트를 아티팩트에 저장해야 합니다. 아티팩트에 로깅된 이러한 체크포인트는 이후에 [모델 레지스트리](../model_registry/intro.md)로 승격될 수 있습니다.

### 저장된 모델 불러오기

`WANDB_LOG_MODEL`로 모델을 W&B 아티팩트에 저장했다면, 추가 학습을 위해 모델 가중치를 다운로드하거나 추론을 실행할 수 있습니다. 이전에 사용한 것과 동일한 Hugging Face 아키텍처로 다시 불러올 수 있습니다.

```python
# 새 실행 생성
with wandb.init(project="amazon_sentiment_analysis") as run:
    # 아티팩트의 이름과 버전 전달
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # 모델 가중치를 폴더에 다운로드하고 경로 반환
    model_dir = my_model_artifact.download()

    # 그 폴더에서 Hugging Face 모델 불러오기
    #  동일한 모델 클래스 사용
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 추가 학습을 하거나 추론 실행
```

### 체크포인트에서 학습 재개하기 
`WANDB_LOG_MODEL='checkpoint'`를 설정했다면 `model_dir`를 `TrainingArguments`의 `model_name_or_path` 인수로 사용하고 `Trainer`에 `resume_from_checkpoint=True`를 전달하여 학습을 재개할 수 있습니다.

```python
last_run_id = "xxxxxxxx"  # wandb 워크스페이스에서 run_id 가져오기

# run_id에서 wandb 실행 재개
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # 실행에 아티팩트 연결
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # 체크포인트를 폴더에 다운로드하고 경로 반환
    checkpoint_dir = my_checkpoint_artifact.download()

    # 모델과 트

#### 학습 중 평가 샘플 보기

다음 섹션에서는 `WandbCallback`을 사용자 지정하여 모델 예측을 실행하고 학습 중 W&B 테이블에 평가 샘플을 기록하는 방법을 보여줍니다. `on_evaluate` 메서드의 트레이너 콜백을 사용하여 `eval_steps`마다 실행할 것입니다.

여기에서는 토크나이저를 사용하여 모델 출력에서 예측값과 라벨을 디코드하는 `decode_predictions` 함수를 작성했습니다.

그런 다음 예측값과 라벨에서 pandas DataFrame을 생성하고 DataFrame에 `epoch` 열을 추가합니다.

마지막으로 DataFrame에서 `wandb.Table`을 생성하고 wandb에 기록합니다.
추가적으로, 로깅 빈도를 제어하여 `freq` 에포크마다 예측을 기록할 수 있습니다.

**주의**: 일반 `WandbCallback`과 달리 이 사용자 정의 콜백은 `Trainer` 초기화 중이 아니라 `Trainer`가 인스턴스화된 **후** 트레이너에 추가해야 합니다.
이는 콜백 초기화 중에 `Trainer` 인스턴스가 콜백에 전달되기 때문입니다.

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """학습 중 모델 예측을 기록하는 사용자 정의 WandbCallback입니다.

    이 콜백은 학습 중 로깅 단계마다 모델 예측값과 라벨을 wandb.Table에 기록하여
    학습이 진행됨에 따라 모델 예측을 시각화할 수 있습니다.

    속성:
        trainer (Trainer): Hugging Face Trainer 인스턴스.
        tokenizer (AutoTokenizer): 모델과 연관된 토크나이저.
        sample_dataset (Dataset): 예측 생성을 위한 검증 데이터세트의 서브세트.
        num_samples (int, optional): 예측 생성을 위해 검증 데이터세트에서 선택할 샘플 수. 기본값은 100입니다.
        freq (int, optional): 로깅 빈도. 기본값은 2입니다.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """WandbPredictionProgressCallback 인스턴스를 초기화합니다.

        인수:
            trainer (Trainer): Hugging Face Trainer 인스턴스.
            tokenizer (AutoTokenizer): 모델과 연관된 토크나이저.
            val_dataset (Dataset): 검증 데이터세트.
            num_samples (int, optional): 예측 생성을 위해 검증 데이터세트에서 선택할 샘플 수.
              기본값은 100입니다.
            freq (int, optional): 로깅 빈도. 기본값은 2입니다.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # `freq` 에포크마다 로깅하는 것으로 로깅 빈도를 제어함
        if state.epoch % self.freq == 0:
            # 예측 생성
            predictions = self.trainer.predict(self.sample_dataset)
            # 예측값과 라벨 디코드
            predictions = decode_predictions(self.tokenizer, predictions)
            # wandb.Table에 예측 추가
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # wandb에 테이블 기록
            self._wandb.log({"sample_predictions": records_table})


# 먼저, Trainer를 인스턴스화합니다
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback을 인스턴스화합니다
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# 콜백을 트레이너에 추가합니다
trainer.add_callback(progress_callback)
```

자세한 예시는 이 [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)을 참조하세요.

### 추가적인 W&B 설정

`Trainer`와 함께 기록되는 내용을 더 구성하려면 환경 변수를 설정할 수 있습니다. W&B 환경 변수의 전체 목록은 [여기에서 찾을 수 있습니다](https://docs.wandb.ai/library/environment-variables).

| 환경 변수               | 사용법                                                                                                                                                                                                                                                                                                    |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | 프로젝트에 이름을 지정합니다(`huggingface`가 기본값)                                                                                                                                                                                                                                                      |
| `WANDB_LOG_MODEL`    | <p>모델 체크포인트를 W&B 아티팩트로 기록합니다(`false`가 기본값)</p><ul><li><code>false</code> (기본값): 모델 체크포인트 없음 </li><li><code>checkpoint</code>: args.save_steps(Trainer의 TrainingArguments에서 설정)마다 체크포인트가 업로드됩니다. </li><li><code>end</code>: 학습 종료 시 최종 모델 체크포인트가 업로드됩니다.</li></ul>                                                                                                                                                                                                                                   |
| `WANDB_WATCH`        | <p>모델의 그레이디언트, 파라미터를 기록할지 여부를 설정합니다</p><ul><li><code>false</code> (기본값): 그레이디언트나 파라미터 기록 없음 </li><li><code>gradients</code>: 그레이디언트의 히스토그램 기록 </li><li><code>all</code>: 그레이디언트와 파라미터의 히스토그램 기록</li></ul> |
| `WANDB_DISABLED`     | 로깅을 완전히 비활성화하려면 `true`로 설정합니다(`false`가 기본값)                                                                                                                                                                                                                                           |
| `WANDB_SILENT`       | wandb가 출력하는 메시지를 숨기려면 `true`로 설정합니다(`false`가 기본값)                                                                                                                                                                                                                                   |

<Tabs
  defaultValue="cli"
  values={[
    {label: '명령줄', value: 'cli'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### `wandb.init` 사용자 지정

`Trainer`가 사용하는 `WandbCallback`은 `Trainer`가 초기화될 때 내부적으로 `wandb.init`을 호출합니다. `Trainer`가 초기화되기 전에 `wandb.init`을 수동으로 호출함으로써 W&B 실행 구성을 완전히 제어할 수 있습니다.

`init`에 전달하고 싶은 것의 예는 아래와 같습니다. `wandb.init` 사용 방법에 대한 자세한 내용은 [참조 문서를 확인하세요](../../ref/python/init.md).

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 주목할 만한 기사

아래는 여러분이 관심가질 만한 6개의 Transformers 및 W&B 관련 기사입니다

<details>

<summary>Hugging Face Transformers를 위한 하이퍼파라미터 최적화</summary>

* Hugging Face Transformers를 위한 하이퍼파라미터 최적화를 위한 세 가지 전략 - 그리드 검색, 베이지안 최적화, 모집단 기반 학습이 비교됩니다.
* 우리는 Hugging Face transformers의 표준 uncased BERT 모델을 사용하고, SuperGLUE 벤치마크의 RTE 데이터세트에서 파인 튜닝하려고 합니다.
* 결과는 모집단 기반 학습이 Hugging Face transformer 모델의 하이퍼파라미터 최적화에 가장 효과적인 접근 방식임을 보여줍니다.

전체 보고서는 [여기에서 읽기](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI).
</details>

<details>

<summary>Hugging Tweets: 모델을 트레이닝하여 트윗 생성하기</summary>

* 이 기사에서 저자는 사전 훈련된 GPT2 HuggingFace Transformer 모델을 누구의 트윗에나 5분 내에 파인 튜닝하는 방법을 보여줍니다.
* 모델은 다음 파이프라인을 사용합니다: 트윗 다운로드, 데이터세트 최적화, 초기 실험, 사용자 간 손실 비교, 모델 파인 튜닝.

전체 보고서는 [여기에서 읽기](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI).
</details>

<details>

<summary>Hugging Face BERT와 WB로 문장 분류하기</summary>

* 이 기사에서는 자연어 처리의 최신 돌파구를 활용하여 문장 분류기를 구축하는 방법에 초점을 맞추며, NLP에 전이 학습을 적용하는 애플리케이션을 다룹니다.
* 우리는 2018년 5월에 처음 발표된 문법적으로 올바른지 여부로 라벨링된 문장 집합인 The Corpus of Linguistic Acceptability (CoLA) 데이터세트를 사용하여 단일 문장 분류를 수행합니다.
* 우리는 Google의 BERT를 사용하여 다양한 NLP 작업에서 최소한의 노력으로 고성능 모델을 생성합니다.

전체 보고서는 [여기에서 읽기](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA).
</details>

<details>

<summary>Hugging Face 모델 성능 추적 가이드</summary>

* 우리는 Weights & Biases와 Hugging Face transformers를 사용하여 GLUE 벤치마크에서 DistilBERT를 훈련시킵니다. DistilBERT는 BERT보다 40% 작지만 BERT의 정확도의 97%를 유지하는 Transformer입니다.
* GLUE 벤치마크는 NLP 모델을 훈련시키기 위한 아홉 개의 데이터세트 및 작업 모음입니다.

전체 보고서는 [여기에서 읽기](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU).
</details>

<details>

<summary>HuggingFace에서 조기 종료 - 예시</summary>

* PyTorch나 TensorFlow에서 Hugging Face Transformer를 조기 종료 정규화를 사용하여 파인 튜닝하는 것이 가능합니다.
* TensorFlow에서 `tf.keras.callbacks.EarlyStopping`콜백을 사용하여 조기 종료 콜백을 사용하는 것은 간단합니다.
* PyTorch에서는 현장에서 사용할 수 있는 조기 종료 방법이 없지만, GitHub Gist에서 사용 가능한 작동하는 조기 종료 후크가 있습니다.

전체 보고서는 [여기에서 읽기](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM).
</details>

<details>

<summary>사용자 지정 데이터세트에서 Hugging Face Transformers를 파인 튜닝하는 방법</summary>

우리는 감정 분석(이진 분류)을 위해 사용자 지정 IMDB 데이터세트에서 DistilBERT transformer를 파인 튜닝합니다.

전체 보고서는 [여기에서 읽기](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc).
</details>

## 문제, 질문, 기능 요청

Hugging Face W&B 통합에 대한 모든 문제, 질문 또는 기능 요청은 [Hugging Face 포럼의 이 스레드](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)에 게시하거나 Hugging Face [Transformers GitHub 저장소](https://github.com/huggingface/transformers)에 이슈를 열어주세요.