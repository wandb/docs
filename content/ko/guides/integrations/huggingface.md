---
title: Hugging Face 트랜스포머
menu:
  default:
    identifier: ko-guides-integrations-huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) 라이브러리는 BERT와 같은 최신 NLP 모델과 혼합 정밀도, 그레이디언트 체크포인팅과 같은 트레이닝 기법을 아주 쉽게 사용할 수 있게 해줍니다. [W&B 인테그레이션](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)을 사용하면 사용자 경험의 편리함을 유지하면서도, 대화형 중앙 대시보드에서 풍부하고 유연한 실험 추적 및 모델 버전 관리를 할 수 있습니다.

## 몇 줄 코드로 한 단계 높은 로깅

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B 프로젝트 이름 지정
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 모든 모델 체크포인트 로깅

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B 로깅 활성화
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="HuggingFace 대시보드" >}}

{{% alert %}}
바로 작동하는 코드를 확인하고 싶다면, [Google Colab](https://wandb.me/hf)을 참고하세요.
{{% /alert %}}

## 시작하기: 실험 추적

### 회원가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증하는 역할을 합니다. 사용자 프로필에서 API 키를 생성하실 수 있습니다.

{{% alert %}}
더 간편하게 API 키를 생성하려면 [W&B 인증 페이지](https://wandb.ai/authorize)에 바로 접속하세요. 화면에 표시된 API 키를 복사해서 비밀번호 관리 프로그램 등의 안전한 위치에 저장하세요.
{{% /alert %}}

1. 화면 우측 상단의 사용자 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택한 후, **API Keys** 섹션까지 아래로 스크롤하세요.
1. **Reveal**을 클릭하고 표시된 API 키를 복사하세요. API 키를 숨기려면 페이지를 새로고침하면 됩니다.

### `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면 다음을 따라하세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_API_KEY`에 API 키를 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인하세요.



    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

처음 W&B를 사용하신다면 [퀵스타트]({{< relref path="/guides/quickstart.md" lang="ko" >}})도 꼭 확인해보세요.


### 프로젝트 이름 지정하기

W&B Project는 관련된 run에서 로깅되는 모든 차트, 데이터, 모델이 저장되는 공간입니다. 프로젝트에 이름을 지정하면 작업을 잘 조직하고 한 곳에서 쉽게 관리할 수 있습니다.

프로젝트에 run을 추가하려면 `WANDB_PROJECT` 환경 변수를 프로젝트 이름으로 설정하세요. `WandbCallback`은 이 프로젝트 이름 환경 변수를 자동으로 감지하여 run 세팅 시 적용합니다.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
import os
os.environ["WANDB_PROJECT"]="amazon_sentiment_analysis"
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
%env WANDB_PROJECT=amazon_sentiment_analysis
```

{{% /tab %}}

{{< /tabpane >}}

{{% alert %}}
반드시 `Trainer`를 초기화하기 _전에_ 프로젝트 이름을 지정하세요.
{{% /alert %}}

프로젝트 이름을 지정하지 않으면 기본값은 `huggingface`로 설정됩니다.

### 트레이닝 run을 W&B에 로깅하기

`Trainer`의 트레이닝 인자를 정의할 때 **가장 중요한 단계**는 `report_to`를 `"wandb"`로 설정해서 W&B 로깅을 활성화하는 것입니다 (코드/커맨드라인 어디서든 가능합니다).

`TrainingArguments`의 `logging_steps` 인수로 트레이닝 중 W&B에 트레이닝 메트릭을 얼마나 자주 보낼지 제어할 수 있습니다. `run_name` 인수로 W&B에서 run의 이름도 지정할 수 있습니다. 

여기까지 하면 이제 트레이닝 중 모델의 loss, 평가 지표, 모델 구조, 그레이디언트 등이 W&B에 모두 자동으로 로깅됩니다.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
python run_glue.py \     # Python 스크립트 실행
  --report_to wandb \    # W&B 로깅 활성화
  --run_name bert-base-high-lr \   # W&B run 이름(선택 사항)
  # 다른 커맨드라인 인자 추가
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 기타 인자 및 kwargs
    report_to="wandb",  # W&B 로깅 켜기
    run_name="bert-base-high-lr",  # W&B run 이름(선택 사항)
    logging_steps=1,  # W&B로 얼마나 자주 로그를 남길지
)

trainer = Trainer(
    # 기타 인자 및 kwargs
    args=args,  # 트레이닝 인자
)

trainer.train()  # 트레이닝 시작 & W&B로 로깅
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlow를 사용 중이신가요? PyTorch `Trainer` 대신 TensorFlow `TFTrainer`를 사용하면 됩니다.
{{% /alert %}}

### 모델 체크포인팅 활성화하기

[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 이용해 최대 100GB의 모델과 데이터셋을 무료로 저장하고 W&B [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})도 활용할 수 있습니다. Registry에서는 모델을 등록해 탐색·평가하고 스테이징 준비, 프로덕션 배포 등에 활용할 수 있습니다.

Hugging Face 모델 체크포인트를 Artifacts에 로깅하려면, `WANDB_LOG_MODEL` 환경 변수를 _아래 중 하나로_ 지정하세요.

- **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)의 `args.save_steps`마다 체크포인트 업로드
- **`end`**: 트레이닝이 끝날 때 (단, `load_best_model_at_end`도 설정되어야 합니다.)
- **`false`**: 모델을 업로드하지 않음

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```bash
WANDB_LOG_MODEL="checkpoint"
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
import os

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
%env WANDB_LOG_MODEL="checkpoint"
```

{{% /tab %}}

{{< /tabpane >}}

이제부터 초기화하는 모든 Transformers `Trainer`는 W&B 프로젝트에 모델을 업로드하게 됩니다. 로그된 모델 체크포인트는 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) UI에서 확인할 수 있고, 전체 모델 계보 정보도 볼 수 있습니다 (UI에서 예시 체크포인트 모델은 [여기](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..) 참고).

{{% alert %}}
기본적으로, `WANDB_LOG_MODEL`이 `end`로 설정되면 모델은 W&B Artifacts에 `model-{run_id}` 형태로, `checkpoint`로 설정되면 `checkpoint-{run_id}` 형태로 저장됩니다.
만약 `TrainingArguments`에 [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)을 넘겼다면, 모델은 `model-{run_name}` 또는 `checkpoint-{run_name}` 형태로 저장됩니다.
{{% /alert %}}

#### W&B Registry
체크포인트를 Artifacts에 로그했다면, [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})에서 최고의 모델 체크포인트들을 등록하여 팀 내에서 중앙 관리할 수 있습니다. Registry로 모델을 태스크별로 정리하고, 모델 라이프사이클을 관리하며, 전체 ML 사이클을 추적·감사하고, [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})된 후처리도 할 수 있습니다.

모델 Artifact 연결 방법은 [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})를 참고하세요.
 
### 트레이닝 중 평가 결과 시각화

트레이닝이나 평가 중 모델 출력 결과를 시각화하는 것은 모델 학습 상태를 깊이 있게 이해하는 데 종종 매우 중요합니다.

Transformers Trainer의 콜백 시스템으로, 모델의 텍스트 생성 출력(또는 기타 예측값)을 W&B Table에 로깅할 수도 있습니다.

트레이닝 중 W&B Table에 평가 출력을 기록하는 전체 가이드는 아래 [커스텀 로깅 섹션]({{< relref path="#custom-logging-log-and-view-evaluation-samples-during-training" lang="ko" >}})에서 확인하세요.

{{< img src="/images/integrations/huggingface_eval_tables.png" alt="평가 출력이 담긴 W&B Table 예시" >}}

### W&B Run 종료하기 (노트북 환경에서) 

만약 트레이닝이 Python 스크립트에 캡슐화된 경우, 스크립트가 마치면 W&B run도 자동으로 종료됩니다.

반면 Jupyter나 Google Colab 노트북을 사용하고 있다면, 트레이닝이 끝났을 때 `run.finish()`를 직접 호출해주셔야 합니다.

```python
run = wandb.init()
trainer.train()  # 트레이닝 시작 & W&B로 로깅

# 트레이닝 후 분석, 테스트, 기타 로깅 코드

run.finish()
```

### 결과 시각화하기

트레이닝 결과를 기록했다면 [W&B 대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 결과를 동적으로 탐색해볼 수 있습니다. 수십 개의 run을 손쉽게 비교하고, 흥미로운 인사이트를 드러내며, 복잡한 데이터에서 유연하게 시각화할 수 있습니다.

## 고급 기능 및 자주 묻는 질문

### 최적의 모델을 저장하려면 어떻게 하나요?
`Trainer`에 `load_best_model_at_end=True`로 `TrainingArguments`를 넘기면, 성능이 가장 좋은 모델 체크포인트가 Artifacts로 W&B에 저장됩니다.

모델 체크포인트가 Artifacts로 저장된 경우, 이를 [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})에도 승격시킬 수 있습니다. Registry에서는
- ML 태스크별 최적 모델 버전을 정리할 수 있고
- 모델을 중앙 집중화하여 팀과 공유
- 모델을 프로덕션에 올리거나 추가 평가를 위해 북마크
- 아래쪽 CI/CD 프로세스를 트리거
할 수 있습니다.

### 저장한 모델을 불러오려면?

`WANDB_LOG_MODEL`로 W&B Artifacts에 모델을 저장했다면, 추가 트레이닝을 하거나 추론을 위해 모델 가중치를 다운로드할 수 있습니다. 이전에 사용한 Hugging Face 아키텍처 그대로 불러올 수 있습니다.

```python
# 새로운 run 생성
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifact 이름과 버전 지정
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # 모델 가중치를 폴더에 다운로드하고 경로 반환
    model_dir = my_model_artifact.download()

    # 동일한 모델 클래스로 Hugging Face 모델을 폴더에서 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 추가 트레이닝 또는 추론 실행
```

### 체크포인트에서 트레이닝을 재개하려면?
`WANDB_LOG_MODEL='checkpoint'`로 설정했다면, `model_dir`을 `TrainingArguments`의 `model_name_or_path` 인자로 사용하고, `Trainer`에 `resume_from_checkpoint=True`를 넘겨 트레이닝을 재개할 수 있습니다.

```python
last_run_id = "xxxxxxxx"  # wandb workspace에서 run_id를 가져옴

# run_id로 wandb run 재개
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # run에 Artifact 연결
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # 체크포인트를 폴더에 다운로드하고 경로 반환
    checkpoint_dir = my_checkpoint_artifact.download()

    # 모델 및 trainer 재초기화
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # 원하는 트레이닝 인자 지정
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # 체크포인트 디렉토리에서 트레이닝 재개
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### 트레이닝 중 평가 샘플을 로깅하고 보려면

Transformers의 `Trainer`에서 W&B로 로깅하는 기능은 라이브러리 내 [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)이 담당합니다. Hugging Face 로깅 기능을 커스터마이징하려면, 이 콜백을 서브클래싱하여 `Trainer` 클래스의 기능을 더 활용할 수 있도록 수정할 수 있습니다.

아래는 HF Trainer에 새로운 콜백을 추가하는 일반적인 패턴이며, 아래쪽에 트레이닝 중 평가 출력을 W&B Table에 기록하는 코드 전체 예시가 있습니다.

```python
# Trainer는 평소와 같이 인스턴스화
trainer = Trainer()

# Trainer 오브젝트를 넘겨서 새로운 로깅 콜백 인스턴스화
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# Trainer에 콜백 추가
trainer.add_callback(evals_callback)

# Trainer 트레이닝 평소처럼 시작
trainer.train()
```

#### 트레이닝 중 평가 샘플 보기

아래 예시는 `WandbCallback`을 커스터마이징해서 모델 예측값을 생성하고, 트레이닝 중 평가 샘플을 W&B Table에 로깅하는 방법을 보여줍니다. 여기서는 `on_evaluate` 메소드에서 매 `eval_steps`마다 실행되도록 했습니다.

먼저 `decode_predictions` 함수를 사용해 토크나이저로 모델 출력값을 디코딩하고,

그다음 예측값과 레이블을 pandas DataFrame으로 만들어 `epoch` 컬럼을 추가한 뒤,

wandb.Table로 변환해 wandb에 로깅합니다. 로깅 빈도는 `freq` 에포크마다 한 번으로 조절할 수 있습니다.

**참고:** 일반 `WandbCallback`과 달리, 이 커스텀 콜백은 Trainer **생성 _후_**에 별도로 추가해야 합니다.
Trainer 인스턴스가 콜백 초기화시 전달되기 때문입니다.

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """트레이닝 중 모델 예측값을 기록하는 커스텀 WandbCallback.

    이 콜백은 트레이닝 중 각 로깅 스텝마다 wandb.Table에
    모델 예측값과 레이블을 기록합니다. 트레이닝이 진행됨에 따라
    모델 예측 결과를 시각화할 수 있습니다.

    Attributes:
        trainer (Trainer): Hugging Face Trainer 인스턴스
        tokenizer (AutoTokenizer): 해당 모델의 토크나이저
        sample_dataset (Dataset): 예측 결과 생성을 위한 validation 서브셋
        num_samples (int, optional): 예측에 사용할 validation 데이터 수 (기본 100)
        freq (int, optional): 로깅 빈도 (기본 2)
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallback 인스턴스 초기화

        Args:
            trainer (Trainer): Hugging Face Trainer 인스턴스
            tokenizer (AutoTokenizer): 해당 모델의 토크나이저
            val_dataset (Dataset): validation 데이터셋
            num_samples (int, optional): 예측 생성에 사용할 샘플 수 (기본 100)
            freq (int, optional): 로깅 빈도 (기본 2)
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # 매 `freq` 에포크마다 예측값을 로깅
        if state.epoch % self.freq == 0:
            # 예측값 생성
            predictions = self.trainer.predict(self.sample_dataset)
            # 예측값과 레이블 디코딩
            predictions = decode_predictions(self.tokenizer, predictions)
            # wandb.Table에 예측값 저장
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # wandb에 테이블로 기록
            self._wandb.log({"sample_predictions": records_table})


# 먼저 Trainer 인스턴스화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback 인스턴스화
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# trainer에 콜백 추가
trainer.add_callback(progress_callback)
```

더 자세한 예시는 [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)에서 확인할 수 있습니다.

### 추가적인 W&B 설정은 무엇이 있나요?

`Trainer`에서 무엇을 로깅할지 더 세밀하게 환경 변수로 설정 가능하며, 전체 W&B 환경 변수 목록은 [여기에서 볼 수 있습니다]({{< relref path="/guides/hosting/env-vars.md" lang="ko" >}}).

| 환경 변수             | 사용법                                                                                                                                                                                                                                                                                                        |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | 프로젝트 이름 지정 (`huggingface`가 기본값)                                                                                                                                                                                                                                |
| `WANDB_LOG_MODEL`    | <p>모델 체크포인트를 W&B Artifact로 저장 (`false`가 기본) </p><ul><li><code>false</code> (기본): 모델 체크포인트 X </li><li><code>checkpoint</code>: Trainer의 TrainingArguments에 지정된 대로 args.save_steps마다 체크포인트 업로드 </li><li><code>end</code>: 트레이닝 종료시 최종 모델만 업로드</li></ul> |
| `WANDB_WATCH`        | <p>모델 그레이디언트, 파라미터 또는 아무것도 로깅할지 설정</p><ul><li><code>false</code> (기본): 그레이디언트/파라미터 로깅 없음 </li><li><code>gradients</code>: 그레이디언트 히스토그램 기록 </li><li><code>all</code>: 그레이디언트와 파라미터 모두 기록</li></ul>              |
| `WANDB_DISABLED`     | `true`로 설정하면 완전히 로깅 비활성화 (기본값 `false`) |
| `WANDB_QUIET`.       | `true`로 설정하면 표준출력에 핵심 메시지만 로깅 (기본값 `false`)                                                                                                                                                                                                         |
| `WANDB_SILENT`       | `true`로 설정 시 wandb에서 출력되는 모든 메시지를 비활성화 (기본값 `false`)                                                                                                                                                                                              |

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

```notebook
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

{{% /tab %}}

{{< /tabpane >}}


### `wandb.init`을 커스터마이즈하고 싶다면?

Trainer에서 사용하는 `WandbCallback`은 `Trainer`가 초기화될 때 내부적으로 `wandb.init`을 호출합니다. 대신, Trainer를 초기화하기 전에 직접 `wandb.init`을 호출하여 run 설정을 수동으로 지정할 수도 있습니다.

`init`에 전달할 수 있는 주요 인자 예시는 아래와 같습니다. 자세한 내용은 [`wandb.init()` 레퍼런스]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})를 참고하세요.

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```


## 추가 자료

아래는 Transformers 및 W&B 관련 추천 아티클 6선입니다.

<details>

<summary>Hugging Face Transformers를 위한 하이퍼파라미터 최적화</summary>

* Hugging Face Transformers의 하이퍼파라미터 최적화 기법 3가지: 그리드 검색, 베이지안 최적화, 모집단적 학습을 비교합니다.
* Hugging Face transformers의 표준 uncased BERT 모델을 사용해 SuperGLUE 벤치마크 중 RTE 데이터셋에 파인튜닝합니다.
* 결과적으로 모집단적 학습이 Hugging Face transformer 모델의 하이퍼파라미터 최적화에 가장 효과적인 방식으로 나왔습니다.

[Hyperparameter Optimization for Hugging Face Transformers report](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)에서 상세히 읽어보세요.
</details>

<details>

<summary>Hugging Tweets: 트윗 생성 모델 훈련하기</summary>

* 이 글에서는 저자가 파인튜닝된 GPT2 HuggingFace Transformer 모델을 활용해 5분 만에 누구의 트윗이든 생성하는 법을 보여줍니다.
* 모델 파이프라인: 트윗 다운로드, 데이터셋 최적화, 초기 실험, 사용자 간의 손실 비교, 모델 파인튜닝

전체 리포트는 [여기](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)에서 확인하세요.
</details>

<details>

<summary>Sentence Classification With Hugging Face BERT와 W&B</summary>

* 이 기사에서는 최신 자연어처리 기술을 기반으로, NLP에 전이 학습을 적용해 문장 분류기를 만듭니다.
* 단일 문장 분류에 The Corpus of Linguistic Acceptability (CoLA) 데이터셋을 사용합니다. 이 데이터셋은 2018년에 출간된 문법적으로 옳고 그름이 판별된 문장 집합입니다.
* Google의 BERT로 다양한 NLP 태스크에서 최소한의 노력으로 고성능 모델을 만들어봅니다.

전체 리포트는 [여기](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)에서 확인할 수 있습니다.
</details>

<details>

<summary>Hugging Face 모델 성능 추적: 단계별 가이드</summary>

* W&B와 Hugging Face transformers를 사용해 DistilBERT(파라미터 수가 BERT보다 40% 적고 정확도의 97%를 유지)를 GLUE 벤치마크에서 훈련합니다.
* GLUE 벤치마크는 9개의 데이터셋 및 태스크로 구성되어 있습니다.

전체 리포트는 [여기](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)에서 읽어볼 수 있습니다.
</details>

<details>

<summary>Examples of Early Stopping in HuggingFace</summary>

* Hugging Face Transformer를 Early Stopping 정규화와 함께 파인튜닝하는 방법: PyTorch 또는 TensorFlow에서 네이티브하게 구현할 수 있습니다.
* TensorFlow에서 EarlyStopping 콜백은 `tf.keras.callbacks.EarlyStopping` 사용으로 아주 간단하게 구현됩니다.
* PyTorch에는 기본 제공 Early Stopping 메소드는 없지만, 사용할 수 있는 early stopping 훅이 GitHub Gist에 올라와 있습니다.

전체 리포트는 [여기](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)에서 확인하세요.
</details>

<details>

<summary>How to Fine-Tune Hugging Face Transformers on a Custom Dataset</summary>

DistilBERT transformer를 커스텀 IMDB 데이터셋으로 감정 분석(이진 분류) 파인튜닝합니다.

전체 리포트는 [여기](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)에서 읽을 수 있습니다.
</details>

## 도움 받기 또는 기능 요청

Hugging Face W&B 인테그레이션 사용 중 궁금한 점이나 버그, 기능 요청이 있다면 [Hugging Face forums의 이 글](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)에 질문을 남겨주시거나 Hugging Face [Transformers GitHub repo](https://github.com/huggingface/transformers)에 이슈를 남겨주세요.