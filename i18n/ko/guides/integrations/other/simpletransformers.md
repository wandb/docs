---
description: How to integrate W&B with the Transformers library by Hugging Face.
slug: /guides/integrations/simpletransformers
displayed_sidebar: default
---

# Simple Transformers

이 라이브러리는 Hugging Face의 Transformers 라이브러리를 기반으로 합니다. Simple Transformers를 사용하면 Transformer 모델을 빠르게 훈련하고 평가할 수 있습니다. 모델을 초기화하고, 훈련하고, 평가하는 데 3줄의 코드만 필요합니다. 시퀀스 분류, 토큰 분류(NER), 질문 응답, 언어 모델 미세 조정, 언어 모델 훈련, 언어 생성, T5 모델, Seq2Seq 작업, 멀티-모달 분류 및 대화형 AI를 지원합니다.

## Weights & Biases 프레임워크

Weights and Biases는 모델 트레이닝 시각화를 지원합니다. 이를 사용하려면, `args` 사전의 `wandb_project` 속성에 W&B 프로젝트 이름을 설정하기만 하면 됩니다. 이렇게 하면 모든 하이퍼파라미터 값, 트레이닝 손실 및 평가 메트릭이 지정된 프로젝트에 로그됩니다.

```text
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init`에 전달되는 추가 인수는 `wandb_kwargs`로 전달할 수 있습니다.

## 구조

이 라이브러리는 각 NLP 작업마다 별도의 클래스를 가지도록 설계되었습니다. 유사한 기능을 제공하는 클래스는 함께 그룹화됩니다.

* `simpletransformers.classification` - 모든 분류 모델을 포함합니다.
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - 모든 명명된 엔티티 인식 모델을 포함합니다.
  * `NERModel`
* `simpletransformers.question_answering` - 모든 질문 응답 모델을 포함합니다.
  * `QuestionAnsweringModel`

여기에 몇 가지 간단한 예시가 있습니다

## 멀티레이블 분류

```text
  model = MultiLabelClassificationModel("distilbert","distilbert-base-uncased",num_labels=6,
    args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs":epochs,'learning_rate':learning_rate,
                'wandb_project': "simpletransformers"},
  )
   # 모델 트레이닝
  model.train_model(train_df)

  # 모델 평가
  result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

위 트레이닝 스크립트를 실행한 후 하이퍼파라미터 스윕을 실행한 후 생성된 몇 가지 시각화입니다.

[![](https://camo.githubusercontent.com/3beab1ca06813523711ff7750cb592430b786834/68747470733a2f2f692e696d6775722e636f6d2f6f63784e676c642e706e67)](https://camo.githubusercontent.com/3beab1ca06813523711ff7750cb592430b786834/68747470733a2f2f692e696d6775722e636f6d2f6f63784e676c642e706e67)

[![](https://camo.githubusercontent.com/b864ca220ddd4228027743790ac30741d1f435ad/68747470733a2f2f692e696d6775722e636f6d2f5252423432374d2e706e67)](https://camo.githubusercontent.com/b864ca220ddd4228027743790ac30741d1f435ad/68747470733a2f2f692e696d6775722e636f6d2f5252423432374d2e706e67)

## 질문 응답

```text
  train_args = {
    'learning_rate': wandb.config.learning_rate,
    'num_train_epochs': 2,
    'max_seq_length': 128,
    'doc_stride': 64,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 2,
    'fp16': False,
    'wandb_project': "simpletransformers"
}

model = QuestionAnsweringModel('distilbert', 'distilbert-base-cased', args=train_args)
model.train_model(train_data)
```

위 트레이닝 스크립트를 실행한 후 하이퍼파라미터 스윕을 실행한 후 생성된 몇 가지 시각화입니다.

[![](https://camo.githubusercontent.com/1411cacec6226ebfa23c2e2dddc76ff5e41c136d/68747470733a2f2f692e696d6775722e636f6d2f7664636d7855532e706e67)](https://camo.githubusercontent.com/1411cacec6226ebfa23c2e2dddc76ff5e41c136d/68747470733a2f2f692e696d6775722e636f6d2f7664636d7855532e706e67)

[![](https://camo.githubusercontent.com/b8e12316520d4ad6d16449db2d13ab70e4d4a6e9/68747470733a2f2f692e696d6775722e636f6d2f395732775677732e706e67)](https://camo.githubusercontent.com/b8e12316520d4ad6d16449db2d13ab70e4d4a6e9/68747470733a2f2f692e696d6775722e636f6d2f395732775677732e706e67)

SimpleTransformers는 모든 일반적인 자연어 작업에 대한 클래스와 트레이닝 스크립트를 제공합니다. 여기에는 라이브러리에서 지원하는 전체 글로벌 인수 목록과 그 기본 인수가 있습니다.

```text
global_args = {
  "adam_epsilon": 1e-8,
  "best_model_dir": "outputs/best_model",
  "cache_dir": "cache_dir/",
  "config": {},
  "do_lower_case": False,
  "early_stopping_consider_epochs": False,
  "early_stopping_delta": 0,
  "early_stopping_metric": "eval_loss",
  "early_stopping_metric_minimize": True,
  "early_stopping_patience": 3,
  "encoding": None,
  "eval_batch_size": 8,
  "evaluate_during_training": False,
  "evaluate_during_training_silent": True,
  "evaluate_during_training_steps": 2000,
  "evaluate_during_training_verbose": False,
  "fp16": True,
  "fp16_opt_level": "O1",
  "gradient_accumulation_steps": 1,
  "learning_rate": 4e-5,
  "local_rank": -1,
  "logging_steps": 50,
  "manual_seed": None,
  "max_grad_norm": 1.0,
  "max_seq_length": 128,
  "multiprocessing_chunksize": 500,
  "n_gpu": 1,
  "no_cache": False,
  "no_save": False,
  "num_train_epochs": 1,
  "output_dir": "outputs/",
  "overwrite_output_dir": False,
  "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
  "reprocess_input_data": True,
  "save_best_model": True,
  "save_eval_checkpoints": True,
  "save_model_every_epoch": True,
  "save_steps": 2000,
  "save_optimizer_and_scheduler": True,
  "silent": False,
  "tensorboard_dir": None,
  "train_batch_size": 8,
  "use_cached_eval_features": False,
  "use_early_stopping": False,
  "use_multiprocessing": True,
  "wandb_kwargs": {},
  "wandb_project": None,
  "warmup_ratio": 0.06,
  "warmup_steps": 0,
  "weight_decay": 0,
}
```

더 자세한 문서는 [simpletransformers on github](https://github.com/ThilinaRajapakse/simpletransformers)를 참조하세요.

가장 인기 있는 GLUE 벤치마크 데이터셋 중 일부에서 트랜스포머를 트레이닝하는 것에 대한 [이 Weights and Biases 리포트](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA)를 확인하십시오. colab에서 직접 시도해 보세요 [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing)