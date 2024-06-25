---
description: W&B を Hugging Face の Transformers ライブラリと統合する方法。
slug: /guides/integrations/simpletransformers
displayed_sidebar: default
---


# Simple Transformers

このライブラリは、Hugging Face の Transformers ライブラリに基づいています。Simple Transformers を使うと、Transformer モデルのトレーニングと評価を素早く行うことができます。モデルの初期化、トレーニング、評価のためには、わずか3行のコードが必要です。Sequence Classification、Token Classification \(NER\)、Question Answering、Language Model Fine-Tuning、Language Model Training、Language Generation、T5 Model、Seq2Seq Tasks、Multi-Modal Classification、Conversational AI をサポートしています。

## Weights & Biases フレームワーク

モデルトレーニングの可視化には、Weights & Biases がサポートされています。これを使用するには、`args` 辞書の `wandb_project` 属性に W&B のプロジェクト名を設定するだけです。これにより、すべてのハイパーパラメータ値、トレーニング損失、および評価メトリクスが指定されたプロジェクトにログされます。

```text
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init` に渡す追加の引数は、`wandb_kwargs` として渡すことができます。

## 構造

このライブラリは、各 NLP タスクごとに個別のクラスを持つように設計されています。同様の機能を提供するクラスは一緒にグループ化されています。

* `simpletransformers.classification` - すべての分類モデルを含みます。
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - すべての名前付きエンティティ認識モデルを含みます。
  * `NERModel`
* `simpletransformers.question_answering` - すべての質問応答モデルを含みます。
  * `QuestionAnsweringModel`

ここにいくつかの最小の例を示します。

## MultiLabel Classification

```text
  model = MultiLabelClassificationModel("distilbert","distilbert-base-uncased",num_labels=6,
    args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs":epochs,'learning_rate':learning_rate,
                'wandb_project': "simpletransformers"},
  )
   # モデルのトレーニング
  model.train_model(train_df)

  # モデルの評価
  result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

ハイパーパラメータ sweep の後、このトレーニングスクリプトから生成された可視化はこちらです。

[![](https://camo.githubusercontent.com/3beab1ca06813523711ff7750cb592430b786834/68747470733a2f2f692e696d6775722e636f6d2f6f63784e676c642e706e67)](https://camo.githubusercontent.com/3beab1ca06813523711ff7750cb592430b786834/68747470733a2f2f692e696d6775722e636f6d2f6f63784e676c642e706e67)

[![](https://camo.githubusercontent.com/b864ca220ddd4228027743790ac30741d1f435ad/68747470733a2f2f692e696d6775722e636f6d2f5252423432374d2e706e67)](https://camo.githubusercontent.com/b864ca220ddd4228027743790ac30741d1f435ad/68747470733a2f2f692e696d6775722e636f6d2f5252423432374d2e706e67)

## Question Answering

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

ハイパーパラメータ sweep の後、このトレーニングスクリプトから生成された可視化はこちらです。

[![](https://camo.githubusercontent.com/1411cacec6226ebfa23c2e2dddc76ff5e41c136d/68747470733a2f2f692e696d6775722e636f6d2f7664636d7855532e706e67)](https://camo.githubusercontent.com/1411cacec6226ebfa23c2e2dddc76ff5e41c136d/68747470733a2f2f692e696d6775722e636f6d2f7664636d7855532e706e67)

[![](https://camo.githubusercontent.com/b8e12316520d4ad6d16449db2d13ab70e4d4a6e9/68747470733a2f2f692e696d6775722e636f6d2f395732775677732e706e67)](https://camo.githubusercontent.com/b8e12316520d4ad6d16449db2d13ab70e4d4a6e9/68747470733a2f2f692e696d6775722e636f6d2f395732775677732e706e67)

SimpleTransformers は、すべての一般的な自然言語タスクのためのクラスおよびトレーニングスクリプトを提供します。ライブラリがサポートするグローバル引数とそのデフォルト引数の完全なリストを以下に示します。

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

より詳細なドキュメントについては、[simpletransformers on github](https://github.com/ThilinaRajapakse/simpletransformers) を参照してください。

Weights & Biases のレポートは [こちら](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA)。最も人気のあるGLUEベンチマークデータセットでトランスフォーマーをトレーニングする方法をカバーしています。Colabでも試してみてください。[![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing)