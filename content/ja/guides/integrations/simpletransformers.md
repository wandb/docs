---
title: Simple Transformers
description: Hugging Face の Transformers ライブラリと W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-simpletransformers
    parent: integrations
weight: 390
---

このライブラリは、Hugging Face の Transformers ライブラリに基づいています。Simple Transformers を使用すると、Transformer モデルを迅速にトレーニングおよび評価できます。モデルの初期化、モデルのトレーニング、およびモデルの評価には、わずか 3 行のコードで済みます。Sequence Classification、Token Classification \(NER\)、Question Answering、Language Model Fine-Tuning、Language Model Training、Language Generation、T5 Model、Seq2Seq Tasks、Multi-Modal Classification、Conversational AI をサポートしています。

モデル トレーニング の可視化に Weights & Biases を使用するには、`args` 辞書の `wandb_project` 属性に W&B のプロジェクト名を設定します。これにより、すべてのハイパーパラメーター 値、トレーニング ロス、および評価メトリクスが指定されたプロジェクトにログされます。

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

`wandb.init` に渡す追加の引数は、`wandb_kwargs` として渡すことができます。

## 構造

このライブラリは、すべての NLP タスクに対して個別のクラスを持つように設計されています。類似の機能を提供するクラスは、グループ化されています。

* `simpletransformers.classification` - すべての Classification モデルを含みます。
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - すべての Named Entity Recognition モデルを含みます。
  * `NERModel`
* `simpletransformers.question_answering` - すべての Question Answering モデルを含みます。
  * `QuestionAnsweringModel`

以下は、いくつかの最小限の例です。

## MultiLabel Classification

```text
  model = MultiLabelClassificationModel("distilbert","distilbert-base-uncased",num_labels=6,
    args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs": epochs, 'learning_rate': learning_rate,
                'wandb_project': "simpletransformers"},
  )
   # モデルをトレーニングする
  model.train_model(train_df)

  # モデルを評価する
  result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

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

SimpleTransformers は、すべての一般的な自然言語タスクのためのクラスとトレーニングスクリプトを提供します。ここでは、ライブラリでサポートされているグローバル引数の完全なリストと、そのデフォルト引数を示します。

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

詳細なドキュメントについては、[github の simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) を参照してください。

最も人気のある GLUE ベンチマークデータセットでのトランスフォーマーのトレーニングをカバーしている [こちらの Weights & Biases レポート](https://app.wandb.ai/cayush/simpletransformers/reports/Using-simpleTransformer-on-common-NLP-applications---Vmlldzo4Njk2NA) をご覧ください。 [ぜひ自分で colab で試してみてください](https://colab.research.google.com/drive/1oXROllqMqVvBFcPgTKJRboTq96uWuqSz?usp=sharing)。