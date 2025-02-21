---
title: How do I make W&B Launch work with Tensorflow on GPU?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_tensorflow_gpu
    parent: launch-faq
---

TensorFlow のジョブで GPU を使用する場合、コンテナビルドのためにカスタムベースイメージを指定してください。これにより、run 中に適切な GPU の利用が確保されます。リソース設定の `builder.accelerator.base_image` キーの下にイメージタグを追加します。例えば：

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

W&B 0.15.6 より前のバージョンでは、`base_image` の親キーとして `accelerator` の代わりに `cuda` を使用します。