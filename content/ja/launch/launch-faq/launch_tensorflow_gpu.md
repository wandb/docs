---
title: How do I make W&B Launch work with Tensorflow on GPU?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_tensorflow_gpu
    parent: launch-faq
---

GPU を使用する TensorFlow ジョブの場合、コンテナビルド用のカスタムベースイメージを指定します。これにより、run 中の適切な GPU 使用率が保証されます。リソース設定の `builder.accelerator.base_image` キーの下にイメージタグを追加します。例：

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

W&B 0.15.6 より前のバージョンでは、`base_image` の親キーとして `accelerator` の代わりに `cuda` を使用してください。
