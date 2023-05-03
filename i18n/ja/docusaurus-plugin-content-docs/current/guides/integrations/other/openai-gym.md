---

slug: /guides/integrations/openai-gym

description: W&BとOpenAI Gymを統合する方法です。

---

# OpenAI Gym

[OpenAI Gym](https://gym.openai.com/)をお使いの場合、`gym.wrappers.Monitor`で生成された環境のビデオを自動的に記録します。[`wandb.init`](../../../ref/python/init.md)の`monitor_gym`キーワード引数を`True`に設定するか、`wandb.gym.monitor()`を呼び出してください。

Gymとの統合は非常に軽量です。`gym`から記録されているビデオファイルの名前を[調べ](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)、それに基づいて名前を付けたり、マッチしない場合は`"videos"`にフォールバックします。さらにコントロールが必要な場合は、手動で[ビデオを記録](../../track/log/media.md)することもできます。

[CleanRL](https://github.com/vwxyzjn/cleanrl)の[OpenRL Benchmark](http://wandb.me/openrl-benchmark-report)は、OpenAI Gymの例でこの統合を使用しています。Gymを使って実行する方法を示すソースコード（[特定のランで使用された特定のコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)を含む）があります。

![詳細はこちら: http://wandb.me/openrl-benchmark-report](/images/integrations/open_ai_report_example.png)