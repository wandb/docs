---
description: OpenAI GymとW&Bを統合する方法
slug: /guides/integrations/openai-gym
displayed_sidebar: default
---


# OpenAI Gym

:::info
"2021年からGymを維持してきたチームは、今後すべての開発を[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)に移行しました。GymnasiumはGymの代替となるもので（import gymnasium as gym）、Gymは今後更新されません。" ([Source](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gymはもう積極的にメンテナンスされていないので、Gymnasiumとのインテグレーションを試してみてください。詳細はこちら # TODO リンクを追加。
:::

[OpenAI Gym](https://gym.openai.com/)を使用している場合、`gym.wrappers.Monitor`によって生成された環境のビデオを自動的にログします。ただし、`monitor_gym`キーワード引数を[`wandb.init`](../../../ref/python/init.md)に`True`として設定するか、`wandb.gym.monitor()`を呼び出してください。

私たちのgymインテグレーションは非常に軽量です。ただ[ビデオファイルの名前を見る](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/\_\_init\_\_.py#L15)だけで、その名前を使用するか、見つからない場合は「videos」としてログします。より詳細なコントロールが必要な場合は、手動で[ビデオをログ](../../track/log/media.md)することもできます。

[CleanRL](https://github.com/vwxyzjn/cleanrl)の[OpenRL Benchmark](http://wandb.me/openrl-benchmark-report)は、このインテグレーションを使用してOpenAI Gymの例を提供しています。gymの使用方法を示すソースコード（特定のRunsで使用される[特定のコード](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)を含む）を見つけることができます。

![詳しくはこちら: http://wandb.me/openrl-benchmark-report](/images/integrations/open_ai_report_example.png)