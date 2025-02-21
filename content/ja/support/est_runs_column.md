---
title: What is the `Est. Runs` column?
menu:
  support:
    identifier: ja-support-est_runs_column
tags:
- sweeps
- hyperparameter
toc_hide: true
type: docs
---

W&B は、離散的な探索空間で W&B Sweep を作成する際に生成される Run の推定数を提供します。この合計は、探索空間のデカルト積を反映しています。

例えば、次の探索空間を考えてみましょう：

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="" >}}

この場合、デカルト積は 9 になります。W&B はこの値を App UI に推定ラン数（**Est. Runs**）として表示します。

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="" >}}

推定された Run 数をプログラムで取得するには、W&B SDK 内の Sweep オブジェクトの `expected_run_count` 属性を使用します。

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```