---
title: What is the `Est. Runs` column?
menu:
  support:
    identifier: ja-support-kb-articles-est_runs_column
support:
- sweeps
- hyperparameter
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、離散的な探索空間を持つ W&B スイープ を作成する際に生成される Run の推定数を提供します。この合計は、探索空間のデカルト積を反映しています。

たとえば、次の探索空間について考えてみましょう。

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="" >}}

この場合、デカルト積は 9 になります。W&B は、この値を推定 Run 数 ( **Est. Runs** ) として App UI に表示します。

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="" >}}

推定 Run 数をプログラムで取得するには、W&B SDK 内の Sweep オブジェクト の `expected_run_count` 属性を使用します。

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```
