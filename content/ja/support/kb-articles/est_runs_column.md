---
title: 「Est. Runs」列とは何ですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
- ハイパーパラメーター
---

W&B では、離散的なサーチスペースで W&B Sweep を作成する際に、生成される Run の推定数が表示されます。この合計はサーチスペースのデカルト積を反映しています。

例えば、次のようなサーチスペースを考えてみましょう。

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="推定 Run の列" >}}

この場合、デカルト積は 9 になります。W&B はこの値を App UI の推定 run 数（**Est. Runs**）として表示します。

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="Sweep の run 推定値" >}}

推定 Run 数をプログラムで取得したい場合は、W&B SDK における Sweep オブジェクトの `expected_run_count` 属性を使用してください。

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```
