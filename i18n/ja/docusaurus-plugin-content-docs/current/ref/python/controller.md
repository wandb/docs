# コントローラ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_sweep.py#L119-L143)
公開スイープコントローラのコンストラクタ。

```python
controller(
 sweep_id_or_config: Optional[Union[str, Dict]] = None,
 entity: Optional[str] = None,
 project: Optional[str] = None
)
```
こちらのMarkdownテキストを日本語に翻訳してください。それ以外のことは何も言わずに、翻訳されたテキストのみを返してください。テキスト：

#### 使い方：

```python
import wandb
以下は、Markdownテキストの一部です。日本語に翻訳してください。翻訳されたテキストだけを返してください。他のことは何も言わないでください。テキスト：

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```