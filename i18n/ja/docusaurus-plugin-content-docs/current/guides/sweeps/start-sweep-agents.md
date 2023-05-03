---
description: 1台以上のマシンでW&Bスイープエージェントを開始または停止します。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# スイープエージェントの開始

<head>
  <title>W&Bスイープの開始または停止</title>
</head>

1台以上のマシンの1つ以上のエージェントでW&Bスイープを開始します。W&Bスイープエージェントは、W&Bスイープを初期化した際に起動されたWeights & Biasesサーバーに問い合わせてハイパーパラメーターを取得し、それらを使用してモデルトレーニングを実行します。

W&Bスイープエージェントを開始するためには、W&Bスイープを初期化した際に返されたW&BスイープIDを提供してください。W&BスイープIDは以下の形式です。

```bash
entity/project/sweep_ID
```

ここで、

* entity: Weights & Biasesのユーザー名またはチーム名。
* project: W&B ランの出力を格納するプロジェクト名。プロジェクトが指定されていない場合、ランは「Uncategorized」のプロジェクトに入れられます。
* sweep\_ID: W&Bが生成する疑似ランダムで一意なID。

W&BスイープエージェントをJupyterノートブックまたはPythonスクリプト内で開始する場合、W&Bスイープが実行する関数名を提供してください。

次のコードスニペットは、Weights & Biasesでエージェントを開始する方法を示しています。すでに設定ファイルがあり、W&Bスイープを初期化したことを前提としています。設定ファイルの定義方法については、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'PythonスクリプトまたはJupyterノートブック', value: 'python'},
  ]}>
  <TabItem value="cli">
`wandb agent` コマンドを使ってスイープを開始してください。スイープを初期化したときに返されたスイープIDを指定します。以下のコードスニペットをコピーして `sweep_id` をあなたのスイープIDに置き換えてください:

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

Weights & Biases Python SDK ライブラリを使ってスイープを開始します。スイープを初期化したときに返されたスイープIDを指定します。さらに、スイープが実行する関数名を指定します。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### W&B エージェントの停止

:::caution
ランダムおよびベイズ探索は無期限に実行されます。コマンドライン、Pythonスクリプト内、または [Sweeps UI](./visualize-sweep-results.md) からプロセスを停止する必要があります。
:::

スイープエージェントが試すべき W&B Runs の数をオプションで指定します。以下のコードスニペットは、CLI および Jupyter Notebook、Pythonスクリプト内で最大の [W&B Runs](../../ref/python/run.md) を設定する方法を示しています。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Pythonスクリプト または Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">
まず、スイープを初期化します。詳細は、[Initialize sweeps](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、スイープジョブを開始します。スイープの開始から生成されたスイープIDを指定してください。countパラメータに整数値を渡して、試行するrunの最大数を設定します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```
  </TabItem>
  <TabItem value="cli">

まず、[`wandb sweep`](https://docs.wandb.ai/ref/cli/wandb-sweep) コマンドでスイープを初期化します。詳細は、[Initialize sweeps](https://docs.wandb.ai/guides/sweeps/initialize-sweeps) を参照してください。

```
wandb sweep config.yaml
```

countフラグに整数値を渡して、試行するrunの最大数を設定します。

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>
How to use Markdown

Markdown is a lightweight markup language that you can use to add formatting elements to your plain text files. With Markdown, you can easily create and maintain well-structured documents without needing to use a rich-text editor or HTML.

#### Basic Syntax

Here are some examples of basic Markdown syntax:

1. Headers

   You can create headers in Markdown by using the '#' symbol. The number of '#' symbols corresponds to the header level.

   ```
   # Heading 1
   ## Heading 2
   ### Heading 3
   ```

2. Bold and Italic text

   You can make text bold by surrounding it with two asterisks `**` or underscores `__`. To make text italic, use one asterisk `*` or underscore `_`.

   ```
   **Bold text**
   __Bold text__

   *Italic text*
   _Italic text_
   ```

3. Lists

   You can create ordered and unordered lists in Markdown. To create an ordered list, use numbers followed by a period. To create an unordered list, use asterisks, plus signs, or hyphens.

   ```
   1. First item
   2. Second item
   3. Third item

   * Unordered item
   + Unordered item
   - Unordered item
   ```

4. Links and Images

   To create a link, wrap the link text in square brackets `[]`, followed by the URL in parentheses `()`.

   ```
   [Example Link](https://example.com)
   ```

   To add an image, add an exclamation mark `!` before the square brackets, and place the image URL inside the parentheses.

   ```
   ![Image description](https://example.com/image.jpg)
   ```

5. Code blocks

   You can create code blocks by wrapping the code in triple backticks ```. You can also add a language identifier for syntax highlighting.

   ```python
   def hello_world():
       print("Hello, world!")
   ```

#### Conclusion

Markdown is an easy-to-use and powerful tool for creating and maintaining well-structured documents. Its simple syntax allows you to focus on the content, while still providing the ability to format your text. Give it a try and see how it can improve your writing experience!