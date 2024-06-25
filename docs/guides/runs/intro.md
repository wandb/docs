---
description: W&Bの基本的な構成要素であるRunについて学びましょう。
slug: /guides/runs
displayed_sidebar: default
---


# Runs

W&Bによって記録される計算の単一単位は*run*と呼ばれます。W&Bのrunをプロジェクト全体の原子的な要素と考えることができます。以下の場合に新しいrunを開始する必要があります：

* モデルをトレーニングする
* ハイパーパラメーターを変更する
* 異なるモデルを使用する
* [W&B Artifact](../artifacts/intro.md)としてデータやモデルをログに記録する
* [W&B Artifactをダウンロードする](../artifacts/download-and-use-an-artifact.md)

例えば、[sweep](../sweeps/intro.md)の間、W&Bは指定されたハイパーパラメーター検索空間を探索します。sweepによって生成された各新しいハイパーパラメーターの組み合わせは、ユニークなrunとして実装および記録されます。

:::tip
runを作成および管理する際に考慮すべき重要なポイント：
* `wandb.log`でログを記録したものはすべてそのrunに記録されます。W&Bでオブジェクトをログに記録する方法の詳細については、[Log Media and Objects](../track/log/intro.md)を参照してください。
* 各runは特定のW&Bプロジェクトに関連付けられています。
* W&B App UIのプロジェクトワークスペース内でrunおよびそのプロパティを表示します。
* 任意のプロセスでアクティブな [`wandb.Run`](../../ref/python/run.md) は多くとも1つしかなく、`wandb.run`としてアクセス可能です。
:::

## runを作成する

[`wandb.init()`](../../ref/python/init.md)を使用してW&Bのrunを作成します：

```python
import wandb

run = wandb.init()
```

新しいrunを作成する際には、プロジェクト名とW&Bエンティティを指定することをお勧めします。W&Bは指定されたエンティティ内に新しいプロジェクト（プロジェクトが既に存在しない場合）を作成します。プロジェクトが既に存在する場合、W&Bはそのプロジェクトにrunを保存します。

例えば、以下のコードスニペットは、`wandbee`エンティティ内にスコープされた`model_registry_example`というプロジェクトに保存されるrunを初期化します：

```python
import wandb

run = wandb.init(entity="wandbee", \
        project="model_registry_example")
```

W&Bは作成されたrunの名前と、その特定のrunに関する詳細情報を得るためのURLパスを出力します。

例えば、上記のコードスニペットは次のような出力を生成します：
![](/images/runs/run_example.png)

## runをrun名とrun IDで整理する
デフォルトでは、新しいrunを初期化するとW&Bはランダムな名前とrun IDを生成します。

前述の例では、W&Bは`likely-lion-9`というrun名と`xlm66ixq`というrun IDを生成します。`likely-lion-9`runは`model_registry_example`というプロジェクトに保存されます。

:::note
W&Bによって生成されるrun名がユニークであることは保証されていません。
:::

runを初期化する際に、`id`パラメータでユニークなrun ID識別子を、`name`パラメータでrunの名前を指定することができます。例えば、

```python 
import wandb

run = wandb.init(
    entity="<project>", 
    project="<project>", 
    name="<run-name>", 
    id="<run-id>"
)
```

run名とrun IDを使用して、W&B App UI内のプロジェクト内で実験をすばやく見つけることができます。特定のrunに関する詳細情報をURLで見つけることができます：

```text title="W&B App URL for a specific run"
https://wandb.ai/entity/project-name/run-id
```

ここで：
* `entity`: runを初期化したW&Bエンティティ。
* `project`: runが保存されるプロジェクト。
* `run-id`: runのrun ID。

:::tip
runを初期化する際にプロジェクト名を指定することをW&Bは推奨しています。プロジェクトが指定されていない場合、W&Bはrunを"Uncategorized"プロジェクトに保存します。
:::

[`wandb.init`](../../ref/python/init.md)のリファレンスドキュメントには、使用可能なパラメータの完全なリストが記載されています。

## runを表示する

runが記録されたプロジェクト内で特定のrunを表示します：

1. W&B App UIの[https://wandb.ai/home](https://wandb.ai/home)に移動します。
2. runを初期化した際に指定したW&Bプロジェクトに移動します。
3. プロジェクトのワークスペース内に**Runs**というラベルの付いたテーブルが表示されます。このテーブルにはプロジェクト内のすべてのrunがリストされています。表示したいrunをリストから選択します。
  ![Example project workspace called 'sweep-demo'](/images/app_ui/workspace_tab_example.png)
4. 次に、**Overview Tab**アイコンを選択します。

次の画像は**sparkling-glade-2**と呼ばれるrunの情報を示しています：

![W&B Dashboard run overview tab](/images/app_ui/wandb_run_overview_page.png)

**Overview Tab**には、選択したrunに関する次の情報が表示されます：

* **Run name**: runの名前。
* **Description**: 提供されたrunの説明。このフィールドは、run作成時に説明が指定されていない場合は初期的に空白のままです。W&B App UIまたはプログラムでrunの説明を追加することができます。
* **Privacy**: runのプライバシー設定。**Private**または**Public**に設定できます。
    * **Private**: (デフォルト) あなただけが表示および貢献できます。
    * **Public**: 誰でも表示できます。
* **Tags**: (リスト、オプション) 文字列のリスト。Tagsはrunを整理するのに役立ちます。また、一時的なラベル（例えば "baseline" や "production"）を適用するのにも役立ちます。
* **Author**: runを作成したW&Bユーザー名。
* **Run state**: runの状態：
  * **finished**: スクリプトが終了し、データが完全に同期されたか、`wandb.finish()`が呼び出された
  * **failed**: スクリプトが非ゼロの終了ステータスで終了した
  * **crashed**: 内部プロセスでスクリプトが心拍を送信しなくなった、これが発生するのはマシンがクラッシュした場合
  * **running**: スクリプトがまだ実行中で最近の心拍を送信した
* **Start time**: runが開始されたタイムスタンプ。
* **Duration**: runが**finish**、**fail**、**crash**するまでにかかった時間（秒）。
* **Run path**: 一意のrun識別子。形式は`entity/project/run-ID`。
* **Host name**: runが起動された場所。ローカルマシンでrunが起動された場合はマシン名が表示されます。
* **Operating system**: runに使用されたオペレーティングシステム。
* **Python version**: runに使用されたPythonバージョン。
* **Python executable**: runを開始したコマンド。
* **System Hardware**: runを作成するために使用されたハードウェア。
* **W&B CLI version**: runコマンドをホストしたマシンにインストールされているW&B CLIバージョン。
* **Job Type**:

概要セクションの下にはさらに以下の情報が表示されます：

* **Artifact Outputs**: runによって生成されたArtifactの出力。
* **Config**: [`wandb.config`](../../guides/track/config.md)で保存された設定パラメータのリスト。
* **Summary**: [`wandb.log()`](../../guides/track/log/intro.md)で保存されたサマリーパラメータのリスト。デフォルトでは、この値は最後にログに記録された値に設定されます。

プロジェクト内の複数のrunを整理する方法の詳細については、[Runs Table](../app/features/runs-table.md)ドキュメントを参照してください。

プロジェクトのワークスペースのライブ例については、[この例のプロジェクト](https://app.wandb.ai/example-team/sweep-demo)を参照してください。

## runを終了する
W&Bは自動的にrunsを終了し、そのrunからW&Bプロジェクトにデータをログに記録します。`run.finish`コマンドを使用して手動でrunを終了できます。例えば：

```python
import wandb

run = wandb.init()
run.finish()
```

:::info
子プロセスから[`wandb.init`](../../ref/python/init.md)を呼び出す場合、子プロセスの終了時に[`wandb.finish`](../../ref/python/finish.md)メソッドを使用することをW&Bは推奨しています。
:::

