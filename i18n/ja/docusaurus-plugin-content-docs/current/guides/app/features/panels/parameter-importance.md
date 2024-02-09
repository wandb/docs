---
description: >-
  Visualize the relationships between your model's hyperparameters and output
  metrics
displayed_sidebar: ja
---

# パラメーター重要度

このパネルは、ハイパーパラメータがメトリクスの望ましい値に対して最良の予測子であり、高い相関があったかどうかを明示します。

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578795733856\_image.png)

**相関** は、ハイパーパラメータと選択されたメトリック（この場合はval\_loss）の間の線形相関です。したがって、相関が高いということは、ハイパーパラメータの値が高い場合、メトリックも高い値であることを意味します。相関は見る価値のあるメトリックですが、入力間の2次相互作用を捉えることができず、範囲が大幅に異なる入力を比較するときには扱いにくくなります。

したがって、**インポータンス**指標も計算します。これは、ハイパーパラメーターを入力として、メトリックを対象とした出力としてランダムフォレストをトレーニングし、ランダムフォレストの特徴重要度の値を報告するものです。

この手法のアイデアは、[Jeremy Howard](https://twitter.com/jeremyphoward) との会話で得たもので、彼は[Fast.ai](http://fast.ai)でランダムフォレストの特徴重要度を使ってハイパーパラメータ空間を調べることを先駆けて行っています。この分析の背後にある動機を理解するために、彼の素晴らしい[講義](http://course18.fast.ai/lessonsml1/lesson4.html)（およびこれらの[ノート](https://forums.fast.ai/t/wiki-lesson-thread-lesson-4/7540)）をぜひチェックしてみてください。

このハイパーパラメーター重要度パネルは、高度に相関したハイパーパラメータの間の複雑な相互作用を解きほぐします。それによって、モデルのパフォーマンスを予測する上で最も重要なハイパーパラメータがどれかを示し、ハイパーパラメータの検索を微調整する手助けをしてくれます。

## ハイパーパラメータ重要度パネルの作成

Weights & Biasesプロジェクトに移動します。もし持っていなければ、[このプロジェクト](https://app.wandb.ai/sweep/simpsons)を使うことができます。

プロジェクトページから、**Add Visualization**をクリックしてください。

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578795570241\_image.png)

次に、**Parameter Importance** を選択してください。

[Weights & Biasesとの統合](https://docs.wandb.com/quickstart) 以外に、新しいコードを書く必要はありません。

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578795636072\_image.png)

:::info
空のパネルが表示される場合、実行がグループ化されていないことを確認してください。
:::

## ハイパーパラメータ重要度パネルの使用

パラメータマネージャーの横にある魔法の杖をクリックすることで、wandbによって最も有用なハイパーパラメータの組を可視化させることができます。そして、重要度に基づいてハイパーパラメータを並べ替えることができます。

![自動パラメータ可視化の利用](/images/app_ui/hyperparameter_importance_panel.gif)

パラメータマネージャを使って、表示されるパラメータと非表示のパラメータを手動で設定できます。

![表示と非表示のフィールドを手動で設定](/images/app_ui/hyperparameter_importance_panel_manual.gif)

## ハイパーパラメータ重要度パネルの解釈

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578798509642\_image.png)

このパネルでは、トレーニングスクリプト内の[wandb.config](https://docs.wandb.com/library/python/config)オブジェクトに渡されるすべてのパラメータが表示されます。次に、選択したモデルメトリック（この場合は`val_loss`）に対するこれらのconfigパラメータの重要度と相関関係が表示されます。

### 重要度

重要度の列は、各ハイパーパラメーターが選択したメトリックの予測にどれだけ役立ったかを示しています。多数のハイパーパラメーターを調整して始め、このプロットを使用してどのパラメーターがさらなる探索に値するかを絞り込むシナリオを想像できます。その後のスイープは、最も重要なハイパーパラメーターに限定されるため、より高速で安価に優れたモデルを見つけることができます。

注: これらの重要度は、ツリーベースのモデルを使って計算します。なぜなら、このモデルはカテゴリカルデータや正規化されていないデータに対しても許容範囲が広いからです。\
先述のパネルでは、`epochs`、`learning_rate`、`batch_size`、そして `weight_decay`がかなり重要であることがわかります。

次に、これらのハイパーパラメーターのより詳細な値を探索する別のスイープを実行することができます。興味深いことに、`learning_rate`と`batch_size`は重要でしたが、それらは出力とあまり相関がありませんでした。\
これにより、相関について考えることになります。

### 相関関係

相関は、個々のハイパーパラメータとメトリック値の間の線形関係を捉えます。それらはこのような質問に答えます - ハイパーパラメータを使うことと、例えばSGDオプティマイザと、val_lossとの間には有意な関係があるのでしょうか（この場合の答えははいです）。相関係数は-1から1の範囲で、正の値は正の線形相関を示し、負の値は負の線形相関を示し、0の値は相関がないことを示します。一般的に、どちらの方向でも0.7以上の値は強い相関を示しています。

このグラフを使用して、メトリックとの相関が高い値をさらに探索するか（この場合、rmspropやnadamよりも確率的勾配降下法やadamを選択するかもしれません）、またはより多くのエポックで学習することができます。

相関を解釈する際の注意点：

* 相関は関連性の証拠を示しますが、必ずしも因果関係ではありません。
* 相関は外れ値に敏感であり、特に試行するハイパーパラメーターのサンプルサイズが小さい場合、強い関係性を緩やかなものに変える可能性があります。
* 最後に、相関はハイパーパラメーターとメトリクスの間の線型関係のみを捉えます。強い多項式関係がある場合、相関では捉えられません。

重要度と相関の違いは、重要度がハイパーパラメーター間の相互作用を考慮しているのに対し、相関は個々のハイパーパラメーターがメトリック値に与える影響だけを測定していることに起因します。また、相関は線型関係性のみを捉えますが、重要度はより複雑な関係性を捉えることができます。

ご覧のように、重要度と相関の両方とも、ハイパーパラメーターがモデルのパフォーマンスにどのように影響するかを理解するための強力なツールです。

このパネルがこれらの洞察を把握し、より迅速に強力なモデルを絞り込むのに役立つことを願っています。