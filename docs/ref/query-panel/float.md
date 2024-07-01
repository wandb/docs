# float

## チェーン可能な操作
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値の不等性を判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を別の番号で除算し、余りを返す

| 引数 |  |
| :--- | :--- |
| `lhs` | 除算する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 除算する [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [番号](https://docs.wandb.ai/ref/weave/number) を掛け合わせます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2つ目の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数 [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 指数 [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
基数の [番号](https://docs.wandb.ai/ref/weave/number) をn乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [番号](https://docs.wandb.ai/ref/weave/number) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2つ目の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の和

<h3 id="number-sub"><code>number-sub</code></h3>

1つの [番号](https://docs.wandb.ai/ref/weave/number) を別の [番号](https://docs.wandb.ai/ref/weave/number) から引きます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 引く元の [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 引く [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の差

<h3 id="number-div"><code>number-div</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を別の [番号](https://docs.wandb.ai/ref/weave/number) で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 分割する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 分割する [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の商

<h3 id="number-less"><code>number-less</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より小さいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より小さいかどうか。

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より小さいか等しいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より小さいか等しいかどうか。

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値の等価性を判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より大きいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より大きいかどうか。

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より大きいか等しいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より大きいか等しいかどうか。

<h3 id="number-negate"><code>number-negate</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を否定します。

| 引数 |  |
| :--- | :--- |
| `val` | 否定する番号 |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toString"><code>number-toString</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を文字列に変換します。

| 引数 |  |
| :--- | :--- |
| `in` | 変換する番号 |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を _タイムスタンプ_ に変換します。値が31536000000 未満の場合は秒に、31536000000000 未満の場合はミリ秒に、31536000000000000 未満の場合はマイクロ秒に、31536000000000000000 未満の場合はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する番号 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number) の絶対値


## リスト操作
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値の不等性を判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を別の番号で除算し、余りを返す

| 引数 |  |
| :--- | :--- |
| `lhs` | 除算する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 除算する [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [番号](https://docs.wandb.ai/ref/weave/number) を掛け合わせます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2つ目の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数 [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 指数 [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
基数の [番号](https://docs.wandb.ai/ref/weave/number) をn乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [番号](https://docs.wandb.ai/ref/weave/number) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2つ目の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の和

<h3 id="number-sub"><code>number-sub</code></h3>

1つの [番号](https://docs.wandb.ai/ref/weave/number) を別の [番号](https://docs.wandb.ai/ref/weave/number) から引きます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 引く元の [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 引く [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の差

<h3 id="number-div"><code>number-div</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を別の [番号](https://docs.wandb.ai/ref/weave/number) で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 分割する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 分割する [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの [番号](https://docs.wandb.ai/ref/weave/number) の商

<h3 id="number-less"><code>number-less</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より小さいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より小さいかどうか。

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より小さいか等しいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より小さいか等しいかどうか。

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値の等価性を判断します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より大きいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より大きいかどうか。

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) が別の番号より大きいか等しいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [番号](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の [番号](https://docs.wandb.ai/ref/weave/number) が2つ目の番号より大きいか等しいかどうか。

<h3 id="number-negate"><code>number-negate</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を否定します。

| 引数 |  |
| :--- | :--- |
| `val` | 否定する番号 |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の [番号](https://docs.wandb.ai/ref/weave/number) のインデックスを見つけます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [番号](https://docs.wandb.ai/ref/weave/number) のインデックスを見つけるための _リスト_ |

#### 戻り値
最大の [番号](https://docs.wandb.ai/ref/weave/number) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の [番号](https://docs.wandb.ai/ref/weave/number) のインデックスを見つけます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [番号](https://docs.wandb.ai/ref/weave/number) のインデックスを見つけるための _リスト_ |

#### 戻り値
最小の [番号](https://docs.wandb.ai/ref/weave/number) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) の平均値

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を取るための _リスト_ の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number) の平均値

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大の番号

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](https://docs.wandb.ai/ref/weave/number) を見つけるための _リスト_ |

#### 戻り値
最大の [番号](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小の番号

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](https://docs.wandb.ai/ref/weave/number) を見つけるための _リスト_ |

#### 戻り値
最小の [番号](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算するための _リスト_ の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) の合計値

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計を求めるための _リスト_ の [番号](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number) の合計値

<h3 id="number-toString"><code>number-toString</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を文字列に変換します。

| 引数 |  |
| :--- | :--- |
| `in` | 変換する番号 |

#### 戻り値
[番号](https://docs.wandb.ai/ref/weave/number) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) を _タイムスタンプ_ に変換します。値が31536000000 未満の場合は秒に、31536000000000 未満の場合はミリ秒に、31536000000000000 未満の場合はマイクロ秒に、31536000000000000000 未満の場合はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する番号 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[番号](https://docs.wandb.ai/ref/weave/number) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [番号](https://docs.wandb.ai/ref/weave/number) |

