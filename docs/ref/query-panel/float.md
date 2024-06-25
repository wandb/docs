
# float

## チェイナブルな操作
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないことを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割り、余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割るための[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割るための別の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を掛けます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を特定の指数で累乗します

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 指数の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
基数の[数値](https://docs.wandb.ai/ref/weave/number)をnth乗したもの

<h3 id="number-add"><code>number-add</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を加算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)から別の数値を減算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 減算される[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の差

<h3 id="number-div"><code>number-div</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割るための[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割るための別の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の商

<h3 id="number-less"><code>number-less</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より小さいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以下かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より大きいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以上かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を否定します

| 引数 |  |
| :--- | :--- |
| `val` | 否定する数値 |

#### 戻り値
否定された[数値](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toString"><code>number-toString</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### 戻り値
数値の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を _タイムスタンプ_ に変換します。31536000000未満の値は秒に変換され、31536000000000未満の値はミリ秒に変換され、31536000000000000未満の値はマイクロ秒に変換され、31536000000000000000未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)の絶対値を計算します

| 引数 |  |
| :--- | :--- |
| `n` | 絶対値を計算する[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[数値](https://docs.wandb.ai/ref/weave/number)の絶対値


## リストの操作
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないことを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割り、余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割るための[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割るための別の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を掛けます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を特定の指数で累乗します

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 指数の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
基数の[数値](https://docs.wandb.ai/ref/weave/number)をnth乗したもの

<h3 id="number-add"><code>number-add</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を加算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)から別の数値を減算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 減算される[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の差

<h3 id="number-div"><code>number-div</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割るための[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割るための別の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の商

<h3 id="number-less"><code>number-less</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より小さいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以下かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より大きいかどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以上かどうかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を否定します

| 引数 |  |
| :--- | :--- |
| `val` | 否定する数値 |

#### 戻り値
否定された[数値](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の[数値](https://docs.wandb.ai/ref/weave/number)のインデックスを見つけます

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の[数値](https://docs.wandb.ai/ref/weave/number)のインデックスを見つける_list_の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最大の[数値](https://docs.wandb.ai/ref/weave/number)のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の[数値