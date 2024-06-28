
# int

## チェーン可能な操作
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値の不等性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割り、余りを返す

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割られる[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を掛け合わせます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値の累乗にします

| 引数 |  |
| :--- | :--- |
| `lhs` | 基本[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 指数[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
基数の[数値](https://docs.wandb.ai/ref/weave/number)を累乗した結果

<h3 id="number-add"><code>number-add</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を加算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)から別の数値を引く

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 引く[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の差

<h3 id="number-div"><code>number-div</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割る

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割る[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の商

<h3 id="number-less"><code>number-less</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より小さいかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以下かどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値の等しさを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より大きいかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以上かどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を否定する

| 引数 |  |
| :--- | :--- |
| `val` | 否定する数値 |

#### 戻り値
否定された[数値](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toString"><code>number-toString</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を文字列に変換する

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### 戻り値
数値の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を_タイムスタンプ_に変換する。31536000000より小さい値は秒に、31536000000000より小さい値はミリ秒に、31536000000000000より小さい値はマイクロ秒に、31536000000000000000より小さい値はナノ秒に変換される。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)の絶対値を計算します

| 引数 |  |
| :--- | :--- |
| `n` | [数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[数値](https://docs.wandb.ai/ref/weave/number)の絶対値

## リスト操作
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値の不等性を判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割り、余りを返す

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割られる[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を掛け合わせます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値の累乗にします

| 引数 |  |
| :--- | :--- |
| `lhs` | 基本[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 指数[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
基数の[数値](https://docs.wandb.ai/ref/weave/number)を累乗した結果

<h3 id="number-add"><code>number-add</code></h3>

2つの[数値](https://docs.wandb.ai/ref/weave/number)を加算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 2番目の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)から別の数値を引く

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 引く[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の差

<h3 id="number-div"><code>number-div</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を別の数値で割る

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 割る[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
2つの[数値](https://docs.wandb.ai/ref/weave/number)の商

<h3 id="number-less"><code>number-less</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より小さいかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以下かどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値の等しさを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値より大きいかどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)が別の数値以上かどうかを確認する

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | 比較対象の[数値](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
最初の[数値](https://docs.wandb.ai/ref/weave/number)が2番目の数値以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

ある[数値](https://docs.wandb.ai/ref/weave/number)を否定する

| 引数 |  |
| :--- | :--- |
| `val` | 否定する数値 |

#### 戻り値
否定された[数値](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の[数値](https://docs.wandb.ai/ref/weave/number)のインデックスを見つける

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の[数値](https://docs.wandb.ai/ref/weave/number)のインデックスを見つけるための_リスト_|

#### 戻り値
最大の[数値](https://docs.wandb.ai/ref/weave/number)のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の[数値](https://docs.wandb.ai/ref/weave/number)のインデックスを見つける

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の[数値](https://docs.wandb.ai/ref/weave/number)のインデックスを見つ

#### 戻り値
最小の [number](https://docs.wandb.ai/ref/weave/number) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[numbers](https://docs.wandb.ai/ref/weave/number) の平均

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を取る [numbers](https://docs.wandb.ai/ref/weave/number) の _リスト_ |

#### 戻り値
[numbers](https://docs.wandb.ai/ref/weave/number) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大 [number](https://docs.wandb.ai/ref/weave/number)

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](https://docs.wandb.ai/ref/weave/number) を探すための [numbers](https://docs.wandb.ai/ref/weave/number) の _リスト_ |

#### 戻り値
最大の [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小 [number](https://docs.wandb.ai/ref/weave/number)

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](https://docs.wandb.ai/ref/weave/number) を探すための [numbers](https://docs.wandb.ai/ref/weave/number) の _リスト_ |

#### 戻り値
最小の [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[numbers](https://docs.wandb.ai/ref/weave/number) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算するための [numbers](https://docs.wandb.ai/ref/weave/number) の _リスト_ |

#### 戻り値
[numbers](https://docs.wandb.ai/ref/weave/number) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[numbers](https://docs.wandb.ai/ref/weave/number) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計する [numbers](https://docs.wandb.ai/ref/weave/number) の _リスト_ |

#### 戻り値
[numbers](https://docs.wandb.ai/ref/weave/number) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[number](https://docs.wandb.ai/ref/weave/number) を文字列に変換

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数字 |

#### 戻り値
[number](https://docs.wandb.ai/ref/weave/number) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](https://docs.wandb.ai/ref/weave/number) を _タイムスタンプ_ に変換。値が 31536000000 未満の場合は秒に、31536000000000 未満の場合はミリ秒に、31536000000000000 未満の場合はマイクロ秒に、31536000000000000000 未満の場合はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数字 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[number](https://docs.wandb.ai/ref/weave/number) の絶対値を計算

| 引数 |  |
| :--- | :--- |
| `n` | [number](https://docs.wandb.ai/ref/weave/number) |

#### 戻り値
[number](https://docs.wandb.ai/ref/weave/number) の絶対値