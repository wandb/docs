---
title: int
menu:
  reference:
    identifier: ja-ref-query-panel-int
---

## Chainable Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2 つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[数値](number.md) を別の数値で割り、余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[数値](number.md) |
| `rhs` | 割る[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) のモジュロ

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [数値](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](number.md) |
| `rhs` | 2 番目の[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[数値](number.md) を指定した指数で累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 底の[数値](number.md) |
| `rhs` | 指数の[数値](number.md) |

#### 戻り値
底の[数値](number.md)が n 乗された値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [数値](number.md) を加えます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](number.md) |
| `rhs` | 2 番目の[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

1 つの [数値](number.md) から別の数値を引きます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる[数値](number.md) |
| `rhs` | 引く[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[数値](number.md) を別の数値で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[数値](number.md) |
| `rhs` | 割る[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

1 つの [数値](number.md) が別の数値より小さいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

1 つの [数値](number.md) が別の数値以下であるかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値以下であるかどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2 つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

1 つの [数値](number.md) が別の数値より大きいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

1 つの [数値](number.md) が別の数値以上であるかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値以上であるかどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[数値](number.md) を負にします。

| 引数 |  |
| :--- | :--- |
| `val` | 負にする数値 |

#### 戻り値
[数値](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[数値](number.md) を文字列に変換します。

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### 戻り値
[数値](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[数値](number.md) を _タイムスタンプ_ に変換します。31536000000 未満の値は秒に、31536000000000 未満の値はミリ秒に、31536000000000000 未満の値はマイクロ秒に、31536000000000000000 未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[数値](number.md) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [数値](number.md) |

#### 戻り値
[数値](number.md) の絶対値


## List Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2 つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[数値](number.md) を別の数値で割り、余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[数値](number.md) |
| `rhs` | 割る[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) のモジュロ

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [数値](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](number.md) |
| `rhs` | 2 番目の[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[数値](number.md) を指定した指数で累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 底の[数値](number.md) |
| `rhs` | 指数の[数値](number.md) |

#### 戻り値
底の[数値](number.md)が n 乗された値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [数値](number.md) を加えます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の[数値](number.md) |
| `rhs` | 2 番目の[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

1 つの [数値](number.md) から別の数値を引きます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる[数値](number.md) |
| `rhs` | 引く[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[数値](number.md) を別の数値で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる[数値](number.md) |
| `rhs` | 割る[数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

1 つの [数値](number.md) が別の数値より小さいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

1 つの [数値](number.md) が別の数値以下であるかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値以下であるかどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2 つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

1 つの [数値](number.md) が別の数値より大きいかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

1 つの [数値](number.md) が別の数値以上であるかどうかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する[数値](number.md) |
| `rhs` | 比較対象の[数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目の数値以上であるかどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[数値](number.md) を負にします。

| 引数 |  |
| :--- | :--- |
| `val` | 負にする数値 |

#### 戻り値
[数値](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の [数値](number.md) のインデックスを見つけます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [数値](number.md) のインデックスを見つけるための[数値](number.md)の_リスト_ |

#### 戻り値
最大の [数値](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の [数値](number.md) のインデックスを見つけます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [数値](number.md) のインデックスを見つけるための[数値](number.md)の_リスト_ |

#### 戻り値
最小の [数値](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[数値](number.md) の平均

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を取るための[数値](number.md)の_リスト_ |

#### 戻り値
[数値](number.md) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大の数値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [数値](number.md) を見つけるための[数値](number.md)の_リスト_ |

#### 戻り値
最大の [数値](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小の数値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [数値](number.md) を見つけるための[数値](number.md)の_リスト_ |

#### 戻り値
最小の [数値](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[数値](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算するための[数値](number.md)の_リスト_ |

#### 戻り値
[数値](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[数値](number.md) の和

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計を求めるための[数値](number.md)の_リスト_ |

#### 戻り値
[数値](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[数値](number.md) を文字列に変換します。

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### 戻り値
[数値](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[数値](number.md) を _タイムスタンプ_ に変換します。31536000000 未満の値は秒に、31536000000000 未満の値はミリ秒に、31536000000000000 未満の値はマイクロ秒に、31536000000000000000 未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[数値](number.md) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [数値](number.md) |

#### 戻り値
[数値](number.md) の絶対値
