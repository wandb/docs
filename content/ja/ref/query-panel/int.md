---
title: int
menu:
  reference:
    identifier: ja-ref-query-panel-int
---

## チェーン可能な Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [数値](number.md) を別の [数値](number.md) で割り、その余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割るための [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [数値](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [数値](number.md) |
| `rhs` | 2 つ目の [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[数値](number.md) を指数で累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 底の [数値](number.md) |
| `rhs` | 指数の [数値](number.md) |

#### 戻り値
底の [数値](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [数値](number.md) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [数値](number.md) |
| `rhs` | 2 つ目の [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [数値](number.md) から別の [数値](number.md) を減算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される側の [数値](number.md) |
| `rhs` | 減算する [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

ある [数値](number.md) を別の [数値](number.md) で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割るための [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [数値](number.md) が別の [数値](number.md) より小さいかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目より小さいかどうか。

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [数値](number.md) が別の [数値](number.md) 以下かを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目以下かどうか。

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [数値](number.md) が別の [数値](number.md) より大きいかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目より大きいかどうか。

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [数値](number.md) が別の [数値](number.md) 以上かを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目以上かどうか。

<h3 id="number-negate"><code>number-negate</code></h3>

[数値](number.md) の符号を反転します。

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する数値 |

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

[数値](number.md) を _タイムスタンプ_ に変換します。値が 31536000000 未満なら秒、31536000000000 未満ならミリ秒、31536000000000000 未満ならマイクロ秒、31536000000000000000 未満ならナノ秒として解釈されます。

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


## リスト Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [数値](number.md) を別の [数値](number.md) で割り、その余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割るための [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [数値](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [数値](number.md) |
| `rhs` | 2 つ目の [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[数値](number.md) を指数で累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 底の [数値](number.md) |
| `rhs` | 指数の [数値](number.md) |

#### 戻り値
底の [数値](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [数値](number.md) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [数値](number.md) |
| `rhs` | 2 つ目の [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [数値](number.md) から別の [数値](number.md) を減算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される側の [数値](number.md) |
| `rhs` | 減算する [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

ある [数値](number.md) を別の [数値](number.md) で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割るための [数値](number.md) |

#### 戻り値
2 つの [数値](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [数値](number.md) が別の [数値](number.md) より小さいかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目より小さいかどうか。

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [数値](number.md) が別の [数値](number.md) 以下かを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目以下かどうか。

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [数値](number.md) が別の [数値](number.md) より大きいかを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目より大きいかどうか。

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [数値](number.md) が別の [数値](number.md) 以上かを確認します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が 2 番目以上かどうか。

<h3 id="number-negate"><code>number-negate</code></h3>

[数値](number.md) の符号を反転します。

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する数値 |

#### 戻り値
[数値](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の [数値](number.md) のインデックスを求めます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [数値](number.md) のインデックスを求める対象となる [数値](number.md) の _リスト_ |

#### 戻り値
最大の [数値](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の [数値](number.md) のインデックスを求めます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [数値](number.md) のインデックスを求める対象となる [数値](number.md) の _リスト_ |

#### 戻り値
最小の [数値](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[数値](number.md) の平均

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均する対象の [数値](number.md) の _リスト_ |

#### 戻り値
[数値](number.md) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大の数値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [数値](number.md) を求める対象となる [数値](number.md) の _リスト_ |

#### 戻り値
最大の [数値](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小の数値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [数値](number.md) を求める対象となる [数値](number.md) の _リスト_ |

#### 戻り値
最小の [数値](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[数値](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する対象の [数値](number.md) の _リスト_ |

#### 戻り値
[数値](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[数値](number.md) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計する対象の [数値](number.md) の _リスト_ |

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

[数値](number.md) を _タイムスタンプ_ に変換します。値が 31536000000 未満なら秒、31536000000000 未満ならミリ秒、31536000000000000 未満ならマイクロ秒、31536000000000000000 未満ならナノ秒として解釈されます。

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