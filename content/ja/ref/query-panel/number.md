---
title: number
menu:
  reference:
    identifier: ja-ref-query-panel-number
---

## Chainable Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### Return Value
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別の [number](number.md) で割り、余りを返します。

| Argument |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### Return Value
2つの [numbers](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [numbers](number.md) を掛けます。

| Argument |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### Return Value
2つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある [number](number.md) を指数で累乗します。

| Argument |  |
| :--- | :--- |
| `lhs` | 基数 [number](number.md) |
| `rhs` | 指数 [number](number.md) |

#### Return Value
基数 [numbers](number.md) の n 乗

<h3 id="number-add"><code>number-add</code></h3>

2つの [numbers](number.md) を加算します。

| Argument |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### Return Value
2つの [numbers](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) から別の [number](number.md) を減算します。

| Argument |  |
| :--- | :--- |
| `lhs` | 減算される [number](number.md) |
| `rhs` | 減算する [number](number.md) |

#### Return Value
2つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

ある [number](number.md) を別の [number](number.md) で割ります。

| Argument |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### Return Value
2つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [number](number.md) が別の [number](number.md) より小さいかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) より小さいかどうか。

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [number](number.md) が別の [number](number.md) 以下かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) 以下かどうか。

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### Return Value
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [number](number.md) が別の [number](number.md) より大きいかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) より大きいかどうか。

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [number](number.md) が別の [number](number.md) 以上かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) 以上かどうか。

<h3 id="number-negate"><code>number-negate</code></h3>

[number](number.md) の符号を反転させます。

| Argument |  |
| :--- | :--- |
| `val` | 符号を反転させる数値 |

#### Return Value
[number](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[number](number.md) を文字列に変換します。

| Argument |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### Return Value
[number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](number.md) を _timestamp_ に変換します。31536000000 未満の値は秒に、31536000000000 未満の値はミリ秒に、31536000000000000 未満の値はマイクロ秒に、31536000000000000000 未満の値はナノ秒に変換されます。

| Argument |  |
| :--- | :--- |
| `val` | timestamp に変換する数値 |

#### Return Value
Timestamp

<h3 id="number-abs"><code>number-abs</code></h3>

[number](number.md) の絶対値を計算します。

| Argument |  |
| :--- | :--- |
| `n` | [number](number.md) |

#### Return Value
[number](number.md) の絶対値

## List Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### Return Value
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別の [number](number.md) で割り、余りを返します。

| Argument |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### Return Value
2つの [numbers](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [numbers](number.md) を掛けます。

| Argument |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### Return Value
2つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

ある [number](number.md) を指数で累乗します。

| Argument |  |
| :--- | :--- |
| `lhs` | 基数 [number](number.md) |
| `rhs` | 指数 [number](number.md) |

#### Return Value
基数 [numbers](number.md) の n 乗

<h3 id="number-add"><code>number-add</code></h3>

2つの [numbers](number.md) を加算します。

| Argument |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2番目の [number](number.md) |

#### Return Value
2つの [numbers](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) から別の [number](number.md) を減算します。

| Argument |  |
| :--- | :--- |
| `lhs` | 減算される [number](number.md) |
| `rhs` | 減算する [number](number.md) |

#### Return Value
2つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

ある [number](number.md) を別の [number](number.md) で割ります。

| Argument |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### Return Value
2つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [number](number.md) が別の [number](number.md) より小さいかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) より小さいかどうか。

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [number](number.md) が別の [number](number.md) 以下かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) 以下かどうか。

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### Return Value
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [number](number.md) が別の [number](number.md) より大きいかどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) より大きいかどうか。

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [number](number.md) が別の [number](number.md) 以上かどうかを確認します。

| Argument |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### Return Value
最初の [number](number.md) が2番目の [number](number.md) 以上かどうか。

<h3 id="number-negate"><code>number-negate</code></h3>

[number](number.md) の符号を反転させます。

| Argument |  |
| :--- | :--- |
| `val` | Number to negate |

#### Return Value
A [number](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大 [number](number.md) のインデックスを見つけます。

| Argument |  |
| :--- | :--- |
| `numbers` | 最大 [number](number.md) のインデックスを見つけるための [numbers](number.md) の _list_ |

#### Return Value
最大 [number](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小 [number](number.md) のインデックスを見つけます。

| Argument |  |
| :--- | :--- |
| `numbers` | 最小 [number](number.md) のインデックスを見つけるための [numbers](number.md) の _list_ |

#### Return Value
最小 [number](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[numbers](number.md) の平均

| Argument |  |
| :--- | :--- |
| `numbers` | 平均を求める [numbers](number.md) の _list_ |

#### Return Value
[numbers](number.md) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大値

| Argument |  |
| :--- | :--- |
| `numbers` | 最大 [number](number.md) を見つけるための [numbers](number.md) の _list_ |

#### Return Value
最大 [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小値

| Argument |  |
| :--- | :--- |
| `numbers` | 最小 [number](number.md) を見つけるための [numbers](number.md) の _list_ |

#### Return Value
最小 [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[numbers](number.md) の標準偏差

| Argument |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算するための [numbers](number.md) の _list_ |

#### Return Value
[numbers](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[numbers](number.md) の合計

| Argument |  |
| :--- | :--- |
| `numbers` | 合計する [numbers](number.md) の _list_ |

#### Return Value
[numbers](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[number](number.md) を文字列に変換します。

| Argument |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### Return Value
[number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](number.md) を _timestamp_ に変換します。31536000000 未満の値は秒に、31536000000000 未満の値はミリ秒に、31536000000000000 未満の値はマイクロ秒に、31536000000000000000 未満の値はナノ秒に変換されます。

| Argument |  |
| :--- | :--- |
| `val` | timestamp に変換する数値 |

#### Return Value
Timestamp

<h3 id="number-abs"><code>number-abs</code></h3>

[number](number.md) の絶対値を計算します。

| Argument |  |
| :--- | :--- |
| `n` | A [number](number.md) |

#### Return Value
The absolute value of the [number](number.md)
