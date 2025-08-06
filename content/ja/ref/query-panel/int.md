---
title: 'int


  整数型（integer）を表すデータ型です。'
menu:
  reference:
    identifier: ja-ref-query-panel-int
---

## チェーン可能な演算
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別の [number](number.md) で割り、その余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る対象の [number](number.md) |
| `rhs` | 割るための [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [numbers](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[Number](number.md) を累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 底となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |

#### 戻り値
base [numbers](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [numbers](number.md) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の合計

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) から別の [number](number.md) を引きます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる [number](number.md) |
| `rhs` | 引く [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[Number](number.md) をもう一方で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る対象の [number](number.md) |
| `rhs` | 割るための [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [number](number.md) が別のものより小さいか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [number](number.md) が別のもの以下かどうか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [number](number.md) が別のものより大きいか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [number](number.md) が別のもの以上かどうか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[Number](number.md) の符号を反転します。

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転させる数値 |

#### 戻り値
[Number](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[Number](number.md) を文字列に変換します。

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### 戻り値
[Number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[Number](number.md) を _timestamp_ に変換します。31536000000 未満の場合は秒、31536000000000 未満の場合はミリ秒、31536000000000000 未満の場合はマイクロ秒、31536000000000000000 未満の場合はナノ秒として変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプへ変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[Number](number.md) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [Number](number.md) |

#### 戻り値
[Number](number.md) の絶対値


## リストの演算
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別の [number](number.md) で割り、その余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る対象の [number](number.md) |
| `rhs` | 割るための [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [numbers](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[Number](number.md) を累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 底となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |

#### 戻り値
base [numbers](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [numbers](number.md) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の合計

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) から別の [number](number.md) を引きます。

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる [number](number.md) |
| `rhs` | 引く [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[Number](number.md) をもう一方で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る対象の [number](number.md) |
| `rhs` | 割るための [number](number.md) |

#### 戻り値
2つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

ある [number](number.md) が別のものより小さいか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

ある [number](number.md) が別のもの以下かどうか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2つ目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

ある [number](number.md) が別のものより大きいか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

ある [number](number.md) が別のもの以上かどうか判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[Number](number.md) の符号を反転します。

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転させる数値 |

#### 戻り値
[Number](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大値の [number](number.md) のインデックスを調べます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値を探す [numbers](number.md) の _list_ |

#### 戻り値
最大値の [number](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小値の [number](number.md) のインデックスを調べます。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値を探す [numbers](number.md) の _list_ |

#### 戻り値
最小値の [number](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[numbers](number.md) の平均値

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を取る [numbers](number.md) の _list_ |

#### 戻り値
[numbers](number.md) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値を探す [numbers](number.md) の _list_ |

#### 戻り値
最大の [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値を探す [numbers](number.md) の _list_ |

#### 戻り値
最小の [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[numbers](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する [numbers](number.md) の _list_ |

#### 戻り値
[numbers](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[numbers](number.md) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計を計算する [numbers](number.md) の _list_ |

#### 戻り値
[numbers](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[Number](number.md) を文字列に変換します。

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |

#### 戻り値
[Number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[Number](number.md) を _timestamp_ に変換します。31536000000 未満の場合は秒、31536000000000 未満の場合はミリ秒、31536000000000000 未満の場合はマイクロ秒、31536000000000000000 未満の場合はナノ秒として変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプへ変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[Number](number.md) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [Number](number.md) |

#### 戻り値
[Number](number.md) の絶対値