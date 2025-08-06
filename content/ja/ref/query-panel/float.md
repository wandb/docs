---
title: 'float


  浮動小数点数型（float）は、小数点数を表します。'
---

## チェーン可能な演算（Chainable Ops）
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[数値](number.md) を他の値で割って、その余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割る [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [数値](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [数値](number.md) |
| `rhs` | 2つ目の [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[数値](number.md) を指定した指数で累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数となる [数値](number.md) |
| `rhs` | 指数となる [数値](number.md) |

#### 戻り値
基数 [数値](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [数値](number.md) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [数値](number.md) |
| `rhs` | 2つ目の [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の合計

<h3 id="number-sub"><code>number-sub</code></h3>

[数値](number.md) から別の [数値](number.md) を減算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される [数値](number.md) |
| `rhs` | 減算する [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[数値](number.md) を他の [数値](number.md) で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割る [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[数値](number.md) が他の [数値](number.md) より小さいかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[数値](number.md) が他の値以下かどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

[数値](number.md) が他の値より大きいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[数値](number.md) が他の値以上かどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[数値](number.md) を反転（符号反転）します。

| 引数 |  |
| :--- | :--- |
| `val` | 反転する数値 |

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

[数値](number.md) を _timestamp_ に変換します。値が 31536000000 未満の場合は秒に、31536000000000 未満はミリ秒に、31536000000000000 未満はマイクロ秒に、31536000000000000000 未満はナノ秒として変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | timestamp に変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[数値](number.md) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [数値](number.md) |

#### 戻り値
[数値](number.md) の絶対値


## リスト演算（List Ops）
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しくないかどうか。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[数値](number.md) を他の値で割って、その余りを返します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割る [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [数値](number.md) を掛け算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [数値](number.md) |
| `rhs` | 2つ目の [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[数値](number.md) を指定した指数で累乗します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数となる [数値](number.md) |
| `rhs` | 指数となる [数値](number.md) |

#### 戻り値
基数 [数値](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [数値](number.md) を加算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [数値](number.md) |
| `rhs` | 2つ目の [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の合計

<h3 id="number-sub"><code>number-sub</code></h3>

[数値](number.md) から別の [数値](number.md) を減算します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される [数値](number.md) |
| `rhs` | 減算する [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[数値](number.md) を他の [数値](number.md) で割ります。

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [数値](number.md) |
| `rhs` | 割る [数値](number.md) |

#### 戻り値
2つの [数値](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[数値](number.md) が他の [数値](number.md) より小さいかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[数値](number.md) が他の値以下かどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する2番目の値。 |

#### 戻り値
2つの値が等しいかどうか。

<h3 id="number-greater"><code>number-greater</code></h3>

[数値](number.md) が他の値より大きいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[数値](number.md) が他の値以上かどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [数値](number.md) |
| `rhs` | 比較対象の [数値](number.md) |

#### 戻り値
最初の [数値](number.md) が2つ目以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[数値](number.md) を反転（符号反転）します。

| 引数 |  |
| :--- | :--- |
| `val` | 反転する数値 |

#### 戻り値
[数値](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の [数値](number.md) のインデックスを取得します。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [数値](number.md) のインデックスを検索する _リスト_ |

#### 戻り値
最大の [数値](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の [数値](number.md) のインデックスを取得します。

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [数値](number.md) のインデックスを検索する _リスト_ |

#### 戻り値
最小の [数値](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[数値](number.md) の平均値

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均をとる [数値](number.md) の _リスト_ |

#### 戻り値
[数値](number.md) の平均値

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値を求める [数値](number.md) の _リスト_ |

#### 戻り値
最大の [数値](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小値

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値を求める [数値](number.md) の _リスト_ |

#### 戻り値
最小の [数値](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[数値](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する [数値](number.md) の _リスト_ |

#### 戻り値
[数値](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[数値](number.md) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計をとる [数値](number.md) の _リスト_ |

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

[数値](number.md) を _timestamp_ に変換します。値が 31536000000 未満の場合は秒に、31536000000000 未満はミリ秒に、31536000000000000 未満はマイクロ秒に、31536000000000000000 未満はナノ秒として変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | timestamp に変換する数値 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[数値](number.md) の絶対値を計算します。

| 引数 |  |
| :--- | :--- |
| `n` | [数値](number.md) |

#### 戻り値
[数値](number.md) の絶対値