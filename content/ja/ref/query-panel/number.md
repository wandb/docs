---
title: 番号
---

## 連結可能な演算 (Chainable Ops)
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2つ目の値 |

#### 戻り値
2つの値が等しくない場合は `true`、等しい場合は `false` です。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[ number ](number.md) を別の number で割り、その余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### 戻り値
2つの [number](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [number](number.md) を掛け算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[ number ](number.md) を累乗します

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |

#### 戻り値
基数 [number](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [number](number.md) を足します

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

1つの [number](number.md) から別の number を引きます

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる [number](number.md) |
| `rhs` | 引く [number](number.md) |

#### 戻り値
2つの [number](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[ number ](number.md) を別の number で割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### 戻り値
2つの [number](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[ number ](number.md) がもう一方より小さいか判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より小さい場合は `true` です

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[ number ](number.md) がもう一方以下か判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以下なら `true` です

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2つ目の値 |

#### 戻り値
2つの値が等しい場合は `true` です。

<h3 id="number-greater"><code>number-greater</code></h3>

[ number ](number.md) がもう一方より大きいか判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より大きい場合は `true` です

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[ number ](number.md) がもう一方以上か判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以上なら `true` です

<h3 id="number-negate"><code>number-negate</code></h3>

[ number ](number.md) の符号を反転します

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転したい number |

#### 戻り値
[ number ](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[ number ](number.md) を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換したい number |

#### 戻り値
[ number ](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[ number ](number.md) を _timestamp_ に変換します。31536000000 未満は秒、31536000000000 未満はミリ秒、31536000000000000 未満はマイクロ秒、31536000000000000000 未満はナノ秒へ変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | timestamp へ変換する number |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[ number ](number.md) の絶対値を求めます

| 引数 |  |
| :--- | :--- |
| `n` | [number](number.md) |

#### 戻り値
[ number ](number.md) の絶対値


## リスト演算 (List Ops)
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2つ目の値 |

#### 戻り値
2つの値が等しくない場合は `true`、等しい場合は `false` です。

<h3 id="number-modulo"><code>number-modulo</code></h3>

[ number ](number.md) を別の number で割り、その余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### 戻り値
2つの [number](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2つの [number](number.md) を掛け算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[ number ](number.md) を累乗します

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |

#### 戻り値
基数 [number](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2つの [number](number.md) を足します

| 引数 |  |
| :--- | :--- |
| `lhs` | 1つ目の [number](number.md) |
| `rhs` | 2つ目の [number](number.md) |

#### 戻り値
2つの [number](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

1つの [number](number.md) から別の number を引きます

| 引数 |  |
| :--- | :--- |
| `lhs` | 引かれる [number](number.md) |
| `rhs` | 引く [number](number.md) |

#### 戻り値
2つの [number](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[ number ](number.md) を別の number で割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### 戻り値
2つの [number](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[ number ](number.md) がもう一方より小さいか判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より小さい場合は `true` です

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[ number ](number.md) がもう一方以下か判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以下なら `true` です

<h3 id="number-equal"><code>number-equal</code></h3>

2つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値 |
| `rhs` | 比較する2つ目の値 |

#### 戻り値
2つの値が等しい場合は `true` です。

<h3 id="number-greater"><code>number-greater</code></h3>

[ number ](number.md) がもう一方より大きいか判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目より大きい場合は `true` です

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[ number ](number.md) がもう一方以上か判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較される [number](number.md) |

#### 戻り値
最初の [number](number.md) が2つ目以上なら `true` です

<h3 id="number-negate"><code>number-negate</code></h3>

[ number ](number.md) の符号を反転します

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転したい number |

#### 戻り値
[ number ](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の [number](number.md) のインデックスを取得します

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値のインデックスを取得したい [numbers](number.md) の _list_ |

#### 戻り値
最大 [number](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の [number](number.md) のインデックスを取得します

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値のインデックスを取得したい [numbers](number.md) の _list_ |

#### 戻り値
最小 [number](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[ numbers ](number.md) の平均値

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を求めたい [numbers](number.md) の _list_ |

#### 戻り値
[ numbers ](number.md) の平均値

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大の number

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値を取得したい [numbers](number.md) の _list_ |

#### 戻り値
最大の [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小の number

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値を取得したい [numbers](number.md) の _list_ |

#### 戻り値
最小の [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[ numbers ](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算したい [numbers](number.md) の _list_ |

#### 戻り値
[ numbers ](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[ numbers ](number.md) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計したい [numbers](number.md) の _list_ |

#### 戻り値
[ numbers ](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[ number ](number.md) を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換したい number |

#### 戻り値
[ number ](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[ number ](number.md) を _timestamp_ に変換します。31536000000 未満は秒、31536000000000 未満はミリ秒、31536000000000000 未満はマイクロ秒、31536000000000000000 未満はナノ秒へ変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | timestamp へ変換する number |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[ number ](number.md) の絶対値を求めます

| 引数 |  |
| :--- | :--- |
| `n` | [number](number.md) |

#### 戻り値
[ number ](number.md) の絶対値