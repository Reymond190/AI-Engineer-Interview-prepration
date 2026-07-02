Perfect. Here are a few progressively harder MCQs that focus on **list comprehensions**, **for loops**, **nested loops**, and **conditions**. Try solving them without running the code.

---

## Question 1 — Basic List Comprehension

What is the output?

```python
nums = [1, 2, 3, 4]

result = [x * 2 for x in nums]

print(result)
```

**A.**
```python
[2, 4, 6, 8]
```

**B.**
```python
[1, 2, 3, 4]
```

**C.**
```python
[1, 4, 9, 16]
```

**D.**
```python
(2, 4, 6, 8)
```

---

## Question 2 — Filtering

```python
nums = [1, 2, 3, 4, 5, 6]

result = [x for x in nums if x % 2 == 0]

print(result)
```

**A.**
```python
[1, 3, 5]
```

**B.**
```python
[2, 4, 6]
```

**C.**
```python
[0, 2, 4]
```

**D.**
```python
[1, 2, 3, 4, 5, 6]
```

---

## Question 3 — if-else Inside List Comprehension

```python
nums = [1, 2, 3, 4]

result = ["Even" if x % 2 == 0 else "Odd" for x in nums]

print(result)
```

**A.**
```python
['Odd', 'Even', 'Odd', 'Even']
```

**B.**
```python
['Even', 'Odd', 'Even', 'Odd']
```

**C.**
```python
['Odd', 'Odd', 'Even', 'Even']
```

**D.**
Raises an exception

---

## Question 4 — Nested Loops

```python
result = [(i, j) for i in range(2) for j in range(3)]

print(result)
```

**A.**
```python
[(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
```

**B.**
```python
[(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
```

**C.**
```python
[(0,1), (0,2), (1,2)]
```

**D.**
```python
[(0,0), (1,1)]
```

---

## Question 5 — Nested Loops with Condition

```python
result = [(i, j)
          for i in range(3)
          for j in range(3)
          if i != j]

print(result)
```

**A.**

```python
[(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
```

**B.**

```python
[(0,0), (1,1), (2,2)]
```

**C.**

```python
[(0,1), (1,2)]
```

**D.**

```python
[(0,2), (2,0)]
```

---

## Question 6 — Order of Loops (Tricky)

```python
result = [i + j
          for i in [1, 2]
          for j in [10, 20]]

print(result)
```

**A.**

```python
[11, 12, 21, 22]
```

**B.**

```python
[11, 21, 12, 22]
```

**C.**

```python
[11, 21, 22, 12]
```

**D.**

```python
[10, 20, 11, 22]
```

---

## Question 7 — Multiple if Conditions

```python
nums = range(10)

result = [x
          for x in nums
          if x % 2 == 0
          if x > 4]

print(result)
```

**A.**

```python
[6, 8]
```

**B.**

```python
[4, 6, 8]
```

**C.**

```python
[2, 4, 6, 8]
```

**D.**

```python
[5, 6, 7, 8, 9]
```

---

## Question 8 — Variable Scope (Very Common Interview Question)

```python
x = 100

result = [x * 2 for x in range(3)]

print(x)
print(result)
```

What is printed?

**A.**

```python
2
[0, 2, 4]
```

**B.**

```python
100
[0, 2, 4]
```

**C.**

```python
100
[100, 200, 300]
```

**D.**

Raises an exception

---

### Bonus (Hard)

Without executing it, determine the output:

```python
matrix = [
    [1, 2],
    [3, 4],
    [5, 6]
]

result = [x
          for row in matrix
          for x in row
          if x % 2 == 0]

print(result)
```

**A.**

```python
[2, 4, 6]
```

**B.**

```python
[1, 3, 5]
```

**C.**

```python
[[2], [4], [6]]
```

**D.**

Raises an exception

---

Nice attempt! You got most of the easier ones right. Let's go through each, especially the ones you weren't sure about.

| Question | Your Answer | Correct |
|----------|-------------|---------|
| Q1 | A ✅ | ✅ |
| Q2 | B ✅ | ✅ |
| Q3 | D ❌ | **A** |
| Q4 | Don't know | **A** |
| Q5 | Don't know | **A** |
| Q6 | A ✅ | ✅ |
| Q7 | Don't know | **A** |
| Bonus | A ✅ | ✅ |

That's **5/8 correct**. Now let's understand the tricky ones.

---

# Q3 — if-else inside a list comprehension

```python
nums = [1, 2, 3, 4]

result = ["Even" if x % 2 == 0 else "Odd" for x in nums]
```

Many people confuse this syntax.

The general form is:

```python
[value_if_true if condition else value_if_false
 for item in iterable]
```

Notice that the `if ... else` comes **before** the `for`.

Let's evaluate it.

```
x = 1
1 % 2 == 0 → False
=> "Odd"

x = 2
True
=> "Even"

x = 3
False
=> "Odd"

x = 4
True
=> "Even"
```

Result:

```python
['Odd', 'Even', 'Odd', 'Even']
```

---

## There are actually two different kinds of `if` in list comprehensions.

### 1. Filtering

```python
[x for x in nums if x % 2 == 0]
```

means

```python
for x in nums:
    if x % 2 == 0:
        ...
```

---

### 2. Conditional expression

```python
["Even" if x % 2 == 0 else "Odd" for x in nums]
```

means

```python
for x in nums:
    if x % 2 == 0:
        append("Even")
    else:
        append("Odd")
```

Very different!

---

# Q4 — Nested loops

```python
result = [(i, j)
          for i in range(2)
          for j in range(3)]
```

The biggest trick is remembering:

A list comprehension follows the same order as nested `for` loops.

Imagine writing it normally:

```python
result = []

for i in range(2):
    for j in range(3):
        result.append((i, j))
```

Now execute it mentally.

```
i = 0

    j = 0
    append (0,0)

    j = 1
    append (0,1)

    j = 2
    append (0,2)

i = 1

    j = 0
    append (1,0)

    j = 1
    append (1,1)

    j = 2
    append (1,2)
```

Result:

```python
[(0,0),
 (0,1),
 (0,2),
 (1,0),
 (1,1),
 (1,2)]
```

**Golden rule:**

> Read list comprehensions exactly like nested `for` loops.

---

# Q5 — Nested loops with a condition

```python
result = [
    (i, j)
    for i in range(3)
    for j in range(3)
    if i != j
]
```

Expand it:

```python
result = []

for i in range(3):
    for j in range(3):
        if i != j:
            result.append((i, j))
```

Let's execute it.

---

### i = 0

```
j = 0

0 != 0 ❌

skip

j = 1

0 != 1 ✅

append (0,1)

j = 2

0 != 2 ✅

append (0,2)
```

Current result

```python
[(0,1), (0,2)]
```

---

### i = 1

```
j = 0

append (1,0)

j = 1

skip

j = 2

append (1,2)
```

Result

```python
[(0,1),
 (0,2),
 (1,0),
 (1,2)]
```

---

### i = 2

```
j = 0

append (2,0)

j = 1

append (2,1)

j = 2

skip
```

Final result

```python
[(0,1),
 (0,2),
 (1,0),
 (1,2),
 (2,0),
 (2,1)]
```

---

# Q7 — Multiple `if`s

```python
nums = range(10)

result = [
    x
    for x in nums
    if x % 2 == 0
    if x > 4
]
```

This looks strange, but it's just equivalent to:

```python
for x in nums:
    if x % 2 == 0:
        if x > 4:
            append(x)
```

Evaluate each number:

```
0

even

not >4

skip

2

even

not >4

skip

4

even

not >4

skip

6

even

>4

append

8

even

>4

append
```

Answer:

```python
[6, 8]
```

---

## A trick that will help in every interview

Whenever you see a complicated list comprehension, rewrite it as ordinary loops first.

For example,

```python
[(i, j)
 for i in range(3)
 for j in range(2)
 if i != j]
```

Immediately rewrite it as:

```python
result = []

for i in range(3):
    for j in range(2):
        if i != j:
            result.append((i, j))
```

Once it's written like this, the answer is usually straightforward.

---

Since you got the basics down, the next step is to practice more challenging patterns. I can give you another set with topics like nested list comprehensions, `enumerate()`, `zip()`, dictionary comprehensions, generator expressions, variable shadowing, and scope—these are common in Python interviews and often trip people up.