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

Absolutely. One of the biggest skills in coding interviews is **recognizing when a nested loop can be eliminated**. Here are the most common tricks that reduce time complexity.

---

# 1. Use a HashMap/Dictionary (Most Important)

### Brute Force — O(n²)

Find if two numbers sum to a target.

```python
nums = [2, 7, 11, 15]
target = 9

for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            print(i, j)
```

Time complexity:

```
O(n²)
```

---

### Optimized — O(n)

```python
seen = {}

for i, num in enumerate(nums):
    if target - num in seen:
        print(seen[target - num], i)

    seen[num] = i
```

This is the classic "trade memory for speed."

---

# 2. Use a Set for Membership

Instead of

```python
for x in list1:
    for y in list2:
        if x == y:
            print(x)
```

O(n²)

Use

```python
s = set(list2)

for x in list1:
    if x in s:
        print(x)
```

Now

```
O(n)
```

because lookup in a set is approximately O(1).

---

# 3. Sort + Two Pointers

Need to find a pair?

Instead of

```python
for i:
    for j:
```

Sort first.

```
left = 0
right = n-1
```

Move pointers depending on the sum.

```
O(n log n)
```

instead of

```
O(n²)
```

This is extremely common in interviews.

---

# 4. Sliding Window

Suppose you need the maximum sum of 5 consecutive elements.

Don't do

```python
for i:
    sum = 0
    for j:
```

Instead

```
current_sum += new_element
current_sum -= old_element
```

Example

```
1 2 3 4 5
```

Window

```
1+2+3

↓

2+3+4

↓

3+4+5
```

Only one addition and one subtraction each step.

Complexity

```
O(n)
```

instead of

```
O(nk)
```

---

# 5. Prefix Sum

Need lots of range sums?

Instead of repeatedly computing

```
2+3+4+5
```

Build

```
prefix

[0,1,3,6,10,15]
```

Then

```
sum(i,j)

=

prefix[j+1]-prefix[i]
```

Every query becomes

```
O(1)
```

---

# 6. Break Early

Sometimes you don't need both loops to finish.

Instead of

```python
for i:
    for j:
```

Use

```python
for i:
    for j:
        if found:
            break
```

or

```python
return
```

This doesn't improve the worst-case complexity, but can greatly improve average performance.

---

# 7. Binary Search

Instead of

```python
for x in list1:
    for y in sorted_list:
```

Use

```python
import bisect

for x in list1:
    pos = bisect.bisect_left(sorted_list, x)
```

Complexity

```
O(n log n)
```

instead of

```
O(n²)
```

---

# 8. Frequency Counting

Need duplicates?

Don't compare every pair.

Instead

```python
freq = {}

for x in nums:
    freq[x] = freq.get(x, 0) + 1
```

Now checking duplicates is

```
O(n)
```

instead of

```
O(n²)
```

---

# 9. Monotonic Stack

Questions involving

- Next Greater Element
- Previous Smaller Element
- Stock Span
- Largest Rectangle

Many beginners write

```python
for i:
    for j:
```

Instead

```
Use a stack.
```

Complexity becomes

```
O(n)
```

---

# 10. Dynamic Programming (Reuse Work)

Instead of recalculating the same subproblems inside nested loops or recursive calls:

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

Use memoization or tabulation to avoid repeated computation.

---

# 11. Recognize Cartesian Product vs. Independent Searches

Sometimes nested loops are unavoidable.

```python
for student in students:
    for course in courses:
```

Every student must be paired with every course.

This is inherently

```
O(n × m)
```

You cannot reduce it because there are `n × m` outputs.

The key question is:

> **Are you comparing every pair, or are you just searching?**

If you're searching, a better data structure (like a set or dictionary) often removes the inner loop.

---

# Interview Cheat Sheet

| Problem Pattern | Brute Force | Better Approach |
|----------------|------------|-----------------|
| Find element | Nested loops | Set |
| Count frequency | Nested loops | Dictionary |
| Two Sum | Nested loops | HashMap |
| Pair in sorted array | Nested loops | Two pointers |
| Consecutive subarray | Nested loops | Sliding window |
| Range sums | Nested loops | Prefix sum |
| Next greater element | Nested loops | Monotonic stack |
| Repeated computation | Recursive/nested work | Dynamic programming |
| Search in sorted list | Linear scan | Binary search |

## A simple decision process

Whenever you write:

```python
for i in ...:
    for j in ...:
```

Pause and ask:

1. **Am I just looking something up?** → Use a **set** or **dictionary**.
2. **Is the data sorted (or can I sort it)?** → Consider **two pointers** or **binary search**.
3. **Am I processing a continuous range?** → Think **sliding window** or **prefix sums**.
4. **Am I recomputing the same thing?** → Consider **dynamic programming**.
5. **Do I truly need every pair?** → If yes, the nested loop may already be optimal.

For Python and coding interviews, mastering these patterns will let you replace many O(n²) solutions with O(n) or O(n log n) alternatives.
