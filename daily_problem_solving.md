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

These questions are very similar to what you'll encounter in Python interviews. If you're preparing seriously, I can also create a **[30-question daily quiz](chatgpt://followup-prompt?start_index=3226&end_index=3248)** that gradually increases in difficulty from basic comprehensions to advanced nested comprehensions, generators, lambda functions, closures, and iterator behavior.