set this straight before interview:
1. work culture prefer remote
2. location

delloite python developer:
1. MRO (Method Resolution Order) 
MRO (Method Resolution Order) solves the ambiguity and unpredictability caused by multiple inheritance. Without it, the computer would not know which parent version of a method to run.

2. Sql alchemy (usage pattern)
3. db relationships (one to many etc)
4. n+1 problem db query
5. sql query
6. problem solving


Interview attending:
cactus communications:
questions unanswered:
1. mutex and semaphores in python: 
- A mutex ensures that only one thread can access a shared resource at a time. 
- A semaphore is a counter that controls access to a limited number of shared resources.
2. whats your indexing type you use for your vector?: 
 db: Milvus: Graph-based index that connects nearby vectors; provides very fast and accurate ANN search. 
3. difference between multi-threading and async:
NA
4. AWS policies for 2 aws instances talking to each other?
5. how did you deploy the llm?
6. uvicorn vs gunicorn

athena health
LTI mindtree
deloitte

offer possible for:


need to prepare:
co-routines and prod.
UCER filter consurme pattern
tree and operations
linked list

questions in interview:
bios Basic Input/Output System:
instructions in firmware which control input and output operations.

Bios firmware used to perform hardware initialization during the booting process (power-on startup), and to provide runtime services for operating systems and programs.

UEFI: (Unified Extensible Firmware Interface)
UEFI defines a new method by which OSes and platform firmware communicate,

Aptio® V represents a hardened production-ready UEFI BIOS Firmware, featuring support for the latest UEFI specifications and the security, fast boot and touch support that today's platforms require.

1. flask (with orm, rest api, and database connection)
2. python (oops concepts, advanced concepts)
3. angular basics
4. sql queries
5. linux  (shell scripts)


factentry:

1. private variables 
2. access specifiers in python : a single underscore ‘_’ symbol before the data member of that class, 

lambda:
add = lambda x, y: x + y

lambda functions advantages:
1. small functions that can be used for small operations/logic to save space.
2.  can be used inside map,filter functions ->list(map(lambda x: x**2, numbers))

4. lambda function disadvantages:
i.Lambda functions can have only one expression.
ii.Lambda functions cannot have a docstring.
iii.Many times lambda functions make code difficult to read. 

6. types of inheritance used in python    
you can create parent and child classes and use super to inherit variables or functions from parent classes.

8. sql query questions - used ORM not queries
9. rest api what are  
put -> 
patch->
post operations-> 



0. how python interpreter works?
Python doesn’t convert its code into machine code, something that hardware can understand. It actually converts it into byte code, and uses Pvm(python virtual machine) to execute the code.

2. binary tree application, country
3.  without loop use the map and filter to iterate and compute and condition


Django mixins:
type of multiple inheritance - combine behaviour and attributes of more than one parentclass


--------------------------------------------------BASICS------------------------------------------------------------

 pep 8:
PEP stands for Python Enhancement Proposal. It is a set of rules that specify how to format Python code for maximum readability.

help():
shows documentation of modules and class, callable function, keyworks

dir():
shows valid list attributes and methods

.py:
contians source code

.pyc:
compiled bytecode


nolocal:
nolocal is often used when you need to access a variable in a nested function



Garbage collection:
Garbage collection is a process in which the interpreter frees up the memory when not in use to make it available for other objects.

How is memory managed in python?
Python uses reference counting and garbage collection to manage memory.

MEmory management:
python uses private heap for storing all objects in python,
Memory allocation means allocating a block of space in the computer memory to a program. memory allocation and deallocation method is automatic  garbage collector for Python so that the user does not have to do manual garbage collection.

Explain id() and object interning (is) ?
id(obj) returns the memory address (or unique ID) of an object.
 Python “interns” small integers and short strings for reuse.
interning:
a = 256
b = 256
print(a is b)  # True

x = 1000
y = 1000
print(x is y)  # False — not interned

whats the diff between 'is' and '=='?
== → Equality 
is → Identity  (check if id's of two objs are same)

finalize method:
used to free up the unmanaged resources and clean up before the garbage collection.

GIL (global interpreter lock):
its a locking mechanism allows only one thread to execute Python code at a time. It simplifies memory management and avoids race conditions but limits true parallel execution in CPU-bound tasks. 

Lambda:
A lambda function is a small anonymous function.
A lambda function can take any number of arguments, but can only have one expression.

How do context managers work (with statement)? 
They implement __enter__ and __exit__. 
class File:
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): 
Used to handle setup/cleanup (like file I/O, locks, DB connections). 


syntax: lambda arguments : expression

example:
lambda x: x % 2 != 0

Filter:
The filter() method filters the given sequence with the help of a function that tests each element in the sequence to be true or not.

syntax: filter(function, iteratable)

example:


Map:
map() function returns a map object(which is an iterator) of the results after applying the given function to each item of a given iterable (list, tuple etc.)
syntax: map(fun, iter)

example:
def myfunc(n):
  return len(n)
x = map(myfunc, ('apple', 'banana', 'cherry'))

Difference between Map and Filter:
Map takes all objects in a list and allows you to apply a function to it whereas Filter takes all objects in a list and runs that through a function to create a new list with all objects that return True in that function.

filter only takes a function that return True or False
map can  take a function that return any objects

difference between a class and an object?
Objects take memory space when they are created, A class does not take memory space when created

different types of inheritance:
Single inheritance
Multiple inheritance
Multilevel inheritance
Hierarchical inheritance
Hybrid inheritance

whats dunder/magic methods in python? 
methods with double underscore are dunder methods, ex: __init__(), __main__()

what is decorator?
refer below

what is unit test? why?
closure in python -> wrapper function
a closure is afunction object that remembers values in encloseing scpoes even if they are not present in memory



what standard/flow of coding are you following?
we following we write the required code first and then unit test in django test.py files. performance test and stress test is done before production.



------------------------------------------------------TESTING-----------------------------------------------------

types of testing?
Unit Testing: It focuses on the smallest unit of software design.It is often done by the programmer by using sample input and observing its corresponding outputs. 

Integration Testing:
The objective is to take unit tested components and build a program structure that has been dictated by design.

White Box testing and Black Box testing?
 Black Box testing: It is used for validation. 
In this we ignore internal working mechanism and 
focuse on what is the output?

White Box testing:- It is used for verification. 
In this we focus on internal mechanism i.e.
how the output is achieved?

Regression Testing:
Every time a new module is added leads to changes in the program. This type of testing makes sure that the whole component works properly even after adding components to the complete program. 


Stress Testing:
In this, we give unfavorable conditions to the system and check how they perform in those conditions.

 Performance Testing
It is designed to test the run-time performance of software within the context of an integrated system.


-------------------------------------------------------processes and threads ------------------------------

diff between multiprocess and multithreading?

lib used:
multithreading:
import threading
multiprocesssing:
import multiprocessing

Multithreading is best for I/O-bound tasks (API calls, database queries, file operations) because threads can work while others wait for responses. However, in CPython, the GIL prevents true parallel execution of Python code.

Multiprocessing is best for CPU-bound tasks because each process has its own Python interpreter and GIL, allowing true parallelism across multiple CPU cores.


In Multiprocessing, Process creation is a time-consuming process.	While in Multithreading, process creation is according to economic.

In Multiprocessing, every process owned a separate address space.	While in Multithreading, a common address space is shared by all the threads.

Threads uniquely run in the same unique memory heap. Whereas Processes run in separate memory heaps. 

example:
multi process:
p1 = multiprocessing.Process(target=print_square, args=(10, ))
p2 = multiprocessing.Process(target=print_cube, args=(10, ))
# starting process 1
p1.start()
# starting process 2
p2.start()
# wait until process 1 is finished
p1.join()
# wait until process 2 is finished
p2.join()

threading:

t1 = threading.Thread(target=print_square, args=(10,))
t2 = threading.Thread(target=print_cube, args=(10,))
 # starting thread 1
t1.start()
# starting thread 2
t2.start()
 
# wait until thread 1 is completely executed
t1.join()
# wait until thread 2 is completely executed
t2.join()



-------------------------------------------------------OOPS---------------------------------------------------------

What’s the difference between @property and a normal method? 
Turns a method into a managed attribute. It lets you access method logic as if it were a regular attribute, without parentheses. 
class Circle:
    def __init__(self, r): self._r = r
    @property
    def area(self): return 3.14 * self._r**2
p = Circle(20)
print(p.area) 

Explain try/except/else/finally flow. 
try:
    ...
except Exception:
    ...
else:
    ...  # runs if no exception
finally:
    ...  # runs always

How does Python handle function default arguments (gotcha)? 
Default mutable arguments are evaluated once at function definition, not each call. 
def f(x=[]):
    x.append(1)
    return x

print(f())  # [1]
print(f())  # [1, 1]  <-- bug



issubclass():
used to check if class is subclass

self keyword:
self represents the instance of the class. By using the “self” keyword we can access the attributes and methods of the class in python.

Class method:
classmethod() methods are bound to a class rather than an object.

used:
1. can call constructor from the inside class function.
2. any value changed affects all the instances created.
3. can be called by class as well as instances

Syntax: 
@classmethod
   def fun(cls, arg1, arg2, ...):

Ex:
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # uses cls instead of self
    @classmethod
    def fromBirthYear(cls, name, year):
        return cls(name, year)           

What is the difference between __init__ and __new__?
__new__ creates an instance (used in singletons, immutables)
__init__ initializes the created instance
ex: 
class A:
    def __new__(cls):
        print("Creating instance")
        return super().__new__(cls)
    def __init__(self):
        print("Initializing instance")




STATIC METHOD:
@staticmethod 
a staticmethod does not have self or class as parameters. these methods are normal methods.

when to use:
1. when you are not using any varables from the class
2. when you have an independant function 

ex:

class cars(object):
   def start(self,a)
          print(a)

   @staticmethod
   def end():
        print('this is a static method')
        return true

inheritance:
Inheritance allows us to define a class that inherits all the methods and properties from another class.
ex:
class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  pass

#parent class call
x = Person("John", "Doe")
x.printname()
# child class call
x = Student("John", "Doe")
x.printname()

Super method:
By using the super() function, you do not have to use the name of the parent element, it will automatically inherit the methods and properties from its parent.

Polymorphism
Polymorphism simply means having many forms. For example, we need to determine if the given species of birds fly or not, using polymorphism we can do this using a single function.

ex:
class Bird:
    def intro(self):
        print("There are many types of birds.")
    def flight(self):
        print("Most of the birds can fly but some cannot.")
 
class sparrow(Bird):
    def flight(self):
        print("Sparrows can fly.")

obj_bird = Bird()
obj_spr = sparrow()
obj_bird.intro()
obj_bird.flight()
obj_spr.intro()
obj_spr.flight()

Encapsultion:
It describes the idea of wrapping data and the methods that work on data within one unit. example: class

This puts restrictions on accessing variables and methods directly and can prevent the accidental modification of data.


Abstraction:

Abstraction is used to hide the internal functionality of the function from the users.

Abstract class:
i. an abstract class is used to make regular clases, an abstract class is like a template for other classes.

i. an abstract class cannot be used by instantiating, you can use by inheriting it to another class and then use it.

ex:
from abc import ABC,   
class Dog(ABC):   
    def bark(self):   
        print('bark')
    

Abstract Method:
an Abstract method is a method that is not implemented. the abstract methods are to be implemented in child classes only,

ex:
from abc import ABC, abstractmethod   
class Dog(ABC):   
    def bark(self):   
        print('bark')
 @abstractmethod
    def poop(self):  # an abstract method should have pass, and has nothing 
        pass


Access specifiers:

Python doesn't have any mechanism that effectively restricts access to any instance variable or method. Python prescribes a convention of prefixing the name of the variable/method with a single or double underscore to emulate the behavior of protected and private access specifiers.

Private Member:
The double underscore __ prefixed to a variable makes it private.
It gives a strong suggestion not to touch it from outside the class. Any attempt to do so will result in an AttributeError:

ex:
class Student:
    __schoolName = 'XYZ School' # private class attribute

    def __init__(self, name, age):
        self.__name=name  # private instance attribute
        self.__salary=age # private instance attribute

>>> std = Student("Bill", 25)
>>> std.__schoolName
AttributeError: 'Student' object has no attribute '__schoolName'

Every member with a double underscore will be changed to _object._class__variable

-----------DATE TIME------------------------

What specific limitation arises when performing 'timedelta' calculations on localized 'datetime' objects in Python using the 'pytz' library?
the utc offset and dst status are not automatically updated, this is because time delt math operates on the absolute time without recalculating timezone specific rules

---------------------------------------------------ADVANCED PYTHON--------------------------------------
GENERATOR FUNCTIONS:
A generator is a special type of function which does not return a single value, instead, it returns an iterator object with a sequence of values. a 'yield' statement is used rather than a return statement

EX:
def mygenerator():
    print('First item')
    yield 10

    print('Second item')
    yield 20

with loop loop:
def get_sequence_upto(x):
    for i in range(x):
        yield i

Execution:

>>> seq = get_sequence_upto(5) 
>>> next(seq) 
0  
>>> next(seq)
1

Generator Expression:
(x*x for x in range(5))



Higher order functions:
A higher order function is a function that takes a function as an argument, or returns a function .

ex: map function

FIRST CLASS FUNCTIONS:
python treats functions as First class objects, thats why its called FIrst class functions

1. you can pass values from outer function and use it inside inner function.
2. outerfunction should always return inner function

ex:

def logger(msg):
    def printa():
        print('hello: '+msg)
    return printa

log = logger('world')
log()

What are annotations in Python functions? 
Metadata stored in __annotations__, often used for type hints.

How does *args and **kwargs work internally?
*args collects positional arguments into a tuple.
**kwargs collects keyword arguments into a dict.
Together, they make flexible APIs. 

whats function overloading?
You can define multiple functions with the same name but different parameter lists, you cannot do function overloading in python but you can simulate it using, args and kwargs

What is memoization and how is it implemented?
Memoization in Python (and in programming generally) is an optimization technique used to speed up functions by storing the results of expensive function calls.
Done using Caching function results to avoid recomputation. 
ex:
from functools import lru_cache
@lru_cache(maxsize=128)
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)

What’s the use of __repr__ vs __str__ ?
__str__: 
User-readable
print(obj)

__repr__ :
Developer-readable
repr(obj) or interactive

Explain __call__ ?
Makes an object callable like a function.
class Adder:
    def __init__(self, n): self.n = n
    def __call__(self, x): return self.n + x

add5 = Adder(5)
print(add5(10))  # 15

What is __del__ ?
Object destructor called when the object is about to be garbage-collected. 

How to reduce memory footprint in Python?
Use __slots__ , Prefer tuple over list for immutables, Use lru_cache to memoize, Use generators for large data streams 

What tools can you use for memory profiling ?
tracemalloc
memory_profiler
 

CLOSURE:
A Closure is a function object that remembers values in enclosing scopes even if they are not present in memory.
ex: multiply remembers factor=2 even after make_multiplier ends. 
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
print(double(5))  # 10

How to modify a nonlocal variable in an inner function? 
Use the nonlocal keyword to modify variables from an enclosing (but non-global) scope.
ex:
def counter():
    n = 0
    def inc():
        nonlocal n
        n += 1
        return n
    return inc

What is a descriptor? 
any class that defines any of __get__, __set__, or __delete__ methods.
They control how attributes are accessed in other classes.

Difference between @property and a descriptor?
 @property is a built-in descriptor that automatically creates getter/setter methods, while a descriptor is customizable for more control. 

What is __slots__ used for? 
It restricts a class to a fixed set of attributes and avoids creating a __dict__, saving memory.
class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

- Slots are not inherited unless explicitly redefined 
-  Saves memory for millions of small objects.
-  You can’t dynamically add new attributes 
