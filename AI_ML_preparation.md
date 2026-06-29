#### How LangChain chains work, tools, memory, agents.
A chain sequences multiple components (e.g., prompts, LLMs, or tools) to process inputs and produce outputs in a structured way.

####Whats langchain?
LangChain is a framework designed to simplify the development of applications powered by large language models (LLMs) by providing modular components for chaining together prompts, tools, memory, and agents.

example chain:
```
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI()
prompt = PromptTemplate(input_variables=["text"], template="Summarize: {text}")
chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run("Long document text...")
```

#### Tools in langchain:
Tools in LangChain allow LLMs to interact with external systems, such as APIs, databases, or search engines, to perform tasks beyond text generation.

example tool:
```
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
tool = Tool(
    name="Google Search",
    func=search.run,
    description="Search Google for recent information."
)
result = tool.run("current weather in New York")
```

#### Langchain Memory:
Memory in LangChain enables applications to retain context across interactions, making conversations or tasks stateful.

Types:
ConversationBufferMemory: Stores the full history of the conversation.
ConversationSummaryMemory: Summarizes the conversation to save tokens.
ConversationBufferWindowMemory: Stores only the last k interactions.

Example:
```
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

llm = OpenAI()
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
chain.run("Hi, Im Bob.")
chain.run("What's my name?")  # LLM remembers "Bob" from memory.
```

#### Langgraph Agents:
 Agents are advanced components that use LLMs to reason about tasks, decide which tools to use, and execute actions dynamically. Agents iterate between reasoning, tool use, and output generation until the task is complete.

```
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Define state
class AgentState(TypedDict):
    messages: List[dict]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key="your-openai-api-key")

# Mock tool
def mock_tool(query: str) -> str:
    return f"Tool result for: {query}"

# Nodes
def llm_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]["content"]
    response = llm.invoke(last_message)
    state["messages"].append({"role": "assistant", "content": response.content})
    return state

# Graph setup
workflow = StateGraph(AgentState)
workflow.add_node("llm_node", llm_node)
workflow.set_entry_point("llm_node")
workflow.add_edge("llm_node", END)
graph = workflow.compile()

# Run agent
initial_state = AgentState(messages=[{"role": "human", "content": "Hello, what's 2+2?"}])
result = graph.invoke(initial_state)
print(result["messages"][-1]["content"])
```



CODING ROUND PREPARATION:
Here's a simple breakdown of each concept with easy-to-understand examples:

Data Structures
Trees

Like a family tree.

Example: Binary Search Tree (BST)

python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
root = Node(10)
root.left = Node(5)
root.right = Node(15)
Graphs

Like a social network (nodes = people, edges = friendships).

Example: Adjacency List

python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A'],
    'D': ['B']
}
Heaps (Priority Queue)

A tree where the smallest (min-heap) or largest (max-heap) element is at the top.

Example:

python
import heapq
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
print(heapq.heappop(heap))  # 2 (smallest comes first)
Tries (Prefix Tree)

Used for autocomplete (e.g., searching words).

Example:

python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
root = TrieNode()
Algorithms
BFS (Breadth-First Search)

Level-order traversal (like ripples in water).

Example (on a tree):

python
from collections import deque
def bfs(root):
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.val)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
DFS (Depth-First Search)

Goes deep before wide (like a maze solver).

Example (recursive):

python
def dfs(node):
    if not node: return
    print(node.val)
    dfs(node.left)
    dfs(node.right)
Recursion

A function calling itself.

Example (Factorial):

python
def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)
Dynamic Programming (DP)

Breaking problems into smaller subproblems (with memoization).

Example (Fibonacci with memoization):

python
memo = {}
def fib(n):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
Greedy Algorithms

Picks the best immediate choice (may not always be optimal).

Example (Coin Change - min coins for amount):

python
def coin_change(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        while amount >= coin:
            amount -= coin
            count += 1
    return count if amount == 0 else -1
Sorting

Arranging elements in order.

Example (QuickSort):

python
def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

Python Fluency
List Comprehensions

Compact way to create lists.

Example:

python
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, ...]
Lambda (Anonymous Function)

Short one-line functions.

Example:

python
add = lambda x, y: x + y
print(add(2, 3))  # 5
Map/Filter/Reduce

map: Apply a function to all items.

python
nums = [1, 2, 3]
squared = list(map(lambda x: x**2, nums))  # [1, 4, 9]
filter: Keep items that meet a condition.

python
evens = list(filter(lambda x: x % 2 == 0, nums))  # [2]
reduce (from functools): Aggregate values.

python
from functools import reduce
sum_all = reduce(lambda x, y: x + y, nums)  # 6
Exception Handling

Gracefully handle errors.

Example:

python
```
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("Done")
```

SYSTEM DESIGN:
Microservices vs Monoliths:
microservices are small lightweight services work together on larger goal, they are very much scalable and require less services

 **Multithreading**                   | **Multiprocessing**                               |
 -------------------------------------| ------------------------------------------------- |
 Multiple threads in a single process | Multiple independent Python processes             |
 Shared memory                        | Separate memory space                             |
 Affected by GIL — only one           | Bypasses GIL — true parallel execution            |
 thread runs at a time
 Low (lightweight threads)            | High (process creation and inter-process comm.)   |
 Low — crash affects all threads      | High — crash in one process doesn't affect others |
 `queue.Queue`, shared variables      | `multiprocessing.Queue`, `Pipe`, `Manager`        |

>>>> 
example for multi threading:
```
import threading
import time

def fetch_data():
    time.sleep(2)
    print("Data fetched")

threads = []
for _ in range(5):
    t = threading.Thread(target=fetch_data)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

example for multiprocessing:

```
import multiprocessing
import time

def compute():
    time.sleep(2)
    print("Computation done")

if __name__ == '__main__':
    processes = []
    for _ in range(5):
        p = multiprocessing.Process(target=compute)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

RAG:
WHATS RAG:
RAG (Retrieval-Augmented Generation) is a technique used to enhance the performance of Large Language Models (LLMs) by combining them with an information retrieval system. The idea is simple: instead of relying solely on the LLM's internal knowledge (which is frozen at training time), RAG supplements it with relevant external documents fetched dynamically during inference.

🧠 Architecture Overview
Input: User query

Vector database + Retriever : Uses dense or sparse retrieval (like FAISS or BM25) to fetch top-k documents.

LLM: Takes the query + documents as context and generates an answer.

Output: Final grounded response.


Chunking stregies:
CHUNKING:
Fixed-length Chunking: Use for fast, simple retrieval when documents are unstructured and uniform.
Sentence-level Chunking: Use when precision and granularity are important, e.g., for simple question answering.
Paragraph-level Chunking: Use when you need more context than sentences provide, especially for moderately complex queries.
Section-based Chunking: Use for long, well-structured documents where sections are meaningful (e.g., legal or academic texts).
Sliding Window Chunking: Use when you need continuity of context, especially for documents with no clear chunking boundaries.
Dynamic Chunking: Use when documents are heterogeneous, requiring more adaptive and context-aware segmentation.
Semantic Chunking: Use for tasks where semantic coherence and understanding of complex topics are critical. Understands Context: Uses AI (e.g., embeddings, NLP models) to analyze the text’s topics, structure, and relationships between sentences/paragraphs. Finds Natural Breaks: Splits text where ideas shift—like between sections, topics, or logical arguments—instead of cutting mid-thought.


vector search strategies:
FAISS (Facebook)

Description: Optimized ANN with GPU support
Best For: Large-scale, high-dimensional data

Annoy (Spotify)

Description: LSH + tree-based index
Best For: Medium-scale, low-latency

HNSWLib

Description: Graph-based (HNSW)
Best For: High recall with fast search

Milvus/Weaviate

Description: Full-featured vector databases
Best For: Production-ready deployments

Pinecone

Description: Managed vector DB
Best For: Serverless/cloud solutions

Top-k

Refers to retrieving the k most relevant items from a dataset
In vector search, returns the k nearest vectors to a query
Common parameter in search systems (e.g., "return top 10 results")
Balances relevance against computational efficiency

Cosine Similarity

Measures similarity between two vectors by calculating the cosine of the angle between them
Ranges from -1 (completely opposite) to 1 (identical), with 0 indicating orthogonality
Formula: cos(θ) = (A·B)/(||A||·||B||)
Widely used in vector databases and embeddings-based search
Focuses on direction rather than magnitude, making it good for text and semantic search

BM25

A probabilistic ranking function used for text retrieval (Best Matching 25)
Improves on TF-IDF by addressing term saturation and document length normalization
Considers term frequency, inverse document frequency, and document length
Standard algorithm in traditional keyword-based search engines
Does not use vector representations like embeddings-based methods
RetryJRand annEditANN (Approximate Nearest Neighbor)

A technique that finds approximate rather than exact nearest neighbors in high-dimensional spaces
Trades perfect accuracy for significant speed improvements
Essential for practical similarity search in large-scale vector databases


What Is a Vector Embedding?
A vector embedding is a way to represent text (words, sentences, or documents) as numerical vectors in a high-dimensional space. These vectors capture semantic meaning, so similar texts have vectors that are close to each other.

what are dimensions in vector embeddings?
- more dimension more quality (only upto a point)
- more dimension more memory
- more dimension more latency

Retrieval Strategies:
Dense Retrieval
Description: Uses dense vector representations (embeddings) of queries and documents to measure similarity. Typically, embeddings are generated using transformer-based models like BERT or sentence transformers.

Sparse Retrieval
Description: Relies on traditional term-based methods like keyword matching or TF-IDF (Term Frequency-Inverse Document Frequency).

1. Dense Retrieval
Description: Uses dense vector representations (embeddings) of queries and documents to measure similarity. Typically, embeddings are generated using transformer-based models like BERT or sentence transformers.
How It Works:
Queries and documents are encoded into dense vectors in a high-dimensional space.
Similarity is computed using metrics like cosine similarity or dot product to identify the most relevant documents.
Advantages:
Captures semantic meaning, allowing retrieval of documents with similar meaning even if exact keywords don’t match.
Effective for complex, context-heavy queries.
Challenges:
Computationally intensive due to embedding generation and similarity computation.
Requires a pre-trained embedding model and a well-indexed vector database.
Example Tools: FAISS, Annoy, or Elasticsearch with dense vector support.

2. Sparse Retrieval
Description: Relies on traditional term-based methods like keyword matching or TF-IDF (Term Frequency-Inverse Document Frequency).
How It Works:
Documents and queries are represented as sparse vectors based on word frequencies or importance.
Retrieval is based on exact or partial matches of terms between the query and documents.
Advantages:
Computationally efficient and works well for keyword-driven queries.
Simple to implement with tools like BM25.
Challenges:
Struggles with semantic understanding, missing documents with similar meaning but different wording.
Less effective for nuanced or complex queries.
Example Tools: BM25, Lucene, Elasticsearch with keyword-based search.

3. Hybrid Retrieval
Description: Combines dense and sparse retrieval to leverage the strengths of both approaches.
How It Works:
Sparse retrieval identifies documents with exact or near-exact term matches.
Dense retrieval refines the results by ranking documents based on semantic similarity.
Results from both methods are merged or re-ranked (e.g., using a weighted scoring system).
Advantages:
Balances keyword-based precision with semantic understanding.
Robust across diverse query types.
Challenges:
Increased complexity due to managing two retrieval systems.
Requires tuning to balance sparse and dense contributions.
Example Tools: Elasticsearch with hybrid search capabilities, custom pipelines combining BM25 and FAISS.

Hierarchical Retrieval
Description: Organizes documents in a hierarchical structure (e.g., by topic or category) to narrow down the search space before applying fine-grained retrieval.

Graph-Based Retrieval
Description: Uses knowledge graphs to enhance retrieval by leveraging relationships between entities.

REST API:


ARCHITECT design coding:

1. Singleton Pattern
Ensures a class has only one instance.

class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


2. Factory Pattern
The Factory Method is a design pattern that lets a superclass define the interface for creating an object, but allows subclasses to change the actual class that is instantiated.

python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class AnimalFactory:
    def create_animal(self, animal_type):
        if animal_type == "dog":
            return Dog()
        # other animals...

3. Observer Pattern
One-to-many dependency between objects.

python
class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

class Observer:
    def update(self, subject):
        pass
