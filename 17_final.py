import numpy as np
import itertools
import random 

def step(x):
    return 1 if x>=0 else 0

def generate_combinations(n):
    return np.array(list(itertools.product([0,1], repeat = n)))

def forward_propagate(X, w1, b1, w2, b2, w3, b3):

    z1 = np.dot(X, w1) + b1
    a1 = np.vectorize(step) (z1) 
    
    z2 = np.dot(a1, w2) + b2
    a2 = np.vectorize(step)(z2)
    
    z3 = np.dot(a2, w3) + b3
    a3 = np.vectorize(step) (z3)

    return a3

num_inputs = int(input("Enter the number of binary inputs: "))
X = generate_combinations(num_inputs)

output = []
for row in X:
    print("Enter the output for the inputs ")
    a = int(input(list(row)))
    output.append(a) 
y = np.array(output).reshape(-1, 1)

found = False
iterations = 0
while not found:
    iterations += 1

    w1 = np.random.uniform(-2, 2, (num_inputs, 2))
    w2 = np.random.uniform(-2, 2, (2, 2))
    w3 = np.random.uniform(-2, 2, (2, 1))

    b1 = np.random.randint(-3, 3)
    b2 = np.random.randint(-3, 3)
    b3 = np.random.randint(-3, 3)

    output = forward_propagate(X, w1, b1, w2, b2, w3, b3)
    if np.array_equal(y, output):
        print("Found at iteration ", iterations)
        print(w1)
        print(b1)
        print(w2)
        print(b2)
        print(w3)
        print(b3)
        found = True
        
    if iterations % 100000 == 0:
        print(f"Iteration {iterations}: still searching...")
