# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:29:39 2019

@author: zhenh
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# %%
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"

# %%
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
# %% 
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]   
# %% List comprehensions can also contain conditions:
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"    
# %%
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
# %% Classes The syntax for defining classes in Python is straightforward:
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"  

# %%
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# plt.subplot(2, 1, 1)
# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
# %%
# cook your dish here
li = [i for i in input().split()]
x = float(li[0])
y = float(li[1])
if x%5 == 0:
    if x+0.5 < y:
        res = (y-x)-0.5
    else:
        res = y
else:
    res = y
print('%.2f'%res)

# %%
l=[str(d) for d in input().split()]
#print(type(float(l[1])))
if(int(l[0])%5==0):
    x=int(l[0])+0.50
    #print(x)
    if(x<float(l[1])):
        #print((float(l[1]))-x)
        print("%.2f"%((float(l[1]))-x))
    else:
        print("%.2f"%((float(l[1]))))
else:
    print("%.2f"%((float(l[1]))))

# %%
def all_nine(number):
    for char in number:
        if char != '9':
            return False
    return True


def solve(number):
    number = list(number)
    if all_nine(number):
        return '1' + ('0' * (len(number) - 1)) + '1'

    middle = len(number) // 2

    i = middle - 1
    j = middle + 1 if len(number) & 1 else middle
    left_small = False
    while i >= 0 and number[i] == number[j]:
        i -= 1
        j += 1

    if i < 0 or number[i] < number[j]:
        left_small = True

    while i >= 0:
        number[j] = number[i]
        j += 1
        i -= 1

    if left_small:
        carry = True
        if len(number) & 1:
            if number[middle] == '9':
                number[middle] = '0'
            else:
                carry = False
                number[middle] = str(int(number[middle]) + 1)

        i = middle - 1
        j = middle + 1 if len(number) % 2 & 1 else middle
        while i >= 0:
            if carry:
                if number[i] == '9':
                    carry = True
                    number[i] = '0'
                else:
                    carry = False
                    number[i] = str(int(number[i]) + 1)
            number[j] = number[i]
            j += 1
            i -= 1
    return ''.join(number)


def main():
    for test_case in range(int(input())):
        print(solve(input().strip()))


if __name__ == '__main__':
    main()

# %%
def rsiFunc(prices, n=14):
    
    
    return 


if __name__ == '__main__':
    print(123)


