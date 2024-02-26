from bisect import insort
# import bisect
import random

# Generate 1000 elements with random confidence scores
elements = [(i, int(random.uniform(1, 20))) for i in range(15)]

# Initialize a sorted list to store elements with lowest confidence
sorted_list = []
keys = []

def insert(seq, keys, item, keyfunc=lambda v: v):
    k = keyfunc(item)  # Get key.
    i = insort(keys, k)  # Determine where to insert item.
    keys.insert(i, k)  # Insert key of item to keys list.
    seq.insert(i, item)  # Insert the item itself in the corresponding place.

# Iterate over the elements
for element in elements:
    # Insert the element into the sorted list while maintaining the sorted order
    insert(sorted_list, keys, element, keyfunc=lambda x: x[1])

# Extract the 100 elements with lowest confidence
dataset = sorted_list

print(dataset)

