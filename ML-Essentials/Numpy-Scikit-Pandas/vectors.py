# Using numpy arrays
# Modern CPUs - single instruction, multiple data (SIMD) - apply in parallel
# NumPy lets us create arrays and memory efficiently with parallel processing on common operations
# "Vectorizing" the code => replacing iterative loops with vector operations that can be done in parallel
import numpy as np
ratings = np.array([
    5,
    4,
    3,
    2,
    4,
    2
])

ratings = ratings * 2

print(ratings)

# Traditional approach
# doing a common operation on each element of an array is inefficient
# ratings = [
#     5,
#     4,
#     3,
#     2,
#     4,
#     2
# ]
#
# for i, rating in enumerate(ratings):
#     ratings[i] = rating * 2
#
# print(ratings)


