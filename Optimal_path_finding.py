#!/usr/bin/env python
# coding: utf-8

#Program aims to find the minimum cost of traversing through the matrix from the top-left corner to the bottom-right corner using Dynamic Programming

import numpy as np
import argparse


# Uncomment to parse similarity matrix using command line
# Parse command line arguments
# parser = argparse.ArgumentParser(description="Find Optimal Path using dynamic programming")
# parser.add_argument("similarity_matrix", help="Path to similarity matrix")
# args = parser.parse_args()


# Loading input matrix, give path to input matrix
# similar_matrix = np.load(args.similarity_matrix)

#Function returns minimum cost of traversing the input matrix from top-left corner to the bottom-right corner
#Also returns the path with minimum cost
def min_cost_path(matrix):
    
    #Find shape of input matrix
    n, m = matrix.shape
    
    #Initialize another array of the same dimension
    dp = np.zeros((n, m))
    
    #Cost of reaching first element at position (0,0) in new matrix will be equal to the value of first element in input matrix
    dp[0, 0] = matrix[0, 0]
    
    # To store traversal directions
    # 1- Move right
    # 2- Move down
    # 3 - Move digonally 
    direction = np.zeros((n, m), dtype=int)

    
    # fill the first row and column of dp array
    #cost of traversal to any cell in the first row or column is the sum of cells above it
    for i in range(1, n):
        #first column
        dp[i, 0] = dp[i-1, 0] + matrix[i, 0]
        direction[i, 0] = 2 # move down
    for j in range(1, m):
        #first row
        dp[0, j] = dp[0, j-1] + matrix[0, j]
        direction[0, j] = 1 # move right
    
    #fill the rest of the dp array
    for i in range(1, n):
        for j in range(1, m):
            # find the minimum cost and update the direction accordingly
            if dp[i-1, j] < dp[i, j-1] and dp[i-1, j] < dp[i-1, j-1]:
                dp[i, j] = dp[i-1, j] + matrix[i, j]
                direction[i, j] = 2 # move down
            elif dp[i, j-1] < dp[i-1, j] and dp[i, j-1] < dp[i-1, j-1]:
                dp[i, j] = dp[i, j-1] + matrix[i, j]
                direction[i, j] = 1 # move right
            else:
                dp[i, j] = dp[i-1, j-1] + matrix[i, j]
                direction[i, j] = 3 # move diagonal
        #print(dp)
    #print("Direction matrix: ", direction)

                
    # backtrack direcrion array to find the path
    path = []
    i, j = n-1, m-1
    while i > 0 or j > 0:
        path.append((i, j))
        if direction[i, j] == 2: # move down
            i -= 1
        elif direction[i, j] == 1: # move right
            j -= 1
        else: # move diagonal
            i -= 1
            j -= 1

    path.reverse()

    #Using threshold to get frames with maximum match, can be changed based on video
    threshold = 0.14

    matching_frames = []
    for i, j in path:
        if matrix[i, j] <= threshold:
            matching_frames.append((i, j))


    #print(matching_frames)

    #return minimum cost, matched frames
    return dp[n-1, m-1], matching_frames


# Test using a smaller matrix
# cost = [[0.1, 0.2, 0.3, 0.4],
#        [0.4, 0.8, 0.2, 0.5],
#        [0.1, 0.5, 0.3, 0.1],
#         [0.3, 0.2, 0.5, 0.3]]
# m = np.array(cost)

#call the function and get the minimum cost and optimal path
# min_cost_sm, path_sm = min_cost_path(similar_matrix)

#print the minimum cost and the optimal path for Similar Matrix
# print("Minimum cost for Similar Matrix:", min_cost_sm)
# print("Optimal path found in Similar Matrix:", path_sm)
# print("------------------------------------------------------------")

