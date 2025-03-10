"""
Hungarian algorithm implementation
Based on information from:
Imperial College London, Transport Systems
"N2.5 Hungarian Algorithm"
URL: https://transport-systems.imperial.ac.uk/tf/60008_21/n2_5_hungarian_algorithm.html
"""

import numpy as np
from copy import deepcopy

def hungarian_step(mat):
    for row_num in range(mat.shape[0]):
        mat[row_num] = mat[row_num] - np.min(mat[row_num])
    
    for col_num in range(mat.shape[1]):
        mat[:,col_num] = mat[:,col_num] - np.min(mat[:,col_num])

    return mat

def min_zeros(zero_mat, mark_zero):
    min_row = [999999, -1]

    for row_num in range(zero_mat.shape[0]):
        if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
            min_row = [np.sum(zero_mat[row_num] == True), row_num]
    
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False

def mark_matrix(mat):
    #Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    #Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
        min_zeros(zero_bool_mat_copy, marked_zero)

    #Recording the row and column indexes seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])
    
    # mark rows not containing zeros
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
    
    # mark columns with zeros
    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                if row_array[j] == True and j not in marked_cols:

                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            if row_num not in non_marked_row and col_num in marked_cols:
                
                non_marked_row.append(row_num)
                check_switch = True
    
    # mark rows with zeros
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))
    
    return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []
    
    # find the minimum value of an element not in a marked column/row 
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    
    min_num = min(non_zero_element)

    # substract to all values not in a marked row/column
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    # add to all values in marked rows/column
    for row in range(len(cover_rows)):  
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num

    return cur_mat

def hungarian_algorithm(cost_matrix):
    n = cost_matrix.shape[0]
    cur_mat = deepcopy(cost_matrix)

    cur_mat = hungarian_step(cur_mat)

    count_zero_lines = 0

    while count_zero_lines < n:
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        count_zero_lines = len(marked_rows) + len(marked_cols)

        if count_zero_lines < n:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
    
    return ans_pos