cimport cython


def nw_align(str seq1, str seq2, int match=1, int miss=-1, int gap=-1):
    
    cdef int seq1_len
    cdef int seq2_len

    cdef list score_matrix
    cdef list candidate

    cdef int i, j, k, m
    
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    score_matrix = [[0 for _ in range(seq1_len+1)] for _ in range(seq2_len+1)]

    for k in range(1, seq1_len+1):
        score_matrix[0][k] = k * gap
    for m in range(1, seq2_len+1):
        score_matrix[m][0] = m * gap

    candidate = [0, 0, 0]

    for i in range(0, seq2_len):
        for j in range(0, seq1_len):
            if seq2[i] == seq1[j]:
                candidate[0] = score_matrix[i][j] + match
            else:
                candidate[0] = score_matrix[i][j] + miss
            candidate[1] = score_matrix[i][j+1] + gap
            candidate[2] = score_matrix[i+1][j] + gap

            score_matrix[i+1][j+1] = max(candidate)
    
    return score_matrix

