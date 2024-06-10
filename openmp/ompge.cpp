//
// Created by shubham on 26.05.21.
//

//
// Created by shubham on 14.05.21.
//

//
// Last Speedup Test Details
// Submission Id : 23165
// Runtime : 1.17954 sec
// Speedup : 14.76846
//

#include <cassert>
#include <chrono>
#include <omp.h>
#include "ompge.h"
#define NUM_THREADS 32

namespace OMP{
void ForwardElimination(double *matrix, double *rhs, int rows, int columns){
    int diag_idx;
    double diag_elem;
    for(int row = 0; row < rows; row++){
        // Extract Diagonal element
        diag_idx = row*rows + row;
        diag_elem = matrix[diag_idx];
        assert(diag_elem!=0);
        int mat_row_ind=row*rows;
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (int lower_rows=row+1; lower_rows<rows; lower_rows++){
            int lower_row_ind=lower_rows*rows;
            int below_diag_idx = lower_row_ind+ row;
            // Compute the factor
            double elimination_factor = matrix[below_diag_idx]/diag_elem;
            for (int column=row+1; column<columns; column++){
                int element_idx = lower_row_ind + column;
                // subtract the row
               matrix[element_idx] = matrix[element_idx] -  elimination_factor*matrix[mat_row_ind+column];
            }
            rhs[lower_rows] -= elimination_factor*rhs[row];
            // set below diagonal elements to 0
            matrix[below_diag_idx] = 0.;
        }
    }
}

void BackwardSubstitution(double *matrix, double *rhs, double*solution, int rows, int columns){
    for(int row=rows-1; row>=0; row--){
        solution[row] = rhs[row];
        int diag_idx = row*rows + row;
        for (int column=row+1; column<columns; column++){
            int element_idx = row*rows + column;
            solution[row] -= matrix[element_idx]*solution[column];
        }
        solution[row] /= matrix[diag_idx];
    }
}
}

void OMP::Solve(double *matrix, double *rhs, double *solution,
                   int rows, int columns){
    OMP::ForwardElimination(matrix, rhs, rows, columns);
    OMP::BackwardSubstitution(matrix, rhs, solution, rows, columns);
}
