#include "mex.h"
#include <time.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

using namespace Eigen;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Input/output variables. */
    double *W;
    double *A;
    
    double *v;
  
    /* Intermediate variables.*/
    int Wm,Wn;
    int Am,An;
    int i, j;
    int m, n, l;
  
    /* Assign pointers to inputs. */
    W = mxGetPr(prhs[0]);
    A = mxGetPr(prhs[1]);
    
    /* Get sizes of input matrices (images, transformations, etc.).*/
    Wm = mxGetM(prhs[0]);
    Wn = mxGetN(prhs[0]);
    Am = mxGetM(prhs[1]);
    An = mxGetN(prhs[1]);    
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(9,1,mxREAL);
         
    /* Assign pointers to output. */
    v = mxGetPr(plhs[0]);

    /* Start computations. */
    /*We first need to multiply the matrix W by the matrix A (in order to obtain WA)*/
    MatrixXf mWA(Am,An);    
    for(i=0;i<Am;i+=2)
    {
        n = i/2;
        for(j=0;j<An;j++)
        {
            m = i+j*Am;
            /*n = i/2;
            l = i+1+j*Am;*/
            l = m + 1;
            mWA(i,j)   = W[n] * A[m];
            mWA(i+1,j) = W[n] * A[l];
        }
    }

    /* Perform SVD on matrix WA. */
    JacobiSVD<MatrixXf,HouseholderQRPreconditioner> svd(mWA, ComputeThinV);
    MatrixXf V = svd.matrixV();
    
    /* Obtain the least significant right singular vector of WA. */
    for(j=0;j<V.rows();j++)
        v[j] = V(j,V.rows()-1);

    /* Bye bye.*/
    return;
}
