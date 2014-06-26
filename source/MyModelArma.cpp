#include <iostream>
#include <algorithm>
#include <limits>
//#include <fftw3.h>
#include "armadillo"
#include <iomanip>
#include <math.h>
//#include "mex.h"

#define SPEEDUP 0;

using namespace arma;
using namespace std;

const double EPS = numeric_limits<double>::epsilon();

struct Opt 
{
  double tol;
  int rank;
  double lambda;
  int flagAlign;
  int moving;
  double card;
} ;


double cubicInterpolation(double pos, double prevstart, double start, double end, double nextend)
{

  double value;
  value = ((3 * start - prevstart - 3 * end + nextend) / 2)* (pos*pos*pos) + (prevstart - (5 * start) / 2 + 2 * end - nextend / 2)*(pos*pos) + (end / 2 - prevstart / 2)*pos + start;
  //cout << value<< endl;
  return value;
}

mat imResizeBicubic(mat input, double ratio, int maxRatio)
{
  // Ratio is expected to be in the interval [0, 1]
  int dest_width = ceil(input.n_cols * ratio);
  int dest_height = ceil(input.n_rows * ratio);
  int src_width = input.n_cols;
  int src_height = input.n_rows;
  //int maxRatio = 8;
  int needed_width = maxRatio * ceil(src_width / (double) maxRatio), needed_height = maxRatio * ceil(src_height / (double) maxRatio); 
  //mat temp = zeros(src_height, dest_width);
  mat out = zeros(dest_height, dest_width);
  double positionN, positionD;

  //cout << needed_width << " " << needed_height << " " << endl;

  /*if (floor(src_width / 2.0) != ceil (src_width / 2.0))
  {
  mat input_new = zeros(src_height, src_width + 1);
  input_new.submat(0,0,src_height - 1, src_width - 1) = input;
  input_new.col(src_width) = input.col(src_width - 1);
  src_width++;
  input = input_new;
  }
  if (floor(src_height / 2.0) != ceil (src_height / 2.0))
  {
  mat input_new = zeros(src_height + 1, src_width);
  input_new.submat(0,0,src_height - 1, src_width - 1) = input;
  input_new.row(src_height) = input.row(src_height - 1);
  src_height++;
  input = input_new;
  }*/

  if (needed_width != src_width)
  {
    mat input_new = zeros(src_height, needed_width);
    input_new.submat(0, 0, src_height - 1, src_width - 1) = input;
    for (int k = src_width; k < needed_width; k++)
    {
      input_new.col(k) = input.col(src_width - 1);
    }
    //src_width++;
    input = input_new;
  }
  if (needed_height != src_height)
  {
    mat input_new = zeros(needed_height, needed_width);
    input_new.submat(0, 0, src_height - 1, needed_width - 1) = input;
    for (int k = src_height; k < needed_height; k++)
    {
      input_new.row(k) = input.row(src_height - 1);
    }
    input = input_new;
  }

  mat temp = zeros(needed_height, needed_width);

  //cout << "dest: " << dest_width << " " << dest_height << "  src: " << src_width << src_height << endl;

  //HORIZONTAL INTERPOLATION:
  for (int i = 0; i < needed_height; ++i)
  {
    for (int j = 0; j < dest_width; ++j)
    {
      positionD = (1 / (2 * ratio)) * (2 * j + 1);
      positionN = positionD - floor(positionD) + 0.5;

      //cout << positionD << endl << positionN << endl;
      //cout << "here" << i  << " " << j << endl;
      if (floor(positionD) - 1 == 0)
      {
        temp(i, j) = cubicInterpolation(positionN, input(i, floor(positionD) - 1), input(i, floor(positionD) - 1), input(i, floor(positionD)), input(i, floor(positionD) + 1));
      }
      else if (floor(positionD) == (src_width - 1)) 
      {
        temp(i, j) = cubicInterpolation(positionN, input(i, floor(positionD) - 2), input(i, floor(positionD) - 1), input(i, floor(positionD)), input(i, floor(positionD)));
      }
      else 
      {
        //cout << "here" << i  << " " << j << " " << positionD << endl << input(i, floor(positionD) - 2) << " " << input(i, floor(positionD) - 1) << " " << input(i, floor(positionD) ) <<" " << input(i, floor(positionD) + 1) << endl; 
        temp(i, j) = cubicInterpolation(positionN, input(i, floor(positionD) - 2), input(i, floor(positionD) - 1), input(i, floor(positionD)), input(i, floor(positionD) + 1));
        //cout << temp(i, j) << " " << input(i,j) << endl;;
      }       //something goes here
    }
    //cout << "stop: " << i << endl;

  }

  temp = trans(temp);
  out = trans(out);
  //VERTICAL INTERPOLATION:
  for (int i = 0; i < dest_width; ++i)
  {
    for (int j = 0; j <dest_height ; ++j)
    {
      positionD = (1 / (2 * ratio)) * (2 * j + 1);
      positionN = positionD - floor(positionD) + 0.5;

      //cout << positionD << endl << positionN << endl;
      //cout << "here" << i  << " " << j << endl;
      if (floor(positionD) - 1 == 0)
      {
        out(i, j) = cubicInterpolation(positionN, temp(i, floor(positionD) - 1), temp(i, floor(positionD) - 1), temp(i, floor(positionD)), temp(i, floor(positionD) + 1));
      }
      else if (floor(positionD) == (src_width - 1)) 
      {
        out(i, j) = cubicInterpolation(positionN, temp(i, floor(positionD) - 2), temp(i, floor(positionD) - 1), temp(i, floor(positionD)), temp(i, floor(positionD)));
      }
      else 
      {
        out(i, j) = cubicInterpolation(positionN, temp(i, floor(positionD) - 2), temp(i, floor(positionD) - 1), temp(i, floor(positionD)), temp(i, floor(positionD) + 1));
      }       //something goes here
    }

  }

  out = trans(out);
  return out;

}


// calculate the cofactor of element (row,col)
void GetMinor(mat src, mat* dest, int row, int col, int order)
{
  // indicate which col and row is being copied to dest
  int colCount = 0, rowCount = 0;
  mat out = zeros(order, order);
  for (int i = 0; i < order; i++)
  {
    if (i != row)
    {
      colCount = 0;
      for (int j = 0; j < order; j++)
      {
        // when j is not the element
        if (j != col)
        {
          out(rowCount, colCount) = src(i, j);
          colCount++;
        }
      }
      rowCount++;
    }
  }
  *dest = out;
  return;
}

// Calculate the determinant recursively.
double CalcDeterminant(mat matrix, int order)
{
  // order must be >= 0
  // stop the recursion when matrix is a single element
  if (order == 1)
    return matrix(0, 0);

  // the determinant value
  double det = 0;

  // allocate the cofactor matrix
  mat minor = zeros(order - 1, order - 1);


  for (int i = 0; i < order; i++)
  {
    // get minor of element (0,i)
    GetMinor(matrix, &minor, 0, i, order);
    // the recusion is here!

    det += (i % 2 == 1 ? -1.0 : 1.0) * matrix(0, i) * CalcDeterminant(minor, order - 1);
    //det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
  }
  return det;
}

// matrix inversioon
// the result is put in Y
void MatrixInversion(mat A, int order, mat* Y)
{
  // get the determinant of a
  double det = 1.0 / CalcDeterminant(A, order);
  mat out = zeros(order, order);
  // memory allocation
  mat temp = zeros(order - 1,order - 1);
  mat minor = zeros(order - 1, order - 1);
  //for (int i = 0; i<order - 1; i++)
  //	minor[i] = temp + (i*(order - 1));

  for (int j = 0; j<order; j++)
  {
    for (int i = 0; i<order; i++)
    {
      // get the co-factor (matrix) of A(j,i)
      GetMinor(A, &minor, j, i, order);
      out(i, j) = det*CalcDeterminant(minor, order - 1);
      if ((i + j) % 2 == 1)
        out(i, j) = -out(i, j);
    }
  }

  *Y = out;
}


// Function for generating the 2D (1D also, in case there is no speedup) Gaussian mask for filtering
mat getGaussian(vec dimension, vec sigma) {
  // Input:
  //        dimension: [column vector] with the dimensions of the Gaussian mask
  //        sigma:     [column vector] with the definition of sigma values for all the dimensions
  //
  // Output:
  //        mask:      [matrix] Gaussian mask specified with the input parameters
  //
  // Speedups: - if the input parameters are correct, their checking can be avoided
  //           - if only the 2D case is possible, switch can be avoided
  //           (-) computing the square could be faster without using the pow() function?
  //
  // Author: Ivan Zupancic: v1 - 30/05/2014
  //                        v2 - 03/06/2014 (added speedup part, code cleaned)
  //=============================================================================================

  int dim = 0;
  mat Mask, X, Y;
  // This should be removed if the 3D Gaussian mask is never used
  //vec xind, yind, zind;
  //cube X1, Y1, Z1, Mask1;

#if !SPEEDUP
  // Checking the input parameters dimension
  if (dimension.n_elem == sigma.n_elem) {
    dim = dimension.n_elem;  
  } 
  else {
    cout << "Vector size and sigma length should have the same length!" << endl;
    exit(-1);
  }
#else
  // Assuming that the dimensions of vectors 'dimension' and 'sigma' are equal
  dim = dimension.n_elem;
#endif

  // Very short for loop for changing the sigma values which are 0 to 0.1
  for (int i = 0; i < dim; i++) {
    if (sigma(i) == 0) {
      sigma(i) = 0.1;
    }
  }

#if !SPEEDUP
  switch (dim) {
  case 1:
    X = trans(linspace(1, dimension(0), dimension(0))) - (1 + dimension(0)) / 2.0;
    //Mask = exp(- pow(X / (sigma(0) + EPS), 2));
    // Version with 'brute' multiplication (may be faster?)
    Mask = exp(- (X / (sigma(0) + EPS)) % (X / (sigma(0) + EPS))); 
    Mask /= accu(Mask);
    break;
  case 2:
    X = (linspace(1, dimension(0), dimension(0)) - (1 + dimension(0)) / 2) * ones(1, dimension(1)); 
    Y = ones(dimension(0), 1) * (trans(linspace(1, dimension(1), dimension(1))) - (1 + dimension(1)) / 2);
    //Mask = exp(- (pow(X / (sigma(0) + EPS), 2) + pow(Y / (sigma(1) + EPS), 2)) / 2.0);
    // Version with 'brute' multiplication (may be faster?)
    Mask = exp(- ((X / (sigma(0) + EPS)) % (X / (sigma(0) + EPS)) + (Y / (sigma(1) + EPS)) % (Y / (sigma(1) + EPS))) / 2.0);
    Mask /= accu(Mask);
    break;
    /*case 3: // That case is not considered! The following code is not working!
    xind = linspace(1, dimension(0), dimension(0)) - (1 + dimension(0)) / 2.0;
    X1 = zeros(dimension(0), dimension(1), dimension(2));

    X1(span(i), span(), span()) = xind(i);
    for(int i = 0; i < xind.n_elem; i++) {
    X1(span(i), span(), span()) = xind(i); 
    }

    yind = linspace(1, dimension(1), dimension(1)) - (1 + dimension(1)) / 2.0;
    Y1 = zeros(dimension(0), dimension(1), dimension(2));
    for(int i = 0; i < yind.n_elem; i++) {
    Y1(span(), span(i), span()) = yind(i); 
    }

    zind = linspace(1, dimension(2), dimension(2)) - (1 + dimension(2)) / 2.0;
    Z1 = zeros(dimension(0), dimension(1), dimension(2));
    for(int i = 0; i < zind.n_elem; i++) {
    Z1(span(), span(), span(i)) = zind(i); 
    }

    Mask1 = exp(- (pow(X1 / (sigma(0) + EPS), 2) + pow(Y1 / (sigma(1) + EPS), 2) + pow(Z1 / (sigma(2) + EPS), 2)) / 2.0);
    // Version with 'brute' multiplication (may be faster?)
    //Mask = exp(- ((X / (sigma(0) + EPS)) % (X / (sigma(0) + EPS)) + (Y / (sigma(1) + EPS)) % (Y / (sigma(1) + EPS))) / 2.0);
    Mask1 /= accu(Mask1);
    break;
    */
  default:
    cout << "This function can only support up to 2 dimensional Gaussian masks!" << endl;
    exit(-1);
  }
#else
  X = (linspace(1, dimension(0), dimension(0)) - (1 + dimension(0)) / 2) * ones(1, dimension(1)); 
  Y = ones(dimension(0), 1) * (trans(linspace(1, dimension(1), dimension(1))) - (1 + dimension(1)) / 2);
  //Mask = exp(- (pow(X / (sigma(0) + EPS), 2) + pow(Y / (sigma(1) + EPS), 2)) / 2.0);
  // Version with 'brute' multiplication (may be faster?)
  Mask = exp(- ((X / (sigma(0) + EPS)) % (X / (sigma(0) + EPS)) + (Y / (sigma(1) + EPS)) % (Y / (sigma(1) + EPS))) / 2.0);
  Mask /= accu(Mask);
#endif

  return Mask;
}

// Function for doing the ifftshift of the inverse transformed image (tailored for 2D input!)
mat ifftshift(mat X) {
  // Input:
  //        I:          [matrix] inverse transformed image
  //
  // Output:
  //        M:          [matrix] ifftshifted version of the input matrix
  //
  // Author: Ivan Zupancic: v1 - 02/06/2014
  //                        v2 - 03/06/2014 (code cleaned)
  //=============================================================================================

  int r = X.n_rows, c = X.n_cols, pr, pc;
  mat M = zeros<mat>(r, c);

  pr = floor(r / 2.0);
  pc = floor(c / 2.0);

  M.submat(0, 0, r - pr - 1, c - pc - 1) = X.submat(pr, pc, r - 1, c - 1);
  M.submat(r - pr, c - pc, r - 1, c - 1) = X.submat(0, 0, pr - 1, pc - 1);

  M.submat(0, c - pc, r - pr - 1, c - 1) = X.submat(pr, 0, r - 1, pc - 1);
  M.submat(r - pr, 0, r - 1, c - pc - 1) = X.submat(0, pc, pr - 1, c - 1);

  return M;
}

// Function for smoothening the image with the 2D GAussian filter specified with the sigma parameter
mat gSmooth(mat I, vec sigma) {
  // Input:
  //        I:          [matrix] image that needs to be smoothen
  //        sigma:      [column vector] with the definition of sigma values for all the dimensions
  //
  // Output:
  //        S:          [matrix] image smoothen with the 2D Gaussian mask
  //
  // Speedups: - if the input parameters are correct, their checking can be avoided
  //           (-) using the FFTW for the FFT computation?
  //
  // Author: Ivan Zupancic: v1 - 30/05/2014
  //                        v2 - 03/06/2014 (code cleaned)
  //=============================================================================================

  vec dimension = zeros<vec>(2,1);
  mat S, Mask;

#if !SPEEDUP
  // Checking if the dimensionality of I is less than 3
  if (I.n_cols * I.n_rows != I.n_elem) {
    cout << "'I' should be 2D!" << endl;
    exit(-1);
  }

  if (sigma.n_elem != 2) {
    cout << "The length of 'sigma' needs to be 2!" << endl;
    exit(-1);
  }
#endif

  dimension(0) = I.n_rows;
  dimension(1) = I.n_cols;

  Mask = getGaussian(dimension, sigma);

  // Using FFTW instead to speed up the computation?
  S = ifftshift(real(ifft2(fft2(Mask) % fft2(I))));

  return S;
}

// Function for performing the image filtering equivalent to the MATLAB function "imfilter(I, mask, 'replicate')" (tailored for mask of type vector with 5 elements! tilored for computing the gradient in both directions!)
void imFilter(mat phi, vec detector, mat *phi_x, mat *phi_y) {
  // Input:
  //        phi:        [matrix] input image which needs to be filtered with the mask specified below
  //        detector:   [column vector] mask for the filtering
  //        phi_x:      [*matrix] pointer to the output gradient in x direction
  //        phi_y:      [*matrix] pointer to the output gradient in y direction
  //
  // Author: Ivan Zupancic: v1 - 03/06/2014
  //=============================================================================================

  mat T = zeros<mat>(phi.n_rows + 4, phi.n_cols + 4);
  mat F1 = zeros<mat>(phi.n_rows, phi.n_cols);
  mat F2 = zeros<mat>(phi.n_rows, phi.n_cols);
  rowvec detectorT = trans(detector);

  // Extended matrix used for filtering, extended outside of the borders according to 'replicate' option in MATLAB function imfilter
  T.submat(2, 2, T.n_rows - 3, T.n_cols - 3) = phi;

  // Extension for the top and bottom rows
  T.submat(0, 2, 0, T.n_cols - 3) = phi.row(0);
  T.submat(1, 2, 1, T.n_cols - 3) = phi.row(0);
  T.submat(T.n_rows - 2, 2, T.n_rows - 2, T.n_cols - 3) = phi.row(phi.n_rows-1);
  T.submat(T.n_rows - 1, 2, T.n_rows - 1, T.n_cols - 3) = phi.row(phi.n_rows-1);

  // Extension for the leftmost and rightmost columns
  T.submat(2, 0, T.n_rows - 3, 0) = phi.col(0);
  T.submat(2, 1, T.n_rows - 3, 1) = phi.col(0);
  T.submat(2, T.n_cols - 2, T.n_rows - 3, T.n_cols - 2) = phi.col(phi.n_cols-1);
  T.submat(2, T.n_cols - 1, T.n_rows - 3, T.n_cols - 1) = phi.col(phi.n_cols-1);

  //if(direction == 0) { // row vector
  for (int i = 2; i < T.n_rows - 2; i++) {
    for (int j = 2; j < T.n_cols - 2; j++) {
      F1(i - 2, j - 2) = accu(detectorT % T.submat(i, j - 2, i, j + 2));
    }
  }

  *phi_y = F1;

  //}
  //else {  // column vector
  for (int i = 2; i < T.n_rows - 2; i++) {
    for (int j = 2; j < T.n_cols - 2; j++) {
      F2(i - 2, j - 2) = accu(detector % T.submat(i - 2, j, i + 2, j));
    }
  }
  //}

  *phi_x = F2;

  return;
}

// Function for computing the gradient of the image
void getGradient(mat phi, mat *phi_x, mat *phi_y, vec sigma) {
  // Input:
  //        phi:        [matrix] input image for which the gradient needs to be computed
  //        sigma:      [column vector] with the values for sigma in both dimensions
  //        phi_x:      [*matrix] pointer to the output gradient in x direction
  //        phi_y:      [*matrix] pointer to the output gradient in y direction
  //
  // Speedups: - if the parameter sigma exists for sure, checking can be avoided
  //
  // Author: Ivan Zupancic: v1 - 03/06/2014
  //=============================================================================================

  vec detector = zeros<vec>(5,1);
  detector << 1 << endr << -8 << endr << 0 << endr << 8 << endr << -1 << endr;
  detector /= 12.0;

  //#if !SPEEDUP
  if (sigma.n_elem != 0) {   
    phi = gSmooth(phi, sigma);
  } 
  //#else
  //  phi = gSmooth(phi, sigma);
  //#endif

  imFilter(phi, detector, phi_x, phi_y);

  return;
}

// Function equivalent to MATLAB  meshgrid function (this can be further optimised!)
void meshgrid(int m, int n, mat *XB, mat *YB) {
  // Input:
  //        m:          [scalar] number of the rows
  //        n:          [scalar] number of the columns
  //        XB:         [*matrix] output grid matrix for X
  //        YB:         [*matrix] output grid matrix for Y
  //
  // Author: Ivan Zupancic: v1 - 04/06/2014
  //=============================================================================================
  rowvec y = trans(linspace(1, m, m));
  vec x = linspace(1, n, n);

  mat M = zeros<mat>(n, m);
  mat N = zeros<mat>(n, m);

  for (int i = 0; i < n; i++) {
    M.row(i) = y;
  }

  for (int j = 0; j < m; j++) {
    N.col(j) = x;
  }

  *YB = M;
  *XB = N;

  return;
}



// Function for computing the warped transform (tailored for affine transform! mode = 0, extrapval = [])
void warpImg(mat I, vec tau, bool mode,  mat *Iwarp, mat *Omega) {
  // Input:
  //        I:          [matrix] input image which needs to be warped
  //        tau:        [matrix] transformation parameters
  //        mode:       [bool] variable for selecting either forward or backward warp
  //        Iwarp:      [*matrix] pointer to the output warped image
  //        Omega:      [*matrix] pointer to the output gradient in y direction
  //
  // Speedups: - if the parameter sigma exists for sure, checking can be avoided
  //
  // Author: Ivan Zupancic: v1 - 04/06/2014
  //=============================================================================================

#if ! SPEEDUP
  if (tau.n_elem != 6) {
    cout << "Only affine transform is considered here! 'tau' should contain 6 elements" << endl;
    exit(-1);
  }
#endif

  mat M = zeros<mat>(3,3);
  vec sizeI = zeros<vec>(2,1);
  sizeI(0) = I.n_rows;
  sizeI(1) = I.n_cols;
  mat O = zeros(sizeI(0), sizeI(1));

  mat XB = zeros<mat>(sizeI(1), sizeI(0)), XA = zeros<mat>(sizeI(1), sizeI(0));
  mat YB = zeros<mat>(sizeI(1), sizeI(0)), YA = zeros<mat>(sizeI(1), sizeI(0));

  mat Iw = zeros(sizeI(0), sizeI(1));

  mat CoordB = zeros<mat>(3, I.n_elem);
  mat CoordA = zeros<mat>(3, I.n_elem);

  M << 1 + tau(0) << tau(2) << tau(4) << endr << tau(1) << 1 + tau(3) << tau(5) << endr << 0 << 0 << 1; 

  meshgrid(sizeI(1), sizeI(0), &XB, &YB);

  CoordB.row(0) = reshape(XB, 1, XB.n_elem) - floor(sizeI(0) / 2.0 + 0.5);
  CoordB.row(1) = reshape(YB, 1, YB.n_elem) - floor(sizeI(1) / 2.0 + 0.5);
  CoordB.row(2) = ones(1, XB.n_elem);

  //cout << CoordB << endl;

  if (mode == 0) {
    CoordA = M * CoordB;
  }
  else {
    CoordA = solve(M, CoordB);
  }

  XA = reshape(CoordA.row(0), sizeI(0), sizeI(1)) + floor(sizeI(0) / 2.0 + 0.5);
  YA = reshape(CoordA.row(1), sizeI(0), sizeI(1)) + floor(sizeI(1) / 2.0 + 0.5);

  //cout << "M" << M << endl;
  //cout << "XB" << XB << endl;
  //cout << "YB" << YB << endl;
  /*cout << "XA" << XA << endl;
  cout << "YA" << YA << endl;*/

  // This may be slow!
  O.elem(find(XA < 1 || XA > sizeI(0) || YA < 1 || YA > sizeI(1))).ones();

  //cout << "O" << O << endl;

  *Omega = O;

  // This may be slow!
  XA.elem(find(XA < 1)).ones();
  uvec q1 = find(XA > sizeI(0));
  XA.elem(q1) = ones(q1.n_elem,1) * sizeI(0);
  YA.elem(find(YA < 1)).ones();
  uvec q2 = find(YA > sizeI(1));
  YA.elem(q2) = ones(q2.n_elem,1) * sizeI(1);

  //cout << "XA" << XA << endl;
  //cout << "YA" << YA << endl;

  XA -= 1;
  YA -= 1;

  double weight, weight_hor, weight_ver, temp0, temp1;

  for (int i = 0; i < sizeI(0); i++) {
    for (int j=0; j<sizeI(1); j++) {
      // Both integers:
      if (XA(i, j) == floor(XA(i, j)) && YA(i, j) == floor(YA(i, j))) {
        Iw(i, j) = I(XA(i, j), YA(i, j));
      }
      //case XA is fractional:
      else if (YA(i, j) == floor(YA(i, j))) {
        weight = XA(i, j) - floor(XA(i, j));
        Iw(i, j) = weight * I(ceil(XA(i, j)), YA(i, j)) + (1 - weight) * I(floor(XA(i, j)), YA(i, j));
      }
      else if (XA(i, j) == floor(XA(i, j))) {
        weight = YA(i, j) - floor(YA(i, j));
        Iw(i, j) = weight * I(XA(i, j), ceil(YA(i, j))) + (1 - weight) * I(XA(i, j), floor(YA(i, j)));
      }
      else {
        weight_hor = XA(i, j) - floor(XA(i, j));
        weight_ver = YA(i, j) - floor(YA(i, j)); 
        temp0 = weight_ver * I(ceil(XA(i, j)), ceil(YA(i, j))) + (1 - weight_ver) * I(ceil(XA(i, j)), floor(YA(i, j)));
        temp1 = weight_ver * I(floor(XA(i, j)), ceil(YA(i, j))) + (1 - weight_ver) * I(floor(XA(i, j)), floor(YA(i, j)));

        Iw(i, j) = weight_hor * temp0 + (1 - weight_hor) * temp1;
      } 
    }
  }

  //cout << "Iw" << Iw << endl;

  *Iwarp = Iw;

  // change elements of A greater than 0.5 to 1
  //A.elem(find(A > 0.5)).ones();

  return;
}
void regImg (mat I1, mat I2, vec tau_old, mat weight, int maxIts, mat *I2Warp, vec *tau_new, mat *residue, mat *OmegaOut) {
  vec sizeI = zeros(2,1);
  sizeI(0) = I1.n_rows;
  sizeI(1) = I1.n_cols;
  int sizeD = I1.n_elem;
  vec sigma;
  vec weight1 = ones(sizeD, 1), y = zeros(sizeD, 1), dtau = zeros(6, 1); 
  vec tau_t = zeros(6, 1);
  vec xC = zeros(sizeD, 1), yC = zeros(sizeD, 1), I2wX = zeros(sizeD, 1), I2wY = zeros(sizeD, 1);
  mat X = ones(sizeD, 6);
  mat A = zeros(6, 6);
  mat TEMP = zeros(3, 2);
  mat I2war = zeros(sizeI(0), sizeI(1)), I2warpX = zeros(sizeI(0), sizeI(1)), I2warpY = zeros(sizeI(0), sizeI(1)), xCoord = zeros(sizeI(0), sizeI(1)), yCoord = zeros(sizeI(0), sizeI(1));


#if !SPEEDUP
  if (tau_old.is_empty() == 1) {
    tau_old = zeros(6, 1);
  }
  if (weight.is_empty() == 0) {
    weight1 = reshape(weight, sizeD, 1);
  }
  if (maxIts == 0) {
    maxIts = 50;
  }
#endif


  for (int i = 1; i <= maxIts; i++) {
    //cout << "i" << i << endl;

    warpImg(I2, tau_old, 0, &I2war, OmegaOut);
    getGradient(I2war, &I2warpX, &I2warpY, sigma);
    //I2warpX = I2war * 1.0023;
    //I2warpY = I2war * 0.9746;

    /* for (int m = 0; m < I2.n_rows; m++)
    {
    for (int n = 0; n < I2.n_cols; n++)
    {
    getRepresentation(I2war(m, n));
    }

    }
    printf("\n ");
    printf("\n ");

    for (int m = 0; m < tau_old.n_rows; m++)
    {
    for (int n = 0; n < tau_old.n_cols; n++)
    {
    getRepresentation(tau_old(m, n));
    }

    }
    printf("\n ");
    printf("\n ");*/

    //for (int m = 0; m < I2warpX.n_rows; m++)
    //{
    // for (int n = 0; n < I2warpX.n_cols; n++)
    // {
    //	 getRepresentation(I2warpX(m, n));
    // }
    //}
    //printf("\n ");
    //printf("\n ");
    //for (int m = 0; m < I2warpY.n_rows; m++)
    //{
    // for (int n = 0; n < I2warpY.n_cols; n++)
    // {
    //	 getRepresentation(I2warpY(m, n));
    // }
    //}
    //printf("\n ");
    //printf("\n ");

    y = reshape(I1 - I2war, sizeD, 1);

    meshgrid(sizeI(1), sizeI(0), &xCoord, &yCoord);
    //xCoord -= (int)(sizeI(0) / 2 + 0.5); /// 2.0 + 0.5);
    //yCoord -= (int)(sizeI(1) / 2 + 0.5); /// 2.0 + 0.5);

    xCoord -= 3;// round((sizeI(0) / 2)); /// 2.0 + 0.5);
    yCoord -= 3;// round((sizeI(1) / 2)); /// 2.0 + 0.5);



    //xCoord.raw_print(cout, "xCoord");
    //yCoord.raw_print(cout, "yCoord");


    //xCoord = (int)(xCoord + 0.5);
    xC = reshape(xCoord, sizeD, 1);
    yC = reshape(yCoord, sizeD, 1);
    I2wX = reshape(I2warpX, sizeD, 1);
    I2wY = reshape(I2warpY, sizeD, 1);

    X.col(0) = (xC % I2wX);
    X.col(1) = (xC % I2wY);
    X.col(2) = (yC % I2wX);
    X.col(3) = (yC % I2wY);
    X.col(4) = (I2wX);
    X.col(5) = (I2wY);

    //for (int m = 0; m < X.n_rows; m++)
    //{
    // for (int n = 0; n < X.n_cols; n++)
    // {
    //	 printf("%.20g ", X(m, n));
    // }
    // printf("\n");

    //}
    //printf("\n ");
    //printf("\n ");
    //X.raw_print(cout, "X");


    //weight1.raw_print(cout, "weight1");
    mat Xtrans = trans(X);
    //X.each_col() %= weight1;
    //A = Xtrans * X;
    for (int m = 0; m < A.n_rows; m++)
    {
      for (int n = 0; n < A.n_cols; n++)
      {
        double temp = 0;
        for (int r = 0; r < X.n_rows; r++)
        {
          temp += Xtrans(m,r) * X(r,n);
        }
        A(m, n) = temp;// accu(trans(Xtrans.row(m)) % X.col(n));

        ;//getRepresentation(A(m, n));
      }
    }
  /*  printf("\n ");

    printf("\n ");*/
    //for (int m = 0; m < A.n_rows; m++)
    //{
    // for (int n = 0; n < A.n_cols; n++)
    // {
    //	 printf("%.20g ", A(m, n));
    // }
    // printf("\n");

    //}
    //printf("\n ");
    //printf("\n ");
    //dtau = solve((A + 0.001 * diagmat(A)), (trans(X) * (weight1 % y)));
    // dtau = inv((A + 0.001 * diagmat(A))) * (trans(X) * (y));

    mat temp1 = zeros(6,6);//   inv((A + 0.001 * diagmat(A)));

    for (int m = 0; m <6; m++)
    {
      for (int n = 0; n < 6; n++)
      {
        if (m == n)
          temp1(m, n) = 0.001 *A(m, n);
        else
          temp1(m, n) = 0;
      }

    }

    for (int m = 0; m <6; m++)
    {
      for (int n = 0; n < 6; n++)
      {
        temp1(m, n) += A(m, n);
      }

    }

    //for (int m = 0; m < 6; m++)
    //{
    //  for (int n = 0; n< 6; n++)
    //    ;//getRepresentation(temp1(m,n));
    //}
    //printf("\n ");

    //printf("\n ");
    //temp1 += A;
    mat invTemp = zeros(6, 6);
    MatrixInversion(temp1, 6, &invTemp);

    temp1 = invTemp;
    //for (int m = 0; m < 6; m++)
    //{
    //  for (int n = 0; n< 6; n++)
    //    ;//getRepresentation(temp1(m, n));
    //}
    //printf("\n ");

    //printf("\n ");
    mat temp2 = zeros(6, 1);
    mat transX = trans(X);
    for (int m = 0; m <6; m++)
    {
      double temp = 0;
      for (int r = 0; r < sizeD; r++)
      {
        temp += transX(m, r)*y(r); 
      }
      temp2(m) = temp;// accu(trans(Xtrans.row(m)) % X.col(n));
    }
    //for (int m = 0; m < 6; m++)
    //{
    //  ;//getRepresentation(temp2(m));
    //}
    //printf("\n ");

    //printf("\n ");

    for (int m = 0; m < 6; m++)
    {

      double temp = 0;
      for (int r = 0; r < 6; r++)
      {
        temp += temp1(m, r) * temp2(r);
      }
      dtau(m) = temp;// accu(trans(Xtrans.row(m)) % X.col(n));
    }



    //cout << std::setprecision(16) << dtau << endl;
    //dtau.raw_print(cout, "dtau");
    //for (int m = 0; m < dtau.n_elem; m++)
    //{
    //  ;//getRepresentation(dtau(m));
    //}
    //printf("\n ");

    //printf("\n ");
    //for (int m = 0; m < dtau.n_rows; m++)
    //{
    // for (int n = 0; n < dtau.n_cols; n++)
    // {
    //	 printf("%.20g ", dtau(m, n));
    // }
    // printf("\n");

    //}
    //printf("\n ");
    //printf("\n ");
    //A.raw_print(cout, "A");
    tau_old += dtau;
    //dtau.raw_print(cout, "dtau");
    // Checking the termination criterion

    if (i == 50)
    {
      printf("break\n");
    }
    tau_t = tau_old + EPS;
    TEMP = reshape((dtau / tau_t), 3, 2);
    TEMP = abs(TEMP);
    if (TEMP.max() < 0.001) {
      break;
    }

    //tau_old.raw_print(cout, "tau");

  }

  *residue = reshape((y - X * dtau), sizeI(0), sizeI(1));

  *tau_new = tau_old;

  *I2Warp = I2war;

  return;
}
mat influence(mat X,  double C) 
{
  mat Y = zeros(X.n_rows, X.n_cols);
  //vec idx = find(abs(X) < C);
  Y(find(abs(X) < C)) = X(find(abs(X) < C)) % ((C * C - X(find(abs(X) < C)) % X(find(abs(X) < C))) % (C * C - X(find(abs(X) < C)) % X(find(abs(X) < C))));
  // change elements of A greater than 0.5 to 1
  //A.elem( find(A > 0.5) ).ones();
  return Y;
}

double MEDIAN(vec X) 
{
  double m = 0;

  if (floor(X.n_elem / 2.0) == ceil(X.n_elem / 2.0)) 
  {
    m = (X(floor((X.n_elem - 1) / 2.0)) + X(ceil((X.n_elem - 1) / 2.0))) / 2.0;
  }
  else 
  {
    m = X(floor((X.n_elem - 1) / 2.0));
  }
  return m;
}

void regMGNC(mat img1, mat img2, vec tau, double numLevel, mat *img2warp, vec *tau_n, double paraTurkey0) {
  double C, paraTurkey = 0, ratio;
  mat I1, I2, weight, img2warpT = zeros(img1.n_rows, img1.n_cols), residue = zeros(img1.n_rows, img1.n_cols), OmegaOut;
  vec tau_old = zeros(tau.n_elem,1);

#if !SPEEDUP
  if (tau.is_empty() == 1) {
    tau = zeros(6, 1);
  }
  if (numLevel == 0) {
    numLevel = 3;
  }
#endif
  if (paraTurkey0 == 0) {
    paraTurkey0 = EPS;
  }

  for (int l = numLevel - 1; l >= 0; l--) {

    // Pyramid construction 
    //cout << pow(0.5, l) << endl;
    if (l != 0) {
      I1 = imResizeBicubic(img1, pow(0.5, l), (int) pow(2, (numLevel - 1)));
      I2 = imResizeBicubic(img2, pow(0.5, l), (int) pow(2, (numLevel - 1)));;
    }
    else
    {
      I1 = img1;
      I2 = img2;
    }

	for (int k = 0; k < 6 ; k++)
	{
	 for (int l = 0; l<6;l++)
	   {
	     printf("%.4f \t", I1(k,l));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
	 }
	 printf("\n");
	}
	printf("end I1\n");

	for (int k = 0; k < 6; k++)
	{
		for (int l = 0; l<6; l++)
		{
			printf("%.4f \t", I2(k, l));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
		}
		printf("\n");
	}
	printf("end I2\n");


    if (l == numLevel - 1) {
      weight = ones(I1.n_rows, I1.n_cols);
      tau(tau.n_elem - 2) *= pow(0.5, l - 1);
      tau(tau.n_elem - 1) *= pow(0.5, l - 1);
      C = max(max(abs(img1)));
    }
    else {
      // Very important part comes here - missing!!!!
      //weight = imResizeBicubic(weight, 0.5); 
      // probably wrong!!!!
      //weight = imResizeBicubic(weight, pow(0.5, l), (int) pow(2, (numLevel - 1)));
      /*ratio = (double) I1.n_rows / ((double) img1.n_rows);
      cout << ratio << endl;
      weight = imResizeBicubic(weight, ratio, (int) pow(2, (numLevel - 1)));
      */
      weight = ones(I1.n_rows, I1.n_cols);

      tau(tau.n_elem - 2) *= 2.0;
      tau(tau.n_elem - 1) *= 2.0;
      C *= 2.0; 
    }

    //cout << "weight " << weight.submat(0,0,5,5) << endl;

    while (1) {

      for (int iter = 1; iter <= 50; iter ++) {
        tau_old = tau;
        //cout << "I1 loop: " << I1.submat(0,0,5,5) << endl;
        //cout << "I2 loop: " << I2.submat(0,0,5,5) << endl;
        regImg(I1, I2, tau_old, weight, 1, &img2warpT, &tau, &residue, &OmegaOut);
    /*    for (int i = 0; i < 6; i++)
        {
          for (int j = 0; j < 6; j++) {
            printf("%f \t", img2warpT(i, j));
          }
          printf("\n");
        }*/

        //cout << "\n \n" << endl;

        //printf("stop\n");
        //cout << "im2warp loop: " << img2warpT.submat(0,0,5,5) << endl;
        weight = (influence(abs(residue), C) + EPS) / (abs(residue) + EPS);
        weight /= max(max(weight));

        if (max(abs((tau_old - tau) / (tau + EPS))) < 0.01) {
          break;
        }
      }

      //cout << "im2warp " << img2warpT.submat(0,0,5,5) << endl;

      paraTurkey = 4.7 * 1.48 * MEDIAN(abs(reshape(residue - MEDIAN(reshape(residue, residue.n_elem, 1)), residue.n_elem, 1)));

      if (C >= max(paraTurkey0, paraTurkey)) {
        C /= 2.0;
      }
      else {
        break;
      }
    }

  }

  *img2warp = img2warpT;

  *tau_n = tau;
}

void preAlign(cube ImData, cube *ImTrans, mat *tau_o) {
  int numFrame = ImData.n_slices;
  double tmp, numLevel;
  int IDcenter = floor(numFrame / 2.0 + 0.5);
  cube ImTransT = ImData;
  vec tau_temp = zeros(6,1);
  mat tau = zeros(6, numFrame), tmp1 = zeros(1,1);
  tmp1 = max(ImData.n_rows, ImData.n_cols);
  tmp1 = ceil(log2(tmp1 / 50.0)) + 1;
  numLevel = tmp1(0,0);

  //double numLevel1 = log2(max(ImData.n_rows, ImData.n_cols) / 50.0);

  for (int i = IDcenter - 2; i >= 0; i--) 
  {
    cout << "Frame" << i << endl;
    //cout << ImData(5,2,0);
    regMGNC(ImTransT.slice(i + 1), ImData.slice(i), tau.col(i + 1), numLevel, &ImTransT.slice(i), &tau_temp, 0);
    tau.col(i) = tau_temp;

    /*cout << "ImData" << ImData.subcube(0,0,i,5,5,i) << endl;
    cout << "ImTrans" << ImTransT.subcube(0,0,i,5,5,i) << endl;*/

  }

  for (int i = IDcenter; i < numFrame; i++) 
  {
    cout << "Frame" << i << endl;
    regMGNC(ImTransT.slice(i - 1), ImData.slice(i), tau.col(i - 1), numLevel, &ImTransT.slice(i), &tau_temp, 0);
    tau.col(i) = tau_temp;

    /*cout << "ImData" << ImData.subcube(0,0,i,5,5,i) << endl;
    cout << "ImTrans" << ImTransT.subcube(0,0,i,5,5,i) << endl;*/
  }

  *ImTrans = ImTransT;

  //cout << "imTrans & tau: " << ImTransT(1,1,1) << endl;
  //cout << tau << endl;

  *tau_o = tau;

  return;
}

void MyModel (mat X, vec isize, Opt options, mat *X_o, mat *L_o, mat *S_o, mat *G_o, mat *tau_o, mat *error_o) {
  int m = X.n_rows, n = X.n_cols, maxOuterIts = 10, cardinality;
  mat tau = zeros(6, n), S = zeros(m, n), Xtau = X, XtauS = zeros(m, n), XtauL = zeros(m, n);
  mat L = zeros(m, n), LP = zeros(m, n), G = zeros(m, n);
  mat weight;
  cube ImData = zeros(isize(0), isize(1), n), ImTrans = zeros(isize(0), isize(1), n), ImDtemp = zeros(isize(0), isize(1), n);
  mat Iwarp, dummy, OmegaO;
  mat U, V, LS = zeros(isize(0), isize(1)), XM = zeros(isize(0), isize(1));
  vec s, tau_temp = zeros(6,1);
  uvec index;
  vec error = zeros(maxOuterIts, 1);


  if (options.tol == 0)
  {
    options.tol = 1e-4;
  }

  if (options.rank == 0)
  {
    options.rank = 1;
  }



  // Pre-alignment
  if (options.moving == 1)
  {
    for (int h = 0; h < n; h++) 
    {
      ImData.slice(h) = reshape(X.col(h), isize(0), isize(1));
      //cout << ImData(0,0,0) << endl;
      //cout << X(0,0) << endl;

    }
    //ImDtemp(find(ImData <= 0)).fill(0);
    //ImDtemp(find(ImData > 0)) = round(ImData(find(ImData > 0)) * 255);

    //cout << "ImData" << ImData.subcube(0,0,0,5,5,0) << endl;
    preAlign(ImData, &ImTrans, &tau); // output is not the same as in matlab

	//for (int k = 0; k < 7 ; k++)
	//{
	// for (int l = 0; l<7;l++)
	//   {
	//     printf("%.4f ", ImData(k,l,0));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
	// }
	// printf("\n");
	//}
	//printf("end ImData\n");

	//for (int k = 0; k < 7; k++)
	//{
	//	for (int l = 0; l<7; l++)
	//	{
	//		printf("%.4f ", ImTrans(k, l, 0));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
	//	}
	//	printf("\n");
	//}
	//printf("end ImTrans\n");


    //cout << "ImTrans" << ImTrans.subcube(0,0,0,5,5,0) << endl;

    for (int i = 0; i < n; i++)
    {
      Xtau.col(i) = reshape(ImTrans.slice(i), m, 1);
    }
  }

  L = Xtau;

  for (int outerIts = 1; outerIts <= maxOuterIts; outerIts++)
  {

    //cout << outerIts << endl;
    if (options.moving == 1)
    {

		for (int i = 0; i < n; i++)
      {
        LS = reshape(L.col(i), isize(0), isize(1)) + reshape(S.col(i), isize(0), isize(1)); 
        XM = reshape(X.col(i), isize(0), isize(1)), tau.col(i);
        //regImg((reshape(L.col(i), isize(0), isize(1)) + reshape(S.col(i), isize(0), isize(1))), reshape(X.col(i), isize(0), isize(1)), tau.col(i), weight, 0, &Iwarp, &tau_temp, &dummy, &OmegaO);
        regImg(LS, XM, tau.col(i), weight, 0, &Iwarp, &tau_temp, &dummy, &OmegaO);
		//for (int k = 0; k < 7 ; k++)
		//{
		// for (int l = 0; l<7;l++)
		//   {
		//     printf("%.4f ", Iwarp(k,l));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
		// }
		// printf("\n");
		//}
		//printf("end\n");

		//for (int k = 0; k < 7; k++)
		//{
		//	for (int l = 0; l<7; l++)
		//	{
		//		printf("%.4f ", LS(k, l));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
		//	}
		//	printf("\n");
		//}
		//printf("end\n");

		//for (int k = 0; k < 7; k++)
		//{
		//	for (int l = 0; l<7; l++)
		//	{
		//		printf("%.4f ", XM(k, l));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
		//	}
		//	printf("\n");
		//}
		//printf("end\n");

        tau.col(i) = tau_temp;
        Xtau.col(i) = reshape(Iwarp, m, 1);
      }
    }

    // Calculating L that solves the minimisation problem
    XtauS = Xtau - S;

    /*cout << "X" << X.submat(0,0,5,5) << endl;
    cout << "L" << L.submat(0,0,5,5) << endl;
    cout << "Xtau" << Xtau.submat(0,0,5,5) << endl;*/
    //cout << "tau" << tau << endl;
    //clock_t begin = clock();
    svd_econ(U, s, V, XtauS);
    //clock_t end = clock();



    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //cout << "time elapsed: " << elapsed_secs << endl;

    LP.fill(0);
    for (int ii = 0; ii < options.rank; ii++) 
    {
      LP += s(ii, ii) * U.col(ii) * trans(V.col(ii));
    }
    L = LP;

    //if (outerIts == 10) {
    //for (int i = 0; i < 7 ; i++)
    //{
    // for (int j = 0; j<7;j++)
    //   {
    //     printf("%.4f ", LP(i,j));//cout <<std::setw(5)<< LP.submat(0,0,10,10);
    // }
    // printf("\n");
    //}
    //}

    // Calculating S that solves the minimisation problem
    XtauL = Xtau - L;
    //begin = clock();
    index = sort_index(abs(vectorise(XtauL)), "descend");
    //end = clock();
    //elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //cout << "time elapsed: " << elapsed_secs << endl;

    cardinality = ceil((X.n_elem) * options.card / 100);


    //cout << index.rows(2,78) << endl;
    //cout << cardinality << endl;
    S.fill(0);
    S(index.rows(0, cardinality - 1)) = XtauL(index.rows(0, cardinality - 1));

    // Calculating the error
    G = Xtau - L - S;
    error(outerIts - 1) = norm(G, 1) / norm(X, 1);

    //cout << error(outerIts - 1) << endl;

    if (error(outerIts - 1) <= options.tol)
    {
      break;
    }

  }

  *L_o = L; 
  *S_o = S; 
  *G_o = G; 
  *X_o = X;
  *error_o = error;
  *tau_o = tau;

}

int main(int argc, char** argv) {

  double elapsed_secs;
  clock_t begin, end;
  int m, n, a, b, tmp;

  cout << setprecision(2);
  /*mat mask;
  mat phi_x, phi_y;*/
  /*vec dimen = ones<vec>(2,1);
  vec sigma = ones<vec>(2,1);

  vec dimen1 = ones<vec>(1,1);
  vec sigma1 = ones<vec>(1,1);

  vec V3 = zeros<vec>(5,1);
  vec V6 = zeros<vec>(6,1);

  mat M4 = zeros<mat>(4,4);
  mat M3 = zeros<mat>(3,3);*/
  mat M5 = zeros<mat>(5,5);
  vec V6 = zeros<vec>(6,1);

  Opt option;
  option.rank = 1;
  option.card = 30;
  option.tol = 1e-3;
  option.moving = 1;

  /*mat X_iii;
  cube X_ii;
  bool status = X_ii.load("pedestrian.mat", arma_ascii);


  cout << X_ii;*/
  vec isize = zeros(3, 1), error = zeros(12,1);

  // Reading the input file
  const char *fileName = "data.txt";
  FILE *FID;

  FID = fopen (fileName,"r");
  fscanf(FID, "%d %d %d", &a, &b, &n);

  cout << a << endl;

  m = a * b;

  isize(0) = (double)a;
  isize(1) = (double)b;
  isize(2) = (double)n;

  mat X_i = zeros(m, n);

  cout << isize << endl;
  cout << m << endl;

  // Measuring the time
  begin = clock();

  for (int i = 0; i < m; ++i) 
  {
    for (int j = 0; j < n; ++j) 
    {
      fscanf(FID, "%d", &tmp);
      X_i(i, j) = (double)tmp;
    } 
  }

  end = clock();
  elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  cout << "time elapsed: " << elapsed_secs << endl;

  fclose(FID);

  /*for (int i = 0; i < n; i++) {
  X_iii.col(i) = vectorise(X_ii.slice(i));
  }
  cout << X_iii(5,5) << endl;*/

  mat X = zeros(isize(0) * isize(1), isize(2)), S = zeros(isize(0) * isize(1), isize(2)), L = zeros(isize(0) * isize(1), isize(2)), G = zeros(isize(0) * isize(1), isize(2)), tau = zeros(6, isize(2));

  begin = clock();

  MyModel(X_i, isize, option, &X, &L, &S, &G, &tau, &error);

  end = clock();
  elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  cout << elapsed_secs << endl;


  X.save("X.mat", raw_ascii);
  L.save("L.mat", raw_ascii);
  S.save("S.mat", raw_ascii);
  G.save("G.mat", raw_ascii);

  /*double k = 1e-17;
  for (int i = 1; i <= 500; i++) {
  k += k;
  }*/

  // mat test= zeros(5,5);
  // test(0,1) = 1;
  // test(0,3) = 1;
  // //test(0,5) = 1;
  // //test(0,7) = 1;

  // test(2,1) = 1;
  // test(2,3) = 1;
  // //test(2,5) = 1;
  // //test(2,7) = 1;

  // test(4,1) = 1;
  // test(4,3) = 1;
  // //test(4,5) = 1;
  //// test(4,7) = 1;

  // //test(6,1) = 1;
  // //test(6,3) = 1;
  // //test(6,5) = 1;
  // //test(6,7) = 1;

  // test(1,0) = 1;
  // test(1,2) = 1;
  // test(1,4) = 1;
  // //test(1,6) = 1;
  // 
  // test(3,0) = 1;
  // test(3,2) = 1;
  // test(3,4) = 1;
  // //test(3,6) = 1;
  // 
  // //test(5,0) = 1;
  // //test(5,2) = 1;
  // //test(5,4) = 1;
  // /*test(5,6) = 1;*/
  // 
  // /*test(7,0) = 1;
  // test(7,2) = 1;
  // test(7,4) = 1;
  // test(7,6) = 1;*/
  // 
  ///* test(3,0) = 1;
  // test(3,2) = 1;
  // test(3,4) = 1;
  // test(5,0) = 1;
  // test(5,2) = 1;
  // test(5,4) = 1;*/

  // cout << test << endl;

  // mat testO;
  // testO = imResizeBicubic(test, 0.25);
  // //double outp = cubicInterpolation(0.5, 0, 1, 0, 1);

  // cout << testO << endl;

  /*cout.precision(32);
  cout.setf(ios::fixed);*/

  //cout << k << endl;

  /*M3(0,0) = 1;
  M3(0,1) = 2;
  M3(0,2) = 3;
  M3(1,0) = 4;
  M3(1,1) = 5;
  M3(1,2) = 6;
  M3(2,0) = 7;
  M3(2,1) = 8;
  M3(2,2) = 9;
  */
  M5(0,0) = 2;
  M5(0,1) = 3;
  M5(0,2) = 4;
  M5(0,3) = 5;
  M5(0,4) = 4;
  M5(1,0) = 3;
  M5(1,1) = 4;
  M5(1,2) = 5;
  M5(1,3) = 3;
  M5(1,4) = 2;
  M5(2,0) = 3;
  M5(2,1) = 4;
  M5(2,2) = 5;
  M5(2,3) = 4;
  M5(2,4) = 3;
  M5(3,0) = 4;
  M5(3,1) = 5;
  M5(3,2) = 3;
  M5(3,3) = 2;
  M5(3,4) = 3;
  M5(4,0) = 4;
  M5(4,1) = 5;
  M5(4,2) = 4;
  M5(4,3) = 3;
  M5(4,4) = 4;

  /*
  double C = 3;
  mat Y;
  Y = influence(M5, C);

  cout << Y << endl;
  */
  /*M4(0,0) = 1;
  M4(0,1) = 2;
  M4(0,2) = 3;
  M4(0,3) = 4;
  M4(1,0) = 5;
  M4(1,1) = 6;
  M4(1,2) = 7;
  M4(1,3) = 8;
  M4(2,0) = 9;
  M4(2,1) = 10;
  M4(2,2) = 11;
  M4(2,3) = 12;
  M4(3,0) = 13;
  M4(3,1) = 14;
  M4(3,2) = 15;
  M4(3,3) = 16;

  V3(0) = -2;
  V3(1) = -1;
  V3(3) = 1;
  V3(4) = 2;
  */
  V6(0) = 0.3;
  V6(1) = 2.2;
  V6(2) = 3.05;
  V6(3) = 2.4;
  V6(4) = 1.8;
  V6(5) = 4;

  //cout << M3 << endl;

  //vec dimen2 = ones<vec>(3,1);
  //vec sigma2 = ones<vec>(3,1);

  /*dimen1(0) = 7;
  sigma1(0) = 0;*/

  /*dimen2(0) = 5; 
  dimen2(1) = 5;
  dimen2(2) = 5;

  sigma2(0) = 2.5; 
  sigma2(1) = 2;
  sigma2(2) = 1.5;*/

  /*dimen(0) = 7; 
  dimen(1) = 5;

  sigma(0) = 0.8;
  sigma(1) = 0.5;*/

  //mask = gSmooth(M3, sigma);

  mat I2 = M5 + 2;

  //cout << M5 << endl;
  //cout << I2 << endl;
  //cout << V3 << endl;
  //mask = imFilter(M5, V3, 1);
  //cout << phi_x << endl;
  //getGradient(M5, &phi_x, &phi_y, sigma);
  //mask = ifftshift(M);

  //warpImg(M5, V6, 0,  &phi_x, &phi_y);

  //meshgrid(4,5,&phi_x, &phi_y);

  /*cout << phi_x << endl;
  cout << phi_y << endl;*/

  vec weight = ones(M5.n_elem,1);

  mat I2Warp, residue, OmegaOut;
  vec tau_new;
  //void regImg (mat I1, mat I2, vec tau_old, vec weight, int maxIts, mat *I2Warp, mat *tau_new, mat *residue, mat *OmegaOut)
  regImg(M5, I2, V6, weight, 50, &I2Warp, &tau_new, &residue, &OmegaOut);

  cout << I2Warp << endl;
  cout << tau_new << endl;
  cout << residue << endl;
  cout << OmegaOut << endl;

  //cout << MEDIAN(M5.col(0)) << endl;

  getchar();
  return 0;
}