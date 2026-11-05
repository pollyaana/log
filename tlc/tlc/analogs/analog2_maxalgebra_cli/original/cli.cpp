#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <random>
#include <ctime>

#include <eigen3/Eigen/Dense>

#include "../lib/struct_a.h"
#include "../lib/struct_b.h"

static Eigen::IOFormat LaTeXFormat(
    Eigen::FullPrecision, Eigen::DontAlignCols,
    " & ", "\\\\\n", "", "", "\\begin{pmatrix}\n", "\n\\end{pmatrix}");

template <typename T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

typedef MatrixT<MaxAlgebraA> MatrixA;
typedef MatrixT<MaxAlgebraB> MatrixB;

template <typename T>
T SpectralRadius(MatrixT<T> &matrix)
{
    T result = matrix.trace();
    MatrixT<T> matrix_pow(matrix);
    for (int i = 2; i <= matrix.cols(); i++)
    {
        matrix_pow *= matrix;
        result += matrix_pow.trace().Root(i);
    }
    return result;
}

template <typename T>
MatrixT<T> Kleene(const MatrixT<T> &matrix)
{
    auto n = matrix.cols();
    MatrixT<T> result(MatrixT<T>::Identity(n, n));
    MatrixT<T> matrix_pow(matrix);
    for (int i = 1; i < matrix.cols(); i++)
    {
        result += matrix_pow;
        matrix_pow *= matrix;
    }
    return result;
}

using namespace std;

// n - matrix size
// m - num limit
// print n m time_structure_A time_structure_B (in ms)
void test(uint n, uint m, std::random_device &dev, ofstream &output)
{
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, m);

    MatrixA A(n, n);
    MatrixB B(n, n);
    for (uint i = 0; i < n; i++)
    {
        A(i, i) = MaxAlgebraA(1);
        B(i, i) = MaxAlgebraB(1);
        for (uint j = 0; j < i; j++)
        {
            int numerator = dist(rng);
            int denominator = dist(rng);
            A(i, j) = StructA(numerator, denominator);
            B(i, j) = StructB(numerator, denominator);
            A(j, i) = StructA(denominator, numerator);
            B(j, i) = StructB(denominator, numerator);
        }
    }

    output << n << " " << m << " ";
    std::cout << n << " " << m << " ";

    std::clock_t startA = std::clock();
    MatrixA resultA(Kleene((A / SpectralRadius(A)).eval()));
    std::clock_t stopA = std::clock();

    output << (stopA - startA) / (double)(CLOCKS_PER_SEC / 1000) << " ";
    std::cout << (stopA - startA) / (double)(CLOCKS_PER_SEC / 1000) << " ";

    std::clock_t startB = std::clock();
    MatrixB resultB(Kleene((B / SpectralRadius(B)).eval()));
    std::clock_t stopB = std::clock();

    output << (stopB - startB) / (double)(CLOCKS_PER_SEC / 1000) << endl;
    std::cout << (stopB - startB) / (double)(CLOCKS_PER_SEC / 1000) << endl;
}

int main(int argc, char *argv[])
{
    string output_file = "out.txt";
    if (argc >= 2)
    {
        output_file = string(argv[1]);
    }

    std::random_device dev;

    ofstream output;
    output.open(output_file);
    // output.open(output_file, std::ios_base::app);

    output << "\"n\" \"m\" \"A\" \"B\"" << endl;
    cout << "\"n\" \"m\" \"A\" \"B\"" << endl;

    for (int n = 10; n <= 100; n += 5)
    {
        for (int m = 100; m <= 100; m *= 10)
        {
            for (int rep = 0; rep < 2; rep++)
            {
                test(n, m, dev, output);
            }
        }
    }

    output.close();

    return 0;
}