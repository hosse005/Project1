/* Random number driven Matrix generator */
#include <iostream>
#include <fstream>
#include <cstdlib>

#define DIM 512

typedef float array_type[DIM][DIM];

class CMatrixGenerator
{
public:
    CMatrixGenerator();
    ~CMatrixGenerator();
        
    float* getMatrixHandle();
    void matrixDumptoCSV();

private:
    array_type m_fMatrix = {{0}};
    std::ofstream m_file;
};
