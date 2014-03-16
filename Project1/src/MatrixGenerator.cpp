/* Random Number driven 512x512 matrix generator */

#include "MatrixGenerator.h"

CMatrixGenerator::CMatrixGenerator()
{
	std::cout << "Invoked CMatrixGenerator constructor!" << std::endl;
	// Initialize the pseudo random number generator
	srand(static_cast<unsigned>(time(0)));

    // Generate random values
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        	m_fMatrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
}

CMatrixGenerator::~CMatrixGenerator()
{
	std::cout << "Invoked CMatrixGenerator destructor!" << std::endl;
}

void CMatrixGenerator::matrixDumptoCSV()
{
	// Open CSV file for matrix dump
	m_file.open("matrixDump.csv", std::ios::out | std::ios::trunc);

	if (!m_file.is_open())
	{
		std::cout << "Error opening matrix dump file.." << std::endl;
		return;
	}

	for (int i = 0; i < DIM; i++)
	{
		for (int j = 0; j < DIM; j++)
			m_file << m_fMatrix[i][j] << ",";

		m_file << "\n";
	}

	m_file.close();
}

float* CMatrixGenerator::getMatrixHandle()
{
	return &m_fMatrix[0][0];
}
