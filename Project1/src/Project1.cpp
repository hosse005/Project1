/*
 ============================================================================
 Name        : Project1.cpp
 Author      : Evan Hosseini
 Class		 : CS708
 Date		 : 3-Mar-2014
 Description : Project 1 Main
 ============================================================================
 */
#include <math.h>
#include "mpi.h"
#include <iostream>
#include "MatrixGenerator.h"

#define ROOT 0

//#define DEBUG

using namespace std;

// Serial matrix multiplication calculates reference result
void matrixMult(array_type &A, array_type &B, float *pC)
{
	for (int i = 0; i < DIM; i++)				// Output stored in row-major order
		for (int j = 0; j < DIM; j++, pC++)
		{
			*pC = 0.0f;
			for (int k = 0; k < DIM; k++)
				*pC += A[i][k] * B[k][j];
		}
}

// Sorts rank ordered MPIResult matrix from gather operation
// for comparison with the reference result
// Note: Need to pass this function the base of the MPIResult array (pMPIOrig)
void MPIResultSort(float *pMPIResult, float *pSortedMPIResult, int &s, int &SUB_DIM)
{
	// Cache reference to base pointer for reset on each loop iteration
	float *pMPIOrig = pMPIResult;

	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++, pSortedMPIResult++)
		{
			// First, determine processor (rank) input belongs to
			int x = j / SUB_DIM;
			int y = i / SUB_DIM;
			int m_rank = y * s + x;		// Processor rank was configured in row-major order

			// Find appropriate index w/in the rank and assign to SortedMPIResult
			pMPIResult += (m_rank * SUB_DIM * SUB_DIM + i % SUB_DIM * SUB_DIM + j % SUB_DIM);
#ifdef DEBUG
			cout << "Generated offset @ element " << i*DIM + j << "  =>  " <<
					m_rank*SUB_DIM*SUB_DIM + i%SUB_DIM*SUB_DIM + j%SUB_DIM << "; i = " << i <<
					", j = " << j << ", m_rank = " << m_rank << endl;
#endif
			*pSortedMPIResult = *pMPIResult;
			pMPIResult = pMPIOrig;
		}
}

// Checks every element of reference and MPI results
bool bCheckResult(float *pGldResult, float *pMPIResult)
{
	float fErrorSum = 0.0f;

	// Now subtract the reference from the MPI calculated result
	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++, pGldResult++, pMPIResult++)
		{
			fErrorSum += fabs(*pGldResult - *pMPIResult);
			//cout << "Difference @ element " << i*DIM + j << " = " << fabs(*pGldResult - *pMPIResult) << endl;
		}
	cout << "Cumulative Error in MPI Matrix multiplication result = " << fErrorSum << endl;

	// Check for differences and report result
	return fErrorSum == 0.0f ? true : false;
}

// Index hash for each processor to use for finding appropriate input range to use in multiplication
float fMatrixHashAndMult(float *a, float *b, int &rank, int &s, int &SUB_DIM, int p, int q, int r)
{
	a += (rank / s * SUB_DIM * DIM + DIM * p + r);
	b += (rank % s * SUB_DIM + DIM * r + q);

#ifdef DEBUG
	if (p == 7 && q == 7 && rank == 3)
		cout << "fMatrixHashAndMult(): a = " << *a << ",	b = " << *b <<
				",	Hash offset for a = " << (rank / s * SUB_DIM * DIM + DIM * p + r) <<
				", 	Hash offset for b = " << (rank % s * SUB_DIM + DIM * r + q) << endl;
#endif

	return *a * *b;
}

 
int main(int argc, char *argv[])
{

	int rank, size;												// MPI parameters
	CMatrixGenerator *pCMatrixGeneratorA, *pCMatrixGeneratorB;	// input generator handles
	float *pA, *pB;												// input matrix generator access pointers
	array_type a,b;												// input matrices a,b;
	float *pCsub, *pCsubOrig;									// sub matrix accessors
	float *pMPIResult, *pMPIOrig, *pGldResult, *pGldOrig;		// ROOT memory accessors
	float *pSortedMPIResult, *pSortedOrig;
	int s = 1;													// Default matrix subdivision parameter
	double dStartTime, dEndTime;								// Time stamps for performance calculations

	MPI::Init(argc, argv);

	size = MPI::COMM_WORLD.Get_size();

	rank = MPI::COMM_WORLD.Get_rank();

	s = atoi(argv[1]);			// Note: cstdlib.h included in MatrixGenerator.h, not so obvious..

	// MPI root node control
	if (rank == ROOT )
	{
		// Check for s parameter input and check for validity
		if (argc > 1 && s != 1 && s != 2 && s != 4 && s != 8)
		{
			cout << "Only acceptable matrix subdivision parameters are 1, 2, 4, or 8 !" << endl;
			return 0;
		}

		// Enforce that mpirun is invoked with s * s processors
		if (s * s != size)
		{
			cout << "Must provide mpirun with s^2 processors!" << endl << "(e.g. mpirun -n 16 ./Project1 4)" << endl;
			return 0;
		}

		// Generate the input matrices
		pCMatrixGeneratorA = new CMatrixGenerator;
		pCMatrixGeneratorB = new CMatrixGenerator;

		// Assign generator outputs to the global inputs
		pA = pCMatrixGeneratorA->getMatrixHandle();
		pB = pCMatrixGeneratorB->getMatrixHandle();

		// Copy matrix generator contents to input and outputs
		for (int i = 0; i < DIM; i++)
			for (int j = 0; j < DIM; j++, pA++, pB++)	// Note: C++ stores matrices in row-major order
			{
				a[i][j] = *pA;
				b[i][j] = *pB;
			}
	}

	// Generate start time stamp
	if (rank == ROOT)
		dStartTime = MPI::Wtime();

	// Broadcast the input matrices to all processors
	MPI::COMM_WORLD.Bcast(&a, DIM * DIM, MPI::FLOAT, ROOT);
	MPI::COMM_WORLD.Bcast(&b, DIM * DIM, MPI::FLOAT, ROOT);

	// Derive sub-matrix dimensions from size parameter
	int SUB_DIM = DIM / s;
	int sub_sz = SUB_DIM * SUB_DIM;
	int sz = DIM * DIM;
	pCsub = new float[sub_sz];
	pCsubOrig = pCsub;

	// Loop through sub-matrices with rank based index and attain sub-matrix c_sub
	for (int p = 0; p < SUB_DIM; p++)
		for (int q = 0; q < SUB_DIM; q++, pCsub++)
		{
			//c_sub[p][q] = 0.0f;
			*pCsub = 0.0f;
			for (int r = 0; r < DIM; r++)
				*pCsub += fMatrixHashAndMult(&a[0][0], &b[0][0], rank, s, SUB_DIM, p, q, r);
#ifdef DEBUG
			if (rank == 0) { printf("c_sub[%d][%d] = %2.5f \n", p, q, c_sub[p][q]); }
#endif
		}

	// Gather results from all processors
	if (rank == ROOT)
	{
		// Allocate memory for result matrices and store reference for cleanup
		pMPIResult = new float[sz];
		pMPIOrig = pMPIResult;

		pSortedMPIResult = new float[sz];
		pSortedOrig = pSortedMPIResult;

		pGldResult =  new float[sz];
		pGldOrig = pGldResult;

		if (pMPIResult == nullptr || pGldResult == nullptr)
		{
			cout << "Memory allocation failure.." << endl;
			return 0;
		}
	}

	// Synchronize and reset pointer before allowing root to grab sub matrix results
	MPI::COMM_WORLD.Barrier();

	pCsub = pCsubOrig;

	MPI::COMM_WORLD.Gather(pCsub, sub_sz, MPI::FLOAT, pMPIResult, sub_sz, MPI::FLOAT, ROOT);


#ifdef DEBUG
	if (rank == ROOT)
	{
		int non_zero_count = 0;
		pMPIResult = pMPIOrig;
		for (int x = 0; x < DIM * DIM; x++, pMPIResult++)
		{
			if (*pMPIResult != 0)
			{
				non_zero_count++;
				cout << "pMPIResult = " << *pMPIResult << " @element = " << x << endl;
			}
		}
		cout << "Non-zero MPIResult count = " << non_zero_count << endl;
		pMPIResult = pMPIOrig;
	}
#endif

	// Master root result check and cleanup
	if (rank == ROOT)
	{
		dEndTime = MPI::Wtime();
		cout << "MPI job run time = " << dEndTime - dStartTime << " s" << endl;

		// Calculate reference result and store in pGldResult
		matrixMult(a, b, pGldResult);

		// Reset pointers to respective origin before comparing
		pMPIResult = pMPIOrig;
		pGldResult = pGldOrig;

		// Sort MPI Result matrix before comparing to reference
		MPIResultSort(pMPIResult, pSortedMPIResult, s, SUB_DIM);
		pSortedMPIResult = pSortedOrig;

		if (bCheckResult(pGldResult, pSortedMPIResult))
			cout << "MPI matrix multiplication success!" << endl;
		else
			cout << "MPI matrix multiplication failed, see above line for cumulative error total" << endl;

		// Reset pointers before releasing resources
		pMPIResult = pMPIOrig;
		pSortedMPIResult = pSortedOrig;
		pGldResult = pGldOrig;
		delete[] pMPIResult;
		delete[] pSortedMPIResult;
		delete[] pGldOrig;

		delete pCMatrixGeneratorA;
		delete pCMatrixGeneratorB;
	}

	delete[] pCsub;
	MPI::Finalize();

	return 0;
}

