/*
 * 	BMR_BGW_aux.cpp
 * 
 *      Author: Aner Ben-Efraim, Satyanarayana
 * 	
 * 	year: 2016
 *	
 *	Modified for crypTFlow and Aramis by Mayank Rathee 
 */

#include "tools.h"
#include <stdint.h>
#include <mutex> 
#include <bitset>
#include "../utils_sgx_port/utils_print_sgx.h"
#include "../utils_sgx_port/utils_malloc_sgx.h"
#include "../utils_sgx_port/utils_abort_sgx.h"
using namespace std;

#define NANOSECONDS_PER_SEC 1E9

//#define SLOW_MULT

bool print_mode = false;
//For time measurements
clock_t tStart;
//struct timespec requestStart, requestEnd;
bool alreadyMeasuringTime = false;

int roundComplexitySend = 0;
int roundComplexityRecv = 0;
bool alreadyMeasuringRounds = false;

//Some globals for threaded sgx matmul
int global_rows, global_common_dim, global_columns;
uint64_t *left_matrix_location, *right_matrix_location, *result_matrix_location;
int start_matmul[NO_CORES];
int end_matmul[NO_CORES];

// For sgx specific threaded populate aes functions
uint64_t* global_populate_target;
string global_r_type = "";
int start_populate_target[NO_CORES];
int end_populate_target[NO_CORES];
uint64_t* global_sharesModuloOdd_target1, *global_sharesModuloOdd_target2;
smallType* global_sharesModuloOdd_target3;
smallType* global_sharesOfBits_target1, *global_sharesOfBits_target2;
uint64_t* global_sharesOfBits_target3;
uint64_t* global_pop_flat_vec_target1;
int global_which_party[NO_CORES];


/************************************ Some MatMul functions ************************/

void matmul_sgx_threaded(int worker_thread_num){
	//print_integer(worker_thread_num);
	int rows = global_rows;
	int common_dim = global_common_dim;
	int columns = global_columns;
	int start_row = start_matmul[worker_thread_num];
	int end_row = end_matmul[worker_thread_num];
	int r_temp = 0;
	int row_temp = 0;
	int col_temp = 0;
	uint64_t cumu_ele = 0;
	for(int r=start_row; r<end_row; r++){
		r_temp = r*columns;
		row_temp = r*common_dim;
		for(int col=0; col<columns; col++){
			cumu_ele = 0;
			col_temp = col*common_dim;
			for(int com=0; com<common_dim; com++){
				cumu_ele += (*(left_matrix_location + (row_temp + com))) * (*(right_matrix_location + (col_temp + com)));
			}
			*(result_matrix_location + (r_temp + col)) = cumu_ele;
		}
	}

}

#ifdef MATMUL_THREADED


//Multithreaded version of 3-nested matmul for SGX
#ifdef SPLIT_LOCAL_MATMUL
void matrixMultEigenNoThread(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<rows; i++){
		offset = i*common_dim;
		for(int j=0; j<common_dim; j++){
			left_matrix_row_major[offset + j] = a[offset + j]; //Row major filling.
		}
	}
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}
	int ra, rb, rc;
	aramisSecretType cumu_ele;
	for(int r=0; r<rows; r++){
		ra = r*common_dim;
		rc = r*columns;
		for(int col=0; col<columns; col++){
			rb = col*common_dim;
			cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += left_matrix_row_major[ra + com] * right_matrix_col_major[rb + com];
			}
			c[rc + col] = cumu_ele;
		}
	}
	
}
void matrixMultEigenSplitBackend(vector<vector<aramisSecretType>> &a, 
		vector<vector<aramisSecretType>> &b, 
		vector<vector<aramisSecretType>> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b);

void matrixMultEigenSplitBackend(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b);

void matrixMultEigenSplitBackend(uint64_t* a, 
		uint64_t* b, 
		uint64_t* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b);

void matrixMultEigen(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{

#ifdef DONT_SPLIT_LOCAL_MATMUL
		matrixMultEigenSplitBackend(a, b, c, rows, common_dim, columns, transpose_a, transpose_b);
#else
	if(common_dim*columns*8 <= 128*1024){
		matrixMultEigenNoThread(a, b, c, rows, common_dim, columns, transpose_a, transpose_b);
		return;
	}
	int columns_per_chunk = columns/LOCAL_MATMUL_CHUNK_COUNT;			
	int columns_last_chunk = columns - (columns_per_chunk*LOCAL_MATMUL_CHUNK_COUNT);
	
	vector<aramisSecretType> b_cut(common_dim*columns_per_chunk, 0);
	vector<aramisSecretType> c_cut(rows*columns_per_chunk, 0);
	
	for(int i=0; i<LOCAL_MATMUL_CHUNK_COUNT; i++){
		int col_start = columns_per_chunk*i;
		for(int j=0; j<common_dim; j++){
			for(int k=col_start; k<col_start+columns_per_chunk; k++){
				b_cut[(j*columns_per_chunk)+k-col_start] = b[(j*columns)+k];
			}	
		}
		matrixMultEigenSplitBackend(a, b_cut, c_cut, rows, common_dim, columns_per_chunk, transpose_a, transpose_b);
		for(int j=0; j<rows; j++){
			for(int k=col_start; k<col_start+columns_per_chunk; k++){
				c[(j*columns)+k] = c_cut[(j*columns_per_chunk)+k-col_start];
			}	
		}
	
	}

	b_cut = vector<aramisSecretType>();
	c_cut = vector<aramisSecretType>();

	vector<aramisSecretType> b_cut_last(common_dim*columns_last_chunk, 0);
	vector<aramisSecretType> c_cut_last(rows*columns_last_chunk, 0);

	if(columns_last_chunk > 0){
		int col_start = columns_per_chunk*LOCAL_MATMUL_CHUNK_COUNT;
		for(int j=0; j<common_dim; j++){
			for(int k=col_start; k<col_start+columns_last_chunk; k++){
				b_cut_last[(j*columns_last_chunk)+k-col_start] = b[(j*columns)+k];
			}	
		}
		matrixMultEigenSplitBackend(a, b_cut_last, c_cut_last, rows, common_dim, columns_last_chunk, transpose_a, transpose_b);
		for(int j=0; j<rows; j++){
			for(int k=col_start; k<col_start+columns_last_chunk; k++){
				c[(j*columns)+k] = c_cut_last[(j*columns_last_chunk)+k-col_start];
			}	
		}

	}
#endif

}

void matrixMultEigen(vector<vector<aramisSecretType>> &a_in, 
		vector<vector<aramisSecretType>> &b, 
		vector<vector<aramisSecretType>> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
#ifdef DONT_SPLIT_LOCAL_MATMUL_2
	matrixMultEigenSplitBackend(a_in, b, c, rows, common_dim, columns, transpose_a, transpose_b);
#else
	int columns_per_chunk = columns/LOCAL_MATMUL_CHUNK_COUNT_2;			
	int columns_last_chunk = columns - (columns_per_chunk*LOCAL_MATMUL_CHUNK_COUNT_2);
	
	vector<aramisSecretType> a(rows*common_dim, 0);
	vector<aramisSecretType> b_cut(common_dim*columns_per_chunk, 0);
	vector<aramisSecretType> c_cut(rows*columns_per_chunk, 0);

	for(int i=0; i<rows; i++){
		for(int j=0; j<common_dim; j++){
			a[i*common_dim + j] = a_in[i][j];
		}
	}

	for(int i=0; i<LOCAL_MATMUL_CHUNK_COUNT_2; i++){
		int col_start = columns_per_chunk*i;
		for(int j=0; j<common_dim; j++){
			for(int k=col_start; k<col_start+columns_per_chunk; k++){
				b_cut[(j*columns_per_chunk)+k-col_start] = b[j][k];
			}	
		}
		matrixMultEigenSplitBackend(a, b_cut, c_cut, rows, common_dim, columns_per_chunk, transpose_a, transpose_b);
		for(int j=0; j<rows; j++){
			for(int k=col_start; k<col_start+columns_per_chunk; k++){
				c[j][k] = c_cut[(j*columns_per_chunk)+k-col_start];
			}	
		}
	
	}

	b_cut = vector<aramisSecretType>();
	c_cut = vector<aramisSecretType>();

	vector<aramisSecretType> b_cut_last(common_dim*columns_last_chunk, 0);
	vector<aramisSecretType> c_cut_last(rows*columns_last_chunk, 0);

	if(columns_last_chunk > 0){
		int col_start = columns_per_chunk*LOCAL_MATMUL_CHUNK_COUNT_2;
		for(int j=0; j<common_dim; j++){
			for(int k=col_start; k<col_start+columns_last_chunk; k++){
				b_cut_last[(j*columns_last_chunk)+k-col_start] = b[j][k];
			}	
		}
		matrixMultEigenSplitBackend(a, b_cut_last, c_cut_last, rows, common_dim, columns_last_chunk, transpose_a, transpose_b);
		for(int j=0; j<rows; j++){
			for(int k=col_start; k<col_start+columns_last_chunk; k++){
				c[j][k] = c_cut_last[(j*columns_last_chunk)+k-col_start];
			}	
		}

	}
#endif

}

void matrixMultEigen(uint64_t* a_in, 
		uint64_t* b, 
		uint64_t* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
#ifdef DONT_SPLIT_LOCAL_MATMUL_2
	matrixMultEigenSplitBackend(a_in, b, c, rows, common_dim, columns, transpose_a, transpose_b);
#else
	int columns_per_chunk = columns/LOCAL_MATMUL_CHUNK_COUNT_2;			
	int columns_last_chunk = columns - (columns_per_chunk*LOCAL_MATMUL_CHUNK_COUNT_2);
	
	vector<aramisSecretType> a(rows*common_dim, 0);
	vector<aramisSecretType> b_cut(common_dim*columns_per_chunk, 0);
	vector<aramisSecretType> c_cut(rows*columns_per_chunk, 0);

	for(int i=0; i<rows; i++){
		for(int j=0; j<common_dim; j++){
			a[i*common_dim + j] = a_in[i*common_dim +j];
		}
	}

	for(int i=0; i<LOCAL_MATMUL_CHUNK_COUNT_2; i++){
		int col_start = columns_per_chunk*i;
		for(int j=0; j<common_dim; j++){
			for(int k=col_start; k<col_start+columns_per_chunk; k++){
				b_cut[(j*columns_per_chunk)+k-col_start] = b[j*columns + k];
			}	
		}
		matrixMultEigenSplitBackend(a, b_cut, c_cut, rows, common_dim, columns_per_chunk, transpose_a, transpose_b);
		for(int j=0; j<rows; j++){
			for(int k=col_start; k<col_start+columns_per_chunk; k++){
				c[j*columns + k] = c_cut[(j*columns_per_chunk)+k-col_start];
			}	
		}
	
	}

	b_cut = vector<aramisSecretType>();
	c_cut = vector<aramisSecretType>();

	vector<aramisSecretType> b_cut_last(common_dim*columns_last_chunk, 0);
	vector<aramisSecretType> c_cut_last(rows*columns_last_chunk, 0);

	if(columns_last_chunk > 0){
		int col_start = columns_per_chunk*LOCAL_MATMUL_CHUNK_COUNT_2;
		for(int j=0; j<common_dim; j++){
			for(int k=col_start; k<col_start+columns_last_chunk; k++){
				b_cut_last[(j*columns_last_chunk)+k-col_start] = b[j*columns + k];
			}	
		}
		matrixMultEigenSplitBackend(a, b_cut_last, c_cut_last, rows, common_dim, columns_last_chunk, transpose_a, transpose_b);
		for(int j=0; j<rows; j++){
			for(int k=col_start; k<col_start+columns_last_chunk; k++){
				c[j*columns + k] = c_cut_last[(j*columns_last_chunk)+k-col_start];
			}	
		}

	}
#endif

}

void matrixMultEigenSplitBackend(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
#ifdef MATMUL_STACK
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	if((left_matrix_row_major == NULL) || (right_matrix_col_major == NULL)){
		abort_sgx(STACK_ALLOC_FAIL_ABORT);			
	}
#else
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(rows*common_dim*sizeof(uint64_t));
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(common_dim*columns*sizeof(uint64_t));
#endif
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = a.data();
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = c.data(); //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
}

void matrixMultEigenSplitBackend(vector<vector<aramisSecretType>> &a, 
		vector<vector<aramisSecretType>> &b, 
		vector<vector<aramisSecretType>> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
#ifdef MATMUL_STACK
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	uint64_t output_matrix[rows*columns];
	if((left_matrix_row_major == NULL) || (right_matrix_col_major == NULL) || (output_matrix == NULL)){
		abort_sgx(STACK_ALLOC_FAIL_ABORT);			
	}
#else
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*common_dim);
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*common_dim*columns);
	uint64_t* output_matrix = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*columns);
#endif
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<rows; i++){
		offset = i*common_dim;
		for(int j=0; j<common_dim; j++){
			left_matrix_row_major[offset + j] = a[i][j]; //Row major filling.
		}
	}
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[i][j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = left_matrix_row_major;
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = output_matrix; //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
	for(int i=0; i<rows; i++){
		int ofs = i*columns;
		for(int j=0; j<columns; j++){
			c[i][j] = output_matrix[ofs + j];
		}
	}
}

void matrixMultEigenSplitBackend(uint64_t* a, 
		uint64_t* b, 
		uint64_t* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
#ifdef MATMUL_STACK
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	if((left_matrix_row_major == NULL) || (right_matrix_col_major == NULL)){
		abort_sgx(STACK_ALLOC_FAIL_ABORT);			
	}
#else
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*common_dim);
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*common_dim*columns);
#endif
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = a;
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = c; //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
}

#else
void matrixMultEigen(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
#ifdef MATMUL_STACK
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	if((left_matrix_row_major == NULL) || (right_matrix_col_major == NULL)){
		abort_sgx(STACK_ALLOC_FAIL_ABORT);			
	}
#else
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(rows*common_dim*sizeof(uint64_t));
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(common_dim*columns*sizeof(uint64_t));
#endif
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = a.data();
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = c.data(); //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
}
void matrixMultEigen(vector<vector<aramisSecretType>> &a, 
		vector<vector<aramisSecretType>> &b, 
		vector<vector<aramisSecretType>> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
#ifdef MATMUL_STACK
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	uint64_t output_matrix[rows*columns];
	if((left_matrix_row_major == NULL) || (right_matrix_col_major == NULL) || (output_matrix == NULL)){
		abort_sgx(STACK_ALLOC_FAIL_ABORT);			
	}
#else
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*common_dim);
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*common_dim*columns);
	uint64_t* output_matrix = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*columns);
#endif
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<rows; i++){
		offset = i*common_dim;
		for(int j=0; j<common_dim; j++){
			left_matrix_row_major[offset + j] = a[i][j]; //Row major filling.
		}
	}
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[i][j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = left_matrix_row_major;
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = output_matrix; //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
	for(int i=0; i<rows; i++){
		int ofs = i*columns;
		for(int j=0; j<columns; j++){
			c[i][j] = output_matrix[ofs + j];
		}
	}
}

void matrixMultEigen(uint64_t* a, 
		uint64_t* b, 
		uint64_t* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
#ifdef MATMUL_STACK
	uint64_t left_matrix_row_major[rows*common_dim];
	uint64_t right_matrix_col_major[common_dim*columns];
	if((left_matrix_row_major == NULL) || (right_matrix_col_major == NULL)){
		abort_sgx(STACK_ALLOC_FAIL_ABORT);			
	}
#else
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*common_dim);
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*common_dim*columns);
#endif
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = a;
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = c; //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
}
#endif


#endif

#ifdef SLOW_MULT_CACHE_OPTI

void matrixMultEigen(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(rows*common_dim*sizeof(uint64_t));
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(common_dim*columns*sizeof(uint64_t));
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}
	int ra, rb, rc;
	aramisSecretType cumu_ele;
	for(int r=0; r<rows; r++){
		ra = r*common_dim;
		rc = r*columns;
		for(int col=0; col<columns; col++){
			rb = col*common_dim;
			cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += left_matrix_row_major[ra + com] * right_matrix_col_major[rb + com];
			}
			c[rc + col] = cumu_ele;
		}
	}
	
}
void matrixMultEigen(vector<vector<aramisSecretType>> &a, 
		vector<vector<aramisSecretType>> &b, 
		vector<vector<aramisSecretType>> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*common_dim);
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*common_dim*columns);
	uint64_t* output_matrix = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*columns);
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<rows; i++){
		offset = i*common_dim;
		for(int j=0; j<common_dim; j++){
			left_matrix_row_major[offset + j] = a[i][j]; //Row major filling.
		}
	}
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[i][j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = left_matrix_row_major;
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = output_matrix; //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
	//copy from output_matrix to c vector of vector.
	for(int i=0; i<rows; i++){
		int ofs = i*columns;
		for(int j=0; j<columns; j++){
			c[i][j] = output_matrix[ofs + j];
		}
	}
}

void matrixMultEigen(uint64_t* a, 
		uint64_t* b, 
		uint64_t* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0);
	uint64_t* left_matrix_row_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*rows*common_dim);
	uint64_t* right_matrix_col_major = (uint64_t*)sgx_malloc(sizeof(uint64_t)*common_dim*columns);
	int offset = 0;
	int tempoffset = 0;
	for(int i=0; i<common_dim; i++){
		offset = i*columns;
		for(int j=0; j<columns; j++){
			right_matrix_col_major[j*common_dim + i] = b[offset + j]; //Column major filling.
		}
	}

	int row_chunk = rows/NO_CORES;
	global_rows = rows;
	global_columns = columns;
	global_common_dim = common_dim;	
	left_matrix_location = a;
	right_matrix_location = right_matrix_col_major;
	result_matrix_location = c; //pointer to the first data of vector c.

	for(int iter=0; iter<NO_CORES; iter++){
		int start = iter*row_chunk;
		int end = 0;	
		if(iter == NO_CORES-1)
			end = rows;
		else
			end = (iter+1)*row_chunk;
		start_matmul[iter] = start;
		end_matmul[iter] = end;
		
		//Now ocall to spawn threads that then enter enclave	
		ocall_matmul_spawn_threads(iter);
	}
	ocall_matmul_join_threads();
}


#endif

#ifdef SLOW_MULT

void matrixMultEigen(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	for(int r=0; r<rows; r++){
		for(int col=0; col<columns; col++){
			aramisSecretType cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += a[r*common_dim + com] * b[com*columns + col];
			}
			c[r*columns + col] = cumu_ele;
		}
	}
	
	
}

void matrixMultEigen(vector<vector<aramisSecretType>> &a, 
		vector<vector<aramisSecretType>> &b, 
		vector<vector<aramisSecretType>> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	for(int r=0; r<rows; r++){
		for(int col=0; col<columns; col++){
			aramisSecretType cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += a[r][com] * b[com][col];
			}
			c[r][col] = cumu_ele;
		}
	}
	
	
}

void matrixMultEigen(uint64_t* a, 
		uint64_t* b, 
		uint64_t* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t transpose_a, 
		size_t transpose_b)
{
	for(int r=0; r<rows; r++){
		for(int col=0; col<columns; col++){
			aramisSecretType cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += a[r*common_dim + com] * b[com*columns + col];
			}
			c[r*columns + col] = cumu_ele;
		}
	}
	
	
}


#endif

/*************************************** End of MatMul functions ***********************************/

/*************************************** Some other STANDALONE EXECTION utility functions **************************/

aramisSecretType divideMyTypeSA(aramisSecretType a, 
		aramisSecretType b)
{
	assert((b != 0) && "Cannot divide by 0");
	return floatToMyType((double)((int64_t)a)/(double)((int64_t)b));
}

aramisSecretType dividePlainSA(aramisSecretType a, 
		int b)
{
	assert((b != 0) && "Cannot divide by 0");
	return static_cast<aramisSecretType>(static_cast<int64_t>(a)/static_cast<int64_t>(b));
}

void dividePlainSA(vector<aramisSecretType> &vec, 
		int divisor)
{
	assert((divisor != 0) && "Cannot divide by 0");
	for (int i = 0; i < vec.size(); ++i)
		vec[i] = (aramisSecretType)((double)((int64_t)vec[i])/(double)((int64_t)divisor)); 	
}

aramisSecretType multiplyMyTypesSA(aramisSecretType a, 
		aramisSecretType b, 
		int shift)
{
	aramisSecretType ret;
	ret = static_cast<aramisSecretType>((static_cast<int64_t>(a) * static_cast<int64_t>(b))/ (1 << shift));
	return ret;
}

/*************************************** Other small utility functions ************************************/

void XORVectors(const vector<smallType> &a, 
		const vector<smallType> &b, 
		vector<smallType> &c, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] ^ b[i];
}

size_t adversary(size_t party)
{
	size_t ret;

	switch(party)
	{
    		case PARTY_A : ret = PARTY_B;
             			break;       
    		case PARTY_B : ret = PARTY_A;
             			break;
	}	
	return ret;
}

smallType subtractModPrime(smallType a, 
		smallType b)
{
	if (b == 0)
		return a;
	else 
	{
		b = (PRIME_NUMBER - b); 
		return additionModPrime[a][b];
	}
}


void wrapAround(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
		vector<smallType> &c, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = wrapAround(a[i], b[i]);
}

/************************************* Some functions with AES and resharing ****************************/

void populateBitsVector(vector<smallType> &vec, 
		string r_type, 
		size_t size)
{
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for populateBitsVector");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_common->getBit();
	}

	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_indep->getBit();
	}
}

//Returns shares of MSB...LSB of first number and so on.  
void sharesOfBits(vector<smallType> &bit_shares_x_1, 
		vector<smallType> &bit_shares_x_2, 
		const vector<aramisSecretType> &x, 
		size_t size, 
		string r_type)
{
	assert((r_type == "SHARE_CONV_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfBits");
	smallType temp;

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				temp = aes_common->randModPrime();
				bit_shares_x_1[i*BIT_SIZE + k] = temp;
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
			}
		}
	}

	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				temp = aes_indep->randModPrime();
				bit_shares_x_1[i*BIT_SIZE + k] = temp;
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
			}
		}
	}
	else if (r_type == "SHARE_CONV_OPTI"){
		
		for(size_t i = 0; i < (size/2); i++){
			for(size_t k=0; k<BIT_SIZE; k++){
				temp = aes_share_conv_bit_shares_p0_p2->randModPrime();
				bit_shares_x_1[i*BIT_SIZE + k] = temp;
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
			}
		}
		for(size_t i = (size/2); i < size; i++){
			for(size_t k=0; k<BIT_SIZE; k++){
				temp = aes_share_conv_bit_shares_p1_p2->randModPrime();
				bit_shares_x_2[i*BIT_SIZE + k] = temp;
				bit_shares_x_1[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
			}
		}

	}
}

//Returns boolean shares of LSB of r.  
void sharesOfLSB(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<aramisSecretType> &r, 
		size_t size, 
		string r_type)
{
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for sharesOfLSB");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->getBit();
			share_2[i] = share_1[i] ^ (r[i] % 2);
		}
	}

	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->getBit();
			share_2[i] = share_1[i] ^ (r[i] % 2);
		}
	}
}

//Returns \Z_L shares of LSB of r.  
void sharesOfLSB(vector<aramisSecretType> &share_1, 
		vector<aramisSecretType> &share_2, 
		const vector<aramisSecretType> &r, 
		size_t size, 
		string r_type)
{
	assert((r_type == "MSB_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfLSB");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->get64Bits();
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
	}

	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->get64Bits();
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
	}

	else if(r_type == "MSB_OPTI"){
		// First half common with P0 and second half common with P1
		for (size_t i = 0; i < (size/2); ++i)
		{
			share_1[i] = aes_comp_msb_shares_lsb_p0_p2->get64Bits();
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}	
		for (size_t i = (size/2); i < size; ++i)
		{
			share_2[i] = aes_comp_msb_shares_lsb_p1_p2->get64Bits();
			share_1[i] = floatToMyType(r[i] % 2) - share_2[i];
		}
	
	}
}



//Returns boolean shares of a bit vector vec.  
void sharesOfBitVector(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type)
{
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for sharesOfLSB");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->getBit();
			share_2[i] = share_1[i] ^ vec[i];
		}
	}

	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->getBit();
			share_2[i] = share_1[i] ^ vec[i];
		}
	}
}

//Returns \Z_L shares of a bit vector vec.  
void sharesOfBitVector(vector<aramisSecretType> &share_1, 
		vector<aramisSecretType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type)
{
	assert((r_type == "MSB_OPTI" ||(r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfLSB");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->get64Bits();
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
	}
	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->get64Bits();
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
	}
	else if(r_type == "MSB_OPTI"){
		for (size_t i = 0; i < (size/2); ++i)
		{
			share_1[i] = aes_comp_msb_shares_bit_vec_p0_p2->get64Bits();
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
		
		for (size_t i = (size/2); i < size; ++i)
		{
			share_2[i] = aes_comp_msb_shares_bit_vec_p1_p2->get64Bits();
			share_1[i] = floatToMyType(vec[i]) - share_2[i];
		}
	
	}
}


//Split shares of a vector of aramisSecretType into shares (randomness is independent)
void splitIntoShares(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &a1, 
		vector<aramisSecretType> &a2, 
		size_t size)
{
	populateRandomVector<aramisSecretType>(a1, size, "INDEP", "POSITIVE");
	subtractVectors<aramisSecretType>(a, a1, a2, size);
}


/***************************** Basic utility functions for Convolution drivers ************************/

void map_reduce_vector(vector<aramisSecretType> in, 
		aramisSecretType* out, 
		uint64_t size){
	*out = 0;
	for(int i=0; i<size; i++){
		*out = ((*out) + in[i]);
	}
}

void zero_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					vec[i][j][k][l] = 0;
				}
			}
		}
	}
}

void subtract_2D_vectors(vector< vector<aramisSecretType> >& inp_l, 
		vector< vector<aramisSecretType> >& inp_r, 
		vector< vector<aramisSecretType> >& out, 
		int d1, 
		int d2)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			out[i][j] = inp_l[i][j] - inp_r[i][j];
		}
	}
}

void add_2D_vectors(vector< vector<aramisSecretType> >& inp_l, 
		vector< vector<aramisSecretType> >& inp_r, 
		vector< vector<aramisSecretType> >& out, 
		int d1, 
		int d2)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			out[i][j] = inp_l[i][j] + inp_r[i][j];
		}
	}
}

void zero_2D_vector(vector< vector<aramisSecretType> >& vec, 
		int d1, 
		int d2)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			vec[i][j] = 0;
		}
	}
}

void add_4D_vectors(vector< vector< vector< vector<aramisSecretType> > > >& inp_l, 
		vector< vector< vector< vector<aramisSecretType> > > >& inp_r, 
		vector< vector< vector< vector<aramisSecretType> > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					out[i][j][k][l] = inp_l[i][j][k][l] + inp_r[i][j][k][l];
				}
			}
		}
	}
}

void subtract_4D_vectors(vector< vector< vector< vector<aramisSecretType> > > >& inp_l, 
		vector< vector< vector< vector<aramisSecretType> > > >& inp_r, 
		vector< vector< vector< vector<aramisSecretType> > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					out[i][j][k][l] = inp_l[i][j][k][l] - inp_r[i][j][k][l];
				}
			}
		}
	}
}

void flatten_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& input, 
		vector<aramisSecretType>& output, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			for(int k=0;k<d3;k++){
				for(int l=0;l<d4;l++){
					output[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = input[i][j][k][l];
				}
			}
		}
	}

}

void deflatten_4D_vector(vector<aramisSecretType>& input, 
		vector< vector< vector< vector<aramisSecretType> > > >& output, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			for(int k=0;k<d3;k++){
				for(int l=0;l<d4;l++){
					output[i][j][k][l] = input[i*d2*d3*d4 + j*d3*d4 + k*d4 + l];
				}
			}
		}
	}

}

void flatten_2D_vector(vector< vector<aramisSecretType> >& input, 
		vector<aramisSecretType>& output, 
		int d1, 
		int d2)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			output[i*d2 + j] = input[i][j];
		}
	}

}

void deflatten_2D_vector(vector<aramisSecretType>& input, 
		vector< vector<aramisSecretType> >& output, 
		int d1, 
		int d2)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			output[i][j] = input[i*d2 + j];
		}
	}

}

void send_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& input, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	vector<aramisSecretType> flat_input(d1*d2*d3*d4, 0);

	//Flatten and send.
	flatten_4D_vector(input, flat_input, d1, d2, d3, d4);

	sendVector<aramisSecretType>(ref(flat_input), adversary(partyNum), d1*d2*d3*d4);

}

void receive_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& recv, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	vector<aramisSecretType> flat_recv(d1*d2*d3*d4, 0);

	//Receive and deflatten.
	receiveVector<aramisSecretType>(ref(flat_recv), adversary(partyNum), d1*d2*d3*d4);

	deflatten_4D_vector(flat_recv, recv, d1, d2, d3, d4);

}

void send_2D_vector(vector< vector<aramisSecretType> >& input, 
		int d1, 
		int d2)
{
	vector<aramisSecretType> flat_input(d1*d2, 0);

	//Flatten and send.
	flatten_2D_vector(input, flat_input, d1, d2);

	sendVector<aramisSecretType>(ref(flat_input), PARTY_B, d1*d2);

}

void receive_2D_vector(vector< vector<aramisSecretType> >& recv, 
		int d1, 
		int d2)
{
	vector<aramisSecretType> flat_recv(d1*d2, 0);

	//Receive and deflatten.
	receiveVector<aramisSecretType>(ref(flat_recv), PARTY_C, d1*d2);

	deflatten_2D_vector(flat_recv, recv, d1, d2);

}

void populate_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4, 
		string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					vec[i][j][k][l] = aesObject->get64Bits();
				}
			}
		}
	}
}

void populate_2D_vector(vector< vector<aramisSecretType> >& vec, 
		int d1, 
		int d2, 
		string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			vec[i][j] = aesObject->get64Bits();
		}
	}
}

void populate_AES_Arr(uint64_t* arr, 
		uint64_t size, 
		string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

	for(uint64_t i=0; i<size; i++){
		arr[i] = aesObject->get64Bits();
	}
}

void add_2_Arr(uint64_t* arr1, 
		uint64_t* arr2, 
		uint64_t* arr, 
		uint64_t size)
{
	for(uint64_t i=0; i<size; i++){
		arr[i] = arr1[i] + arr2[i];
	}
}

void subtract_2_Arr(uint64_t* arr1, 
		uint64_t* arr2, 
		uint64_t* arr, 
		uint64_t size)
{
	for(uint64_t i=0; i<size; i++){
		arr[i] = arr1[i] - arr2[i];
	}
}

/**********************************************************
 * Populate AES random vector parallelized
 * *******************************************************/

void populateRandomVectorThreaded(aramisSecretType* vec, 
		size_t size, 
		string r_type, 
		int thread_id)
{	
	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_common[thread_id]->get64Bits();		
	}
	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_indep[thread_id]->get64Bits();		
	}
	if (r_type == "a_1")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_a_1[thread_id]->get64Bits();
	}
	else if (r_type == "b_1")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_b_1[thread_id]->get64Bits();
	}
	else if (r_type == "c_1")
	{	
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_c_1[thread_id]->get64Bits();
	}
	else if (r_type == "a_2")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_a_2[thread_id]->get64Bits();
	}
	else if (r_type == "b_2")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = threaded_aes_b_2[thread_id]->get64Bits();
	}
}


void aes_parallel_populate_thread_dispatcher(int worker_thread_num){
	assert(NO_CORES == 4);
	populateRandomVectorThreaded(&(global_populate_target[start_populate_target[worker_thread_num]]), end_populate_target[worker_thread_num]-start_populate_target[worker_thread_num], global_r_type, worker_thread_num);
}

void populateRandomVectorParallel(vector<uint64_t> vec, 
		uint64_t size, 
		string type, 
		string extra){
	int chunk = size/NO_CORES;
	global_populate_target = vec.data();
	global_r_type = type;
	for(int i=0; i<NO_CORES; i++){
		int start = i*chunk;
		int end = 0;
		if(i == NO_CORES-1)
			end = size;
		else
			end = (i+1)*chunk;
		start_populate_target[i] = start;
		end_populate_target[i] = end;
	
		ocall_populate_aes_spawn_threads(i);
	}	
	ocall_join_threads();
}

template<typename T1, typename T2>
void addModuloOddArray(T1 *a, 
		T2* b, 
		aramisSecretType* c, 
		size_t size)
{
	assert((sizeof(T1) == sizeof(aramisSecretType) or sizeof(T2) == sizeof(aramisSecretType)) && "At least one type should be aramisSecretType for typecast to work");

	for (size_t i = 0; i < size; ++i)
	{
		if (a[i] == MINUS_ONE and b[i] == MINUS_ONE)
			c[i] = 0;
		else 
			c[i] = (a[i] + b[i] + wrapAround(a[i], b[i])) % MINUS_ONE;
	}
}

template <typename T1, typename T2>
void subtractModuloOddArray(T1 *a, 
		T2 *b, 
		aramisSecretType* c, 
		size_t size)
{
	vector<aramisSecretType> temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = MINUS_ONE - b[i];

	addModuloOddArray<T1, aramisSecretType>(a, temp.data(), c, size);
}

void sharesModuloOddThreaded(aramisSecretType* shares_1, 
		aramisSecretType* shares_2, 
		smallType* x, 
		uint64_t size, 
		string r_type, 
		int thread_id)
{
	assert((r_type == "SHARE_CONV_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfBits");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = threaded_aes_common[thread_id]->randModuloOdd();
		subtractModuloOddArray<smallType, aramisSecretType>(x, shares_1, shares_2, size);
	}

	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = threaded_aes_indep[thread_id]->randModuloOdd();
		subtractModuloOddArray<smallType, aramisSecretType>(x, shares_1, shares_2, size);
	}

	
	if(r_type == "SHARE_CONV_OPTI"){
		if(partyNum == PARTY_C){
			vector<aramisSecretType> shares1_temp_first(1+size/2);
			vector<aramisSecretType> shares2_temp_first(1+size/2);
			vector<aramisSecretType> shares1_temp_last(1+size/2);
			vector<aramisSecretType> shares2_temp_last(1+size/2);
			for(size_t i=0; i<(size/2); i++){
				//First half common with P0
				shares1_temp_first[i] = aes_share_conv_shares_mod_odd_p0_p2->randModuloOdd();
			}
			for(size_t i=0; i<(size-(size/2)); i++){
				//Last half common with P1
				shares2_temp_last[i] = aes_share_conv_shares_mod_odd_p1_p2->randModuloOdd();
			}
			vector<smallType> x_first(1+size/2);
			vector<smallType> x_last(1+size/2);
			for(size_t i=0; i<(size/2); i++){
				x_first[i] = x[i];
			}
			for(size_t i=0; i<(size-(size/2)); i++){
				x_last[i] = x[(size/2)+i];
			}
			subtractModuloOddArray<smallType, aramisSecretType>(x_first.data(), shares1_temp_first.data(), shares2_temp_first.data(), size/2);
			subtractModuloOddArray<smallType, aramisSecretType>(x_last.data(), shares2_temp_last.data(), shares1_temp_last.data(), size-(size/2));	
			for(size_t i=0; i<(size/2); i++){
				shares_1[i] = shares1_temp_first[i];
				shares_2[i] = shares2_temp_first[i];
			}	
			for(size_t i=0; i<(size-(size/2)); i++){
				shares_1[i+(size/2)] = shares1_temp_last[i];
				shares_2[i+(size/2)] = shares2_temp_last[i];
			}	
		}
		if(partyNum == PARTY_A){
			vector<aramisSecretType> shares1_temp_first(1+size/2);
			for(size_t i=0; i<(size/2); i++){
				//First half common with P0
				shares1_temp_first[i] = aes_share_conv_shares_mod_odd_p0_p2->randModuloOdd();
			}
			for(size_t i=0; i<(size/2); i++){
				shares_1[i] = shares1_temp_first[i];
			}	
			for(size_t i=0; i<(size-(size/2)); i++){
				shares_1[i+(size/2)] = 0;
			}
		}
		if(partyNum == PARTY_B){
			vector<aramisSecretType> shares2_temp_last(1+size/2);
			for(size_t i=0; i<(size-(size/2)); i++){
				//last half common with P0
				shares2_temp_last[i] = aes_share_conv_shares_mod_odd_p1_p2->randModuloOdd();
			}
			for(size_t i=0; i<(size/2); i++){
				shares_2[i] = 0;
			}	
			for(size_t i=0; i<(size-(size/2)); i++){
				shares_2[i+(size/2)] = shares2_temp_last[i];
			}
		}

	}
}

void aes_parallel_sharesModuloOdd_thread_dispatcher(int worker_thread_num){
	assert(NO_CORES == 4);
	sharesModuloOddThreaded(&(global_sharesModuloOdd_target1[start_populate_target[worker_thread_num]]), &(global_sharesModuloOdd_target2[start_populate_target[worker_thread_num]]), &(global_sharesModuloOdd_target3[start_populate_target[worker_thread_num]]), end_populate_target[worker_thread_num]-start_populate_target[worker_thread_num], global_r_type, worker_thread_num);
}

void sharesModuloOddParallel(vector<aramisSecretType> &shares_1, 
		vector<aramisSecretType> &shares_2, 
		vector<smallType> &x, 
		size_t size, 
		string r_type)
{
	int chunk = size/NO_CORES;
	global_sharesModuloOdd_target1 = shares_1.data();
	global_sharesModuloOdd_target2 = shares_2.data();
	global_sharesModuloOdd_target3 = x.data();
	global_r_type = r_type;
	for(int i=0; i<NO_CORES; i++){
		int start = i*chunk;
		int end = 0;
		if(i == NO_CORES-1)
			end = size;
		else
			end = (i+1)*chunk;
		start_populate_target[i] = start;
		end_populate_target[i] = end;
	
		ocall_sharesModuloOdd_aes_spawn_threads(i);
	}	
	ocall_join_threads();
	
}

void sharesOfBitsThreaded(smallType *bit_shares_x_1, 
		smallType* bit_shares_x_2, 
		aramisSecretType* x, 
		size_t size, 
		string r_type, 
		int thread_id, 
		int which_party)
{
	assert((r_type == "SHARE_CONV_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfBits");
	smallType temp;
#ifdef CACHE_OPTI
	if (r_type == "COMMON")
	{
		threaded_aes_common[thread_id]->fillWithRandomBitsModuloPrime(bit_shares_x_1, size*BIT_SIZE);

		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), bit_shares_x_1[i*BIT_SIZE + k]);
			}
		}
	}

	if (r_type == "INDEP")
	{
		threaded_aes_indep[thread_id]->fillWithRandomBitsModuloPrime(bit_shares_x_1, size*BIT_SIZE);

		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), bit_shares_x_1[i*BIT_SIZE + k]);
			}
		}
	}
#else
	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				temp = threaded_aes_common[thread_id]->randModPrime();
				bit_shares_x_1[i*BIT_SIZE + k] = temp;
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
			}
		}
	}

	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				temp = threaded_aes_indep[thread_id]->randModPrime();
				bit_shares_x_1[i*BIT_SIZE + k] = temp;
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
			}
		}
	}

#endif
	else if (r_type == "SHARE_CONV_OPTI"){
		
		//assert((size%2 == 0) && "Share convert optimization written for even size only");
		if(which_party == 0){
			for(size_t i = 0; i < (size); i++){
				for(size_t k=0; k<BIT_SIZE; k++){
					temp = threaded_aes_share_conv_bit_shares_p0_p2[thread_id]->randModPrime();
					bit_shares_x_1[i*BIT_SIZE + k] = temp;
					bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
				}
			}
		}
		else{
			for(size_t i = 0; i < size; i++){
				for(size_t k=0; k<BIT_SIZE; k++){
					temp = threaded_aes_share_conv_bit_shares_p1_p2[thread_id]->randModPrime();
					bit_shares_x_2[i*BIT_SIZE + k] = temp;
					bit_shares_x_1[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), temp);
				}
			}

		}
	}
}

void aes_parallel_sharesOfBits_thread_dispatcher(int worker_thread_num){
	assert(NO_CORES == 4);
	sharesOfBitsThreaded(&(global_sharesOfBits_target1[BIT_SIZE*start_populate_target[worker_thread_num]]), &(global_sharesOfBits_target2[BIT_SIZE*start_populate_target[worker_thread_num]]), &(global_sharesOfBits_target3[start_populate_target[worker_thread_num]]), end_populate_target[worker_thread_num]-start_populate_target[worker_thread_num], global_r_type, worker_thread_num, global_which_party[worker_thread_num]);
}

void sharesOfBitsParallel(vector<smallType> &bit_shares_x_1, 
		vector<smallType> &bit_shares_x_2, 
		vector<aramisSecretType> &x, 
		size_t size, 
		string r_type)
{
	int l_chunk_size = (size/2);
	int r_chunk_size = size-(size/2);
	int half_cores = NO_CORES/2;
	int l_chunk = (l_chunk_size)/half_cores;
	int r_chunk = (r_chunk_size)/half_cores;
	global_sharesOfBits_target1 = bit_shares_x_1.data();
	global_sharesOfBits_target2 = bit_shares_x_2.data();
	global_sharesOfBits_target3 = x.data();
	global_r_type = r_type;
	for(int i=0; i<NO_CORES; i++){
		if(i<half_cores){
			int start = i*l_chunk;
			int end = 0;
			if(i==half_cores-1)
				end = l_chunk_size;
			else
				end = (i+1)*l_chunk;
			start_populate_target[i] = start;
			end_populate_target[i] = end;
			global_which_party[i] = 0;
		}
		else{
			int start = l_chunk_size+(i-half_cores)*r_chunk;
			int end = 0;
			if(i==NO_CORES-1)
				end = size;
			else
				end = l_chunk_size+(i+1-half_cores)*r_chunk;
			start_populate_target[i] = start;
			end_populate_target[i] = end;
			global_which_party[i] = 1;
	
		}
		ocall_sharesOfBits_aes_spawn_threads(i);
	}	
	ocall_join_threads();
	
}



void populate_flat_vectorThreaded(uint64_t* vec, 
		int start, 
		int end, 
		string type, 
		int thread_id)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = threaded_aes_conv_opti_a_1[thread_id];
	else if(type == "a2") aesObject = threaded_aes_conv_opti_a_2[thread_id];
	else if(type == "b1") aesObject = threaded_aes_conv_opti_b_1[thread_id];
	else if(type == "b2") aesObject = threaded_aes_conv_opti_b_2[thread_id];
	else if(type == "c1") aesObject = threaded_aes_conv_opti_c_1[thread_id];
	else assert(false);

	for(int i=start; i<end; i++){
		vec[i] = aesObject->get64Bits();
	}
}

void aes_parallel_populate_flat_vector_thread_dispatcher(int thread_id){
	populate_flat_vectorThreaded(global_pop_flat_vec_target1, start_populate_target[thread_id], end_populate_target[thread_id], global_r_type, thread_id);
}

void populate_4D_vectorParallel(vector< vector< vector< vector<aramisSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4, 
		string type)
{
	aramisSecretType* flatarray = (aramisSecretType*)malloc(d1*d2*d3*d4*sizeof(aramisSecretType));	
	int size = d1*d2*d3*d4;	
	int chunk = size/NO_CORES;
	global_pop_flat_vec_target1 = flatarray;
	global_r_type = type;
	for(int i=0; i<NO_CORES; i++){
		int start = i*chunk;
		int end = 0;
		if(i == NO_CORES-1)
			end = size;
		else
			end = (i+1)*chunk;
		start_populate_target[i] = start;
		end_populate_target[i] = end;
	
		ocall_populate_flat_vector_aes_spawn_threads(i);
	}	
	ocall_join_threads();
	
	
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					vec[i][j][k][l] = flatarray[i*d2*d3*d4 + j*d3*d4 + k*d4 + l];
				}
			}
		}
	}
	free(flatarray);
}

void populate_2D_vectorParallel(vector< vector<aramisSecretType> >& vec, 
		int d1, 
		int d2, 
		string type)
{
	aramisSecretType* flatarray = (aramisSecretType*)malloc(d1*d2*sizeof(aramisSecretType));	
	int size = d1*d2;	
	int chunk = size/NO_CORES;
	global_pop_flat_vec_target1 = flatarray;
	global_r_type = type;
	for(int i=0; i<NO_CORES; i++){
		int start = i*chunk;
		int end = 0;
		if(i == NO_CORES-1)
			end = size;
		else
			end = (i+1)*chunk;
		start_populate_target[i] = start;
		end_populate_target[i] = end;
	
		ocall_populate_flat_vector_aes_spawn_threads(i);
	}	
	ocall_join_threads();
	
	
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			vec[i][j] = flatarray[i*d2 + j];
		}
	}
	free(flatarray);

}

void populate_AES_ArrParallel(uint64_t* arr, 
		uint64_t size, 
		string type)
{
	int chunk = size/NO_CORES;
	global_pop_flat_vec_target1 = arr;
	global_r_type = type;
	for(int i=0; i<NO_CORES; i++){
		int start = i*chunk;
		int end = 0;
		if(i == NO_CORES-1)
			end = size;
		else
			end = (i+1)*chunk;
		start_populate_target[i] = start;
		end_populate_target[i] = end;
	
		ocall_populate_flat_vector_aes_spawn_threads(i);
	}	
	ocall_join_threads();
	
}

void sharesOfBitsPrimaryThreaded(smallType *bit_shares_x_1, 
		size_t size, 
		int thread_id, 
		int which_party)
{
	if(which_party == 0){
		for(size_t i = 0; i < (size); i++){
				bit_shares_x_1[i] = threaded_aes_share_conv_bit_shares_p0_p2[thread_id]->randModPrime();
		}
	}
	else{
		for(size_t i = 0; i < size; i++){
				bit_shares_x_1[i] = threaded_aes_share_conv_bit_shares_p1_p2[thread_id]->randModPrime();
		}

	}
}

void aes_parallel_sharesOfBits_primary_thread_dispatcher(int worker_thread_num){
	assert(NO_CORES == 4);
	sharesOfBitsPrimaryThreaded(&(global_sharesOfBits_target1[start_populate_target[worker_thread_num]]), end_populate_target[worker_thread_num]-start_populate_target[worker_thread_num], worker_thread_num, global_which_party[worker_thread_num]);
}

void sharesOfBitsPrimaryParallel(smallType *arr, 
		size_t size, 
		int which_party)
{
	//This can only use 2 cores.
	assert(((NO_CORES%4)==0) && "Aramis only runs on multiples of 4 threads right now");
	int cores_here = NO_CORES/2;
	int chunk = size/cores_here;
	global_sharesOfBits_target1 = arr;
	for(int i=0; i<cores_here; i++){
		int start = i*chunk;
		int end = 0;
		if(i == cores_here-1)
			end = size;
		else
			end = (i+1)*chunk;
		start_populate_target[i] = start;
		end_populate_target[i] = end;
		global_which_party[i] = which_party;
		ocall_sharesOfBits_primary_aes_spawn_threads(i);
	}	
	ocall_join_threads_half();
	
}

/********************************** Aramis info display functions *****************************/

void show_aramis_mode()
{
	if(print_mode){
		return;
	}
	print_mode = true;
	bool all_opti, share_conv_opti, msb_opti, conv_opti, parallel_aes, parallelize_crit_opti;
	print_string("\n**********************ARAMIS MODE**********************\n>>> Running Aramis in the following mode: ");
#ifndef ALL_OPTI
	print_string("ARAMIS OPTIMIZATIONS         OFF");
#else
#ifdef RUN_SHARECONV_OPTI
	print_string("SHARE CONVERT OPTIMIZATION   ON");
#else
	print_string("SHARE CONVERT OPTIMIZATION   OFF");
#endif
#ifdef RUN_MSB_OPTI
	print_string("COMPUTE MSB OPTIMIZATION     ON");
#else
	print_string("COMPUTE MSB OPTIMIZATION     OFF");
#endif
#ifdef CONV_OPTI
	print_string("CONVOLUTION OPTIMIZATION     ON");
#else
	print_string("CONVOLUTION OPTIMIZATION     OFF");
#endif
#ifdef PARALLEL_AES
	print_string("PARALLEL AES OPTIMIZATION    ON");
#else
	print_string("PARALLEL AES OPTIMIZATION    OFF");
#endif
#ifdef PARALLELIZE_CRITICAL
	print_string("PRIVATE COMP. OPTIMIZATION   ON");
#else
	print_string("PRIVATE COMP. OPTIMIZATION   OFF");
#endif


#ifdef DEBUG
	print_string("ARAMIS DEBUG BUILD(ASSERTS)  ON");
#else
	print_string("ARAMIS DEBUG BUILD(ASSERTS)  OFF");
#endif
#ifdef MATMUL_THREADED
	print_string("ARAMIS FASTER MATMUL SUPPORT ON");
#else
	print_string("ARAMIS FASTER MATMUL SUPPORT OFF");
#endif
#ifdef SPLIT_RELU
	print_string("ARAMIS FASTER RELU SUPPORT   ON");
#else
	print_string("ARAMIS FASTER RELU SUPPORT   OFF");
#endif
#ifdef SPLIT_MAXPOOL
	print_string("ARAMIS FASTER MAXP SUPPORT   ON");
#else
	print_string("ARAMIS FASTER MAXP SUPPORT   OFF");
#endif
#ifdef SPLIT_CONV
	print_string("ARAMIS FASTER CONV SUPPORT   ON");
#else
	print_string("ARAMIS FASTER CONV SUPPORT   OFF");
#endif
#ifdef SPLIT_LOCAL_MATMUL
	print_string("ARAMIS FASTER L.MM SUPPORT   ON");
#else
	print_string("ARAMIS FASTER CONV SUPPORT   OFF");
#endif

#endif

	if(sizeof(uint64_t) == sizeof(aramisSecretType)){
		print_string("Running Aramis in Z_2^64");
	}
	else{
		print_string("Support for custom ring not available");
	}
	print_string("");
}

/********************************** Some helper functions invoked only by test functions **************/

void peek_inside_enclave(int worker_thread_num){
	return;
}

void maxPoolReshape(const vector<aramisSecretType> &vec, 
		vector<aramisSecretType> &vecShaped, 
		size_t ih, 
		size_t iw, 
		size_t D, 
		size_t B, 
		size_t fh, 
		size_t fw, 
		size_t sy, 
		size_t sx)
{
	assert(fw >= sx and fh >= sy && "Check implementation");
	assert((iw - fw)%sx == 0 && "Check implementations for this unmet condition");
	assert((ih - fh)%sy == 0 && "Check implementations for this unmet condition");
	assert(vec.size() == vecShaped.size() && "Dimension issue with convolutionReshape");

	size_t loc = 0, counter = 0;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < D; ++j)
			for (size_t k = 0; k < ih-fh+1; k += sy) 
				for (size_t l = 0; l < iw-fw+1; l += sx)
				{
					loc = i*iw*ih*D + j*iw*ih + k*iw + l;
					for (size_t a = 0; a < fh; ++a)
						for (size_t b = 0; b < fw; ++b)
							vecShaped[counter++] = vec[loc + a*iw + b];
				}
}


