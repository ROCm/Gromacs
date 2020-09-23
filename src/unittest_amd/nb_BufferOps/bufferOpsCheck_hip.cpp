#include <hip/hip_runtime.h>

#include <fstream>

#include "myHipUtil.h"

/*! \brief CUDA kernel to sum up the force components
 *
 * \tparam        accumulateForce  If the initial forces in \p gm_fTotal should be saved.
 * \tparam        addPmeForce      Whether the PME force should be added to the total.
 *
 * \param[in]     gm_fNB     Non-bonded forces in nbnxm format.
 * \param[in]     gm_fPme    PME forces.
 * \param[in,out] gm_fTotal  Force buffer to be reduced into.
 * \param[in]     cell       Cell index mapping.
 * \param[in]     atomStart  Start atom index.
 * \param[in]     numAtoms   Number of atoms.
 */
template<bool accumulateForce, bool addPmeForce>
static __global__ void nbnxn_gpu_add_nbat_f_to_f_kernel(const float3* __restrict__ gm_fNB,
                                                        const float3* __restrict__ gm_fPme,
                                                        float3* gm_fTotal,
                                                        const int* __restrict__ gm_cell,
                                                        const int atomStart,
                                                        const int numAtoms)
{

    /* map particle-level parallelism to 1D CUDA thread and block index */
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    /* perform addition for each particle*/
    if (threadIndex < numAtoms)
    {

        const int i        = gm_cell[atomStart + threadIndex];
        float3*   gm_fDest = &gm_fTotal[atomStart + threadIndex];
        float3    temp;

        if (accumulateForce)
        {
            temp = *gm_fDest;
            //temp += gm_fNB[i]; //cm todo
            temp = temp + gm_fNB[i];
        }
        else
        {
            temp = gm_fNB[i];
        }
        if (addPmeForce)
        {
            //temp += gm_fPme[atomStart + threadIndex]; //cm todo
            temp = temp + gm_fPme[atomStart + threadIndex];
        }
        *gm_fDest = temp;
    }
    return;
}

struct kernelTest {
     size_t blockSize[3]; 
     size_t gridSize[3]; 

     int totalSize; 

     // kernel parameters
     int atomStart;
     int numAtoms; 
     float3 *d_fNB; 
     float3 *d_fPme; 
     float3 *d_fTotal; 
     int *d_cell; 
}; 

static void setup_kernel_test(std::string dataDir, kernelTest &test, bool accumulateForce, bool addPmeForce)
{
    std::string fPath(dataDir); 
    std::string key1; 

    fPath += "/gpu_bufOps_ff_kernel_indata_"; 

    fPath += accumulateForce? "A1_" : "A0_" ; 
    fPath += addPmeForce? "P1.txt" : "P0.txt" ; 

    std::ifstream istr; 

    istr.open(fPath.c_str(), std::ofstream::in); 

    //  block and grid sizes used for scheduling the kernel
    istr >> key1 >> test.blockSize[0] >> test.blockSize[1] >> test.blockSize[2]; 
    istr >> key1 >> test.gridSize[0] >> test.gridSize[1] >> test.gridSize[2]; 

    // totalSize
    istr >> key1 >> test.totalSize; 
 
    // numAtoms 
    istr >> key1 >> test.numAtoms; 
   
    // atomStart
    istr >> key1 >> test.atomStart;  

    // d_fNB
    istr >> key1;  
    float3 *h_fNB = new float3[test.totalSize]; 

    for (int i=0; i < test.totalSize; i++)  
         istr >> h_fNB[i].x >> h_fNB[i].y >> h_fNB[i].z; 

    MY_HIP_CHECK( hipMalloc((void**)&test.d_fNB, sizeof(float3) * test.totalSize) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.d_fNB, h_fNB, sizeof(float3) * test.totalSize) ); 

    delete [] h_fNB; 

    // d_fPme
    istr >> key1; 
    float3 *h_fPme = new float3[test.totalSize]; 

    for (int i=0; i < test.totalSize; i++)  
         istr >> h_fPme[i].x >> h_fPme[i].y >> h_fPme[i].z; 

    MY_HIP_CHECK( hipMalloc((void**)&test.d_fPme, sizeof(float3) * test.totalSize) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.d_fPme, h_fPme, sizeof(float3) * test.totalSize) ); 

    delete [] h_fPme; 
      
    // d_cell
    istr >> key1; 
    int *h_cell = new int[test.totalSize]; 

    for (int i=0; i < test.totalSize; i++)  
         istr >> h_cell[i]; 

    MY_HIP_CHECK( hipMalloc((void**)&test.d_cell, sizeof(int) * test.totalSize) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.d_cell, h_cell, sizeof(int) * test.totalSize) ); 

    delete [] h_cell; 

    // d_fTotal
    istr >> key1; 
    float3 *h_fTotal = new float3[test.totalSize]; 

    for (int i=0; i < test.totalSize; i++)  
         istr >> h_fTotal[i].x >> h_fTotal[i].y >> h_fTotal[i].z; 

    MY_HIP_CHECK( hipMalloc((void**)&test.d_fTotal, sizeof(float3) * test.totalSize) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.d_fTotal, h_fTotal, sizeof(float3) * test.totalSize) ); 

    delete [] h_fTotal; 

    istr.close(); 
}; 

static void save_kernel_test_output(std::string dataDir, kernelTest &test, bool accumulateForce, bool addPmeForce)
{
    std::string fPath(dataDir); 

    fPath += "/kernel_test_outdata_";
    
    fPath += accumulateForce? "A1_" : "A0_" ;
    fPath += addPmeForce? "P1.txt" : "P0.txt" ;
    
    std::ofstream ostr;
    
    ostr.open(fPath.c_str(), std::ofstream::out | std::ofstream::trunc);

    // d_fTotal     
    ostr << "d_fTotal: " << test.totalSize << std::endl; 
    float3 *h_fTotal = new float3[test.totalSize]; 

    MY_HIP_CHECK( hipMemcpyDtoH((void*)h_fTotal, (void*)test.d_fTotal, sizeof(float3) * test.totalSize) ); 
    for (int i=0; i < test.totalSize; i++)   
         ostr << h_fTotal[i].x << " " << h_fTotal[i].y << " " << h_fTotal[i].z << std::endl; 

    delete [] h_fTotal; 

    ostr.close(); 
}; 

static void destroy_kernel_test(kernelTest &test)
{
    MY_HIP_CHECK( hipFree(test.d_fNB) ); 
    MY_HIP_CHECK( hipFree(test.d_fPme) ); 
    MY_HIP_CHECK( hipFree(test.d_fTotal) ); 
    MY_HIP_CHECK( hipFree(test.d_cell) ); 
}; 

int main(int argc, char *argv[])
{
    kernelTest  test;  
    hipStream_t stream; 

    if ( argc != 4 ) {
         std::cerr << "Invalid commandline parameters!" << std::endl; 
         throw std::runtime_error(""); 
    }
    
    std::string dataDir(argv[1]); 

    bool accumulateForce, addPmeForce; 

    accumulateForce = atoi(argv[2]) !=0 ? true : false; 
    addPmeForce = atoi(argv[3]) !=0 ? true : false; 

    MY_HIP_CHECK( hipStreamCreate(&stream) ); 

    void (*kernelFunc)(const float3* gm_fNB, const float3* gm_fPme, float3* gm_fTotal, const int* gm_cell, const int atomStart, const int numAtoms);

    if ( accumulateForce ) { 
         if ( addPmeForce ) 
              kernelFunc = nbnxn_gpu_add_nbat_f_to_f_kernel<true, true>; 
         else 
              kernelFunc = nbnxn_gpu_add_nbat_f_to_f_kernel<true, false>; 
    }
    else 
        if ( addPmeForce ) 
             kernelFunc = nbnxn_gpu_add_nbat_f_to_f_kernel<false, true>; 
        else 
             kernelFunc = nbnxn_gpu_add_nbat_f_to_f_kernel<false, false>; 

    setup_kernel_test(dataDir, test, accumulateForce, addPmeForce); 

    hipLaunchKernelGGL(kernelFunc, dim3(test.gridSize[0],test.gridSize[1],test.gridSize[2]), dim3(test.blockSize[0],test.blockSize[1],test.blockSize[2]), 
                      0, stream, test.d_fNB, test.d_fPme, test.d_fTotal, test.d_cell, test.atomStart, test.numAtoms);  

    MY_HIP_CHECK( hipStreamSynchronize(stream) ); 

    save_kernel_test_output(dataDir, test, accumulateForce, addPmeForce); 

    destroy_kernel_test(test); 
}; 

