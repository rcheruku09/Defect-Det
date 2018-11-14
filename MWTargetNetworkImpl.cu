#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWCNNLayerImpl.hpp"
 void MWTargetNetworkImpl::preSetup(int BufSize,int numBufsToAlloc) { numBufs = 
numBufsToAlloc; for(int i = 0; i < numBufs; i++) { float *memPtr ; 
cudaMalloc((void**)&memPtr, sizeof(float)*BufSize); 
memBuffer.push_back(memPtr); } QjgQHaUACFNSteMrRtRj = new cublasHandle_t; 
cublasCreate(QjgQHaUACFNSteMrRtRj); QwUuNuQNtlPXrIwRNiSZ = new 
cudnnHandle_t; cudnnCreate(QwUuNuQNtlPXrIwRNiSZ); } void 
MWTargetNetworkImpl::postSetup(MWCNNLayer* layers[],int numLayers) { 
createWorkSpace(&xkUNToJIgvoLoUQuzKRF); for (int i = 0; i < numLayers; i++) { 
layers[i]->postSetup();  } if ((GnxRkpzrPZimKtYYHSuG != 
*getWorkSpaceSize() && GnxRkpzrPZimKtYYHSuG > 0)) { 
setWorkSpaceSize(GnxRkpzrPZimKtYYHSuG); if (xkUNToJIgvoLoUQuzKRF) 
{ cudaFree(xkUNToJIgvoLoUQuzKRF); xkUNToJIgvoLoUQuzKRF = 0; } 
CUDA_CALL(cudaMalloc((void**)&xkUNToJIgvoLoUQuzKRF, *getWorkSpaceSize())); }  
if (!xkUNToJIgvoLoUQuzKRF && (*getWorkSpaceSize() > 0)) { throw 
std::runtime_error("Out of memory. Unable to allocate workspace."); } } void 
MWTargetNetworkImpl::createWorkSpace(float** yCdIUfwoZFngCRRRkCTg) { 
cudaError_t qWwjVYwfnvEnFKlgpqwA = cudaMalloc((void**)yCdIUfwoZFngCRRRkCTg, 
omxlPZbBePZdWaJOBUUG); if (qWwjVYwfnvEnFKlgpqwA != cudaSuccess) { 
*yCdIUfwoZFngCRRRkCTg = 0;  } } void 
MWTargetNetworkImpl::setWorkSpaceSize(size_t wss) { omxlPZbBePZdWaJOBUUG 
= wss;  } size_t* MWTargetNetworkImpl::getWorkSpaceSize() { return 
&omxlPZbBePZdWaJOBUUG; } float* MWTargetNetworkImpl::getWorkSpace() { 
return xkUNToJIgvoLoUQuzKRF; } size_t* 
MWTargetNetworkImpl::getPostSetupWorkSpaceSize() { return 
&GnxRkpzrPZimKtYYHSuG; } void 
MWTargetNetworkImpl::setPostSetupWorkSpaceSize(size_t psWSize) { 
GnxRkpzrPZimKtYYHSuG = psWSize; } cublasHandle_t* 
MWTargetNetworkImpl::getCublasHandle() { return QjgQHaUACFNSteMrRtRj; } 
cudnnHandle_t* MWTargetNetworkImpl::getCudnnHandle() { return 
QwUuNuQNtlPXrIwRNiSZ; } void MWTargetNetworkImpl::setAutoTune(bool 
autotune) { MW_autoTune = autotune; } bool MWTargetNetworkImpl::getAutoTune() 
const { return MW_autoTune; } void MWTargetNetworkImpl::cleanup() { if 
(xkUNToJIgvoLoUQuzKRF) { cudaFree(xkUNToJIgvoLoUQuzKRF); } if 
(QjgQHaUACFNSteMrRtRj) { cublasDestroy(*QjgQHaUACFNSteMrRtRj); } if 
(QwUuNuQNtlPXrIwRNiSZ) { cudnnDestroy(*QwUuNuQNtlPXrIwRNiSZ); } for(int 
i = 0; i < numBufs; i++) { float *memPtr = memBuffer[i]; cudaError_t 
qWwjVYwfnvEnFKlgpqwA = cudaFree(memPtr); if (qWwjVYwfnvEnFKlgpqwA != 
cudaErrorCudartUnloading) { CUDA_CALL(qWwjVYwfnvEnFKlgpqwA); } } }