#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#ifdef RANDOM
#include <curand.h>
 curandGenerator_t VCbcPxtPsBLTrHYdEvqn; void 
curand_call_line_file(curandStatus_t qWwjVYwfnvEnFKlgpqwA, const int 
eFaDPmxDdzHlRYSAoMmX, const char *UEESbUvbMihFnquvuFij) { if (qWwjVYwfnvEnFKlgpqwA != 
CURAND_STATUS_SUCCESS) { printf("%d, line: %d, file: %s\n", qWwjVYwfnvEnFKlgpqwA, 
eFaDPmxDdzHlRYSAoMmX, UEESbUvbMihFnquvuFij); exit(EXIT_FAILURE); } }
#endif
 float* malloc_call_line_file(size_t msize, const int eFaDPmxDdzHlRYSAoMmX, const 
char *UEESbUvbMihFnquvuFij) { float * mem = (float*)malloc(msize); if (!mem) { 
printf("%s, line: %d, file: %s\n", "Memory allocation failed. ", 
eFaDPmxDdzHlRYSAoMmX, UEESbUvbMihFnquvuFij); exit(EXIT_FAILURE); } return mem; } void 
call_cuda_free(float* mem) { cudaError_t qWwjVYwfnvEnFKlgpqwA = cudaFree(mem); if 
(qWwjVYwfnvEnFKlgpqwA != cudaErrorCudartUnloading) { CUDA_CALL(qWwjVYwfnvEnFKlgpqwA); 
} } void cuda_call_line_file(cudaError_t qWwjVYwfnvEnFKlgpqwA, const int 
eFaDPmxDdzHlRYSAoMmX, const char *UEESbUvbMihFnquvuFij) { if (qWwjVYwfnvEnFKlgpqwA != 
cudaSuccess) { printf("%s, line: %d, file: %s\n", 
cudaGetErrorString(qWwjVYwfnvEnFKlgpqwA), eFaDPmxDdzHlRYSAoMmX, UEESbUvbMihFnquvuFij); 
exit(EXIT_FAILURE); } } void cudnn_call_line_file(cudnnStatus_t 
qWwjVYwfnvEnFKlgpqwA, const int eFaDPmxDdzHlRYSAoMmX, const char *UEESbUvbMihFnquvuFij) { if 
(qWwjVYwfnvEnFKlgpqwA != CUDNN_STATUS_SUCCESS) { 
printf("%s, line: %d, file: %s\n", cudnnGetErrorString(qWwjVYwfnvEnFKlgpqwA), 
eFaDPmxDdzHlRYSAoMmX, UEESbUvbMihFnquvuFij); exit(EXIT_FAILURE); } } const char* 
cublasGetErrorString(cublasStatus_t qWwjVYwfnvEnFKlgpqwA) { 
switch(qWwjVYwfnvEnFKlgpqwA) { case CUBLAS_STATUS_SUCCESS: return 
"CUBLAS_STATUS_SUCCESS"; case CUBLAS_STATUS_NOT_INITIALIZED: return 
"CUBLAS_STATUS_NOT_INITIALIZED"; case CUBLAS_STATUS_ALLOC_FAILED: return 
"CUBLAS_STATUS_ALLOC_FAILED"; case CUBLAS_STATUS_INVALID_VALUE: return 
"CUBLAS_STATUS_INVALID_VALUE";  case CUBLAS_STATUS_ARCH_MISMATCH: return 
"CUBLAS_STATUS_ARCH_MISMATCH";  case CUBLAS_STATUS_MAPPING_ERROR: return 
"CUBLAS_STATUS_MAPPING_ERROR"; case CUBLAS_STATUS_EXECUTION_FAILED: return 
"CUBLAS_STATUS_EXECUTION_FAILED";  case CUBLAS_STATUS_INTERNAL_ERROR: return 
"CUBLAS_STATUS_INTERNAL_ERROR";  case CUBLAS_STATUS_NOT_SUPPORTED: return 
"CUBLAS_STATUS_NOT_SUPPORTED";  case CUBLAS_STATUS_LICENSE_ERROR: return 
"CUBLAS_STATUS_LICENSE_ERROR";  } return "unknown error"; } void 
cublas_call_line_file(cublasStatus_t qWwjVYwfnvEnFKlgpqwA, const int 
eFaDPmxDdzHlRYSAoMmX, const char *UEESbUvbMihFnquvuFij) { if (qWwjVYwfnvEnFKlgpqwA != 
CUBLAS_STATUS_SUCCESS) { printf("%s, line: %d, file: %s\n", 
cublasGetErrorString(qWwjVYwfnvEnFKlgpqwA), eFaDPmxDdzHlRYSAoMmX, UEESbUvbMihFnquvuFij); 
exit(EXIT_FAILURE); } } MWCNNLayerImpl::MWCNNLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl) : SUleyRyvAggTFnSdxLru(0.0), SGsAudmgjmvcUXzzrUtf(1.0), 
SDWKEQTZaTFZByPlzUDR(-1.0), dMxIKDGTITyhdLqIHBLA(layer), 
fYaOQTeunPwVjnhhTECh(ntwk_impl), RAtlBpdedvgxUsgDTsch(0)  { } 
MWCNNLayerImpl::~MWCNNLayerImpl() { for(std::map<int, 
cudnnTensorDescriptor_t*>::iterator it = lteHjcLsItGbVPMQtGDB.begin(); it != 
lteHjcLsItGbVPMQtGDB.end(); ++it) { delete it->second; it->second = 0; } } 
float* MWCNNLayerImpl::getZeroPtr() { return &SUleyRyvAggTFnSdxLru; } float* 
MWCNNLayerImpl::getOnePtr() { return &SGsAudmgjmvcUXzzrUtf; } float* 
MWCNNLayerImpl::getNegOnePtr() { return &SDWKEQTZaTFZByPlzUDR; } 
cudnnTensorDescriptor_t* MWCNNLayerImpl::getOutputDescriptor(int index) { 
std::map<int, cudnnTensorDescriptor_t*>::iterator it = 
lteHjcLsItGbVPMQtGDB.find(index); if (it == lteHjcLsItGbVPMQtGDB.end()) { 
cudnnTensorDescriptor_t* tmp = new cudnnTensorDescriptor_t;  
lteHjcLsItGbVPMQtGDB[index] = tmp; return tmp; } else { return it->second; } } 
cudnnTensorDescriptor_t* MWCNNLayerImpl::getCuDNNDescriptor(MWTensor* tensor) { 
MWCNNLayerImpl* impl = tensor->getOwner()->getImpl(); if (!impl || 
dynamic_cast<MWPassthroughLayer*>(tensor->getOwner())) { 
assert(dynamic_cast<MWPassthroughLayer*>(tensor->getOwner())); return 
getCuDNNDescriptor(tensor->getOwner()->getInputTensor(0)); } return 
impl->getOutputDescriptor(tensor->getSourcePortIndex()); } void __global__ 
__launch_bounds__(1024) padInputImpl(float* in, int inputH, int inputW, int 
inputCh, int outputH, int outputW, int offsetH, int offsetW, float* out, int 
inputElems) { for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < 
inputElems; i+= blockDim.x*gridDim.x) { int idxB = i/(inputH*inputW*inputCh); 
int rem = (i - idxB*(inputH*inputW*inputCh)); int idxCh = rem/(inputH*inputW); 
int rem1 = rem - idxCh*(inputH*inputW); int idxH = rem1/inputW; int idxCol = 
rem1 - idxH*inputW; if ((idxH < inputH) && (idxCol < inputW)) { int outputR = 
idxH + offsetH; int outputCol = idxCol + offsetW; int outputCh = inputCh; *(out 
+ idxB*(outputH*outputW*outputCh) + idxCh*(outputH*outputW) + outputR*(outputW) 
+ outputCol) = *(in + i); } } } void MWCNNLayerImpl::padInput(float* 
XLJXOFXdnZOyJvtltbyr, int bDTIjtxZiSHtjwzgEluE, int bMAyVFGSPDjmUbziYLAy, int 
atVCyzqXZAZxwlkRLBRA, int nNULvWnBXnnWdpEkHPAH, int nlIRrOJaFuVaywxOqOyb, int 
jaqKGCwoANNDMHgAsehk, int jhFUWlztBndwjbXwYNaJ, float* kNsviQGMPdXzNMRixGWR, int 
gzSTokDHvkXefhiGDcWL) { int tqZLvfMHdgZzbchUyDzd = (gzSTokDHvkXefhiGDcWL < 
1024) ? gzSTokDHvkXefhiGDcWL : 1024; int NldNILHvuQqQPSAHXxdT = 
(gzSTokDHvkXefhiGDcWL + tqZLvfMHdgZzbchUyDzd - 
1)/tqZLvfMHdgZzbchUyDzd; padInputImpl<<<NldNILHvuQqQPSAHXxdT, 
tqZLvfMHdgZzbchUyDzd>>>(XLJXOFXdnZOyJvtltbyr, bDTIjtxZiSHtjwzgEluE, 
bMAyVFGSPDjmUbziYLAy, atVCyzqXZAZxwlkRLBRA, nNULvWnBXnnWdpEkHPAH, nlIRrOJaFuVaywxOqOyb, 
jaqKGCwoANNDMHgAsehk, jhFUWlztBndwjbXwYNaJ, kNsviQGMPdXzNMRixGWR, gzSTokDHvkXefhiGDcWL); } 
MWInputLayerImpl::MWInputLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int fSKMHAqIghbYYgyIpNDw, int WprSrhAStKGxyXeoxETy, int vjDFlBZzKvbpPseAtMBP, int 
OumvfgWXDdmsQaciHMHx, int xHViLEwTujGGrPZZgmbF, const char* avg_file_name, int outbufIdx) 
: MWCNNLayerImpl(layer, ntwk_impl) { createInputLayer(fSKMHAqIghbYYgyIpNDw, 
WprSrhAStKGxyXeoxETy, vjDFlBZzKvbpPseAtMBP, OumvfgWXDdmsQaciHMHx, xHViLEwTujGGrPZZgmbF, avg_file_name, 
outbufIdx); } MWInputLayerImpl::~MWInputLayerImpl() { } void 
MWInputLayerImpl::createInputLayer(int fSKMHAqIghbYYgyIpNDw, int WprSrhAStKGxyXeoxETy, int 
vjDFlBZzKvbpPseAtMBP, int OumvfgWXDdmsQaciHMHx, int xHViLEwTujGGrPZZgmbF, const char* 
avg_file_name, int outbufIdx){ if (outbufIdx < 0) { 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP*OumvfgWXDdmsQaciHMHx*fSKMHAqIghbYYgyIpNDw)); } else { 
setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); } 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
CUDNN_CALL(cudnnCreateTensorDescriptor(&MIBnYCbKBdUrlfqlHdoo)); 
dJcdBfQQLhIAYHPxwQeg = xHViLEwTujGGrPZZgmbF; 
fYaOQTeunPwVjnhhTECh->setWorkSpaceSize(0); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fSKMHAqIghbYYgyIpNDw, OumvfgWXDdmsQaciHMHx, WprSrhAStKGxyXeoxETy, 
vjDFlBZzKvbpPseAtMBP)); if( dJcdBfQQLhIAYHPxwQeg == 1) { 
CUDNN_CALL(cudnnSetTensor4dDescriptor(MIBnYCbKBdUrlfqlHdoo, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, 1, OumvfgWXDdmsQaciHMHx, WprSrhAStKGxyXeoxETy, vjDFlBZzKvbpPseAtMBP)); 
CUDA_CALL(cudaMalloc((void**)&IbSWJNMuIiKbocfQKqXb, 
sizeof(float)*OumvfgWXDdmsQaciHMHx*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP)); int fhikqqlnUKCjleVKDqiG = 
OumvfgWXDdmsQaciHMHx*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP;  loadAvg(avg_file_name, 
fhikqqlnUKCjleVKDqiG); } else if (dJcdBfQQLhIAYHPxwQeg == 2){ 
CUDA_CALL(cudaMalloc((void**)&IbSWJNMuIiKbocfQKqXb, sizeof(float)*OumvfgWXDdmsQaciHMHx)); int 
fhikqqlnUKCjleVKDqiG = OumvfgWXDdmsQaciHMHx;  loadAvg(avg_file_name, fhikqqlnUKCjleVKDqiG); }
#ifdef RANDOM
 curandGenerateUniform(VCbcPxtPsBLTrHYdEvqn, MW_data, fSKMHAqIghbYYgyIpNDw*OumvfgWXDdmsQaciHMHx*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP);
#endif
 fYaOQTeunPwVjnhhTECh->setWorkSpaceSize(0); return; } void 
MWInputLayerImpl::loadAvg(const char* UKtMXCCqdjeyaVHabkxg, int fhikqqlnUKCjleVKDqiG) 
{ FILE* UzaGmBLFEwmwaFXebUma = MWCNNLayer::openBinaryFile(UKtMXCCqdjeyaVHabkxg); 
assert(UzaGmBLFEwmwaFXebUma); float* OKaRVOctKLlnIyGmjRNW = 
MALLOC_CALL(sizeof(float)*fhikqqlnUKCjleVKDqiG); fread(OKaRVOctKLlnIyGmjRNW, 
sizeof(float), fhikqqlnUKCjleVKDqiG, UzaGmBLFEwmwaFXebUma); 
CUDA_CALL(cudaMemcpy(IbSWJNMuIiKbocfQKqXb, OKaRVOctKLlnIyGmjRNW, 
sizeof(float)*fhikqqlnUKCjleVKDqiG, cudaMemcpyHostToDevice)); 
free(OKaRVOctKLlnIyGmjRNW); fclose(UzaGmBLFEwmwaFXebUma); return; } void __global__ 
__launch_bounds__(1024) subtractMeanPerChannelImpl(float * 
eybNKlJCSDUvsznWynwK, float * REXdEoRjxuQJkqgIDihy, const int 
fxxCPKTclxXPxrdMAkwi, const int fvTCtkwXgyScJYogJVFU, const int 
CDJtexcMbXMWAmnNZsNf) {  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < 
CDJtexcMbXMWAmnNZsNf; i+= blockDim.x*gridDim.x) {  int idx = static_cast<int>((i % 
fvTCtkwXgyScJYogJVFU) / fxxCPKTclxXPxrdMAkwi); 
REXdEoRjxuQJkqgIDihy[i] -= eybNKlJCSDUvsznWynwK[idx]; } } void 
MWInputLayerImpl::predict() { if ( dJcdBfQQLhIAYHPxwQeg == 1) 
CUDNN_CALL(cudnnAddTensor(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
getNegOnePtr(), MIBnYCbKBdUrlfqlHdoo, IbSWJNMuIiKbocfQKqXb, getOnePtr(), 
*getOutputDescriptor(), getData())); else if( dJcdBfQQLhIAYHPxwQeg == 2){ 
MWInputLayer* thisLayer = static_cast<MWInputLayer*>(getLayer()); MWTensor* 
opTensor = thisLayer->getOutputTensor(0); int fjfzkUfcCOqjrkAVGfuc = 
opTensor->getHeight()*opTensor->getWidth()* 
opTensor->getChannels()*opTensor->getBatchSize(); int 
fxxCPKTclxXPxrdMAkwi = opTensor->getHeight() * opTensor->getWidth(); 
int fvTCtkwXgyScJYogJVFU = 
fxxCPKTclxXPxrdMAkwi*opTensor->getChannels(); int 
tqZLvfMHdgZzbchUyDzd = (fjfzkUfcCOqjrkAVGfuc < 1024) ? fjfzkUfcCOqjrkAVGfuc : 
1024; int NldNILHvuQqQPSAHXxdT = (fjfzkUfcCOqjrkAVGfuc + 
tqZLvfMHdgZzbchUyDzd - 1)/tqZLvfMHdgZzbchUyDzd; 
subtractMeanPerChannelImpl<<<NldNILHvuQqQPSAHXxdT, 
tqZLvfMHdgZzbchUyDzd>>>( IbSWJNMuIiKbocfQKqXb, getData(), 
fxxCPKTclxXPxrdMAkwi, fvTCtkwXgyScJYogJVFU, fjfzkUfcCOqjrkAVGfuc); 
} return; } void MWInputLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); for(int idx = 
0; idx < dMxIKDGTITyhdLqIHBLA->getNumOutputs(); idx++) {  float* data = 
dMxIKDGTITyhdLqIHBLA->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } if ( dJcdBfQQLhIAYHPxwQeg == 1) { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(MIBnYCbKBdUrlfqlHdoo)); if (IbSWJNMuIiKbocfQKqXb) 
{ call_cuda_free(IbSWJNMuIiKbocfQKqXb); } } return; } 
MWReLULayerImpl::MWReLULayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int inPlace, int outbufIdx)  : MWCNNLayerImpl(layer, ntwk_impl) , 
XYbzSmRQGatVJtGmDZSo(inPlace)  { 
CUDNN_CALL(cudnnCreateActivationDescriptor(&olKGEIcsxmLSoMhRhEtP)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
createReLULayer(outbufIdx); } MWReLULayerImpl::~MWReLULayerImpl() { } void 
MWReLULayerImpl::createReLULayer(int outbufIdx) { MWReLULayer* reluLayer = 
static_cast<MWReLULayer*>(getLayer()); MWTensor* ipTensor = 
reluLayer->getInputTensor(0); MWTensor* opTensor = 
reluLayer->getOutputTensor(0); 
CUDNN_CALL(cudnnSetActivationDescriptor(olKGEIcsxmLSoMhRhEtP, 
CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));  
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, opTensor->getBatchSize(), 
opTensor->getChannels(), opTensor->getHeight(), opTensor->getWidth())); if 
(outbufIdx < 0) { if(XYbzSmRQGatVJtGmDZSo){ RAtlBpdedvgxUsgDTsch = 
getLayer()->getInputTensor()->getData(); } else{ 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*opTensor->getHeight()* 
opTensor->getWidth()*opTensor->getChannels()*opTensor->getBatchSize())); } } 
else { setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
reluLayer->getOutputTensor(0)->setopBufIndex(outbufIdx); }  } void 
MWReLULayerImpl::predict() { MWReLULayer* reluLayer = 
static_cast<MWReLULayer*>(getLayer()); cudnnTensorDescriptor_t ipDesc = 
*getCuDNNDescriptor(reluLayer->getInputTensor()); 
CUDNN_CALL(cudnnActivationForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
olKGEIcsxmLSoMhRhEtP, getOnePtr(), ipDesc, 
reluLayer->getInputTensor()->getData(), getZeroPtr(), *getOutputDescriptor(), 
RAtlBpdedvgxUsgDTsch)); } void MWReLULayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyActivationDescriptor(olKGEIcsxmLSoMhRhEtP)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); MWTensor* op 
= getLayer()->getOutputTensor(0); float* data = op->getData(); if (data) { 
if((op->getopBufIndex() < 0) && !XYbzSmRQGatVJtGmDZSo) call_cuda_free(data); } } 
MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, unsigned IAlDgIFcchbwRGBSfVfA,  double AFQBkxwYGKLsACiDKwRM,  
double AHqhysOOIgbDpWZoPUFT,  double BUOdotSvmFyUWQKMUdra, int outbufIdx) : 
MWCNNLayerImpl(layer, ntwk_impl)  { 
CUDNN_CALL(cudnnCreateLRNDescriptor(&fSbUUBgjKRbNXrHrlOLo)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
createNormLayer(IAlDgIFcchbwRGBSfVfA, AFQBkxwYGKLsACiDKwRM, 
AHqhysOOIgbDpWZoPUFT, BUOdotSvmFyUWQKMUdra, outbufIdx); } 
MWNormLayerImpl::~MWNormLayerImpl() { } void MWNormLayerImpl::createNormLayer( 
unsigned IAlDgIFcchbwRGBSfVfA,  double AFQBkxwYGKLsACiDKwRM,  double 
AHqhysOOIgbDpWZoPUFT,  double BUOdotSvmFyUWQKMUdra, int outbufIdx) { MWNormLayer* normLayer 
= static_cast<MWNormLayer*>(getLayer()); MWTensor* ipTensor = 
normLayer->getInputTensor(0); MWTensor* opTensor = 
normLayer->getOutputTensor(0); int numOutputFeatures = opTensor->getChannels(); 
if (outbufIdx < 0) { CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*opTensor->getHeight()*opTensor->getWidth()*numOutputFeatures*opTensor->getBatchSize())); 
} else { setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
normLayer->getOutputTensor(0)->setopBufIndex(outbufIdx); } 
CUDNN_CALL(cudnnSetLRNDescriptor(fSbUUBgjKRbNXrHrlOLo, 
IAlDgIFcchbwRGBSfVfA, AFQBkxwYGKLsACiDKwRM, AHqhysOOIgbDpWZoPUFT, 
BUOdotSvmFyUWQKMUdra)); CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, opTensor->getBatchSize(), 
opTensor->getChannels(), opTensor->getHeight(), opTensor->getWidth())); return; 
} void MWNormLayerImpl::predict() { MWNormLayer* normLayer = 
static_cast<MWNormLayer*>(getLayer()); cudnnTensorDescriptor_t ipDesc = 
*getCuDNNDescriptor(normLayer->getInputTensor()); 
CUDNN_CALL(cudnnLRNCrossChannelForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
fSbUUBgjKRbNXrHrlOLo, CUDNN_LRN_CROSS_CHANNEL_DIM1, getOnePtr(), ipDesc, 
normLayer->getInputTensor()->getData(),getZeroPtr(), *getOutputDescriptor(), 
normLayer->getOutputTensor()->getData())); } void MWNormLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyLRNDescriptor(fSbUUBgjKRbNXrHrlOLo)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); for(int idx = 
0; idx < getLayer()->getNumOutputs(); idx++) {  MWTensor* op = 
getLayer()->getOutputTensor(idx); float* data = op->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0 ) 
call_cuda_free(data); } }  } void __global__ MWSetDyForBackPropImpl(float * 
SIBpKtDURUWQaaenbwrC, const int hljcfGWsvZXJZNrImpJB); void __global__ 
doMWMaxPoolingLayerImpl(float * ZinudJuZuGitiNTsJpBR, float * 
ZDWLzHUkuZuIUZHfbGDY, const int CDJtexcMbXMWAmnNZsNf); 
MWMaxPoolingLayerImpl::MWMaxPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int GFienSVKLlDQuZeqAdLC,  int GeeOVBfQrpMacIFBLKOo,  
int GsZlHFuhbvjLtRMDjXnW,  int HJHXkKmgFxxIOsIvRRnF, int EvebzoroiuKkIxwjkGnD, int 
ECTnqgWHyHCHCLBZlffd,  int FrpxvsDMwwgbpqHXWxmN, int FwLnexHgxHRquTKmNpoa, 
bool JgLfgHrHMEMmMYTettJF, int iPqBiFnIJMxelVhQBZex, const std::vector<int>& 
NtWaRGCHLeTapjWdEHHS) : MWCNNLayerImpl(layer, ntwk_impl) , 
BRSPqxNffoBYKqpSVHne(JgLfgHrHMEMmMYTettJF) , ZinudJuZuGitiNTsJpBR(0) 
, SIBpKtDURUWQaaenbwrC(0) , ZDWLzHUkuZuIUZHfbGDY(0)  {  
CUDNN_CALL(cudnnCreatePoolingDescriptor(&npGnQZLrEfVTQnEbwqij)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
createMaxPoolingLayer(GFienSVKLlDQuZeqAdLC,GeeOVBfQrpMacIFBLKOo,GsZlHFuhbvjLtRMDjXnW,HJHXkKmgFxxIOsIvRRnF,EvebzoroiuKkIxwjkGnD,ECTnqgWHyHCHCLBZlffd,FrpxvsDMwwgbpqHXWxmN,FwLnexHgxHRquTKmNpoa, 
iPqBiFnIJMxelVhQBZex, NtWaRGCHLeTapjWdEHHS); } 
MWMaxPoolingLayerImpl::~MWMaxPoolingLayerImpl() { } void 
MWMaxPoolingLayerImpl::createMaxPoolingLayer(int GFienSVKLlDQuZeqAdLC,  int 
GeeOVBfQrpMacIFBLKOo,  int GsZlHFuhbvjLtRMDjXnW, int HJHXkKmgFxxIOsIvRRnF, int 
EvebzoroiuKkIxwjkGnD, int ECTnqgWHyHCHCLBZlffd,  int FrpxvsDMwwgbpqHXWxmN, 
int FwLnexHgxHRquTKmNpoa, int iPqBiFnIJMxelVhQBZex, const std::vector<int>& 
NtWaRGCHLeTapjWdEHHS) { MWMaxPoolingLayer* maxpoolLayer = 
static_cast<MWMaxPoolingLayer*>(getLayer()); MWTensor* ipTensor = 
maxpoolLayer->getInputTensor(0); int oJUVMnJggjhEdQLWzIUC = 
EvebzoroiuKkIxwjkGnD; int oYbqYsqgVhrUzFEKbBbR = 
FrpxvsDMwwgbpqHXWxmN; cudnnTensorDescriptor_t bUVPfnrJhLfHzOLUUrKk = 
*getCuDNNDescriptor(ipTensor);  
CUDNN_CALL(cudnnSetPooling2dDescriptor(npGnQZLrEfVTQnEbwqij, CUDNN_POOLING_MAX, 
CUDNN_NOT_PROPAGATE_NAN, GFienSVKLlDQuZeqAdLC, GeeOVBfQrpMacIFBLKOo, 
oJUVMnJggjhEdQLWzIUC, oYbqYsqgVhrUzFEKbBbR, GsZlHFuhbvjLtRMDjXnW, 
HJHXkKmgFxxIOsIvRRnF)); int fSKMHAqIghbYYgyIpNDw, OumvfgWXDdmsQaciHMHx, WprSrhAStKGxyXeoxETy, 
vjDFlBZzKvbpPseAtMBP; CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(npGnQZLrEfVTQnEbwqij, 
bUVPfnrJhLfHzOLUUrKk, &fSKMHAqIghbYYgyIpNDw ,&OumvfgWXDdmsQaciHMHx, &WprSrhAStKGxyXeoxETy, 
&vjDFlBZzKvbpPseAtMBP)); WprSrhAStKGxyXeoxETy = getLayer()->getOutputTensor(0)->getHeight(); 
vjDFlBZzKvbpPseAtMBP = getLayer()->getOutputTensor(0)->getWidth(); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fSKMHAqIghbYYgyIpNDw, OumvfgWXDdmsQaciHMHx, WprSrhAStKGxyXeoxETy, 
vjDFlBZzKvbpPseAtMBP)); int outbufIdx = NtWaRGCHLeTapjWdEHHS[0]; if (outbufIdx < 0) { 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*fSKMHAqIghbYYgyIpNDw*OumvfgWXDdmsQaciHMHx*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP)); } else { 
setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
maxpoolLayer->getOutputTensor(0)->setopBufIndex(outbufIdx); } if 
(BRSPqxNffoBYKqpSVHne){ 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor(1))); const int 
fjfzkUfcCOqjrkAVGfuc = 
(ipTensor->getHeight())*(ipTensor->getWidth())*(ipTensor->getChannels())*(ipTensor->getBatchSize()); 
CUDA_CALL(cudaMalloc((void**)&ZinudJuZuGitiNTsJpBR, 
sizeof(float)*fjfzkUfcCOqjrkAVGfuc)); assert(iPqBiFnIJMxelVhQBZex == 2); int 
bufIndex = NtWaRGCHLeTapjWdEHHS[1]; if (bufIndex < 0) { 
CUDA_CALL(cudaMalloc((void**)&ZDWLzHUkuZuIUZHfbGDY, 
sizeof(float)*fSKMHAqIghbYYgyIpNDw*OumvfgWXDdmsQaciHMHx*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP)); } else { 
ZDWLzHUkuZuIUZHfbGDY = fYaOQTeunPwVjnhhTECh->memBuffer[bufIndex]; 
maxpoolLayer->getOutputTensor(1)->setopBufIndex(bufIndex); } 
assert((OumvfgWXDdmsQaciHMHx == ipTensor->getChannels()) && (fSKMHAqIghbYYgyIpNDw == 
ipTensor->getBatchSize()));  const int hljcfGWsvZXJZNrImpJB = 
vjDFlBZzKvbpPseAtMBP*WprSrhAStKGxyXeoxETy*OumvfgWXDdmsQaciHMHx*fSKMHAqIghbYYgyIpNDw; 
CUDA_CALL(cudaMalloc((void**)&SIBpKtDURUWQaaenbwrC, 
sizeof(float)*hljcfGWsvZXJZNrImpJB)); int tqZLvfMHdgZzbchUyDzd = 
(hljcfGWsvZXJZNrImpJB < 1024) ? hljcfGWsvZXJZNrImpJB : 1024; int 
NldNILHvuQqQPSAHXxdT = (hljcfGWsvZXJZNrImpJB + tqZLvfMHdgZzbchUyDzd - 
1)/tqZLvfMHdgZzbchUyDzd; 
MWSetDyForBackPropImpl<<<NldNILHvuQqQPSAHXxdT, 
tqZLvfMHdgZzbchUyDzd>>>( SIBpKtDURUWQaaenbwrC, hljcfGWsvZXJZNrImpJB); } } void 
MWMaxPoolingLayerImpl::predict() { MWMaxPoolingLayer* maxpoolLayer = 
static_cast<MWMaxPoolingLayer*>(getLayer()); cudnnTensorDescriptor_t 
bUVPfnrJhLfHzOLUUrKk = *getCuDNNDescriptor(maxpoolLayer->getInputTensor()); 
MWTensor* ipTensor = getLayer()->getInputTensor(0); 
CUDNN_CALL(cudnnPoolingForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
npGnQZLrEfVTQnEbwqij, getOnePtr(), bUVPfnrJhLfHzOLUUrKk, ipTensor->getData(), 
getZeroPtr(), *getOutputDescriptor(), 
maxpoolLayer->getOutputTensor()->getData())); if (BRSPqxNffoBYKqpSVHne) { 
CUDNN_CALL(cudnnPoolingBackward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
npGnQZLrEfVTQnEbwqij, getOnePtr(), *getOutputDescriptor(0), 
getLayer()->getOutputTensor(0)->getData(), *getOutputDescriptor(0), 
SIBpKtDURUWQaaenbwrC, bUVPfnrJhLfHzOLUUrKk, ipTensor->getData(), getZeroPtr(), 
bUVPfnrJhLfHzOLUUrKk, ZinudJuZuGitiNTsJpBR)); int fjfzkUfcCOqjrkAVGfuc = 
ipTensor->getHeight()*(ipTensor->getWidth())*(ipTensor->getChannels())*(ipTensor->getBatchSize()); 
int tqZLvfMHdgZzbchUyDzd = (fjfzkUfcCOqjrkAVGfuc < 1024) ? 
fjfzkUfcCOqjrkAVGfuc : 1024; int NldNILHvuQqQPSAHXxdT = (fjfzkUfcCOqjrkAVGfuc + 
tqZLvfMHdgZzbchUyDzd - 1)/tqZLvfMHdgZzbchUyDzd; 
doMWMaxPoolingLayerImpl<<<NldNILHvuQqQPSAHXxdT, 
tqZLvfMHdgZzbchUyDzd>>>( ZinudJuZuGitiNTsJpBR, 
maxpoolLayer->getOutputTensor(1)->getData(), fjfzkUfcCOqjrkAVGfuc); } return; } 
void MWMaxPoolingLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyPoolingDescriptor(npGnQZLrEfVTQnEbwqij)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); if 
(BRSPqxNffoBYKqpSVHne){ 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor(1))); } for(int 
idx = 0; idx < getLayer()->getNumOutputs(); idx++) {  float* data = 
getLayer()->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } if (ZinudJuZuGitiNTsJpBR){ 
call_cuda_free(ZinudJuZuGitiNTsJpBR); } if (SIBpKtDURUWQaaenbwrC){ 
call_cuda_free(SIBpKtDURUWQaaenbwrC); }  } float* 
MWMaxPoolingLayerImpl::getIndexData()  { return ZDWLzHUkuZuIUZHfbGDY; } void 
__global__ __launch_bounds__(1024) MWSetDyForBackPropImpl(float * 
SIBpKtDURUWQaaenbwrC, const int hljcfGWsvZXJZNrImpJB) { for(int i = blockDim.x * 
blockIdx.x + threadIdx.x; i < hljcfGWsvZXJZNrImpJB; i+= blockDim.x*gridDim.x) { 
SIBpKtDURUWQaaenbwrC[i] = i+1; } } void __global__ __launch_bounds__(1024) 
doMWMaxPoolingLayerImpl(float * ZinudJuZuGitiNTsJpBR, float * 
ZDWLzHUkuZuIUZHfbGDY, const int CDJtexcMbXMWAmnNZsNf) { for(int i = blockDim.x * 
blockIdx.x + threadIdx.x; i < CDJtexcMbXMWAmnNZsNf; i+= blockDim.x*gridDim.x) { if 
(static_cast<int>(ZinudJuZuGitiNTsJpBR[i]) != 0){ 
ZDWLzHUkuZuIUZHfbGDY[static_cast<int>(ZinudJuZuGitiNTsJpBR[i])-1] = 
i; } } } MWFCLayerImpl::MWFCLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int CpMjJjtGOeWOzwxpAAQP, const char* 
wMySyzzledUmSLTWhuYH,  const char* NZjOkZPwLzQsdEVkwMcX, int outbufIdx) : 
MWCNNLayerImpl(layer, ntwk_impl)  { 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
CUDNN_CALL(cudnnCreateTensorDescriptor(&NMMfJylfQjiIUAKhXCJb)); 
createFCLayer(CpMjJjtGOeWOzwxpAAQP, wMySyzzledUmSLTWhuYH, 
NZjOkZPwLzQsdEVkwMcX, outbufIdx); } MWFCLayerImpl::~MWFCLayerImpl() { } void 
MWFCLayerImpl::createFCLayer( int CpMjJjtGOeWOzwxpAAQP, const char* 
wMySyzzledUmSLTWhuYH, const char* NZjOkZPwLzQsdEVkwMcX, int outbufIdx) { 
MWFCLayer* fcLayer = static_cast<MWFCLayer*>(getLayer()); MWTensor* opTensor = 
fcLayer->getOutputTensor(0); if (outbufIdx < 0) { 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*fcLayer->getOutputTensor()->getBatchSize()*fcLayer->getOutputTensor()->getChannels())); 
} else { setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
fcLayer->getOutputTensor(0)->setopBufIndex(outbufIdx); } 
CUDA_CALL(cudaMalloc((void**)&vjDFlBZzKvbpPseAtMBP, 
sizeof(float)*CpMjJjtGOeWOzwxpAAQP* 
fcLayer->getOutputTensor()->getChannels())); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
fcLayer->getOutputTensor()->getBatchSize(),fcLayer->getOutputTensor()->getChannels(), 
1, 1)); CUDNN_CALL(cudnnSetTensor4dDescriptor(NMMfJylfQjiIUAKhXCJb, 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 
fcLayer->getOutputTensor()->getChannels(), 1, 1)); 
CUDA_CALL(cudaMalloc((void**)&MNuwXDSoGEYeABeVTwOh, sizeof(float)*fcLayer->getOutputTensor()->getChannels()));
#ifdef RANDOM
 curandGenerateNormal(VCbcPxtPsBLTrHYdEvqn, vjDFlBZzKvbpPseAtMBP, 
fcLayer->getInputTensor()->getChannels()*fcLayer->getInputTensor()->getWidth()*fcLayer->getInputTensor()->getHeight()*fcLayer->getOutputTensor()->getChannels(), 
0, 0.1); curandGenerateNormal(VCbcPxtPsBLTrHYdEvqn, MNuwXDSoGEYeABeVTwOh, 
fcLayer->getOutputTensor()->getChannels(), -0.5, 1);
#endif
 int fhikqqlnUKCjleVKDqiG = CpMjJjtGOeWOzwxpAAQP*opTensor->getChannels();  
loadWeights(fhikqqlnUKCjleVKDqiG, wMySyzzledUmSLTWhuYH); 
loadBias(NZjOkZPwLzQsdEVkwMcX); return; } void MWFCLayerImpl::loadWeights(int 
fhikqqlnUKCjleVKDqiG, const char* UKtMXCCqdjeyaVHabkxg) {  MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(0); FILE* UzaGmBLFEwmwaFXebUma = 
MWCNNLayer::openBinaryFile(UKtMXCCqdjeyaVHabkxg); assert(UzaGmBLFEwmwaFXebUma); float* 
OKaRVOctKLlnIyGmjRNW = MALLOC_CALL(sizeof(float)*fhikqqlnUKCjleVKDqiG); 
fread(OKaRVOctKLlnIyGmjRNW, sizeof(float), fhikqqlnUKCjleVKDqiG, UzaGmBLFEwmwaFXebUma); if( 
ipTensor->getHeight() != 1 && ipTensor->getWidth() != 1 ) { float* 
ONvcEjLBnVNUdjMKOAwF = 
MALLOC_CALL(sizeof(float)*ipTensor->getHeight()*ipTensor->getWidth()); for(int 
k=0; k<fhikqqlnUKCjleVKDqiG/ipTensor->getHeight()/ipTensor->getWidth(); k++) { 
for(int i=0; i<ipTensor->getHeight()*ipTensor->getWidth(); i++) 
ONvcEjLBnVNUdjMKOAwF[i]=OKaRVOctKLlnIyGmjRNW[k*ipTensor->getHeight()*ipTensor->getWidth()+i]; 
for(int j=0; j<ipTensor->getHeight(); j++) for(int i=0; i<ipTensor->getWidth(); 
i++) 
OKaRVOctKLlnIyGmjRNW[k*ipTensor->getHeight()*ipTensor->getWidth()+j*ipTensor->getWidth()+i]=ONvcEjLBnVNUdjMKOAwF[j+i*ipTensor->getHeight()]; 
} free(ONvcEjLBnVNUdjMKOAwF); } CUDA_CALL(cudaMemcpy(vjDFlBZzKvbpPseAtMBP, 
OKaRVOctKLlnIyGmjRNW, sizeof(float)*fhikqqlnUKCjleVKDqiG, cudaMemcpyHostToDevice));
#if 0
 printf("%s loaded. Size = %d. %f\n", UKtMXCCqdjeyaVHabkxg, fhikqqlnUKCjleVKDqiG, OKaRVOctKLlnIyGmjRNW[0]);
#endif
 free(OKaRVOctKLlnIyGmjRNW); fclose(UzaGmBLFEwmwaFXebUma); return; } void 
MWFCLayerImpl::loadBias(const char* UKtMXCCqdjeyaVHabkxg) { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* opTensor = 
fcLayer->getOutputTensor(0); FILE* UzaGmBLFEwmwaFXebUma = 
MWCNNLayer::openBinaryFile(UKtMXCCqdjeyaVHabkxg); assert(UzaGmBLFEwmwaFXebUma); int 
fhikqqlnUKCjleVKDqiG = opTensor->getChannels();  float* OKaRVOctKLlnIyGmjRNW = 
MALLOC_CALL(sizeof(float)*fhikqqlnUKCjleVKDqiG); fread(OKaRVOctKLlnIyGmjRNW, 
sizeof(float), fhikqqlnUKCjleVKDqiG, UzaGmBLFEwmwaFXebUma); 
CUDA_CALL(cudaMemcpy(MNuwXDSoGEYeABeVTwOh, OKaRVOctKLlnIyGmjRNW, 
sizeof(float)*fhikqqlnUKCjleVKDqiG, cudaMemcpyHostToDevice)); 
free(OKaRVOctKLlnIyGmjRNW); fclose(UzaGmBLFEwmwaFXebUma); return; } void 
MWFCLayerImpl::predict() { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(0); MWTensor* opTensor = fcLayer->getOutputTensor(0); 
int CpMjJjtGOeWOzwxpAAQP = 
ipTensor->getChannels()*ipTensor->getHeight()*ipTensor->getWidth(); int 
DSsxcjIrUgZCKZovyNQf = opTensor->getChannels(); int YgcpEBUCwCLaPhyntIio=1; 
int ZCArwzdUdwQuFQUWjnUE=1; if( opTensor->getBatchSize()==1 ) { 
CUDA_CALL(cudaMemcpy(getData(), MNuwXDSoGEYeABeVTwOh, 
sizeof(float)*DSsxcjIrUgZCKZovyNQf, cudaMemcpyDeviceToDevice)); 
CUBLAS_CALL(cublasSgemv(*fYaOQTeunPwVjnhhTECh->getCublasHandle(), CUBLAS_OP_T, 
CpMjJjtGOeWOzwxpAAQP, DSsxcjIrUgZCKZovyNQf, getOnePtr(), 
vjDFlBZzKvbpPseAtMBP, CpMjJjtGOeWOzwxpAAQP, ipTensor->getData(), 
YgcpEBUCwCLaPhyntIio, getOnePtr(),getData(), ZCArwzdUdwQuFQUWjnUE)); } else { 
CUBLAS_CALL(cublasSgemm(*fYaOQTeunPwVjnhhTECh->getCublasHandle(), CUBLAS_OP_T, 
CUBLAS_OP_N, DSsxcjIrUgZCKZovyNQf, opTensor->getBatchSize(), 
CpMjJjtGOeWOzwxpAAQP, getOnePtr(), vjDFlBZzKvbpPseAtMBP, 
CpMjJjtGOeWOzwxpAAQP, ipTensor->getData(), CpMjJjtGOeWOzwxpAAQP, 
getZeroPtr(),getData(), DSsxcjIrUgZCKZovyNQf)); 
CUDNN_CALL(cudnnAddTensor(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), getOnePtr(), 
NMMfJylfQjiIUAKhXCJb, MNuwXDSoGEYeABeVTwOh, getOnePtr(), 
*getOutputDescriptor(),getData())); } return; } void MWFCLayerImpl::cleanup() { 
if (vjDFlBZzKvbpPseAtMBP) { call_cuda_free(vjDFlBZzKvbpPseAtMBP); }  
CUDNN_CALL(cudnnDestroyTensorDescriptor(NMMfJylfQjiIUAKhXCJb)); if 
(MNuwXDSoGEYeABeVTwOh) { call_cuda_free(MNuwXDSoGEYeABeVTwOh); } 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); for(int idx = 
0; idx < getLayer()->getNumOutputs(); idx++) {  float* data = 
getLayer()->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } } MWSoftmaxLayerImpl::MWSoftmaxLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int outbufIdx) : MWCNNLayerImpl(layer, 
ntwk_impl)  {  CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
createSoftmaxLayer(outbufIdx); } MWSoftmaxLayerImpl::~MWSoftmaxLayerImpl() { } 
void MWSoftmaxLayerImpl::createSoftmaxLayer(int outbufIdx) { MWSoftmaxLayer* 
sfmxLayer = static_cast<MWSoftmaxLayer*>(getLayer()); MWTensor* ipTensor = 
sfmxLayer->getInputTensor(0); MWTensor* opTensor = 
sfmxLayer->getOutputTensor(0); int numOutputFeatures = ipTensor->getChannels(); 
if (outbufIdx < 0) { CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*ipTensor->getHeight()*ipTensor->getWidth()*numOutputFeatures*ipTensor->getBatchSize())); 
} else { setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
opTensor->setopBufIndex(outbufIdx); } 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, opTensor->getBatchSize(), 
opTensor->getChannels(), opTensor->getHeight(), opTensor->getWidth()));  
return; } void MWSoftmaxLayerImpl::predict() { MWSoftmaxLayer* sfmxLayer = 
static_cast<MWSoftmaxLayer*>(getLayer()); MWTensor* ipTensor = 
sfmxLayer->getInputTensor(0); MWTensor* opTensor = 
sfmxLayer->getOutputTensor(0); cudnnTensorDescriptor_t ipDesc = 
*getCuDNNDescriptor(ipTensor);  
CUDNN_CALL(cudnnSoftmaxForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, getOnePtr(), ipDesc, 
ipTensor->getData(), getZeroPtr(), *getOutputDescriptor(), getData())); } void 
MWSoftmaxLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); for(int idx = 
0; idx < getLayer()->getNumOutputs(); idx++) {  float* data = 
getLayer()->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } } MWAvgPoolingLayerImpl::MWAvgPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int GFienSVKLlDQuZeqAdLC,  int GeeOVBfQrpMacIFBLKOo,  
int GsZlHFuhbvjLtRMDjXnW,  int HJHXkKmgFxxIOsIvRRnF,  int DqxLTLaJwwgQqmrtCDuu,  int 
FeVcBgtQmTLtmnNcJGMY, int outbufIdx) : MWCNNLayerImpl(layer, ntwk_impl)  { 
CUDNN_CALL(cudnnCreatePoolingDescriptor(&npGnQZLrEfVTQnEbwqij)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
createAvgPoolingLayer(GFienSVKLlDQuZeqAdLC, GeeOVBfQrpMacIFBLKOo, GsZlHFuhbvjLtRMDjXnW, 
HJHXkKmgFxxIOsIvRRnF, DqxLTLaJwwgQqmrtCDuu, FeVcBgtQmTLtmnNcJGMY, outbufIdx); } 
MWAvgPoolingLayerImpl::~MWAvgPoolingLayerImpl() { } void 
MWAvgPoolingLayerImpl::createAvgPoolingLayer(int GFienSVKLlDQuZeqAdLC, int 
GeeOVBfQrpMacIFBLKOo, int GsZlHFuhbvjLtRMDjXnW, int HJHXkKmgFxxIOsIvRRnF, int 
DqxLTLaJwwgQqmrtCDuu, int FeVcBgtQmTLtmnNcJGMY, int outbufIdx) { 
MWAvgPoolingLayer* avgpoolLayer = static_cast<MWAvgPoolingLayer*>(getLayer()); 
MWTensor* ipTensor = avgpoolLayer->getInputTensor(0); 
CUDNN_CALL(cudnnSetPooling2dDescriptor(npGnQZLrEfVTQnEbwqij, 
CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 
GFienSVKLlDQuZeqAdLC, GeeOVBfQrpMacIFBLKOo, DqxLTLaJwwgQqmrtCDuu, FeVcBgtQmTLtmnNcJGMY, 
GsZlHFuhbvjLtRMDjXnW, HJHXkKmgFxxIOsIvRRnF)); int fSKMHAqIghbYYgyIpNDw, OumvfgWXDdmsQaciHMHx, 
WprSrhAStKGxyXeoxETy, vjDFlBZzKvbpPseAtMBP;  cudnnTensorDescriptor_t bUVPfnrJhLfHzOLUUrKk = 
*getCuDNNDescriptor(ipTensor); 
CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(npGnQZLrEfVTQnEbwqij, 
bUVPfnrJhLfHzOLUUrKk, &fSKMHAqIghbYYgyIpNDw ,&OumvfgWXDdmsQaciHMHx, &WprSrhAStKGxyXeoxETy, 
&vjDFlBZzKvbpPseAtMBP)); CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fSKMHAqIghbYYgyIpNDw, OumvfgWXDdmsQaciHMHx, WprSrhAStKGxyXeoxETy, 
vjDFlBZzKvbpPseAtMBP)); if (outbufIdx < 0) { 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*fSKMHAqIghbYYgyIpNDw*OumvfgWXDdmsQaciHMHx*WprSrhAStKGxyXeoxETy*vjDFlBZzKvbpPseAtMBP)); } else { 
setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
avgpoolLayer->getOutputTensor(0)->setopBufIndex(outbufIdx); } } void 
MWAvgPoolingLayerImpl::predict() { MWAvgPoolingLayer* avgpoolLayer = 
static_cast<MWAvgPoolingLayer*>(getLayer()); MWTensor* ipTensor = 
avgpoolLayer->getInputTensor(0); MWTensor* opTensor = 
avgpoolLayer->getOutputTensor(0); cudnnTensorDescriptor_t bUVPfnrJhLfHzOLUUrKk = 
*getCuDNNDescriptor(ipTensor); 
CUDNN_CALL(cudnnPoolingForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
npGnQZLrEfVTQnEbwqij, getOnePtr(), bUVPfnrJhLfHzOLUUrKk, ipTensor->getData(), 
getZeroPtr(), *getOutputDescriptor(),opTensor->getData())); } void 
MWAvgPoolingLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyPoolingDescriptor(npGnQZLrEfVTQnEbwqij)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); for(int idx = 
0; idx < getLayer()->getNumOutputs(); idx++) {  float* data = 
getLayer()->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } } MWOutputLayerImpl::MWOutputLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int ) : MWCNNLayerImpl(layer, ntwk_impl) { 
createOutputLayer(); } MWOutputLayerImpl::~MWOutputLayerImpl() { } void 
MWOutputLayerImpl::createOutputLayer() { MWOutputLayer* opLayer = 
static_cast<MWOutputLayer*>(getLayer()); MWTensor* ipTensor = 
opLayer->getInputTensor(0); setData(ipTensor->getData()); return; } void 
MWOutputLayerImpl::predict() { } void MWOutputLayerImpl::cleanup() { }