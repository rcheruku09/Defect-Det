#include "MWMaxUnpoolingLayerImpl.hpp"
#include "MWMaxUnpoolingLayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include <cassert>
 MWMaxUnpoolingLayerImpl::MWMaxUnpoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int outbufIdx) : MWCNNLayerImpl(layer, 
ntwk_impl) { createUnpoolingLayer(outbufIdx); } 
MWMaxUnpoolingLayerImpl::~MWMaxUnpoolingLayerImpl() {  } void 
MWMaxUnpoolingLayerImpl::createUnpoolingLayer(int outbufIdx) { MWTensor* 
opTensor = getLayer()->getOutputTensor(0); if (outbufIdx < 0) { 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, 
sizeof(float)*opTensor->getBatchSize()* opTensor->getChannels()* 
opTensor->getHeight()* opTensor->getWidth())); } else { 
setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
opTensor->setopBufIndex(outbufIdx); } CUDA_CALL(cudaMemset(getData(),0.0f, 
sizeof(float)*opTensor->getBatchSize()* opTensor->getChannels()* 
opTensor->getHeight()* opTensor->getWidth() ));  
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, opTensor->getBatchSize(), 
opTensor->getChannels(), opTensor->getHeight(), opTensor->getWidth())); } void 
MWMaxUnpoolingLayerImpl::predict() { assert(this->getData() != 
getLayer()->getInputTensor(0)->getData()); 
doMaxUnpoolingForwardImpl(getLayer()->getInputTensor(0)->getData(), 
getLayer()->getInputTensor(1)->getData(), 
getLayer()->getOutputTensor(0)->getData(), 
getLayer()->getInputTensor(0)->getHeight(), 
getLayer()->getInputTensor(0)->getWidth(), 
getLayer()->getInputTensor(0)->getChannels(), 
getLayer()->getInputTensor(0)->getBatchSize()); return; } void __global__ 
__launch_bounds__(1024) MaxUnpoolingImpl(float * inputBuffer, float * 
indexBuffer, float * outputBuffer, const int CDJtexcMbXMWAmnNZsNf) { for(int i = 
blockDim.x * blockIdx.x + threadIdx.x; i < CDJtexcMbXMWAmnNZsNf; i+= 
blockDim.x*gridDim.x) { outputBuffer[static_cast<int>(indexBuffer[i])] = 
inputBuffer[i]; } } void 
MWMaxUnpoolingLayerImpl::doMaxUnpoolingForwardImpl(float* inputBuffer, float* 
indexBuffer, float* outputBuffer, int XCLDbxHBtWRStETWIkId, int wtNPjzxHKNoJIigzXrEl, 
int hDaNSVZAofAENeIAiWEw, int MdSWZSOAjugbWppryHbR ) {  
CUDA_CALL(cudaMemset(outputBuffer,0, 
sizeof(float)*getLayer()->getOutputTensor(0)->getBatchSize()* 
getLayer()->getOutputTensor(0)->getChannels()* 
getLayer()->getOutputTensor(0)->getHeight()* 
getLayer()->getOutputTensor(0)->getWidth() )); int fjfzkUfcCOqjrkAVGfuc = 
XCLDbxHBtWRStETWIkId*wtNPjzxHKNoJIigzXrEl* 
hDaNSVZAofAENeIAiWEw*MdSWZSOAjugbWppryHbR; int 
tqZLvfMHdgZzbchUyDzd = (fjfzkUfcCOqjrkAVGfuc < 1024) ? fjfzkUfcCOqjrkAVGfuc : 
1024; int NldNILHvuQqQPSAHXxdT = (fjfzkUfcCOqjrkAVGfuc + 
tqZLvfMHdgZzbchUyDzd - 1)/tqZLvfMHdgZzbchUyDzd; 
MaxUnpoolingImpl<<<NldNILHvuQqQPSAHXxdT, tqZLvfMHdgZzbchUyDzd>>>( 
inputBuffer, indexBuffer, outputBuffer, fjfzkUfcCOqjrkAVGfuc); } void 
MWMaxUnpoolingLayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); for(int idx = 
0; idx < getLayer()->getNumOutputs(); idx++) { float* data = 
getLayer()->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } }