#include "MWFusedConvReLULayer.hpp"
#include "MWFusedConvReLULayerImpl.hpp"
#include <cassert>
#include <stdio.h>
 MWFusedConvReLULayerImpl::MWFusedConvReLULayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int filt_H, int filt_W, int numGrps, int 
numChnls, int numFilts, int GsZlHFuhbvjLtRMDjXnW, int HJHXkKmgFxxIOsIvRRnF, int 
EvebzoroiuKkIxwjkGnD, int ECTnqgWHyHCHCLBZlffd, int FrpxvsDMwwgbpqHXWxmN, 
int FwLnexHgxHRquTKmNpoa, int ATEikvMQPqBefhJzjzhc, int 
AwZQzUhuWVLGrWgLHRuM, const char* wMySyzzledUmSLTWhuYH, const char* 
NZjOkZPwLzQsdEVkwMcX, int outbufIdx) : MWCNNLayerImpl(layer, ntwk_impl) , 
zzWugmJRYlNEuAzHMpeQ(NULL) , vjDFlBZzKvbpPseAtMBP(NULL) , MNuwXDSoGEYeABeVTwOh(NULL) , 
vxtNGOWYjhKeBBSzuIMB(NULL) , MUmglsoWcEiRiAZsclur(NULL) , XLJXOFXdnZOyJvtltbyr(NULL) , 
aLsOwwcceEmRSYzllBNs(NULL) , cQBKlCKXxecGPJrXBXdk(0) , 
AzTsxYcYjIEJsGQbeYHm(filt_H) , BLjrjqvCcCommiXWQLjs (filt_W) , 
ClEhcJFlvGCgiavziIag (numGrps) , CGbFsczkgkhjcHoCKzBx (numChnls) , 
CZNYmBcNFSZWvaCklqeM (numFilts) { fYaOQTeunPwVjnhhTECh = ntwk_impl; 
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&QMgBqCuvjnbWHWiVPEwn)); 
CUDNN_CALL(cudnnCreateFilterDescriptor(&UpnEytIWGokwbTFkBcSx)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(&NMMfJylfQjiIUAKhXCJb)); 
CUDNN_CALL(cudnnCreateActivationDescriptor(&olKGEIcsxmLSoMhRhEtP)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); 
createFusedConvReLULayer(GsZlHFuhbvjLtRMDjXnW, HJHXkKmgFxxIOsIvRRnF, 
EvebzoroiuKkIxwjkGnD, ECTnqgWHyHCHCLBZlffd, FrpxvsDMwwgbpqHXWxmN, 
FwLnexHgxHRquTKmNpoa, ATEikvMQPqBefhJzjzhc, AwZQzUhuWVLGrWgLHRuM, 
wMySyzzledUmSLTWhuYH, NZjOkZPwLzQsdEVkwMcX, outbufIdx); } 
MWFusedConvReLULayerImpl::~MWFusedConvReLULayerImpl() { } float 
MWFusedConvReLULayerImpl::getIsGrouped() { return cQBKlCKXxecGPJrXBXdk; } void 
MWFusedConvReLULayerImpl::setIsGrouped(float ig) { cQBKlCKXxecGPJrXBXdk = ig; 
return; } void MWFusedConvReLULayerImpl::setOutput2(float* out2) { 
zzWugmJRYlNEuAzHMpeQ = out2; return; } float* MWFusedConvReLULayerImpl::getOutput2() { 
return zzWugmJRYlNEuAzHMpeQ; } cudnnTensorDescriptor_t* 
MWFusedConvReLULayerImpl::getGroupDescriptor() { return &WOJynDmqVUPWjAGVIuMQ; 
} void MWFusedConvReLULayerImpl::createFusedConvReLULayer(int 
GsZlHFuhbvjLtRMDjXnW, int HJHXkKmgFxxIOsIvRRnF, int EvebzoroiuKkIxwjkGnD, int 
ECTnqgWHyHCHCLBZlffd , int FrpxvsDMwwgbpqHXWxmN, int FwLnexHgxHRquTKmNpoa, 
int ATEikvMQPqBefhJzjzhc, int AwZQzUhuWVLGrWgLHRuM, const char* 
wMySyzzledUmSLTWhuYH, const char* NZjOkZPwLzQsdEVkwMcX, int outbufIdx) { 
MWTensor* ipTensor = getLayer()->getInputTensor(0); int 
QVgVGfoCXYiYXzPhvVPX = EvebzoroiuKkIxwjkGnD; int 
QhTesEEIHwhNmHSeYbRR = FrpxvsDMwwgbpqHXWxmN; if 
((EvebzoroiuKkIxwjkGnD != ECTnqgWHyHCHCLBZlffd) || (FrpxvsDMwwgbpqHXWxmN != 
FwLnexHgxHRquTKmNpoa)) { float* newInput; int inputH = ipTensor->getHeight() + 
EvebzoroiuKkIxwjkGnD + ECTnqgWHyHCHCLBZlffd; int inputW = 
ipTensor->getWidth() + FrpxvsDMwwgbpqHXWxmN + FwLnexHgxHRquTKmNpoa; 
CUDA_CALL(cudaMalloc((void**)&newInput, sizeof(float)*ipTensor->getBatchSize() 
* ipTensor->getChannels() * inputH * inputW)); CUDA_CALL(cudaMemset(newInput, 
0, 
sizeof(float)*ipTensor->getBatchSize()*ipTensor->getChannels()*inputH*inputW)); 
XLJXOFXdnZOyJvtltbyr = new MWTensor(inputH, inputW, ipTensor->getChannels(), 
ipTensor->getBatchSize(), newInput,getLayer(), 0); 
CUDNN_CALL(cudnnCreateTensorDescriptor(&bUVPfnrJhLfHzOLUUrKk)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(bUVPfnrJhLfHzOLUUrKk, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, XLJXOFXdnZOyJvtltbyr->getBatchSize(), XLJXOFXdnZOyJvtltbyr->getChannels(), 
XLJXOFXdnZOyJvtltbyr->getHeight(), XLJXOFXdnZOyJvtltbyr->getWidth())); 
QVgVGfoCXYiYXzPhvVPX = 0; QhTesEEIHwhNmHSeYbRR = 0; } else { 
XLJXOFXdnZOyJvtltbyr = ipTensor; bUVPfnrJhLfHzOLUUrKk = 
*getCuDNNDescriptor(XLJXOFXdnZOyJvtltbyr); } eVAFqeShtGZAZluKdMvQ = 
EvebzoroiuKkIxwjkGnD; eqOmMKQRpqBqRQCnJmxt = FrpxvsDMwwgbpqHXWxmN; 
assert(XLJXOFXdnZOyJvtltbyr != NULL); MWFusedConvReLULayer* fusedConvReluLayer = static_cast<MWFusedConvReLULayer*>(getLayer());
#if (CUDNN_MAJOR <= 5)
 { if ((ATEikvMQPqBefhJzjzhc != 1) && (AwZQzUhuWVLGrWgLHRuM != 1)){ 
printf("Dilated Convolution only supported for cuDNN 6 or greater "); throw 
std::runtime_error("Unsupported Dilation Factor"); } 
CUDNN_CALL(cudnnSetConvolution2dDescriptor(QMgBqCuvjnbWHWiVPEwn, 
QVgVGfoCXYiYXzPhvVPX, QhTesEEIHwhNmHSeYbRR, GsZlHFuhbvjLtRMDjXnW, 
HJHXkKmgFxxIOsIvRRnF, 1, 1, CUDNN_CROSS_CORRELATION));  }
#else
 { CUDNN_CALL(cudnnSetConvolution2dDescriptor(QMgBqCuvjnbWHWiVPEwn, 
QVgVGfoCXYiYXzPhvVPX, QhTesEEIHwhNmHSeYbRR, GsZlHFuhbvjLtRMDjXnW, 
HJHXkKmgFxxIOsIvRRnF, ATEikvMQPqBefhJzjzhc, AwZQzUhuWVLGrWgLHRuM, 
CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); }
#endif
 CUDNN_CALL(cudnnSetActivationDescriptor(olKGEIcsxmLSoMhRhEtP, 
CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0)); int sxuOMwKXOKfuExclRaSe, 
sRECVoNNtDdcBOWgDyar; int numInputFeatures = 
CGbFsczkgkhjcHoCKzBx*ClEhcJFlvGCgiavziIag; int 
hnewnpwgzKmOdualajhn,shEncNmxJsMuJKwbrwok,tnTPxeDjBsqLAPkJcPJX; MWTensor* 
opTensor = fusedConvReluLayer->getOutputTensor(0); hnewnpwgzKmOdualajhn 
= opTensor->getChannels(); shEncNmxJsMuJKwbrwok = opTensor->getHeight(); 
tnTPxeDjBsqLAPkJcPJX = opTensor->getWidth(); size_t ugnnrhsgTeWucrMPCJUc = 0; if( 
ClEhcJFlvGCgiavziIag == 1 ) { 
CUDNN_CALL(cudnnSetFilter4dDescriptor(UpnEytIWGokwbTFkBcSx, CUDNN_DATA_FLOAT, 
CUDNN_TENSOR_NCHW, hnewnpwgzKmOdualajhn, numInputFeatures, 
AzTsxYcYjIEJsGQbeYHm, BLjrjqvCcCommiXWQLjs)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(NMMfJylfQjiIUAKhXCJb, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, 1, hnewnpwgzKmOdualajhn, 1, 1)); 
CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(QMgBqCuvjnbWHWiVPEwn, 
bUVPfnrJhLfHzOLUUrKk, UpnEytIWGokwbTFkBcSx, &sxuOMwKXOKfuExclRaSe, 
&sRECVoNNtDdcBOWgDyar, &shEncNmxJsMuJKwbrwok, &tnTPxeDjBsqLAPkJcPJX)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sxuOMwKXOKfuExclRaSe, sRECVoNNtDdcBOWgDyar, 
opTensor->getHeight(), opTensor->getWidth())); assert(opTensor->getHeight() == 
shEncNmxJsMuJKwbrwok); assert(opTensor->getWidth() == tnTPxeDjBsqLAPkJcPJX);
#if (CUDNN_MAJOR < 7)
 { 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
bUVPfnrJhLfHzOLUUrKk, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, 
*getOutputDescriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, 
&PmFfARVzoHVAYkfpuvqK)); }
#else
 { cudnnConvolutionFwdAlgoPerf_t perf_results[3]; int returnedAlgoCount; 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
bUVPfnrJhLfHzOLUUrKk, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, 
*getOutputDescriptor(), 3, &returnedAlgoCount, perf_results)); 
PmFfARVzoHVAYkfpuvqK = perf_results[0].algo; }
#endif
 
CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
bUVPfnrJhLfHzOLUUrKk, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, 
*getOutputDescriptor(), PmFfARVzoHVAYkfpuvqK, &ugnnrhsgTeWucrMPCJUc)); } else { 
setIsGrouped(1); MWTensor* ipTensor = XLJXOFXdnZOyJvtltbyr; aLsOwwcceEmRSYzllBNs = 
ipTensor->getData() + ipTensor->getChannels()/ClEhcJFlvGCgiavziIag * 
ipTensor->getHeight() * ipTensor->getWidth(); 
CUDNN_CALL(cudnnCreateTensorDescriptor(&cCXqPFPPcoHzYMDpnUxQ)); 
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(cCXqPFPPcoHzYMDpnUxQ, 
CUDNN_DATA_FLOAT, ipTensor->getBatchSize(), 
ipTensor->getChannels()/ClEhcJFlvGCgiavziIag, ipTensor->getHeight(), 
ipTensor->getWidth(), 
ipTensor->getChannels()*ipTensor->getHeight()*ipTensor->getWidth(), 
ipTensor->getHeight()*ipTensor->getWidth(), ipTensor->getWidth(), 1)); 
CUDNN_CALL(cudnnCreateTensorDescriptor(getGroupDescriptor())); 
CUDNN_CALL(cudnnSetFilter4dDescriptor(UpnEytIWGokwbTFkBcSx, CUDNN_DATA_FLOAT, 
CUDNN_TENSOR_NCHW, CZNYmBcNFSZWvaCklqeM, CGbFsczkgkhjcHoCKzBx, 
AzTsxYcYjIEJsGQbeYHm, BLjrjqvCcCommiXWQLjs)); 
CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(QMgBqCuvjnbWHWiVPEwn, 
cCXqPFPPcoHzYMDpnUxQ, UpnEytIWGokwbTFkBcSx, &sxuOMwKXOKfuExclRaSe, 
&sRECVoNNtDdcBOWgDyar, &shEncNmxJsMuJKwbrwok, &tnTPxeDjBsqLAPkJcPJX)); 
assert(opTensor->getHeight() == shEncNmxJsMuJKwbrwok); assert(opTensor->getWidth() 
== tnTPxeDjBsqLAPkJcPJX); 
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(*getGroupDescriptor(), 
CUDNN_DATA_FLOAT, sxuOMwKXOKfuExclRaSe, sRECVoNNtDdcBOWgDyar, shEncNmxJsMuJKwbrwok, 
tnTPxeDjBsqLAPkJcPJX, 
sRECVoNNtDdcBOWgDyar*ClEhcJFlvGCgiavziIag*shEncNmxJsMuJKwbrwok*tnTPxeDjBsqLAPkJcPJX, 
shEncNmxJsMuJKwbrwok*tnTPxeDjBsqLAPkJcPJX, tnTPxeDjBsqLAPkJcPJX, 1)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sxuOMwKXOKfuExclRaSe, 
sRECVoNNtDdcBOWgDyar*ClEhcJFlvGCgiavziIag, shEncNmxJsMuJKwbrwok, tnTPxeDjBsqLAPkJcPJX)); 
assert(CZNYmBcNFSZWvaCklqeM == sRECVoNNtDdcBOWgDyar); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(NMMfJylfQjiIUAKhXCJb, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, 1, sRECVoNNtDdcBOWgDyar, 1, 1));
#if (CUDNN_MAJOR < 7)
 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
cCXqPFPPcoHzYMDpnUxQ, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, 
*getGroupDescriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &PmFfARVzoHVAYkfpuvqK));
#else
 cudnnConvolutionFwdAlgoPerf_t perf_results[3]; int returnedAlgoCount; 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
cCXqPFPPcoHzYMDpnUxQ, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, 
*getGroupDescriptor(), 3, &returnedAlgoCount,perf_results)); 
PmFfARVzoHVAYkfpuvqK = perf_results[0].algo;
#endif
 
CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
cCXqPFPPcoHzYMDpnUxQ, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, 
*getGroupDescriptor(), PmFfARVzoHVAYkfpuvqK, &ugnnrhsgTeWucrMPCJUc)); } if( 
ugnnrhsgTeWucrMPCJUc > *fYaOQTeunPwVjnhhTECh->getWorkSpaceSize() ) { 
fYaOQTeunPwVjnhhTECh->setWorkSpaceSize(ugnnrhsgTeWucrMPCJUc); } 
assert(sxuOMwKXOKfuExclRaSe == ipTensor->getBatchSize()); 
assert(hnewnpwgzKmOdualajhn == sRECVoNNtDdcBOWgDyar * 
ClEhcJFlvGCgiavziIag); if (outbufIdx < 0) { 
CUDA_CALL(cudaMalloc((void**)&RAtlBpdedvgxUsgDTsch, sizeof(float) * 
opTensor->getBatchSize() * opTensor->getChannels() * opTensor->getHeight() * 
opTensor->getWidth())); } else { 
setData(fYaOQTeunPwVjnhhTECh->memBuffer[outbufIdx]); 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); } 
CUDA_CALL(cudaMalloc((void**)&vjDFlBZzKvbpPseAtMBP, 
sizeof(float)*CGbFsczkgkhjcHoCKzBx*hnewnpwgzKmOdualajhn*AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs)); 
CUDA_CALL(cudaMalloc((void**)&MNuwXDSoGEYeABeVTwOh, sizeof(float)*hnewnpwgzKmOdualajhn));
#ifdef RANDOM
 curandGenerateNormal(VCbcPxtPsBLTrHYdEvqn, vjDFlBZzKvbpPseAtMBP, 
CGbFsczkgkhjcHoCKzBx*hnewnpwgzKmOdualajhn*AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs, 
0, 0.1); curandGenerateNormal(VCbcPxtPsBLTrHYdEvqn, MNuwXDSoGEYeABeVTwOh, 
hnewnpwgzKmOdualajhn, -0.5, 1);
#endif
 if( ClEhcJFlvGCgiavziIag == 2 ) { vxtNGOWYjhKeBBSzuIMB = vjDFlBZzKvbpPseAtMBP + 
CZNYmBcNFSZWvaCklqeM * CGbFsczkgkhjcHoCKzBx * AzTsxYcYjIEJsGQbeYHm * 
BLjrjqvCcCommiXWQLjs; MUmglsoWcEiRiAZsclur = MNuwXDSoGEYeABeVTwOh + CZNYmBcNFSZWvaCklqeM; 
setOutput2(getData() + hnewnpwgzKmOdualajhn/ 2 * shEncNmxJsMuJKwbrwok * 
tnTPxeDjBsqLAPkJcPJX); setIsGrouped(1); } loadWeights(wMySyzzledUmSLTWhuYH); 
loadBias(NZjOkZPwLzQsdEVkwMcX); return; } void 
MWFusedConvReLULayerImpl::predict() { MWFusedConvReLULayer* fusedConvReluLayer 
= static_cast<MWFusedConvReLULayer*>(getLayer()); if (XLJXOFXdnZOyJvtltbyr != 
fusedConvReluLayer->getInputTensor()) { 
CUDA_CALL(cudaMemset(XLJXOFXdnZOyJvtltbyr->getData(), 0, 
sizeof(float)*XLJXOFXdnZOyJvtltbyr->getBatchSize()*XLJXOFXdnZOyJvtltbyr->getChannels()*XLJXOFXdnZOyJvtltbyr->getHeight()*XLJXOFXdnZOyJvtltbyr->getWidth())); 
int gzSTokDHvkXefhiGDcWL = 
fusedConvReluLayer->getInputTensor()->getHeight()*fusedConvReluLayer->getInputTensor()->getWidth()*fusedConvReluLayer->getInputTensor()->getBatchSize()*fusedConvReluLayer->getInputTensor()->getChannels(); 
MWCNNLayerImpl::padInput(fusedConvReluLayer->getInputTensor()->getData(), 
fusedConvReluLayer->getInputTensor()->getHeight(), 
fusedConvReluLayer->getInputTensor()->getWidth(), 
fusedConvReluLayer->getInputTensor()->getChannels(), 
XLJXOFXdnZOyJvtltbyr->getHeight(), XLJXOFXdnZOyJvtltbyr->getWidth(), eVAFqeShtGZAZluKdMvQ, 
eqOmMKQRpqBqRQCnJmxt, XLJXOFXdnZOyJvtltbyr->getData(), gzSTokDHvkXefhiGDcWL); } 
if(ClEhcJFlvGCgiavziIag == 1 ) { assert(getData() != XLJXOFXdnZOyJvtltbyr->getData()); 
CUDNN_CALL(cudnnConvolutionBiasActivationForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
getOnePtr(), bUVPfnrJhLfHzOLUUrKk, XLJXOFXdnZOyJvtltbyr->getData(), 
UpnEytIWGokwbTFkBcSx, vjDFlBZzKvbpPseAtMBP, QMgBqCuvjnbWHWiVPEwn, PmFfARVzoHVAYkfpuvqK, 
fYaOQTeunPwVjnhhTECh->getWorkSpace(), *fYaOQTeunPwVjnhhTECh->getWorkSpaceSize(), 
getZeroPtr(), *getOutputDescriptor(), XLJXOFXdnZOyJvtltbyr->getData(), 
NMMfJylfQjiIUAKhXCJb, MNuwXDSoGEYeABeVTwOh, olKGEIcsxmLSoMhRhEtP, *getOutputDescriptor(), 
getData())); } else { assert(getData() != XLJXOFXdnZOyJvtltbyr->getData()); 
CUDNN_CALL(cudnnConvolutionBiasActivationForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
getOnePtr(), cCXqPFPPcoHzYMDpnUxQ, XLJXOFXdnZOyJvtltbyr->getData(), 
UpnEytIWGokwbTFkBcSx, vjDFlBZzKvbpPseAtMBP, QMgBqCuvjnbWHWiVPEwn, PmFfARVzoHVAYkfpuvqK, 
fYaOQTeunPwVjnhhTECh->getWorkSpace(), *fYaOQTeunPwVjnhhTECh->getWorkSpaceSize(), 
getZeroPtr(), *getGroupDescriptor(), XLJXOFXdnZOyJvtltbyr->getData(), 
NMMfJylfQjiIUAKhXCJb, MNuwXDSoGEYeABeVTwOh, olKGEIcsxmLSoMhRhEtP, *getGroupDescriptor(), 
getData())); 
CUDNN_CALL(cudnnConvolutionBiasActivationForward(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
getOnePtr(), cCXqPFPPcoHzYMDpnUxQ, aLsOwwcceEmRSYzllBNs, UpnEytIWGokwbTFkBcSx, 
vxtNGOWYjhKeBBSzuIMB, QMgBqCuvjnbWHWiVPEwn, PmFfARVzoHVAYkfpuvqK, 
fYaOQTeunPwVjnhhTECh->getWorkSpace(), *fYaOQTeunPwVjnhhTECh->getWorkSpaceSize(), 
getZeroPtr(), *getGroupDescriptor(), aLsOwwcceEmRSYzllBNs, NMMfJylfQjiIUAKhXCJb, 
MUmglsoWcEiRiAZsclur, olKGEIcsxmLSoMhRhEtP, *getGroupDescriptor(), getOutput2())); } } 
void MWFusedConvReLULayerImpl::cleanup() { 
CUDNN_CALL(cudnnDestroyConvolutionDescriptor(QMgBqCuvjnbWHWiVPEwn)); 
CUDNN_CALL(cudnnDestroyFilterDescriptor(UpnEytIWGokwbTFkBcSx)); 
CUDNN_CALL(cudnnDestroyActivationDescriptor(olKGEIcsxmLSoMhRhEtP)); if 
(vjDFlBZzKvbpPseAtMBP) { call_cuda_free(vjDFlBZzKvbpPseAtMBP); } 
CUDNN_CALL(cudnnDestroyTensorDescriptor(NMMfJylfQjiIUAKhXCJb)); if 
(MNuwXDSoGEYeABeVTwOh) { call_cuda_free(MNuwXDSoGEYeABeVTwOh); } 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); if 
(XLJXOFXdnZOyJvtltbyr != getLayer()->getInputTensor(0)) { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(bUVPfnrJhLfHzOLUUrKk)); 
call_cuda_free(XLJXOFXdnZOyJvtltbyr->getData()); } if (getIsGrouped()) { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(cCXqPFPPcoHzYMDpnUxQ)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getGroupDescriptor())); } for(int idx 
= 0; idx < getLayer()->getNumOutputs(); idx++) { float* data = 
getLayer()->getOutputTensor(idx)->getData(); if (data) { 
if(getLayer()->getOutputTensor(idx)->getopBufIndex() < 0) call_cuda_free(data); 
} } return; } void MWFusedConvReLULayerImpl::loadWeights(const char* 
UKtMXCCqdjeyaVHabkxg) { MWFusedConvReLULayer* fusedConvReluLayer = 
static_cast<MWFusedConvReLULayer*>(getLayer()); FILE* UzaGmBLFEwmwaFXebUma = 
MWCNNLayer::openBinaryFile(UKtMXCCqdjeyaVHabkxg); assert(UzaGmBLFEwmwaFXebUma); 
assert(CGbFsczkgkhjcHoCKzBx == 
XLJXOFXdnZOyJvtltbyr->getChannels()/ClEhcJFlvGCgiavziIag); int fhikqqlnUKCjleVKDqiG = 
CGbFsczkgkhjcHoCKzBx*fusedConvReluLayer->getOutputTensor()->getChannels()*AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs; 
 float* OKaRVOctKLlnIyGmjRNW = MALLOC_CALL(sizeof(float)*fhikqqlnUKCjleVKDqiG); 
fread(OKaRVOctKLlnIyGmjRNW, sizeof(float), fhikqqlnUKCjleVKDqiG, UzaGmBLFEwmwaFXebUma); if( 
AzTsxYcYjIEJsGQbeYHm != 1 && BLjrjqvCcCommiXWQLjs != 1 ) { float* 
ONvcEjLBnVNUdjMKOAwF = 
MALLOC_CALL(sizeof(float)*AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs); 
for(int k=0; k<fhikqqlnUKCjleVKDqiG/AzTsxYcYjIEJsGQbeYHm/BLjrjqvCcCommiXWQLjs; 
k++) { for(int i=0; i<AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs; i++) 
ONvcEjLBnVNUdjMKOAwF[i]=OKaRVOctKLlnIyGmjRNW[k*AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs+i]; 
for(int j=0; j<AzTsxYcYjIEJsGQbeYHm; j++) for(int i=0; 
i<BLjrjqvCcCommiXWQLjs; i++) 
OKaRVOctKLlnIyGmjRNW[k*AzTsxYcYjIEJsGQbeYHm*BLjrjqvCcCommiXWQLjs+j*BLjrjqvCcCommiXWQLjs+i]=ONvcEjLBnVNUdjMKOAwF[j+i*AzTsxYcYjIEJsGQbeYHm]; 
} free(ONvcEjLBnVNUdjMKOAwF); } CUDA_CALL(cudaMemcpy(vjDFlBZzKvbpPseAtMBP, 
OKaRVOctKLlnIyGmjRNW, sizeof(float)*fhikqqlnUKCjleVKDqiG, cudaMemcpyHostToDevice));
#if 0
 printf("%s loaded. Size = %d. %f\n", UKtMXCCqdjeyaVHabkxg, fhikqqlnUKCjleVKDqiG, OKaRVOctKLlnIyGmjRNW[0]);
#endif
 free(OKaRVOctKLlnIyGmjRNW); fclose(UzaGmBLFEwmwaFXebUma); return; } void 
MWFusedConvReLULayerImpl::loadBias(const char* UKtMXCCqdjeyaVHabkxg) { 
MWFusedConvReLULayer* fusedConvReluLayer = 
static_cast<MWFusedConvReLULayer*>(getLayer()); FILE* UzaGmBLFEwmwaFXebUma = 
MWCNNLayer::openBinaryFile(UKtMXCCqdjeyaVHabkxg); assert(UzaGmBLFEwmwaFXebUma); int 
fhikqqlnUKCjleVKDqiG = fusedConvReluLayer->getOutputTensor()->getChannels();  float* 
OKaRVOctKLlnIyGmjRNW = MALLOC_CALL(sizeof(float)*fhikqqlnUKCjleVKDqiG); 
fread(OKaRVOctKLlnIyGmjRNW, sizeof(float), fhikqqlnUKCjleVKDqiG, UzaGmBLFEwmwaFXebUma); 
CUDA_CALL(cudaMemcpy(MNuwXDSoGEYeABeVTwOh, OKaRVOctKLlnIyGmjRNW, 
sizeof(float)*fhikqqlnUKCjleVKDqiG, cudaMemcpyHostToDevice)); 
free(OKaRVOctKLlnIyGmjRNW); fclose(UzaGmBLFEwmwaFXebUma); return; } void 
MWFusedConvReLULayerImpl::postSetup() { if(fYaOQTeunPwVjnhhTECh->getAutoTune()) 
{ getConvAlgoTuned(); } else if(!fYaOQTeunPwVjnhhTECh->getWorkSpace()) { 
getConvAlgoNoWorkSpace(); } cudnnTensorDescriptor_t tmpInDesc = getIsGrouped() 
? cCXqPFPPcoHzYMDpnUxQ : bUVPfnrJhLfHzOLUUrKk; cudnnTensorDescriptor_t 
juRPduBvIGpwaZiftkzr = getIsGrouped() ? *getGroupDescriptor() : 
*getOutputDescriptor(); size_t ugnnrhsgTeWucrMPCJUc; 
CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
tmpInDesc, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, juRPduBvIGpwaZiftkzr, 
PmFfARVzoHVAYkfpuvqK, &ugnnrhsgTeWucrMPCJUc)); if( ugnnrhsgTeWucrMPCJUc > 
*fYaOQTeunPwVjnhhTECh->getPostSetupWorkSpaceSize() ) { 
fYaOQTeunPwVjnhhTECh->setPostSetupWorkSpaceSize(ugnnrhsgTeWucrMPCJUc); } } void 
MWFusedConvReLULayerImpl::getConvAlgoTuned() { cudnnConvolutionFwdAlgoPerf_t 
perf_results[3]; cudnnTensorDescriptor_t tempInDesc = getIsGrouped() ? 
cCXqPFPPcoHzYMDpnUxQ : bUVPfnrJhLfHzOLUUrKk; cudnnTensorDescriptor_t 
juRPduBvIGpwaZiftkzr = getIsGrouped() ? *getGroupDescriptor() : 
*getOutputDescriptor(); int returnedAlgoCount; 
CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
tempInDesc, XLJXOFXdnZOyJvtltbyr->getData(), UpnEytIWGokwbTFkBcSx, vjDFlBZzKvbpPseAtMBP, 
QMgBqCuvjnbWHWiVPEwn, juRPduBvIGpwaZiftkzr, getData(), 3, &returnedAlgoCount, 
&perf_results[0], fYaOQTeunPwVjnhhTECh->getWorkSpace(), 
*fYaOQTeunPwVjnhhTECh->getWorkSpaceSize())); PmFfARVzoHVAYkfpuvqK = 
perf_results[0].algo; } void MWFusedConvReLULayerImpl::getConvAlgoNoWorkSpace() 
{ assert(fYaOQTeunPwVjnhhTECh->getWorkSpace() == 0); cudnnTensorDescriptor_t 
tempInDesc = getIsGrouped() ? cCXqPFPPcoHzYMDpnUxQ : bUVPfnrJhLfHzOLUUrKk; 
cudnnTensorDescriptor_t juRPduBvIGpwaZiftkzr = getIsGrouped() ? 
*getGroupDescriptor() : *getOutputDescriptor(); 
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*fYaOQTeunPwVjnhhTECh->getCudnnHandle(), 
tempInDesc, UpnEytIWGokwbTFkBcSx, QMgBqCuvjnbWHWiVPEwn, juRPduBvIGpwaZiftkzr, 
CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &PmFfARVzoHVAYkfpuvqK)); }