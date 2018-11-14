//
// File: DeepLearningNetwork.cu
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//

// Include Files
#include "segnet_predict.h"
#include "DeepLearningNetwork.h"

// Type Definitions
#include "MWFusedConvReLULayer.hpp"
#include "cnn_api.hpp"
#include "MWMaxUnpoolingLayer.hpp"
#include "MWTargetNetworkImpl.hpp"

// Function Declarations

// Function Definitions

//
// Arguments    : MWCNNLayer *this
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *this
// Return Type  : void
//

//
// Arguments    : MWCNNLayer *this
// Return Type  : void
//

//
// Arguments    : MWFusedConvReLULayer *this
//                MWTargetNetworkImpl *targetImpl
//                Tensor *b
//                int32_T FilterSizeH
//                int32_T FilterSizeW
//                int32_T NumChannels
//                int32_T NumFilters
//                int32_T StrideH
//                int32_T StrideW
//                int32_T PaddingH_Top
//                int32_T PaddingH_Bottom
//                int32_T PaddingW_Left
//                int32_T PaddingW_Right
//                int32_T DilationFactorH
//                int32_T DilationFactorW
//                int32_T NumGroups
//                const char * c_a___codegen_exe_segnet_predic
//                const char * d_a___codegen_exe_segnet_predic
//                int32_T c
// Return Type  : void
//

//
// Arguments    : MWInputLayer *this
//                MWTargetNetworkImpl *targetImpl
//                int32_T n
//                int32_T h
//                int32_T w
//                int32_T c
//                int32_T withAvg
//                const char * c_a___codegen_exe_segnet_predic
//                int32_T b
// Return Type  : void
//

//
// Arguments    : MWMaxUnpoolingLayer *this
//                MWTargetNetworkImpl *targetImpl
//                Tensor *b
//                Tensor *c
//                int32_T d
// Return Type  : void
//

//
// Arguments    : MWOutputLayer *this
//                MWTargetNetworkImpl *targetImpl
//                Tensor *b
//                int32_T c
// Return Type  : void
//

//
// Arguments    : MWSoftmaxLayer *this
//                MWTargetNetworkImpl *targetImpl
//                Tensor *b
//                int32_T c
// Return Type  : void
//

//
// Arguments    : MWCNNLayer *this
//                int32_T handle
// Return Type  : void
//

//
// Arguments    : MWCNNLayer *this
//                int32_T b_index
// Return Type  : Tensor *
//

//
// Arguments    : MWTargetNetworkImpl *this
//                MWCNNLayer *layers[39]
//                int32_T numLayers
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *this
//                int32_T MaxBufSize
//                int32_T numBufstoAllocate
// Return Type  : void
//

//
// Arguments    : MWCNNLayer *this
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *this
//                boolean_T autoTune
// Return Type  : void
//

//
// Arguments    : b_SegNet_0 *obj
// Return Type  : void
//
void DeepLearningNetwork_setup(b_SegNet_0 *obj)
{
  obj->setup();
  obj->batchSize = 1;
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
b_SegNet_0::b_SegNet_0()
{
  this->numLayers = 39;
  this->targetImpl = 0;
  this->layers[0] = new MWInputLayer;
  this->layers[1] = new MWFusedConvReLULayer;
  this->layers[2] = new MWFusedConvReLULayer;
  this->layers[3] = new MWMaxPoolingLayer;
  this->layers[4] = new MWFusedConvReLULayer;
  this->layers[5] = new MWFusedConvReLULayer;
  this->layers[6] = new MWMaxPoolingLayer;
  this->layers[7] = new MWFusedConvReLULayer;
  this->layers[8] = new MWFusedConvReLULayer;
  this->layers[9] = new MWFusedConvReLULayer;
  this->layers[10] = new MWMaxPoolingLayer;
  this->layers[11] = new MWFusedConvReLULayer;
  this->layers[12] = new MWFusedConvReLULayer;
  this->layers[13] = new MWFusedConvReLULayer;
  this->layers[14] = new MWMaxPoolingLayer;
  this->layers[15] = new MWFusedConvReLULayer;
  this->layers[16] = new MWFusedConvReLULayer;
  this->layers[17] = new MWFusedConvReLULayer;
  this->layers[18] = new MWMaxPoolingLayer;
  this->layers[19] = new MWMaxUnpoolingLayer;
  this->layers[20] = new MWFusedConvReLULayer;
  this->layers[21] = new MWFusedConvReLULayer;
  this->layers[22] = new MWFusedConvReLULayer;
  this->layers[23] = new MWMaxUnpoolingLayer;
  this->layers[24] = new MWFusedConvReLULayer;
  this->layers[25] = new MWFusedConvReLULayer;
  this->layers[26] = new MWFusedConvReLULayer;
  this->layers[27] = new MWMaxUnpoolingLayer;
  this->layers[28] = new MWFusedConvReLULayer;
  this->layers[29] = new MWFusedConvReLULayer;
  this->layers[30] = new MWFusedConvReLULayer;
  this->layers[31] = new MWMaxUnpoolingLayer;
  this->layers[32] = new MWFusedConvReLULayer;
  this->layers[33] = new MWFusedConvReLULayer;
  this->layers[34] = new MWMaxUnpoolingLayer;
  this->layers[35] = new MWFusedConvReLULayer;
  this->layers[36] = new MWFusedConvReLULayer;
  this->layers[37] = new MWSoftmaxLayer;
  this->layers[38] = new MWOutputLayer;
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
b_SegNet_0::~b_SegNet_0()
{
  int32_T idx;
  this->cleanup();
  for (idx = 0; idx < 39; idx++) {
    delete this->layers[idx];
  }

  if (this->targetImpl) {
    delete this->targetImpl;
  }
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
void b_SegNet_0::cleanup()
{
  int32_T idx;
  for (idx = 0; idx < 39; idx++) {
    this->layers[idx]->cleanup();
  }

  if (this->targetImpl) {
    this->targetImpl->cleanup();
  }
}

//
// Arguments    : b_SegNet_0 *this
//                int32_T layerIndex
//                int32_T portIndex
// Return Type  : real32_T *
//
real32_T *b_SegNet_0::getLayerOutput(int32_T layerIndex, int32_T portIndex)
{
  return this->layers[layerIndex]->getData(portIndex);
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
void b_SegNet_0::postsetup()
{
  int32_T idx;
  this->targetImpl->postSetup(this->layers, this->numLayers);
  for (idx = 0; idx < 39; idx++) {
    this->layers[idx]->allocate();
  }
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
void b_SegNet_0::predict()
{
  int32_T idx;
  for (idx = 0; idx < 39; idx++) {
    this->layers[idx]->predict();
  }
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
void b_SegNet_0::presetup()
{
  this->targetImpl->preSetup(11059200, 7);
  this->targetImpl->setAutoTune(true);
}

//
// Arguments    : b_SegNet_0 *this
// Return Type  : void
//
void b_SegNet_0::setup()
{
  this->targetImpl = new MWTargetNetworkImpl;
  this->presetup();
  (dynamic_cast<MWInputLayer *>(this->layers[0]))->createInputLayer
    (this->targetImpl, 1, 360, 480, 3, 1,
     "./codegen/exe/segnet_predict/cnn_SegNet_avg", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[1]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[0]
    ->getOutputTensor(0), 3, 3, 3, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv1_1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv1_1_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[2]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[1]
    ->getOutputTensor(0), 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv1_2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv1_2_b", 0);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[3]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[2]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0,
     0, 1, 2, 1, 2);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[4]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[3]
    ->getOutputTensor(0), 3, 3, 64, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv2_1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv2_1_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[5]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[4]
    ->getOutputTensor(0), 3, 3, 128, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv2_2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv2_2_b", 1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[6]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[5]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0,
     0, 1, 2, 0, 3);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[7]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[6]
    ->getOutputTensor(0), 3, 3, 128, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv3_1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv3_1_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[8]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[7]
    ->getOutputTensor(0), 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv3_2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv3_2_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[9]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[8]
    ->getOutputTensor(0), 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv3_3_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv3_3_b", 1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[10]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[9]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0,
     0, 1, 2, 0, 4);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[11]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[10]
    ->getOutputTensor(0), 3, 3, 256, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv4_1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv4_1_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[12]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[11]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv4_2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv4_2_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[13]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[12]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv4_3_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv4_3_b", 1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[14]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[13]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0,
     0, 1, 2, 0, 5);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[15]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[14]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv5_1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv5_1_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[16]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[15]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv5_2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv5_2_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[17]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[16]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_conv5_3_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_conv5_3_b", 1);
  (dynamic_cast<MWMaxPoolingLayer *>(this->layers[18]))->createMaxPoolingLayer
    (this->targetImpl, this->layers[17]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0,
     0, 1, 2, 0, 6);
  (dynamic_cast<MWMaxUnpoolingLayer *>(this->layers[19]))
    ->createMaxUnpoolingLayer(this->targetImpl, this->layers[18]
    ->getOutputTensor(0), this->layers[18]->getOutputTensor(1), 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[20]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[19]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder5_conv3_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder5_conv3_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[21]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[20]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder5_conv2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder5_conv2_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[22]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[21]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder5_conv1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder5_conv1_b", 0);
  (dynamic_cast<MWMaxUnpoolingLayer *>(this->layers[23]))
    ->createMaxUnpoolingLayer(this->targetImpl, this->layers[22]
    ->getOutputTensor(0), this->layers[14]->getOutputTensor(1), 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[24]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[23]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder4_conv3_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder4_conv3_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[25]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[24]
    ->getOutputTensor(0), 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder4_conv2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder4_conv2_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[26]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[25]
    ->getOutputTensor(0), 3, 3, 512, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder4_conv1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder4_conv1_b", 0);
  (dynamic_cast<MWMaxUnpoolingLayer *>(this->layers[27]))
    ->createMaxUnpoolingLayer(this->targetImpl, this->layers[26]
    ->getOutputTensor(0), this->layers[10]->getOutputTensor(1), 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[28]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[27]
    ->getOutputTensor(0), 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder3_conv3_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder3_conv3_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[29]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[28]
    ->getOutputTensor(0), 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder3_conv2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder3_conv2_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[30]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[29]
    ->getOutputTensor(0), 3, 3, 256, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder3_conv1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder3_conv1_b", 0);
  (dynamic_cast<MWMaxUnpoolingLayer *>(this->layers[31]))
    ->createMaxUnpoolingLayer(this->targetImpl, this->layers[30]
    ->getOutputTensor(0), this->layers[6]->getOutputTensor(1), 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[32]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[31]
    ->getOutputTensor(0), 3, 3, 128, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder2_conv2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder2_conv2_b", 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[33]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[32]
    ->getOutputTensor(0), 3, 3, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder2_conv1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder2_conv1_b", 1);
  (dynamic_cast<MWMaxUnpoolingLayer *>(this->layers[34]))
    ->createMaxUnpoolingLayer(this->targetImpl, this->layers[33]
    ->getOutputTensor(0), this->layers[3]->getOutputTensor(1), 0);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[35]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[34]
    ->getOutputTensor(0), 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder1_conv2_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder1_conv2_b", 1);
  (dynamic_cast<MWFusedConvReLULayer *>(this->layers[36]))
    ->createFusedConvReLULayer(this->targetImpl, this->layers[35]
    ->getOutputTensor(0), 3, 3, 64, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder1_conv1_w",
    "./codegen/exe/segnet_predict/cnn_SegNet_decoder1_conv1_b", 0);
  (dynamic_cast<MWSoftmaxLayer *>(this->layers[37]))->createSoftmaxLayer
    (this->targetImpl, this->layers[36]->getOutputTensor(0), 1);
  (dynamic_cast<MWOutputLayer *>(this->layers[38]))->createOutputLayer
    (this->targetImpl, this->layers[37]->getOutputTensor(0), 1);
  this->postsetup();
  this->inputData = this->layers[0]->getData(0);
  this->outputData = this->layers[38]->getData(0);
}

//
// File trailer for DeepLearningNetwork.cu
//
// [EOF]
//
