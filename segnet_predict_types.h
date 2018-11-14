//
// File: segnet_predict_types.h
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//
#ifndef SEGNET_PREDICT_TYPES_H
#define SEGNET_PREDICT_TYPES_H

// Include Files
#include "rtwtypes.h"

// Type Definitions
#include "cnn_api.hpp"
#include "MWTargetNetworkImpl.hpp"

class b_SegNet_0
{
 public:
  int32_T batchSize;
  int32_T numLayers;
  real32_T *inputData;
  real32_T *outputData;
  MWCNNLayer *layers[39];
 private:
  MWTargetNetworkImpl *targetImpl;
 public:
  b_SegNet_0();
  void presetup();
  void postsetup();
  void setup();
  void predict();
  void cleanup();
  real32_T *getLayerOutput(int32_T layerIndex, int32_T portIndex);
  ~b_SegNet_0();
};

#endif

//
// File trailer for segnet_predict_types.h
//
// [EOF]
//
