//
// File: DeepLearningNetwork.h
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//
#ifndef DEEPLEARNINGNETWORK_H
#define DEEPLEARNINGNETWORK_H

// Include Files
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "segnet_predict_types.h"

// Type Definitions
#include "MWFusedConvReLULayer.hpp"
#include "cnn_api.hpp"
#include "MWMaxUnpoolingLayer.hpp"
#include "MWTargetNetworkImpl.hpp"

// Function Declarations
extern void DeepLearningNetwork_setup(b_SegNet_0 *obj);

#endif

//
// File trailer for DeepLearningNetwork.h
//
// [EOF]
//
