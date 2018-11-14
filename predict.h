//
// File: predict.h
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//
#ifndef PREDICT_H
#define PREDICT_H

// Include Files
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "segnet_predict_types.h"

// Function Declarations
extern void DeepLearningNetwork_predict(b_SegNet_0 *obj, const uint8_T
  inputdata[518400], real32_T outT[1900800]);

#endif

//
// File trailer for predict.h
//
// [EOF]
//
