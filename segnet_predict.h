//
// File: segnet_predict.h
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//
#ifndef SEGNET_PREDICT_H
#define SEGNET_PREDICT_H

// Include Files
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "segnet_predict_types.h"

// Function Declarations
extern void segnet_predict(const uint8_T in[518400], real32_T out[1900800]);
extern void segnet_predict_init();

#endif

//
// File trailer for segnet_predict.h
//
// [EOF]
//
