//
// File: segnet_predict_initialize.cu
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//

// Include Files
#include "segnet_predict.h"
#include "segnet_predict_initialize.h"

// Function Definitions

//
// Arguments    : void
// Return Type  : void
//
void segnet_predict_initialize()
{
  cudaSetDevice(0);
  segnet_predict_init();
}

//
// File trailer for segnet_predict_initialize.cu
//
// [EOF]
//
