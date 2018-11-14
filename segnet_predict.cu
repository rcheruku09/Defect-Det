//
// File: segnet_predict.cu
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//

// Include Files
#include "segnet_predict.h"
#include "predict.h"
#include "DeepLearningNetwork.h"

// Variable Definitions
static b_SegNet_0 mynet;
static boolean_T mynet_not_empty;

// Function Definitions

//
// A persistent object mynet is used to load the DAG network object.
//  At the first call to this function, the persistent object is constructed and
//  setup. When the function is called subsequent times, the same object is reused
//  to call predict on inputs, thus avoiding reconstructing and reloading the
//  network object.
// Arguments    : const uint8_T in[518400]
//                real32_T out[1900800]
// Return Type  : void
//
void segnet_predict(const uint8_T in[518400], real32_T out[1900800])
{
  //  Copyright 2018 The MathWorks, Inc.
  //  Update buildinfo with the OpenCV library flags.
  if (!mynet_not_empty) {
    DeepLearningNetwork_setup(&mynet);
    mynet_not_empty = true;
  }

  //  pass in input
  DeepLearningNetwork_predict(&mynet, in, out);
}

//
// Arguments    : void
// Return Type  : void
//
void segnet_predict_init()
{
  mynet_not_empty = false;
}

//
// File trailer for segnet_predict.cu
//
// [EOF]
//
