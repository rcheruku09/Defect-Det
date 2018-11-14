/*
 * File: _coder_segnet_predict_api.h
 *
 * GPU Coder version                    : 1.2
 * CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
 */

#ifndef _CODER_SEGNET_PREDICT_API_H
#define _CODER_SEGNET_PREDICT_API_H

/* Include Files */
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include <stddef.h>
#include <stdlib.h>
#include "_coder_segnet_predict_api.h"

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void segnet_predict(uint8_T in[518400], real32_T out[1900800]);
extern void segnet_predict_api(const mxArray * const prhs[1], int32_T nlhs,
  const mxArray *plhs[1]);
extern void segnet_predict_atexit(void);
extern void segnet_predict_initialize(void);
extern void segnet_predict_terminate(void);
extern void segnet_predict_xil_terminate(void);

#endif

/*
 * File trailer for _coder_segnet_predict_api.h
 *
 * [EOF]
 */
