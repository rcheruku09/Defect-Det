/*
 * File: _coder_segnet_predict_api.c
 *
 * GPU Coder version                    : 1.2
 * CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
 */

/* Include Files */
#include "tmwtypes.h"
#include "_coder_segnet_predict_api.h"
#include "_coder_segnet_predict_mex.h"

/* Variable Definitions */
emlrtCTX emlrtRootTLSGlobal = NULL;
emlrtContext emlrtContextGlobal = { true,/* bFirstTime */
  false,                               /* bInitialized */
  131467U,                             /* fVersionInfo */
  NULL,                                /* fErrorFunction */
  "segnet_predict",                    /* fFunctionName */
  NULL,                                /* fRTCallStack */
  false,                               /* bDebugMode */
  { 2045744189U, 2170104910U, 2743257031U, 4284093946U },/* fSigWrd */
  NULL                                 /* fSigMem */
};

/* Function Declarations */
static uint8_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
  const emlrtMsgIdentifier *parentId))[518400];
static uint8_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[518400];
static uint8_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *in, const
  char_T *identifier))[518400];
static const mxArray *emlrt_marshallOut(const real32_T u[1900800]);

/* Function Definitions */

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *u
 *                const emlrtMsgIdentifier *parentId
 * Return Type  : uint8_T (*)[518400]
 */
static uint8_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
  const emlrtMsgIdentifier *parentId))[518400]
{
  uint8_T (*y)[518400];
  y = c_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *src
 *                const emlrtMsgIdentifier *msgId
 * Return Type  : uint8_T (*)[518400]
 */
  static uint8_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[518400]
{
  uint8_T (*ret)[518400];
  static const int32_T dims[3] = { 360, 480, 3 };

  emlrtCheckBuiltInR2012b(sp, (const emlrtMsgIdentifier *)msgId, src, "uint8",
    false, 3U, *(int32_T (*)[3])&dims[0]);
  ret = (uint8_T (*)[518400])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *in
 *                const char_T *identifier
 * Return Type  : uint8_T (*)[518400]
 */
static uint8_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *in, const
  char_T *identifier))[518400]
{
  uint8_T (*y)[518400];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(sp, emlrtAlias(in), &thisId);
  emlrtDestroyArray(&in);
  return y;
}
/*
 * Arguments    : const real32_T u[1900800]
 * Return Type  : const mxArray *
 */
  static const mxArray *emlrt_marshallOut(const real32_T u[1900800])
{
  const mxArray *y;
  const mxArray *m0;
  static const int32_T iv0[3] = { 0, 0, 0 };

  static const int32_T iv1[3] = { 360, 480, 11 };

  y = NULL;
  m0 = emlrtCreateNumericArray(3, iv0, mxSINGLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m0, (void *)&u[0]);
  emlrtSetDimensions((mxArray *)m0, *(int32_T (*)[3])&iv1[0], 3);
  emlrtAssign(&y, m0);
  return y;
}

/*
 * Arguments    : const mxArray * const prhs[1]
 *                int32_T nlhs
 *                const mxArray *plhs[1]
 * Return Type  : void
 */
void segnet_predict_api(const mxArray * const prhs[1], int32_T nlhs, const
  mxArray *plhs[1])
{
  real32_T (*out)[1900800];
  uint8_T (*in)[518400];
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  (void)nlhs;
  st.tls = emlrtRootTLSGlobal;
  out = (real32_T (*)[1900800])mxMalloc(sizeof(real32_T [1900800]));

  /* Marshall function inputs */
  in = emlrt_marshallIn(&st, emlrtAlias(prhs[0]), "in");

  /* Invoke the target function */
  segnet_predict(*in, *out);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(*out);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void segnet_predict_atexit(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  segnet_predict_xil_terminate();
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void segnet_predict_initialize(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  cudaSetDevice(0);
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, 0);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void segnet_predict_terminate(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/*
 * File trailer for _coder_segnet_predict_api.c
 *
 * [EOF]
 */
