//
// File: predict.cu
//
// GPU Coder version                    : 1.2
// CUDA/C/C++ source code generated on  : 02-Nov-2018 17:52:01
//

// Include Files
#include "MWCudaDimUtility.h"
#include "segnet_predict.h"
#include "predict.h"
#include "DeepLearningNetwork.h"

// Function Declarations
static __global__ void c_DeepLearningNetwork_predict_k(const uint8_T inputdata
  [518400], uint8_T b_inputdata[518400]);
static __global__ void d_DeepLearningNetwork_predict_k(uint8_T inputdata[518400],
  real32_T inputT[518400]);
static __global__ void e_DeepLearningNetwork_predict_k(real32_T out[1900800],
  real32_T outT[1900800]);

// Function Definitions

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const uint8_T inputdata[518400]
//                uint8_T b_inputdata[518400]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void c_DeepLearningNetwork_predict_k
  (const uint8_T inputdata[518400], uint8_T b_inputdata[518400])
{
  uint32_T threadId;
  int32_T i0;
  threadId = (uint32_T)mwGetGlobalThreadIndex();
  i0 = (int32_T)threadId;
  if (i0 < 518400) {
    b_inputdata[i0] = inputdata[i0];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                uint8_T inputdata[518400]
//                real32_T inputT[518400]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void d_DeepLearningNetwork_predict_k
  (uint8_T inputdata[518400], real32_T inputT[518400])
{
  uint32_T threadId;
  int32_T i0;
  int32_T i1;
  int32_T p;
  uint32_T tmpIndex;
  threadId = (uint32_T)mwGetGlobalThreadIndex();
  i0 = (int32_T)(threadId % 480U);
  tmpIndex = (threadId - (uint32_T)i0) / 480U;
  i1 = (int32_T)(tmpIndex % 360U);
  tmpIndex = (tmpIndex - (uint32_T)i1) / 360U;
  p = (int32_T)tmpIndex;
  if (p < 3) {
    inputT[(i0 + 480 * i1) + 172800 * p] = (real32_T)inputdata[(i1 + 360 * i0) +
      172800 * p];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                real32_T out[1900800]
//                real32_T outT[1900800]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void e_DeepLearningNetwork_predict_k
  (real32_T out[1900800], real32_T outT[1900800])
{
  uint32_T threadId;
  int32_T i0;
  int32_T i1;
  int32_T p;
  uint32_T tmpIndex;
  threadId = (uint32_T)mwGetGlobalThreadIndex();
  i0 = (int32_T)(threadId % 360U);
  tmpIndex = (threadId - (uint32_T)i0) / 360U;
  i1 = (int32_T)(tmpIndex % 480U);
  tmpIndex = (tmpIndex - (uint32_T)i1) / 480U;
  p = (int32_T)tmpIndex;
  if (p < 11) {
    outT[(i0 + 360 * i1) + 172800 * p] = out[(i1 + 480 * i0) + 172800 * p];
  }
}

//
// Arguments    : b_SegNet_0 *obj
//                const uint8_T inputdata[518400]
//                real32_T outT[1900800]
// Return Type  : void
//
void DeepLearningNetwork_predict(b_SegNet_0 *obj, const uint8_T inputdata[518400],
  real32_T outT[1900800])
{
  real32_T (*gpu_inputT)[518400];
  real32_T (*gpu_out)[1900800];
  uint8_T (*gpu_inputdata)[518400];
  uint8_T (*b_gpu_inputdata)[518400];
  real32_T (*gpu_outT)[1900800];
  cudaMalloc(&gpu_outT, 7603200UL);
  cudaMalloc(&gpu_out, 7603200UL);
  cudaMalloc(&gpu_inputT, 2073600UL);
  cudaMalloc(&b_gpu_inputdata, 518400UL);
  cudaMalloc(&gpu_inputdata, 518400UL);
  cudaMemcpy(gpu_inputdata, (void *)&inputdata[0], 518400UL,
             cudaMemcpyHostToDevice);
  c_DeepLearningNetwork_predict_k<<<dim3(1013U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_inputdata, *b_gpu_inputdata);
  d_DeepLearningNetwork_predict_k<<<dim3(1013U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*b_gpu_inputdata, *gpu_inputT);
  cudaMemcpy(obj->inputData, *gpu_inputT, 518400UL * sizeof(real32_T),
             cudaMemcpyDeviceToDevice);
  obj->predict();
  cudaMemcpy(*gpu_out, obj->outputData, 1900800UL * sizeof(real32_T),
             cudaMemcpyDeviceToDevice);
  e_DeepLearningNetwork_predict_k<<<dim3(3713U, 1U, 1U), dim3(512U, 1U, 1U)>>>
    (*gpu_out, *gpu_outT);
  cudaMemcpy(&outT[0], gpu_outT, 7603200UL, cudaMemcpyDeviceToHost);
  cudaFree(*gpu_inputdata);
  cudaFree(*b_gpu_inputdata);
  cudaFree(*gpu_inputT);
  cudaFree(*gpu_out);
  cudaFree(*gpu_outT);
}

//
// File trailer for predict.cu
//
// [EOF]
//
