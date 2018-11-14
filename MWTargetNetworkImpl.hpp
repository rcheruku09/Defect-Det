/* Copyright 2017 The MathWorks, Inc. */

#ifndef CNN_TARGET_NTWK_IMPL
#define CNN_TARGET_NTWK_IMPL

#include <cudnn.h>
#include <cublas_v2.h>
#include "cnn_api.hpp"

#include <vector>

class MWTargetNetworkImpl
{
  public:
    
    MWTargetNetworkImpl()
        : xkUNToJIgvoLoUQuzKRF(0)
        , QjgQHaUACFNSteMrRtRj(0)
        , QwUuNuQNtlPXrIwRNiSZ(0)
        , MW_autoTune(true)
        , GnxRkpzrPZimKtYYHSuG(0)
    {}
    ~MWTargetNetworkImpl() {}
    void preSetup(int, int);
    void postSetup(MWCNNLayer* layers[],int numLayers);
    void cleanup();

    void setWorkSpaceSize(size_t);  // Set the workspace size of this layer and previous layers   
    size_t* getWorkSpaceSize();     // Get the workspace size of this layer and previous layers
    size_t* getPostSetupWorkSpaceSize();
    void setPostSetupWorkSpaceSize(size_t psWSize);
    float* getWorkSpace();          // Get the workspace buffer in GPU memory    
    cublasHandle_t* getCublasHandle();      // Get the cuBLAS handle to use for GPU computation
    cudnnHandle_t* getCudnnHandle();        // Get the cuDNN handle to use for GPU computation    
    std::vector<float *> memBuffer;
    int numBufs;

    void setAutoTune(bool);
    bool getAutoTune() const;
       
  private:    
    size_t omxlPZbBePZdWaJOBUUG;    
    float* xkUNToJIgvoLoUQuzKRF;    
    cublasHandle_t* QjgQHaUACFNSteMrRtRj;
    cudnnHandle_t* QwUuNuQNtlPXrIwRNiSZ;
    bool MW_autoTune;
    size_t GnxRkpzrPZimKtYYHSuG;
    
  private:
    void createWorkSpace(float**);  // Create the workspace needed for this layer
    
};
#endif
