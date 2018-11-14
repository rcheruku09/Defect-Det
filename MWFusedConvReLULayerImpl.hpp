/* Copyright 2018 The MathWorks, Inc. */

#ifndef __FUSED_CONV_RELU_LAYER_IMPL_HPP
#define __FUSED_CONV_RELU_LAYER_IMPL_HPP

#include "MWFusedConvReLULayer.hpp"
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"


class MWFusedConvReLULayerImpl: public MWCNNLayerImpl
{

public:
  int AzTsxYcYjIEJsGQbeYHm;           //Filter height for CONV and FC
  int BLjrjqvCcCommiXWQLjs;            //Filter width for CONV and FC

  int ClEhcJFlvGCgiavziIag;
  int CGbFsczkgkhjcHoCKzBx;
  int CZNYmBcNFSZWvaCklqeM;

private:

  float* zzWugmJRYlNEuAzHMpeQ;
  float* vjDFlBZzKvbpPseAtMBP;
  float* MNuwXDSoGEYeABeVTwOh;
  float* vxtNGOWYjhKeBBSzuIMB;
  float* MUmglsoWcEiRiAZsclur;
  MWTensor* XLJXOFXdnZOyJvtltbyr; // for pre-padded input
  float* aLsOwwcceEmRSYzllBNs;
  float  cQBKlCKXxecGPJrXBXdk;
  int eVAFqeShtGZAZluKdMvQ;
  int eqOmMKQRpqBqRQCnJmxt;


  public:
    MWFusedConvReLULayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int, int, int, int, int,  int, int, int, int, int, int, int, int, const char*, const char*, int outbufIdx);
    ~MWFusedConvReLULayerImpl();

    void createFusedConvReLULayer(int, int, int, int, int, int, int, int, const char*, const char*, int outbufIdx);
    void predict();
    void cleanup();
    virtual void postSetup();

    void setOutput2(float*);  // Set the pointer to the second half of the output for grouped convolution
    float* getOutput2();     // Get the pointer to the second half of the output for grouped convolution
    cudnnTensorDescriptor_t* getGroupDescriptor();  // Get the cuDNN descriptor of the output for grouped convolution

    // xxx tbd
    float  getIsGrouped();          // Get the isGrouped parameter
    void   setIsGrouped(float);     // Set the isGrouped parameter

  private:
    void loadWeights(const char*);
    void loadBias(const char*);
    void getConvAlgoTuned();
    void getConvAlgoNoWorkSpace();

  private:
    cudnnConvolutionDescriptor_t  QMgBqCuvjnbWHWiVPEwn;
    cudnnConvolutionFwdAlgo_t     PmFfARVzoHVAYkfpuvqK;

    cudnnFilterDescriptor_t       UpnEytIWGokwbTFkBcSx;
    cudnnTensorDescriptor_t       NMMfJylfQjiIUAKhXCJb;

    cudnnTensorDescriptor_t       cCXqPFPPcoHzYMDpnUxQ;
    cudnnTensorDescriptor_t       WOJynDmqVUPWjAGVIuMQ;

    cudnnTensorDescriptor_t      bUVPfnrJhLfHzOLUUrKk;

    cudnnActivationDescriptor_t   olKGEIcsxmLSoMhRhEtP;

};

#endif
