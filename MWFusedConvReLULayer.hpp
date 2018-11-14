/* Copyright 2018 The MathWorks, Inc. */

#include "cnn_api.hpp"

#ifndef __FUSED_CONV_RELU_LAYER_HPP
#define __FUSED_CONV_RELU_LAYER_HPP

/**
  *  Codegen class for Fused Convolution2D-ReLU
**/
class MWFusedConvReLULayer : public MWCNNLayer
{
public:
    MWFusedConvReLULayer();
    ~MWFusedConvReLULayer();
    virtual void postSetup();
    void createFusedConvReLULayer(MWTargetNetworkImpl* , MWTensor*, int, int, int, int, int, int,
                                  int, int, int, int, int, int, int, const char*, const char*, int);
};

#endif
