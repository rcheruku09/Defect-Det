/* Copyright 2017 The MathWorks, Inc. */

#include "MWMaxUnpoolingLayer.hpp"
#include "MWMaxUnpoolingLayerImpl.hpp"

#include <stdarg.h>
#include <cassert>

MWMaxUnpoolingLayer::MWMaxUnpoolingLayer()
{
}

MWMaxUnpoolingLayer::~MWMaxUnpoolingLayer()
{    
}

void MWMaxUnpoolingLayer::createMaxUnpoolingLayer(MWTargetNetworkImpl* ntwk_impl,
                                                  MWTensor* dataInput,
                                                  MWTensor* indexInput,
                                                  int outbufIdx)
{

    setInputTensor(dataInput, 0);
    setInputTensor(indexInput, 1);                                      

    // Get height and width of input to max pool layer
    int outH = indexInput->getOwner()->getInputTensor(0)->getHeight();
    int outW = indexInput->getOwner()->getInputTensor(0)->getWidth();
    int numOutputFeatures = getInputTensor()->getChannels();
    int BatchSize = getInputTensor()->getBatchSize();
    allocateOutputTensor(outH, outW, numOutputFeatures, BatchSize, NULL, 0);
     
    m_impl = new MWMaxUnpoolingLayerImpl(this, ntwk_impl, outbufIdx);
                                                                                                      
    MWTensor *opTensor = getOutputTensor();
    opTensor->setData(m_impl->getData());   
}
