/* Copyright 2018 The MathWorks, Inc. */

#include "MWFusedConvReLULayer.hpp"
#include "MWFusedConvReLULayerImpl.hpp"

// utils
#include <stdio.h>
#include <cassert>


MWFusedConvReLULayer::MWFusedConvReLULayer()
{
}

MWFusedConvReLULayer::~MWFusedConvReLULayer()
{
}


void MWFusedConvReLULayer::createFusedConvReLULayer(MWTargetNetworkImpl* ntwk_impl,
                                                    MWTensor* m_in,
                                                    int m_r,
                                                    int m_s,
                                                    int m_c,
                                                    int m_k,
                                                    int m_StrideH,
                                                    int m_StrideW,
                                                    int m_PaddingH_T,
                                                    int m_PaddingH_B,
                                                    int m_PaddingW_L,
                                                    int m_PaddingW_R,
                                                    int m_DilationFactorH,
                                                    int m_DilationFactorW,
                                                    int m_g,
                                                    const char* m_weights_file,
                                                    const char* m_bias_file,
                                                    int outbufIdx)
{

    setInputTensor(m_in);

    int m_FilterHeight        = m_r;
    int m_FilterWidth         = m_s;
    int m_NumGroups           = m_g;
    int m_NumChannels =  m_c;
    int m_NumFilters  =  m_k;

    int m_filterH_temp = ((m_FilterHeight-1)*m_DilationFactorH)+1;
    int m_filterW_temp = ((m_FilterWidth-1)*m_DilationFactorW)+1;
    int outputH = ((getInputTensor()->getHeight()- m_filterH_temp + m_PaddingH_B + m_PaddingH_T)/m_StrideH) + 1;
    int outputW = ((getInputTensor()->getWidth()- m_filterW_temp + m_PaddingW_L + m_PaddingW_R)/m_StrideW) + 1;

    // allocate output tensor
    allocateOutputTensor(outputH, outputW, m_NumFilters*m_NumGroups , getInputTensor()->getBatchSize(), NULL);

    m_impl = new MWFusedConvReLULayerImpl(this, ntwk_impl, m_FilterHeight, m_FilterWidth, m_NumGroups, m_NumChannels, m_NumFilters, m_StrideH, m_StrideW,
                                    m_PaddingH_T, m_PaddingH_B, m_PaddingW_L, m_PaddingW_R, m_DilationFactorH, m_DilationFactorW, m_weights_file, m_bias_file, outbufIdx);

    /*Setting the data pointer */
    MWTensor *opTensor = getOutputTensor();
    opTensor->setData(m_impl->getData());


}

void MWFusedConvReLULayer::postSetup()
{
   m_impl->postSetup();
}
