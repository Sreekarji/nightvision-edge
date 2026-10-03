/**
  ******************************************************************************
  * @file    nirdet.h
  * @author  STEdgeAI
  * @date    2026-10-03 16:30:27
  * @brief   Minimal description of the generated c-implemention of the network
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef LL_ATON_NIRDET_H
#define LL_ATON_NIRDET_H

/******************************************************************************/
#define LL_ATON_NIRDET_C_MODEL_NAME        "nirdet"
#define LL_ATON_NIRDET_ORIGIN_MODEL_NAME   "nirdetint8qdq"

/************************** USER ALLOCATED IOs ********************************/
// No user allocated inputs
// No user allocated outputs

/************************** INPUTS ********************************************/
#define LL_ATON_NIRDET_IN_NUM        (1)    // Total number of input buffers
// Input buffer 1 -- Input_1_out_0
#define LL_ATON_NIRDET_IN_1_ALIGNMENT   (32)
#define LL_ATON_NIRDET_IN_1_SIZE_BYTES  (147456)

/************************** OUTPUTS *******************************************/
#define LL_ATON_NIRDET_OUT_NUM        (9)    // Total number of output buffers
// Output buffer 1 -- Quantize_221_out_0
#define LL_ATON_NIRDET_OUT_1_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_1_SIZE_BYTES  (2304)
// Output buffer 2 -- Quantize_229_out_0
#define LL_ATON_NIRDET_OUT_2_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_2_SIZE_BYTES  (4608)
// Output buffer 3 -- Quantize_227_out_0
#define LL_ATON_NIRDET_OUT_3_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_3_SIZE_BYTES  (4608)
// Output buffer 4 -- Quantize_200_out_0
#define LL_ATON_NIRDET_OUT_4_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_4_SIZE_BYTES  (576)
// Output buffer 5 -- Quantize_208_out_0
#define LL_ATON_NIRDET_OUT_5_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_5_SIZE_BYTES  (1152)
// Output buffer 6 -- Quantize_206_out_0
#define LL_ATON_NIRDET_OUT_6_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_6_SIZE_BYTES  (1152)
// Output buffer 7 -- Quantize_152_out_0
#define LL_ATON_NIRDET_OUT_7_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_7_SIZE_BYTES  (144)
// Output buffer 8 -- Quantize_160_out_0
#define LL_ATON_NIRDET_OUT_8_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_8_SIZE_BYTES  (288)
// Output buffer 9 -- Quantize_158_out_0
#define LL_ATON_NIRDET_OUT_9_ALIGNMENT   (32)
#define LL_ATON_NIRDET_OUT_9_SIZE_BYTES  (288)

#endif /* LL_ATON_NIRDET_H */
