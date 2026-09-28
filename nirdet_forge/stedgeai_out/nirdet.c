/**
  ******************************************************************************
  * @file    nirdet.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-27T13:59:54+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "nirdet.h"
#include "nirdet_details.h"
#include "nirdet_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NIRDET_EVENT_NODE_START_CB
  #define _STAI_NIRDET_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NIRDET_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NIRDET_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NIRDET_EVENT_NODE_STOP_CB
  #define _STAI_NIRDET_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NIRDET_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NIRDET_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NIRDET_MODEL_SIGNATURE     "0x8003b4db0638478ea8b8c5154e9dca25"
#define _STAI_NIRDET_DATETIME            "2026-09-27T13:59:54+0530"
#define _STAI_NIRDET_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NIRDET_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_nirdet_activations_1     (NULL)




#if defined(HAVE_NIRDET_INFO)
/*****************************************************************************/
static const stai_network_info g_nirdet_info = {
  .model_signature = _STAI_NIRDET_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NIRDET_COMPILE_DATETIME,
  .c_model_name = STAI_NIRDET_MODEL_NAME,
  .c_model_datetime = _STAI_NIRDET_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 1),
  .tool_version = STAI_INIT_VERSION(4, 0, 1),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NIRDET_MACC_NUM,
  .n_nodes = STAI_NIRDET_NODES_NUM,
  .flags = STAI_NIRDET_FLAGS,
  .n_inputs = STAI_NIRDET_IN_NUM,
  .n_outputs = STAI_NIRDET_OUT_NUM,
  .n_activations = STAI_NIRDET_ACTIVATIONS_NUM,
  .n_weights = STAI_NIRDET_WEIGHTS_NUM,
  .n_states = STAI_NIRDET_STATES_NUM,
  .inputs = (stai_tensor[STAI_NIRDET_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NIRDET_IN_1_NAME,
      STAI_NIRDET_IN_1_FLAGS,
      STAI_NIRDET_IN_1_FORMAT,
      STAI_NIRDET_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 1, 288, 512),
      STAI_DECLARE_ARRAY(float, 1, 0.003921568859368563f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NIRDET_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_1_NAME,
      STAI_NIRDET_OUT_1_FLAGS,
      STAI_NIRDET_OUT_1_FORMAT,
      STAI_NIRDET_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 1, 36, 64),
      STAI_DECLARE_ARRAY(float, 1, 0.055070631206035614f),
      STAI_DECLARE_ARRAY(int16_t, 1, 81)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_2_NAME,
      STAI_NIRDET_OUT_2_FLAGS,
      STAI_NIRDET_OUT_2_FORMAT,
      STAI_NIRDET_OUT_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 36, 64),
      STAI_DECLARE_ARRAY(float, 1, 0.18284200131893158f),
      STAI_DECLARE_ARRAY(int16_t, 1, 2)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_3_NAME,
      STAI_NIRDET_OUT_3_FLAGS,
      STAI_NIRDET_OUT_3_FORMAT,
      STAI_NIRDET_OUT_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 36, 64),
      STAI_DECLARE_ARRAY(float, 1, 0.017869222909212112f),
      STAI_DECLARE_ARRAY(int16_t, 1, 119)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_4_NAME,
      STAI_NIRDET_OUT_4_FLAGS,
      STAI_NIRDET_OUT_4_FORMAT,
      STAI_NIRDET_OUT_4_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 1, 18, 32),
      STAI_DECLARE_ARRAY(float, 1, 0.044113121926784515f),
      STAI_DECLARE_ARRAY(int16_t, 1, 75)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_5_NAME,
      STAI_NIRDET_OUT_5_FLAGS,
      STAI_NIRDET_OUT_5_FORMAT,
      STAI_NIRDET_OUT_5_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 18, 32),
      STAI_DECLARE_ARRAY(float, 1, 0.14666779339313507f),
      STAI_DECLARE_ARRAY(int16_t, 1, -4)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_6_NAME,
      STAI_NIRDET_OUT_6_FLAGS,
      STAI_NIRDET_OUT_6_FORMAT,
      STAI_NIRDET_OUT_6_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 18, 32),
      STAI_DECLARE_ARRAY(float, 1, 0.016024518758058548f),
      STAI_DECLARE_ARRAY(int16_t, 1, 124)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_7_NAME,
      STAI_NIRDET_OUT_7_FLAGS,
      STAI_NIRDET_OUT_7_FORMAT,
      STAI_NIRDET_OUT_7_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 1, 9, 16),
      STAI_DECLARE_ARRAY(float, 1, 0.02662220038473606f),
      STAI_DECLARE_ARRAY(int16_t, 1, 127)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_8_NAME,
      STAI_NIRDET_OUT_8_FLAGS,
      STAI_NIRDET_OUT_8_FORMAT,
      STAI_NIRDET_OUT_8_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 9, 16),
      STAI_DECLARE_ARRAY(float, 1, 0.03284941241145134f),
      STAI_DECLARE_ARRAY(int16_t, 1, 43)),
    STAI_INIT_TENSOR(
      STAI_NIRDET_OUT_9_NAME,
      STAI_NIRDET_OUT_9_FLAGS,
      STAI_NIRDET_OUT_9_FORMAT,
      STAI_NIRDET_OUT_9_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 9, 16),
      STAI_DECLARE_ARRAY(float, 1, 0.015480712987482548f),
      STAI_DECLARE_ARRAY(int16_t, 1, 127)),
    },
  .activations = (stai_tensor[STAI_NIRDET_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NIRDET_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NIRDET_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 1019584),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NIRDET_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NIRDET_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NIRDET_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 844780),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_nirdet_context* _net_ctx = (_stai_nirdet_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_nirdet_check(_stai_nirdet_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NIRDET_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NIRDET_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NIRDET_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NIRDET_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NIRDET_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NIRDET_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NIRDET_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NIRDET_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_nirdet_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_nirdet_context* net_ctx = (_stai_nirdet_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_nirdet_context))

  _STAI_SET_ERROR(net_ctx, STAI_NIRDET_CONTEXT_SIZE != sizeof(_stai_nirdet_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_nirdet_context _nirdet_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NIRDET_MODEL_SIGNATURE,
      ._flags = STAI_NIRDET_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_nirdet_activations_1
      },
      ._weights = {
      (stai_ptr)g_nirdet_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _nirdet_context;

    _stai_nirdet_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NIRDET_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027351221069693565f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_AveragePool_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027351221069693565f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_AveragePool_3_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027351221069693565f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_proj_2_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05105074867606163f),
    AI_PACK_INTQ_ZP(-101)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_proj_2_Conv_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10549747943878174f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Sigmoid_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921534400433302f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00588231859728694f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019607844296842813f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_proj_1_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07647889107465744f),
    AI_PACK_INTQ_ZP(-110)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Sigmoid_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0058823530562222f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_proj_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0974404588341713f),
    AI_PACK_INTQ_ZP(-114)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Sigmoid_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Add_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0058823530562222f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage1_cv1_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage1_blocks_blocks_0_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage1_cv2_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage1_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_cv1_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_blocks_blocks_0_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_blocks_blocks_1_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_cv2_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_cv1_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_blocks_blocks_0_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_blocks_blocks_1_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_cv2_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage4_cv1_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage4_blocks_blocks_0_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage4_cv2_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019807321950793266f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage4_Concat_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage4_cv3_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Clip_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_cls_branch_blocks_0_act_2_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020054398104548454f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cls32_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02662220038473606f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cls32_QuantizeLinear_Input_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023094723001122475f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(size32_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015480712987482548f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(size32_QuantizeLinear_Input_Transpose_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015480712987482548f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(off32_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03284941241145134f),
    AI_PACK_INTQ_ZP(43)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(off32_QuantizeLinear_Input_Transpose_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03284941241145134f),
    AI_PACK_INTQ_ZP(43)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_lat5_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08638046681880951f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_Resize_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08638046681880951f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage3_cv3_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Clip_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_lat4_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09602516144514084f),
    AI_PACK_INTQ_ZP(18)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0470588244497776f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_Resize_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0470588244497776f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_backbone_stage2_cv3_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_lat3_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.119174525141716f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_Clip_1_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0470588244497776f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_pan3to4_block_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_neck_Clip_2_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0470588244497776f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_cls_branch_blocks_0_act_1_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cls16_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.044113121926784515f),
    AI_PACK_INTQ_ZP(75)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cls16_QuantizeLinear_Input_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007462222129106522f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(size16_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016024518758058548f),
    AI_PACK_INTQ_ZP(124)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(size16_QuantizeLinear_Input_Transpose_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016024518758058548f),
    AI_PACK_INTQ_ZP(124)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(off16_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14666779339313507f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(off16_QuantizeLinear_Input_Transpose_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14666779339313507f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_cls_branch_blocks_0_act_Clip_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cls8_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.055070631206035614f),
    AI_PACK_INTQ_ZP(81)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cls8_QuantizeLinear_Input_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005376456771045923f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(size8_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017869222909212112f),
    AI_PACK_INTQ_ZP(119)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(size8_QuantizeLinear_Input_Transpose_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017869222909212112f),
    AI_PACK_INTQ_ZP(119)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(off8_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.18284200131893158f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(off8_QuantizeLinear_Input_Transpose_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.18284200131893158f),
    AI_PACK_INTQ_ZP(2)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  _Abs_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 147456, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _AveragePool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36864, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _AveragePool_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _AveragePool_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _AveragePool_3_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _proj_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _proj_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _proj_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _proj_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _Sigmoid_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _Add_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _proj_1_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _proj_1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _Sigmoid_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _proj_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _proj_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _Add_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage1_cv1_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage1_blocks_blocks_0_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage1_cv2_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage1_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 442368, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_cv1_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 110592, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 110592, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_0_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 110592, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 110592, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_1_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 110592, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_cv2_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 110592, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_cv1_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_0_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_1_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_cv2_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 27648, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 55296, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage4_cv1_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage4_blocks_blocks_0_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage4_cv2_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage4_Concat_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage4_cv3_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _Clip_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _cls_branch_blocks_0_act_2_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 144, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  size32_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  size32_QuantizeLinear_Input_Transpose_8_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 288, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  off32_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  off32_QuantizeLinear_Input_Transpose_7_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 288, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _neck_lat5_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _neck_Resize_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage3_cv3_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 55296, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _Clip_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 55296, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _neck_lat4_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _neck_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _neck_Resize_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 147456, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _backbone_stage2_cv3_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 221184, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _neck_lat3_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 147456, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _neck_Clip_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 147456, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _neck_pan3to4_block_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _neck_Clip_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _cls_branch_blocks_0_act_1_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 576, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  size16_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  size16_QuantizeLinear_Input_Transpose_5_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1152, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  off16_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  off16_QuantizeLinear_Input_Transpose_4_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1152, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _cls_branch_blocks_0_act_Clip_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 147456, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2304, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  size8_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  size8_QuantizeLinear_Input_Transpose_2_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 4608, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  off8_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  off8_QuantizeLinear_Input_Transpose_1_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 4608, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Abs_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 256, 144), AI_STRIDE_INIT(4, 4, 4, 16, 4096),
  1, &_Abs_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _AveragePool_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 128, 72), AI_STRIDE_INIT(4, 4, 4, 16, 2048),
  1, &_AveragePool_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _AveragePool_1_output_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 64, 36), AI_STRIDE_INIT(4, 4, 4, 16, 1024),
  1, &_AveragePool_1_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 64, 36), AI_STRIDE_INIT(4, 1, 1, 4, 256),
  1, &_AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array, &_AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _AveragePool_2_output_0_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 32, 18), AI_STRIDE_INIT(4, 1, 1, 4, 128),
  1, &_AveragePool_2_output_0_output_array, &_AveragePool_2_output_0_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _AveragePool_3_output_0_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 16, 9), AI_STRIDE_INIT(4, 1, 1, 4, 64),
  1, &_AveragePool_3_output_0_output_array, &_AveragePool_3_output_0_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _proj_2_Conv_output_0_bias, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_proj_2_Conv_output_0_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _proj_2_Conv_output_0_output, AI_STATIC,
  188, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 16, 9), AI_STRIDE_INIT(4, 1, 1, 1, 16),
  1, &_proj_2_Conv_output_0_output_array, &_proj_2_Conv_output_0_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _proj_2_Conv_output_0_scratch0, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_proj_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _proj_2_Conv_output_0_weights, AI_STATIC,
  190, 0x1,
  AI_SHAPE_INIT(4, 4, 1, 1, 1), AI_STRIDE_INIT(4, 1, 4, 4, 4),
  1, &_proj_2_Conv_output_0_weights_array, &_proj_2_Conv_output_0_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _Sigmoid_2_output_0_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 16, 9), AI_STRIDE_INIT(4, 1, 1, 1, 16),
  1, &_Sigmoid_2_output_0_output_array, &_Sigmoid_2_output_0_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Add_2_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 16, 9), AI_STRIDE_INIT(4, 1, 1, 1, 16),
  1, &_Add_2_output_0_output_array, &_Add_2_output_0_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _proj_1_Conv_output_0_output, AI_STATIC,
  185, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 32, 18), AI_STRIDE_INIT(4, 1, 1, 1, 32),
  1, &_proj_1_Conv_output_0_output_array, &_proj_1_Conv_output_0_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _proj_1_Conv_output_0_scratch0, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_proj_1_Conv_output_0_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _Sigmoid_1_output_0_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 32, 18), AI_STRIDE_INIT(4, 1, 1, 1, 32),
  1, &_Sigmoid_1_output_0_output_array, &_Sigmoid_1_output_0_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 32, 18), AI_STRIDE_INIT(4, 1, 1, 1, 32),
  1, &_Add_1_output_0_output_array, &_Add_1_output_0_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _proj_Conv_output_0_output, AI_STATIC,
  191, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 36), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &_proj_Conv_output_0_output_array, &_proj_Conv_output_0_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _proj_Conv_output_0_scratch0, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_proj_Conv_output_0_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _Sigmoid_output_0_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 36), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &_Sigmoid_output_0_output_array, &_Sigmoid_output_0_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _Add_output_0_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 36), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &_Add_output_0_output_array, &_Add_output_0_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage1_blocks_blocks_0_Clip_output_0_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 128, 72), AI_STRIDE_INIT(4, 1, 1, 24, 3072),
  1, &_backbone_stage1_blocks_blocks_0_Clip_output_0_output_array, &_backbone_stage1_blocks_blocks_0_Clip_output_0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 128, 72), AI_STRIDE_INIT(4, 1, 1, 24, 3072),
  1, &_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output_array, &_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage1_cv1_act_Clip_output_0_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 128, 72), AI_STRIDE_INIT(4, 1, 1, 24, 3072),
  1, &_backbone_stage1_cv1_act_Clip_output_0_output_array, &_backbone_stage1_cv1_act_Clip_output_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage1_Concat_output_0_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 128, 72), AI_STRIDE_INIT(4, 1, 1, 48, 6144),
  1, &_backbone_stage1_Concat_output_0_output_array, &_backbone_stage1_Concat_output_0_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage1_cv2_act_Clip_output_0_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 128, 72), AI_STRIDE_INIT(4, 1, 1, 24, 3072),
  1, &_backbone_stage1_cv2_act_Clip_output_0_output_array, &_backbone_stage1_cv2_act_Clip_output_0_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_0_Clip_output_0_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 64, 36), AI_STRIDE_INIT(4, 1, 1, 48, 3072),
  1, &_backbone_stage2_blocks_blocks_0_Clip_output_0_output_array, &_backbone_stage2_blocks_blocks_0_Clip_output_0_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 64, 36), AI_STRIDE_INIT(4, 1, 1, 48, 3072),
  1, &_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output_array, &_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_cv1_act_Clip_output_0_output, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 64, 36), AI_STRIDE_INIT(4, 1, 1, 48, 3072),
  1, &_backbone_stage2_cv1_act_Clip_output_0_output_array, &_backbone_stage2_cv1_act_Clip_output_0_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_1_Clip_output_0_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 64, 36), AI_STRIDE_INIT(4, 1, 1, 48, 3072),
  1, &_backbone_stage2_blocks_blocks_1_Clip_output_0_output_array, &_backbone_stage2_blocks_blocks_1_Clip_output_0_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 64, 36), AI_STRIDE_INIT(4, 1, 1, 48, 3072),
  1, &_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output_array, &_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_Concat_output_0_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 64, 36), AI_STRIDE_INIT(4, 1, 1, 96, 6144),
  1, &_backbone_stage2_Concat_output_0_output_array, &_backbone_stage2_Concat_output_0_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_cv2_act_Clip_output_0_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 64, 36), AI_STRIDE_INIT(4, 1, 1, 48, 3072),
  1, &_backbone_stage2_cv2_act_Clip_output_0_output_array, &_backbone_stage2_cv2_act_Clip_output_0_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_0_Clip_output_0_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 32, 18), AI_STRIDE_INIT(4, 1, 1, 48, 1536),
  1, &_backbone_stage3_blocks_blocks_0_Clip_output_0_output_array, &_backbone_stage3_blocks_blocks_0_Clip_output_0_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 32, 18), AI_STRIDE_INIT(4, 1, 1, 48, 1536),
  1, &_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output_array, &_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_cv1_act_Clip_output_0_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 32, 18), AI_STRIDE_INIT(4, 1, 1, 48, 1536),
  1, &_backbone_stage3_cv1_act_Clip_output_0_output_array, &_backbone_stage3_cv1_act_Clip_output_0_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_1_Clip_output_0_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 32, 18), AI_STRIDE_INIT(4, 1, 1, 48, 1536),
  1, &_backbone_stage3_blocks_blocks_1_Clip_output_0_output_array, &_backbone_stage3_blocks_blocks_1_Clip_output_0_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output, AI_STATIC,
  84, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 32, 18), AI_STRIDE_INIT(4, 1, 1, 48, 1536),
  1, &_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output_array, &_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_Concat_output_0_output, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 32, 18), AI_STRIDE_INIT(4, 1, 1, 96, 3072),
  1, &_backbone_stage3_Concat_output_0_output_array, &_backbone_stage3_Concat_output_0_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_cv2_act_Clip_output_0_output, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 32, 18), AI_STRIDE_INIT(4, 1, 1, 48, 1536),
  1, &_backbone_stage3_cv2_act_Clip_output_0_output_array, &_backbone_stage3_cv2_act_Clip_output_0_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage4_blocks_blocks_0_Clip_output_0_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 16, 9), AI_STRIDE_INIT(4, 1, 1, 48, 768),
  1, &_backbone_stage4_blocks_blocks_0_Clip_output_0_output_array, &_backbone_stage4_blocks_blocks_0_Clip_output_0_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 16, 9), AI_STRIDE_INIT(4, 1, 1, 48, 768),
  1, &_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output_array, &_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage4_cv1_act_Clip_output_0_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 16, 9), AI_STRIDE_INIT(4, 1, 1, 48, 768),
  1, &_backbone_stage4_cv1_act_Clip_output_0_output_array, &_backbone_stage4_cv1_act_Clip_output_0_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage4_Concat_output_0_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 16, 9), AI_STRIDE_INIT(4, 1, 1, 96, 1536),
  1, &_backbone_stage4_Concat_output_0_output_array, &_backbone_stage4_Concat_output_0_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage4_cv2_act_Clip_output_0_output, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 16, 9), AI_STRIDE_INIT(4, 1, 1, 48, 768),
  1, &_backbone_stage4_cv2_act_Clip_output_0_output_array, &_backbone_stage4_cv2_act_Clip_output_0_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _Clip_2_output_0_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 16, 9), AI_STRIDE_INIT(4, 1, 1, 96, 1536),
  1, &_Clip_2_output_0_output_array, &_Clip_2_output_0_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage4_cv3_act_Clip_output_0_output, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 16, 9), AI_STRIDE_INIT(4, 1, 1, 96, 1536),
  1, &_backbone_stage4_cv3_act_Clip_output_0_output_array, &_backbone_stage4_cv3_act_Clip_output_0_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _cls_branch_blocks_0_act_2_Clip_output_0_output, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 9), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &_cls_branch_blocks_0_act_2_Clip_output_0_output_array, &_cls_branch_blocks_0_act_2_Clip_output_0_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_bias, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &cls32_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_output, AI_STATIC,
  225, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 16, 9), AI_STRIDE_INIT(4, 1, 1, 1, 16),
  1, &cls32_QuantizeLinear_Input_output_array, &cls32_QuantizeLinear_Input_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_scratch0, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &cls32_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_weights, AI_STATIC,
  227, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &cls32_QuantizeLinear_Input_weights_array, &cls32_QuantizeLinear_Input_weights_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  size32_QuantizeLinear_Input_Transpose_8_output, AI_STATIC,
  253, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 9, 2), AI_STRIDE_INIT(4, 1, 1, 16, 144),
  1, &size32_QuantizeLinear_Input_Transpose_8_output_array, &size32_QuantizeLinear_Input_Transpose_8_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  size32_QuantizeLinear_Input_output, AI_STATIC,
  255, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 16, 9), AI_STRIDE_INIT(4, 1, 1, 2, 32),
  1, &size32_QuantizeLinear_Input_output_array, &size32_QuantizeLinear_Input_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  off32_QuantizeLinear_Input_Transpose_7_output, AI_STATIC,
  238, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 9, 2), AI_STRIDE_INIT(4, 1, 1, 16, 144),
  1, &off32_QuantizeLinear_Input_Transpose_7_output_array, &off32_QuantizeLinear_Input_Transpose_7_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  off32_QuantizeLinear_Input_output, AI_STATIC,
  240, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 16, 9), AI_STRIDE_INIT(4, 1, 1, 2, 32),
  1, &off32_QuantizeLinear_Input_output_array, &off32_QuantizeLinear_Input_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _neck_Resize_output_0_output, AI_STATIC,
  152, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 18), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_neck_Resize_output_0_output_array, &_neck_Resize_output_0_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _neck_lat5_Conv_output_0_output, AI_STATIC,
  162, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 9), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &_neck_lat5_Conv_output_0_output_array, &_neck_lat5_Conv_output_0_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _Clip_1_output_0_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 32, 18), AI_STRIDE_INIT(4, 1, 1, 96, 3072),
  1, &_Clip_1_output_0_output_array, &_Clip_1_output_0_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage3_cv3_act_Clip_output_0_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 32, 18), AI_STRIDE_INIT(4, 1, 1, 96, 3072),
  1, &_backbone_stage3_cv3_act_Clip_output_0_output_array, &_backbone_stage3_cv3_act_Clip_output_0_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _neck_Clip_output_0_output, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 18), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_neck_Clip_output_0_output_array, &_neck_Clip_output_0_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _neck_lat4_Conv_output_0_output, AI_STATIC,
  158, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 18), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_neck_lat4_Conv_output_0_output_array, &_neck_lat4_Conv_output_0_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _neck_Resize_1_output_0_output, AI_STATIC,
  151, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 64, 36), AI_STRIDE_INIT(4, 1, 1, 64, 4096),
  1, &_neck_Resize_1_output_0_output_array, &_neck_Resize_1_output_0_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _Clip_output_0_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 64, 36), AI_STRIDE_INIT(4, 1, 1, 96, 6144),
  1, &_Clip_output_0_output_array, &_Clip_output_0_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _backbone_stage2_cv3_act_Clip_output_0_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 64, 36), AI_STRIDE_INIT(4, 1, 1, 96, 6144),
  1, &_backbone_stage2_cv3_act_Clip_output_0_output_array, &_backbone_stage2_cv3_act_Clip_output_0_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _neck_Clip_1_output_0_output, AI_STATIC,
  148, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 64, 36), AI_STRIDE_INIT(4, 1, 1, 64, 4096),
  1, &_neck_Clip_1_output_0_output_array, &_neck_Clip_1_output_0_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _neck_lat3_Conv_output_0_output, AI_STATIC,
  154, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 64, 36), AI_STRIDE_INIT(4, 1, 1, 64, 4096),
  1, &_neck_lat3_Conv_output_0_output_array, &_neck_lat3_Conv_output_0_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _neck_Clip_2_output_0_output, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 18), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_neck_Clip_2_output_0_output_array, &_neck_Clip_2_output_0_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _neck_pan3to4_block_act_Clip_output_0_output, AI_STATIC,
  181, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 18), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_neck_pan3to4_block_act_Clip_output_0_output_array, &_neck_pan3to4_block_act_Clip_output_0_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _cls_branch_blocks_0_act_1_Clip_output_0_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 18), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_cls_branch_blocks_0_act_1_Clip_output_0_output_array, &_cls_branch_blocks_0_act_1_Clip_output_0_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_bias, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &cls16_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_output, AI_STATIC,
  221, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 32, 18), AI_STRIDE_INIT(4, 1, 1, 1, 32),
  1, &cls16_QuantizeLinear_Input_output_array, &cls16_QuantizeLinear_Input_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_scratch0, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &cls16_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_weights, AI_STATIC,
  223, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &cls16_QuantizeLinear_Input_weights_array, &cls16_QuantizeLinear_Input_weights_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  size16_QuantizeLinear_Input_Transpose_5_output, AI_STATIC,
  248, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 18, 2), AI_STRIDE_INIT(4, 1, 1, 32, 576),
  1, &size16_QuantizeLinear_Input_Transpose_5_output_array, &size16_QuantizeLinear_Input_Transpose_5_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  size16_QuantizeLinear_Input_output, AI_STATIC,
  250, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 32, 18), AI_STRIDE_INIT(4, 1, 1, 2, 64),
  1, &size16_QuantizeLinear_Input_output_array, &size16_QuantizeLinear_Input_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  off16_QuantizeLinear_Input_Transpose_4_output, AI_STATIC,
  233, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 18, 2), AI_STRIDE_INIT(4, 1, 1, 32, 576),
  1, &off16_QuantizeLinear_Input_Transpose_4_output_array, &off16_QuantizeLinear_Input_Transpose_4_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  off16_QuantizeLinear_Input_output, AI_STATIC,
  235, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 32, 18), AI_STRIDE_INIT(4, 1, 1, 2, 64),
  1, &off16_QuantizeLinear_Input_output_array, &off16_QuantizeLinear_Input_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _cls_branch_blocks_0_act_Clip_output_0_output, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 64, 36), AI_STRIDE_INIT(4, 1, 1, 64, 4096),
  1, &_cls_branch_blocks_0_act_Clip_output_0_output_array, &_cls_branch_blocks_0_act_Clip_output_0_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_bias, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &cls8_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_output, AI_STATIC,
  229, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 36), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &cls8_QuantizeLinear_Input_output_array, &cls8_QuantizeLinear_Input_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_scratch0, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &cls8_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_weights, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &cls8_QuantizeLinear_Input_weights_array, &cls8_QuantizeLinear_Input_weights_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  size8_QuantizeLinear_Input_Transpose_2_output, AI_STATIC,
  258, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &size8_QuantizeLinear_Input_Transpose_2_output_array, &size8_QuantizeLinear_Input_Transpose_2_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  size8_QuantizeLinear_Input_output, AI_STATIC,
  260, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 64, 36), AI_STRIDE_INIT(4, 1, 1, 2, 128),
  1, &size8_QuantizeLinear_Input_output_array, &size8_QuantizeLinear_Input_output_array_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  off8_QuantizeLinear_Input_Transpose_1_output, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &off8_QuantizeLinear_Input_Transpose_1_output_array, &off8_QuantizeLinear_Input_Transpose_1_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  off8_QuantizeLinear_Input_output, AI_STATIC,
  245, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 64, 36), AI_STRIDE_INIT(4, 1, 1, 2, 128),
  1, &off8_QuantizeLinear_Input_output_array, &off8_QuantizeLinear_Input_output_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _AveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Abs_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _AveragePool_output_0_layer, 113,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &_AveragePool_output_0_chain,
  NULL, &_AveragePool_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _AveragePool_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _AveragePool_1_output_0_layer, 115,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &_AveragePool_1_output_0_chain,
  NULL, &_AveragePool_1_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _AveragePool_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _AveragePool_2_output_0_layer, 124,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_AveragePool_2_output_0_chain,
  NULL, &_AveragePool_2_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _AveragePool_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _AveragePool_3_output_0_layer, 135,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_AveragePool_3_output_0_chain,
  NULL, &_AveragePool_3_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _proj_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_proj_2_Conv_output_0_weights, &_proj_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_2_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _proj_2_Conv_output_0_layer, 147,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_proj_2_Conv_output_0_chain,
  NULL, &_proj_2_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _Sigmoid_2_output_0_nl_params_data[] = { -77, -75, -72, -70, -68, -65, -63, -60, -58, -55, -53, -50, -47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -13, -10, -7, -4, 0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 54, 57, 59, 62, 64, 67, 69, 71, 74, 76, 78, 80, 82, 84, 85, 87, 89, 90, 92, 94, 95, 96, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 109, 110, 111, 112, 112, 113, 114, 114, 115, 116, 116, 117, 117, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _Sigmoid_2_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _Sigmoid_2_output_0_nl_params_data, _Sigmoid_2_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Sigmoid_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sigmoid_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Sigmoid_2_output_0_layer, 158,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_Sigmoid_2_output_0_chain,
  NULL, &_Sigmoid_2_output_0_layer, AI_STATIC, 
  .nl_params = &_Sigmoid_2_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Sigmoid_2_output_0_output, &_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_2_output_0_layer, 166,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_2_output_0_chain,
  NULL, &_Add_2_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _proj_1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_proj_2_Conv_output_0_weights, &_proj_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_1_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _proj_1_Conv_output_0_layer, 134,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_proj_1_Conv_output_0_chain,
  NULL, &_proj_1_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _Sigmoid_1_output_0_nl_params_data[] = { -77, -73, -70, -67, -63, -59, -55, -51, -47, -43, -38, -34, -29, -25, -20, -15, -10, -5, -1, 4, 9, 14, 19, 24, 28, 33, 37, 42, 46, 50, 54, 58, 62, 66, 69, 72, 76, 79, 82, 84, 87, 90, 92, 94, 96, 98, 100, 102, 104, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 116, 117, 118, 118, 119, 120, 120, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _Sigmoid_1_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _Sigmoid_1_output_0_nl_params_data, _Sigmoid_1_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Sigmoid_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sigmoid_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Sigmoid_1_output_0_layer, 146,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_Sigmoid_1_output_0_chain,
  NULL, &_Sigmoid_1_output_0_layer, AI_STATIC, 
  .nl_params = &_Sigmoid_1_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Sigmoid_1_output_0_output, &_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_1_output_0_layer, 157,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_1_output_0_chain,
  NULL, &_Add_1_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _proj_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_proj_2_Conv_output_0_weights, &_proj_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _proj_Conv_output_0_layer, 123,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &_proj_Conv_output_0_chain,
  NULL, &_proj_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _Sigmoid_output_0_nl_params_data[] = { -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -7, -1, 6, 12, 18, 24, 30, 36, 41, 47, 52, 57, 62, 67, 71, 75, 79, 83, 86, 89, 92, 95, 98, 100, 102, 105, 106, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    _Sigmoid_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _Sigmoid_output_0_nl_params_data, _Sigmoid_output_0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_proj_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Sigmoid_output_0_layer, 133,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &_Sigmoid_output_0_chain,
  NULL, &_Sigmoid_output_0_layer, AI_STATIC, 
  .nl_params = &_Sigmoid_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Sigmoid_output_0_output, &_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_output_0_layer, 145,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Add_output_0_chain,
  NULL, &_Add_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage1_blocks_blocks_0_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage1_cv1_act_Clip_output_0_output, &_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage1_blocks_blocks_0_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage1_blocks_blocks_0_Clip_output_0_layer, 140,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_backbone_stage1_blocks_blocks_0_Clip_output_0_chain,
  NULL, &_backbone_stage1_blocks_blocks_0_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage1_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage1_blocks_blocks_0_Clip_output_0_output, &_backbone_stage1_cv2_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage1_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage1_Concat_output_0_layer, 152,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_backbone_stage1_Concat_output_0_chain,
  NULL, &_backbone_stage1_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_0_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage2_cv1_act_Clip_output_0_output, &_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage2_blocks_blocks_0_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_0_Clip_output_0_layer, 182,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_backbone_stage2_blocks_blocks_0_Clip_output_0_chain,
  NULL, &_backbone_stage2_blocks_blocks_0_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_1_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage2_blocks_blocks_0_Clip_output_0_output, &_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage2_blocks_blocks_1_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage2_blocks_blocks_1_Clip_output_0_layer, 188,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_backbone_stage2_blocks_blocks_1_Clip_output_0_chain,
  NULL, &_backbone_stage2_blocks_blocks_1_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage2_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage2_blocks_blocks_1_Clip_output_0_output, &_backbone_stage2_cv2_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage2_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage2_Concat_output_0_layer, 191,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_backbone_stage2_Concat_output_0_chain,
  NULL, &_backbone_stage2_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_0_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage3_cv1_act_Clip_output_0_output, &_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage3_blocks_blocks_0_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_0_Clip_output_0_layer, 215,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_backbone_stage3_blocks_blocks_0_Clip_output_0_chain,
  NULL, &_backbone_stage3_blocks_blocks_0_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_1_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage3_blocks_blocks_0_Clip_output_0_output, &_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage3_blocks_blocks_1_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage3_blocks_blocks_1_Clip_output_0_layer, 221,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_backbone_stage3_blocks_blocks_1_Clip_output_0_chain,
  NULL, &_backbone_stage3_blocks_blocks_1_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage3_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage3_blocks_blocks_1_Clip_output_0_output, &_backbone_stage3_cv2_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage3_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage3_Concat_output_0_layer, 224,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_backbone_stage3_Concat_output_0_chain,
  NULL, &_backbone_stage3_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage4_blocks_blocks_0_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage4_cv1_act_Clip_output_0_output, &_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage4_blocks_blocks_0_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage4_blocks_blocks_0_Clip_output_0_layer, 248,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_backbone_stage4_blocks_blocks_0_Clip_output_0_chain,
  NULL, &_backbone_stage4_blocks_blocks_0_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _backbone_stage4_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage4_blocks_blocks_0_Clip_output_0_output, &_backbone_stage4_cv2_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_backbone_stage4_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _backbone_stage4_Concat_output_0_layer, 251,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_backbone_stage4_Concat_output_0_chain,
  NULL, &_backbone_stage4_Concat_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Clip_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage4_cv3_act_Clip_output_0_output, &_Add_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Clip_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Clip_2_output_0_layer, 257,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Clip_2_output_0_chain,
  NULL, &_Clip_2_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_cls_branch_blocks_0_act_2_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cls32_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &cls32_QuantizeLinear_Input_weights, &cls32_QuantizeLinear_Input_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cls32_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  cls32_QuantizeLinear_Input_layer, 285,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &cls32_QuantizeLinear_Input_chain,
  NULL, &cls32_QuantizeLinear_Input_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  size32_QuantizeLinear_Input_Transpose_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &size32_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &size32_QuantizeLinear_Input_Transpose_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  size32_QuantizeLinear_Input_Transpose_8_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &size32_QuantizeLinear_Input_Transpose_8_chain,
  NULL, &size32_QuantizeLinear_Input_Transpose_8_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  off32_QuantizeLinear_Input_Transpose_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &off32_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &off32_QuantizeLinear_Input_Transpose_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  off32_QuantizeLinear_Input_Transpose_7_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &off32_QuantizeLinear_Input_Transpose_7_chain,
  NULL, &off32_QuantizeLinear_Input_Transpose_7_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_float _neck_Resize_output_0_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _neck_Resize_output_0_scales, AI_ARRAY_FORMAT_FLOAT,
    _neck_Resize_output_0_scales_data, _neck_Resize_output_0_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _neck_Resize_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_lat5_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_Resize_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _neck_Resize_output_0_layer, 263,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &_neck_Resize_output_0_chain,
  NULL, &_neck_Resize_output_0_layer, AI_STATIC, 
  .scales = &_neck_Resize_output_0_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Clip_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage3_cv3_act_Clip_output_0_output, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Clip_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Clip_1_output_0_layer, 231,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Clip_1_output_0_chain,
  NULL, &_Clip_1_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _neck_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_neck_lat4_Conv_output_0_output, &_neck_Resize_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _neck_Clip_output_0_layer, 269,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_neck_Clip_output_0_chain,
  NULL, &_neck_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_float _neck_Resize_1_output_0_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _neck_Resize_1_output_0_scales, AI_ARRAY_FORMAT_FLOAT,
    _neck_Resize_1_output_0_scales_data, _neck_Resize_1_output_0_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _neck_Resize_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_Resize_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _neck_Resize_1_output_0_layer, 275,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &_neck_Resize_1_output_0_chain,
  NULL, &_neck_Resize_1_output_0_layer, AI_STATIC, 
  .scales = &_neck_Resize_1_output_0_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_backbone_stage2_cv3_act_Clip_output_0_output, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Clip_output_0_layer, 198,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_Clip_output_0_chain,
  NULL, &_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _neck_Clip_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_neck_lat3_Conv_output_0_output, &_neck_Resize_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_Clip_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _neck_Clip_1_output_0_layer, 284,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_neck_Clip_1_output_0_chain,
  NULL, &_neck_Clip_1_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _neck_Clip_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_neck_Clip_output_0_output, &_neck_pan3to4_block_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_neck_Clip_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _neck_Clip_2_output_0_layer, 302,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_neck_Clip_2_output_0_chain,
  NULL, &_neck_Clip_2_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_cls_branch_blocks_0_act_1_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cls16_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &cls16_QuantizeLinear_Input_weights, &cls16_QuantizeLinear_Input_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cls16_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  cls16_QuantizeLinear_Input_layer, 335,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &cls16_QuantizeLinear_Input_chain,
  NULL, &cls16_QuantizeLinear_Input_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  size16_QuantizeLinear_Input_Transpose_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &size16_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &size16_QuantizeLinear_Input_Transpose_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  size16_QuantizeLinear_Input_Transpose_5_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &size16_QuantizeLinear_Input_Transpose_5_chain,
  NULL, &size16_QuantizeLinear_Input_Transpose_5_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  off16_QuantizeLinear_Input_Transpose_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &off16_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &off16_QuantizeLinear_Input_Transpose_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  off16_QuantizeLinear_Input_Transpose_4_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &off16_QuantizeLinear_Input_Transpose_4_chain,
  NULL, &off16_QuantizeLinear_Input_Transpose_4_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_cls_branch_blocks_0_act_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cls8_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &cls8_QuantizeLinear_Input_weights, &cls8_QuantizeLinear_Input_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cls8_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  cls8_QuantizeLinear_Input_layer, 318,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &cls8_QuantizeLinear_Input_chain,
  NULL, &cls8_QuantizeLinear_Input_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  size8_QuantizeLinear_Input_Transpose_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &size8_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &size8_QuantizeLinear_Input_Transpose_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  size8_QuantizeLinear_Input_Transpose_2_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &size8_QuantizeLinear_Input_Transpose_2_chain,
  NULL, &size8_QuantizeLinear_Input_Transpose_2_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  off8_QuantizeLinear_Input_Transpose_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &off8_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &off8_QuantizeLinear_Input_Transpose_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  off8_QuantizeLinear_Input_Transpose_1_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &off8_QuantizeLinear_Input_Transpose_1_chain,
  NULL, &off8_QuantizeLinear_Input_Transpose_1_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_ap__AveragePool_output_0(_stai_nirdet_context* net_ctx)
{
  _Abs_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _Abs_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _STAI_NIRDET_EVENT_NODE_START_CB(113, 1, { _Abs_output_0_output.data->data});
  forward_ap(&_AveragePool_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(113, 1, { _AveragePool_output_0_output.data->data});
}
void forward_lite_ap__AveragePool_1_output_0(_stai_nirdet_context* net_ctx)
{
  _AveragePool_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 366336);
  _AveragePool_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 366336);
  _STAI_NIRDET_EVENT_NODE_START_CB(115, 1, { _AveragePool_output_0_output.data->data});
  forward_ap(&_AveragePool_1_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(115, 1, { _AveragePool_1_output_0_output.data->data});
}
void forward_lite_ap_integer_INT8__AveragePool_2_output_0(_stai_nirdet_context* net_ctx)
{
  _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228096);
  _AveragePool_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228096);
  _STAI_NIRDET_EVENT_NODE_START_CB(124, 1, { _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output.data->data});
  forward_ap_integer_INT8(&_AveragePool_2_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(124, 1, { _AveragePool_2_output_0_output.data->data});
}
void forward_lite_ap_integer_INT8__AveragePool_3_output_0(_stai_nirdet_context* net_ctx)
{
  _AveragePool_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228096);
  _AveragePool_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228096);
  _AveragePool_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230400);
  _AveragePool_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230400);
  _STAI_NIRDET_EVENT_NODE_START_CB(135, 1, { _AveragePool_2_output_0_output.data->data});
  forward_ap_integer_INT8(&_AveragePool_3_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(135, 1, { _AveragePool_3_output_0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA__proj_2_Conv_output_0(_stai_nirdet_context* net_ctx)
{
  _AveragePool_3_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230400);
  _AveragePool_3_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230400);
  _proj_2_Conv_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 56);
  _proj_2_Conv_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 56);
  _proj_2_Conv_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 60);
  _proj_2_Conv_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 60);
  _proj_2_Conv_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 230976);
  _proj_2_Conv_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 230976);
  _proj_2_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230992);
  _proj_2_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230992);
  _STAI_NIRDET_EVENT_NODE_START_CB(147, 1, { _AveragePool_3_output_0_output.data->data});
  forward_conv2d_integer_SSSA(&_proj_2_Conv_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(147, 1, { _proj_2_Conv_output_0_output.data->data});
}
void forward_lite_nl_integer__Sigmoid_2_output_0(_stai_nirdet_context* net_ctx)
{
  _proj_2_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230992);
  _proj_2_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230992);
  _Sigmoid_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230400);
  _Sigmoid_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230400);
  _STAI_NIRDET_EVENT_NODE_START_CB(158, 1, { _proj_2_Conv_output_0_output.data->data});
  forward_nl_integer(&_Sigmoid_2_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(158, 1, { _Sigmoid_2_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__Add_2_output_0(_stai_nirdet_context* net_ctx)
{
  _Sigmoid_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230400);
  _Sigmoid_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230400);
  _Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  _Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  _Add_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230544);
  _Add_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230544);
  _STAI_NIRDET_EVENT_NODE_START_CB(166, 2, { _Sigmoid_2_output_0_output.data->data,_Constant_output_0_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&_Add_2_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(166, 1, { _Add_2_output_0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA__proj_1_Conv_output_0(_stai_nirdet_context* net_ctx)
{
  _AveragePool_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228096);
  _AveragePool_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228096);
  _proj_2_Conv_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 56);
  _proj_2_Conv_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 56);
  _proj_2_Conv_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 60);
  _proj_2_Conv_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 60);
  _proj_1_Conv_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 230400);
  _proj_1_Conv_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 230400);
  _proj_1_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _proj_1_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _STAI_NIRDET_EVENT_NODE_START_CB(134, 1, { _AveragePool_2_output_0_output.data->data});
  forward_conv2d_integer_SSSA(&_proj_1_Conv_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(134, 1, { _proj_1_Conv_output_0_output.data->data});
}
void forward_lite_nl_integer__Sigmoid_1_output_0(_stai_nirdet_context* net_ctx)
{
  _proj_1_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _proj_1_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _Sigmoid_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228096);
  _Sigmoid_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228096);
  _STAI_NIRDET_EVENT_NODE_START_CB(146, 1, { _proj_1_Conv_output_0_output.data->data});
  forward_nl_integer(&_Sigmoid_1_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(146, 1, { _Sigmoid_1_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__Add_1_output_0(_stai_nirdet_context* net_ctx)
{
  _Sigmoid_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228096);
  _Sigmoid_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228096);
  _Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  _Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  _Add_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228672);
  _Add_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228672);
  _STAI_NIRDET_EVENT_NODE_START_CB(157, 2, { _Sigmoid_1_output_0_output.data->data,_Constant_output_0_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&_Add_1_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(157, 1, { _Add_1_output_0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA__proj_Conv_output_0(_stai_nirdet_context* net_ctx)
{
  _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _proj_2_Conv_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 56);
  _proj_2_Conv_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 56);
  _proj_2_Conv_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 60);
  _proj_2_Conv_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 60);
  _proj_Conv_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 228096);
  _proj_Conv_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 228096);
  _proj_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _proj_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _STAI_NIRDET_EVENT_NODE_START_CB(123, 1, { _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_output.data->data});
  forward_conv2d_integer_SSSA(&_proj_Conv_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(123, 1, { _proj_Conv_output_0_output.data->data});
}
void forward_lite_nl_integer__Sigmoid_output_0(_stai_nirdet_context* net_ctx)
{
  _proj_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _proj_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _Sigmoid_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _Sigmoid_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _STAI_NIRDET_EVENT_NODE_START_CB(133, 1, { _proj_Conv_output_0_output.data->data});
  forward_nl_integer(&_Sigmoid_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(133, 1, { _Sigmoid_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__Add_output_0(_stai_nirdet_context* net_ctx)
{
  _Sigmoid_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 218880);
  _Sigmoid_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 218880);
  _Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  _Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  _Add_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 221184);
  _Add_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 221184);
  _STAI_NIRDET_EVENT_NODE_START_CB(145, 2, { _Sigmoid_output_0_output.data->data,_Constant_output_0_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&_Add_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(145, 1, { _Add_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__backbone_stage1_blocks_blocks_0_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage1_cv1_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage1_cv1_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 461568);
  _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 461568);
  _backbone_stage1_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage1_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(140, 2, { _backbone_stage1_cv1_act_Clip_output_0_output.data->data,_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_backbone_stage1_blocks_blocks_0_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(140, 1, { _backbone_stage1_blocks_blocks_0_Clip_output_0_output.data->data});
}
void forward_lite_concat__backbone_stage1_Concat_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage1_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage1_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage1_cv2_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 798400);
  _backbone_stage1_cv2_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 798400);
  _backbone_stage1_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 356032);
  _backbone_stage1_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 356032);
  _STAI_NIRDET_EVENT_NODE_START_CB(152, 2, { _backbone_stage1_blocks_blocks_0_Clip_output_0_output.data->data,_backbone_stage1_cv2_act_Clip_output_0_output.data->data});
  forward_concat(&_backbone_stage1_Concat_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(152, 1, { _backbone_stage1_Concat_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__backbone_stage2_blocks_blocks_0_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage2_cv1_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _backbone_stage2_cv1_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 461664);
  _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 461664);
  _backbone_stage2_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 341280);
  _backbone_stage2_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 341280);
  _STAI_NIRDET_EVENT_NODE_START_CB(182, 2, { _backbone_stage2_cv1_act_Clip_output_0_output.data->data,_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_backbone_stage2_blocks_blocks_0_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(182, 1, { _backbone_stage2_blocks_blocks_0_Clip_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__backbone_stage2_blocks_blocks_1_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage2_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 341280);
  _backbone_stage2_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 341280);
  _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _backbone_stage2_blocks_blocks_1_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 451872);
  _backbone_stage2_blocks_blocks_1_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 451872);
  _STAI_NIRDET_EVENT_NODE_START_CB(188, 2, { _backbone_stage2_blocks_blocks_0_Clip_output_0_output.data->data,_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_backbone_stage2_blocks_blocks_1_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(188, 1, { _backbone_stage2_blocks_blocks_1_Clip_output_0_output.data->data});
}
void forward_lite_concat__backbone_stage2_Concat_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage2_blocks_blocks_1_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 451872);
  _backbone_stage2_blocks_blocks_1_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 451872);
  _backbone_stage2_cv2_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 384);
  _backbone_stage2_cv2_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 384);
  _backbone_stage2_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230688);
  _backbone_stage2_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230688);
  _STAI_NIRDET_EVENT_NODE_START_CB(191, 2, { _backbone_stage2_blocks_blocks_1_Clip_output_0_output.data->data,_backbone_stage2_cv2_act_Clip_output_0_output.data->data});
  forward_concat(&_backbone_stage2_Concat_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(191, 1, { _backbone_stage2_Concat_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__backbone_stage3_blocks_blocks_0_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage3_cv1_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 87168);
  _backbone_stage3_cv1_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87168);
  _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 114816);
  _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 114816);
  _backbone_stage3_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage3_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(215, 2, { _backbone_stage3_cv1_act_Clip_output_0_output.data->data,_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_backbone_stage3_blocks_blocks_0_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(215, 1, { _backbone_stage3_blocks_blocks_0_Clip_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__backbone_stage3_blocks_blocks_1_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage3_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage3_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 29760);
  _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 29760);
  _backbone_stage3_blocks_blocks_1_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 87168);
  _backbone_stage3_blocks_blocks_1_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87168);
  _STAI_NIRDET_EVENT_NODE_START_CB(221, 2, { _backbone_stage3_blocks_blocks_0_Clip_output_0_output.data->data,_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_backbone_stage3_blocks_blocks_1_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(221, 1, { _backbone_stage3_blocks_blocks_1_Clip_output_0_output.data->data});
}
void forward_lite_concat__backbone_stage3_Concat_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage3_blocks_blocks_1_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 87168);
  _backbone_stage3_blocks_blocks_1_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87168);
  _backbone_stage3_cv2_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 59520);
  _backbone_stage3_cv2_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 59520);
  _backbone_stage3_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage3_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(224, 2, { _backbone_stage3_blocks_blocks_1_Clip_output_0_output.data->data,_backbone_stage3_cv2_act_Clip_output_0_output.data->data});
  forward_concat(&_backbone_stage3_Concat_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(224, 1, { _backbone_stage3_Concat_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__backbone_stage4_blocks_blocks_0_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage4_cv1_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 24960);
  _backbone_stage4_cv1_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24960);
  _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _backbone_stage4_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 6912);
  _backbone_stage4_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6912);
  _STAI_NIRDET_EVENT_NODE_START_CB(248, 2, { _backbone_stage4_cv1_act_Clip_output_0_output.data->data,_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_output.data->data});
  forward_eltwise_integer_INT8(&_backbone_stage4_blocks_blocks_0_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(248, 1, { _backbone_stage4_blocks_blocks_0_Clip_output_0_output.data->data});
}
void forward_lite_concat__backbone_stage4_Concat_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage4_blocks_blocks_0_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 6912);
  _backbone_stage4_blocks_blocks_0_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6912);
  _backbone_stage4_cv2_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 18048);
  _backbone_stage4_cv2_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18048);
  _backbone_stage4_Concat_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 24960);
  _backbone_stage4_Concat_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24960);
  _STAI_NIRDET_EVENT_NODE_START_CB(251, 2, { _backbone_stage4_blocks_blocks_0_Clip_output_0_output.data->data,_backbone_stage4_cv2_act_Clip_output_0_output.data->data});
  forward_concat(&_backbone_stage4_Concat_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(251, 1, { _backbone_stage4_Concat_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__Clip_2_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage4_cv3_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 768);
  _backbone_stage4_cv3_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 768);
  _Add_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 230544);
  _Add_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 230544);
  _Clip_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 14592);
  _Clip_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14592);
  _STAI_NIRDET_EVENT_NODE_START_CB(257, 2, { _backbone_stage4_cv3_act_Clip_output_0_output.data->data,_Add_2_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_Clip_2_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(257, 1, { _Clip_2_output_0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_cls32_QuantizeLinear_Input(_stai_nirdet_context* net_ctx)
{
  _cls_branch_blocks_0_act_2_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 34432);
  _cls_branch_blocks_0_act_2_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 34432);
  cls32_QuantizeLinear_Input_weights_array.data = AI_PTR(net_ctx->_weights[0] + 535600);
  cls32_QuantizeLinear_Input_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 535600);
  cls32_QuantizeLinear_Input_bias_array.data = AI_PTR(net_ctx->_weights[0] + 535664);
  cls32_QuantizeLinear_Input_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 535664);
  cls32_QuantizeLinear_Input_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  cls32_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  cls32_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_outputs[6] + 0);
  cls32_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_outputs[6] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(285, 1, { _cls_branch_blocks_0_act_2_Clip_output_0_output.data->data});
  forward_conv2d_integer_SSSA(&cls32_QuantizeLinear_Input_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(285, 1, { cls32_QuantizeLinear_Input_output.data->data});
}
void forward_lite_transpose_size32_QuantizeLinear_Input_Transpose_8(_stai_nirdet_context* net_ctx)
{
  size32_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_activations[0] + 9728);
  size32_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9728);
  size32_QuantizeLinear_Input_Transpose_8_output_array.data = AI_PTR(net_ctx->_outputs[8] + 0);
  size32_QuantizeLinear_Input_Transpose_8_output_array.data_start = AI_PTR(net_ctx->_outputs[8] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(1, 1, { size32_QuantizeLinear_Input_output.data->data});
  forward_transpose(&size32_QuantizeLinear_Input_Transpose_8_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(1, 1, { size32_QuantizeLinear_Input_Transpose_8_output.data->data});
}
void forward_lite_transpose_off32_QuantizeLinear_Input_Transpose_7(_stai_nirdet_context* net_ctx)
{
  off32_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_activations[0] + 9728);
  off32_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9728);
  off32_QuantizeLinear_Input_Transpose_7_output_array.data = AI_PTR(net_ctx->_outputs[7] + 0);
  off32_QuantizeLinear_Input_Transpose_7_output_array.data_start = AI_PTR(net_ctx->_outputs[7] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(1, 1, { off32_QuantizeLinear_Input_output.data->data});
  forward_transpose(&off32_QuantizeLinear_Input_Transpose_7_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(1, 1, { off32_QuantizeLinear_Input_Transpose_7_output.data->data});
}
void forward_lite_upsample_nearest__neck_Resize_output_0(_stai_nirdet_context* net_ctx)
{
  _neck_lat5_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  _neck_lat5_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  _neck_Resize_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10592);
  _neck_Resize_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10592);
  _STAI_NIRDET_EVENT_NODE_START_CB(263, 1, { _neck_lat5_Conv_output_0_output.data->data});
  forward_upsample_nearest(&_neck_Resize_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(263, 1, { _neck_Resize_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__Clip_1_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage3_cv3_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 56064);
  _backbone_stage3_cv3_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 56064);
  _Add_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 228672);
  _Add_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 228672);
  _Clip_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 111360);
  _Clip_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 111360);
  _STAI_NIRDET_EVENT_NODE_START_CB(231, 2, { _backbone_stage3_cv3_act_Clip_output_0_output.data->data,_Add_1_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_Clip_1_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(231, 1, { _Clip_1_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__neck_Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _neck_lat4_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 47456);
  _neck_lat4_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 47456);
  _neck_Resize_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10592);
  _neck_Resize_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10592);
  _neck_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 84320);
  _neck_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 84320);
  _STAI_NIRDET_EVENT_NODE_START_CB(269, 2, { _neck_lat4_Conv_output_0_output.data->data,_neck_Resize_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_neck_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(269, 1, { _neck_Clip_output_0_output.data->data});
}
void forward_lite_upsample_nearest__neck_Resize_1_output_0(_stai_nirdet_context* net_ctx)
{
  _neck_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 84320);
  _neck_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 84320);
  _neck_Resize_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 223488);
  _neck_Resize_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 223488);
  _STAI_NIRDET_EVENT_NODE_START_CB(275, 1, { _neck_Clip_output_0_output.data->data});
  forward_upsample_nearest(&_neck_Resize_1_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(275, 1, { _neck_Resize_1_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__Clip_output_0(_stai_nirdet_context* net_ctx)
{
  _backbone_stage2_cv3_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 451872);
  _backbone_stage2_cv3_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 451872);
  _Add_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 221184);
  _Add_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 221184);
  _Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 673056);
  _Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 673056);
  _STAI_NIRDET_EVENT_NODE_START_CB(198, 2, { _backbone_stage2_cv3_act_Clip_output_0_output.data->data,_Add_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_Clip_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(198, 1, { _Clip_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__neck_Clip_1_output_0(_stai_nirdet_context* net_ctx)
{
  _neck_lat3_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 370944);
  _neck_lat3_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 370944);
  _neck_Resize_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 223488);
  _neck_Resize_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 223488);
  _neck_Clip_1_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 518400);
  _neck_Clip_1_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 518400);
  _STAI_NIRDET_EVENT_NODE_START_CB(284, 2, { _neck_lat3_Conv_output_0_output.data->data,_neck_Resize_1_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_neck_Clip_1_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(284, 1, { _neck_Clip_1_output_0_output.data->data});
}
void forward_lite_eltwise_integer_INT8__neck_Clip_2_output_0(_stai_nirdet_context* net_ctx)
{
  _neck_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 84320);
  _neck_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 84320);
  _neck_pan3to4_block_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10592);
  _neck_pan3to4_block_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10592);
  _neck_Clip_2_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 47456);
  _neck_Clip_2_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 47456);
  _STAI_NIRDET_EVENT_NODE_START_CB(302, 2, { _neck_Clip_output_0_output.data->data,_neck_pan3to4_block_act_Clip_output_0_output.data->data});
  forward_eltwise_integer_INT8(&_neck_Clip_2_output_0_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(302, 1, { _neck_Clip_2_output_0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_cls16_QuantizeLinear_Input(_stai_nirdet_context* net_ctx)
{
  _cls_branch_blocks_0_act_1_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 10592);
  _cls_branch_blocks_0_act_1_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10592);
  cls16_QuantizeLinear_Input_weights_array.data = AI_PTR(net_ctx->_weights[0] + 701572);
  cls16_QuantizeLinear_Input_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 701572);
  cls16_QuantizeLinear_Input_bias_array.data = AI_PTR(net_ctx->_weights[0] + 701636);
  cls16_QuantizeLinear_Input_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 701636);
  cls16_QuantizeLinear_Input_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  cls16_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  cls16_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_outputs[3] + 0);
  cls16_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_outputs[3] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(335, 1, { _cls_branch_blocks_0_act_1_Clip_output_0_output.data->data});
  forward_conv2d_integer_SSSA(&cls16_QuantizeLinear_Input_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(335, 1, { cls16_QuantizeLinear_Input_output.data->data});
}
void forward_lite_transpose_size16_QuantizeLinear_Input_Transpose_5(_stai_nirdet_context* net_ctx)
{
  size16_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_activations[0] + 976);
  size16_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 976);
  size16_QuantizeLinear_Input_Transpose_5_output_array.data = AI_PTR(net_ctx->_outputs[5] + 0);
  size16_QuantizeLinear_Input_Transpose_5_output_array.data_start = AI_PTR(net_ctx->_outputs[5] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(1, 1, { size16_QuantizeLinear_Input_output.data->data});
  forward_transpose(&size16_QuantizeLinear_Input_Transpose_5_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(1, 1, { size16_QuantizeLinear_Input_Transpose_5_output.data->data});
}
void forward_lite_transpose_off16_QuantizeLinear_Input_Transpose_4(_stai_nirdet_context* net_ctx)
{
  off16_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_activations[0] + 976);
  off16_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 976);
  off16_QuantizeLinear_Input_Transpose_4_output_array.data = AI_PTR(net_ctx->_outputs[4] + 0);
  off16_QuantizeLinear_Input_Transpose_4_output_array.data_start = AI_PTR(net_ctx->_outputs[4] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(1, 1, { off16_QuantizeLinear_Input_output.data->data});
  forward_transpose(&off16_QuantizeLinear_Input_Transpose_4_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(1, 1, { off16_QuantizeLinear_Input_Transpose_4_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_cls8_QuantizeLinear_Input(_stai_nirdet_context* net_ctx)
{
  _cls_branch_blocks_0_act_Clip_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 318560);
  _cls_branch_blocks_0_act_Clip_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 318560);
  cls8_QuantizeLinear_Input_weights_array.data = AI_PTR(net_ctx->_weights[0] + 807320);
  cls8_QuantizeLinear_Input_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 807320);
  cls8_QuantizeLinear_Input_bias_array.data = AI_PTR(net_ctx->_weights[0] + 807384);
  cls8_QuantizeLinear_Input_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 807384);
  cls8_QuantizeLinear_Input_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  cls8_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  cls8_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  cls8_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(318, 1, { _cls_branch_blocks_0_act_Clip_output_0_output.data->data});
  forward_conv2d_integer_SSSA(&cls8_QuantizeLinear_Input_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(318, 1, { cls8_QuantizeLinear_Input_output.data->data});
}
void forward_lite_transpose_size8_QuantizeLinear_Input_Transpose_2(_stai_nirdet_context* net_ctx)
{
  size8_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_activations[0] + 158048);
  size8_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 158048);
  size8_QuantizeLinear_Input_Transpose_2_output_array.data = AI_PTR(net_ctx->_outputs[2] + 0);
  size8_QuantizeLinear_Input_Transpose_2_output_array.data_start = AI_PTR(net_ctx->_outputs[2] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(1, 1, { size8_QuantizeLinear_Input_output.data->data});
  forward_transpose(&size8_QuantizeLinear_Input_Transpose_2_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(1, 1, { size8_QuantizeLinear_Input_Transpose_2_output.data->data});
}
void forward_lite_transpose_off8_QuantizeLinear_Input_Transpose_1(_stai_nirdet_context* net_ctx)
{
  off8_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_activations[0] + 158048);
  off8_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 158048);
  off8_QuantizeLinear_Input_Transpose_1_output_array.data = AI_PTR(net_ctx->_outputs[1] + 0);
  off8_QuantizeLinear_Input_Transpose_1_output_array.data_start = AI_PTR(net_ctx->_outputs[1] + 0);
  _STAI_NIRDET_EVENT_NODE_START_CB(1, 1, { off8_QuantizeLinear_Input_output.data->data});
  forward_transpose(&off8_QuantizeLinear_Input_Transpose_1_layer);
  _STAI_NIRDET_EVENT_NODE_STOP_CB(1, 1, { off8_QuantizeLinear_Input_Transpose_1_output.data->data});
}

/*****************************************************************************/


static const ai_u16 _edge_conv_Conv_output_0_t_in_0_shape_w_const_u16 = 512;
static const ai_u16 _edge_conv_Conv_output_0_t_in_0_shape_h_const_u16 = 288;
static const ai_u16 _edge_conv_Conv_output_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 _edge_conv_Conv_output_0_t_out_0_shape_ch_const_u16 = 4;
static const ai_u16 _edge_conv_Conv_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _edge_conv_Conv_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _edge_conv_Conv_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _edge_conv_Conv_output_0_l_stride_0_const_u16 = 2;
static const ai_i32 _edge_conv_Conv_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _edge_conv_Conv_output_0_l_pad_H_0_const_s32 = 1;
static const ai_i8 _edge_conv_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _edge_conv_Conv_output_0_t_out_0_fmt_zero_const_s8 = -1;
static const ai_float _edge_conv_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_float _edge_conv_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.010223041288554668f;
static const ai_float _edge_conv_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00511681055650115f, 0.005314781796187162f, 0.005147160030901432f, 0.005128808319568634f);
static const ai_layer_format_type _edge_conv_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _edge_conv_Conv_output_0_t_out_0_shape_w_const_u16 = 256;
static const ai_u16 _edge_conv_Conv_output_0_t_out_0_shape_h_const_u16 = 144;

static const ai_u32 _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 147456;
static const ai_float _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.010223041288554668f;
static const ai_i8 _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_fmt_zero_const_s8 = -1;

static const ai_i32 _Abs_output_0_t_in_0_shape_ch_h_w_prod_const_s32 = 147456;



static const ai_u32 _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 9216;
static const ai_float _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_fmt_scale_const_f32 = 0.0027351221069693565f;
static const ai_i8 _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_fmt_zero_const_s8 = -128;












static const ai_u16 _backbone_stem_act_Clip_output_0_t_in_0_shape_w_const_u16 = 512;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_in_0_shape_h_const_u16 = 288;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _backbone_stem_act_Clip_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _backbone_stem_act_Clip_output_0_l_stride_0_const_u16 = 2;
static const ai_i32 _backbone_stem_act_Clip_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _backbone_stem_act_Clip_output_0_l_pad_H_0_const_s32 = 1;
static const ai_i8 _backbone_stem_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stem_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stem_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_float _backbone_stem_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stem_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.07009059935808182f, 0.09812542051076889f, 0.08129969984292984f, 0.07973592728376389f, 0.035010915249586105f, 0.18208417296409607f, 0.017581097781658173f, 0.12781274318695068f, 0.032066427171230316f, 0.14904747903347015f, 0.14682775735855103f, 0.09950324147939682f, 0.017695629969239235f, 0.08600021153688431f, 0.1075814962387085f, 0.04863646253943443f);
static const ai_layer_format_type _backbone_stem_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_out_0_shape_w_const_u16 = 256;
static const ai_u16 _backbone_stem_act_Clip_output_0_t_out_0_shape_h_const_u16 = 144;

static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_in_0_shape_w_const_u16 = 256;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_in_0_shape_h_const_u16 = 144;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_l_stride_0_const_u16 = 2;
static const ai_i32 _backbone_widen1_block_act_Clip_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _backbone_widen1_block_act_Clip_output_0_l_pad_H_0_const_s32 = 1;
static const ai_i8 _backbone_widen1_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_widen1_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_widen1_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_widen1_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_widen1_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002297396305948496f, 0.0021180170588195324f, 0.0020629349164664745f, 0.0013959544012323022f, 0.0017461309907957911f, 0.0022023669444024563f, 0.00132941419724375f, 0.0019184394041076303f, 0.0015716827474534512f, 0.0020452518947422504f, 0.0025453732814639807f, 0.0019067447865381837f, 0.0029409693088382483f, 0.00231772824190557f, 0.0022819540463387966f, 0.0015451285289600492f, 0.0011689300881698728f, 0.0025352146476507187f, 0.0019313909579068422f, 0.0016539448406547308f, 0.002413066104054451f, 0.002526348689571023f, 0.001962596783414483f, 0.0021714328322559595f, 0.0028128589037805796f, 0.002176472684368491f, 0.0013819385785609484f, 0.0026118061505258083f, 0.0015573151176795363f, 0.002194356871768832f, 0.0022292304784059525f, 0.0015090208034962416f, 0.0021297158673405647f, 0.001932709594257176f, 0.002313098404556513f, 0.002644370077177882f, 0.0018860059790313244f, 0.0022842271719127893f, 0.0018772982293739915f, 0.0018140074098482728f, 0.00215101707726717f, 0.0015912491362541914f, 0.0030470467172563076f, 0.0016010480467230082f, 0.002716687275096774f, 0.002696183742955327f, 0.002006873255595565f, 0.0022211538162082434f);
static const ai_layer_format_type _backbone_widen1_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_out_0_shape_w_const_u16 = 128;
static const ai_u16 _backbone_widen1_block_act_Clip_output_0_t_out_0_shape_h_const_u16 = 72;

static const ai_u16 _backbone_stage1_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16 = 128;
static const ai_u16 _backbone_stage1_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16 = 72;
static const ai_u16 _backbone_stage1_cv2_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage1_cv2_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage1_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage1_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _backbone_stage1_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage1_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage1_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.003678420791402459f, 0.004433027934283018f, 0.0036546734627336264f, 0.004431112669408321f, 0.00374915124848485f, 0.004128127358853817f, 0.00431860564276576f, 0.005121853668242693f, 0.004073516931384802f, 0.003979184664785862f, 0.003583580255508423f, 0.0032766261138021946f, 0.0030983781907707453f, 0.0045870100148022175f, 0.004082147963345051f, 0.005156051367521286f, 0.004680533893406391f, 0.0019232301274314523f, 0.0020911293104290962f, 0.004564030095934868f, 0.0028761387802660465f, 0.003992917016148567f, 0.004002747125923634f, 0.003283656667917967f);
static const ai_layer_format_type _backbone_stage1_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _backbone_stage1_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16 = 128;
static const ai_u16 _backbone_stage1_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16 = 72;
static const ai_u16 _backbone_stage1_cv1_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage1_cv1_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage1_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage1_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _backbone_stage1_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage1_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage1_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002975974464789033f, 0.00305876019410789f, 0.003279756987467408f, 0.0035812363494187593f, 0.003262060461565852f, 0.0025823514442890882f, 0.0028978607151657343f, 0.0030819650273770094f, 0.003514984855428338f, 0.0035185031592845917f, 0.0028808533679693937f, 0.0035546990111470222f, 0.003052774118259549f, 0.0026957341469824314f, 0.0037356633692979813f, 0.003318645292893052f, 0.004756628070026636f, 0.0035172072239220142f, 0.002507949247956276f, 0.0019257462117820978f, 0.003011387772858143f, 0.002300660125911236f, 0.00355217675678432f, 0.004386461805552244f);
static const ai_layer_format_type _backbone_stage1_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 72;

static const ai_u16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u16 = 130;
static const ai_u16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u16 = 74;
static const ai_u16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0020459634251892567f, 0.002281493740156293f, 0.002181149087846279f, 0.0025408000219613314f, 0.0024733389727771282f, 0.0025732149370014668f, 0.0030380564276129007f, 0.0017533208010718226f, 0.0021786161232739687f, 0.0014729393878951669f, 0.001984878908842802f, 0.002332524163648486f, 0.002170429565012455f, 0.0023743961937725544f, 0.0031050085090100765f, 0.0023029004223644733f, 0.002018798841163516f, 0.002289424417540431f, 0.0019767330959439278f, 0.0019368521170690656f, 0.002151859225705266f, 0.0020875767804682255f, 0.0033881054259836674f, 0.0024354399647563696f);
static const ai_layer_format_type _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u16 = 128;
static const ai_u16 _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u16 = 72;



static const ai_u16 _backbone_stage1_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16 = 128;
static const ai_u16 _backbone_stage1_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16 = 72;
static const ai_u16 _backbone_stage1_cv3_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage1_cv3_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage1_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage1_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage1_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage1_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage1_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage1_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0027008657343685627f, 0.0027572419494390488f, 0.0018628282705321908f, 0.0025871586985886097f, 0.002259208122268319f, 0.0036186748184263706f, 0.0026154571678489447f, 0.0036725527606904507f, 0.0026320533361285925f, 0.0024267069529742002f, 0.003188655013218522f, 0.003055748762562871f, 0.0023269890807569027f, 0.002403971506282687f, 0.002333472715690732f, 0.0023324997164309025f, 0.003120502457022667f, 0.003478148952126503f, 0.0028499080799520016f, 0.003313433611765504f, 0.0024228175170719624f, 0.002193603664636612f, 0.002165014622732997f, 0.0028865148779004812f, 0.003583659650757909f, 0.004187657963484526f, 0.002232028404250741f, 0.0028471502009779215f, 0.0027034871745854616f, 0.0035427927505224943f, 0.0029094533529132605f, 0.002018848666921258f, 0.003029843559488654f, 0.0024387165904045105f, 0.003074284642934799f, 0.002875687787309289f, 0.0019343687454238534f, 0.0037925008218735456f, 0.002456974470987916f, 0.0024604140780866146f, 0.004070284776389599f, 0.002418793737888336f, 0.00284645427018404f, 0.0033949040807783604f, 0.003350470680743456f, 0.002435336820781231f, 0.0019482317147776484f, 0.002351050265133381f);
static const ai_layer_format_type _backbone_stage1_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_down2_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 72;

static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_in_0_shape_w_const_u16 = 130;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_in_0_shape_h_const_u16 = 74;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_l_stride_0_const_u16 = 2;
static const ai_i8 _backbone_down2_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_down2_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_down2_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_down2_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_down2_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001132879639044404f, 0.0011779641499742866f, 0.0005602611927315593f, 0.0009335811482742429f, 0.0013777371495962143f, 0.0010425884975120425f, 0.0010461017955094576f, 0.0010412975680083036f, 0.001036915578879416f, 0.0009373876382596791f, 0.00110002257861197f, 0.0010049144038930535f, 0.0012800367549061775f, 0.001344701973721385f, 0.0011908848537132144f, 0.0009394519729539752f, 0.0010695677483454347f, 0.0010322873713448644f, 0.0013933677691966295f, 0.0010690928902477026f, 0.0011466385331004858f, 0.001184404594823718f, 0.0010221051052212715f, 0.001147752394899726f, 0.0010580482194200158f, 0.0009710435406304896f, 0.0009920238517224789f, 0.0007997758220881224f, 0.0009472848614677787f, 0.0008914510835893452f, 0.0012409050250425935f, 0.0013358539436012506f, 0.0012412909418344498f, 0.0009970437968149781f, 0.0011899726232513785f, 0.0011681908508762717f, 0.0011414500186219811f, 0.0013685020385310054f, 0.001188631053082645f, 0.0012081293389201164f, 0.0009161568596027792f, 0.00124836852774024f, 0.0010191467590630054f, 0.0010016707237809896f, 0.0008421939564868808f, 0.0008602477028034627f, 0.0010793461697176099f, 0.0012172652641311288f, 0.0006546378135681152f, 0.0009977787267416716f, 0.0011129732010886073f, 0.0009373693610541523f, 0.000960672041401267f, 0.0009346826118417084f, 0.0009222376393154263f, 0.0012163394130766392f, 0.001009016647003591f, 0.0010292677907273173f, 0.0010662280255928636f, 0.000817885622382164f, 0.0011347236577421427f, 0.000997080933302641f, 0.0010735810501500964f, 0.0010685727465897799f, 0.0009575904696248472f, 0.001379321445710957f, 0.001005774480290711f, 0.0012462357990443707f, 0.0011953632347285748f, 0.0010657744714990258f, 0.0006064550252631307f, 0.0011387019185349345f, 0.0009224242530763149f, 0.0007516897167079151f, 0.0010304632596671581f, 0.0011705263750627637f, 0.0009232126758433878f, 0.0011328599648550153f, 0.0012034062528982759f, 0.0009328542510047555f, 0.0009557849261909723f, 0.0015710656298324466f, 0.0011746158124879003f, 0.0009344092686660588f, 0.0012177187018096447f, 0.0008509759791195393f, 0.0014742797939106822f, 0.000960835546720773f, 0.0010244989534839988f, 0.0008122024592012167f, 0.0015299617080017924f, 0.0010147043503820896f, 0.001057153451256454f, 0.001256834832020104f, 0.0013141912641003728f, 0.000991859007626772f);
static const ai_layer_format_type _backbone_down2_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 _backbone_down2_block_act_Clip_output_0_t_out_0_shape_h_const_u16 = 36;

static const ai_u16 _backbone_stage2_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 _backbone_stage2_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 _backbone_stage2_cv2_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage2_cv2_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage2_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage2_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage2_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage2_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage2_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017946980660781264f, 0.001712235389277339f, 0.001355291111394763f, 0.0014652324607595801f, 0.002148842206224799f, 0.0017820027424022555f, 0.0021531216334551573f, 0.0021678286138921976f, 0.002265492221340537f, 0.0018030756618827581f, 0.002978979144245386f, 0.002749878680333495f, 0.001886092941276729f, 0.001875416375696659f, 0.002101754304021597f, 0.0017729535466060042f, 0.002315550111234188f, 0.0025266441516578197f, 0.002092755865305662f, 0.0021010925993323326f, 0.0016615798231214285f, 0.0015399146359413862f, 0.0025646365247666836f, 0.0028626781422644854f, 0.0022961737122386694f, 0.0014801608631387353f, 0.0018732991302385926f, 0.0013138351496309042f, 0.002724527381360531f, 0.0023433093447238207f, 0.002131157089024782f, 0.002438530558720231f, 0.001757232821546495f, 0.0020228701177984476f, 0.0022121223155409098f, 0.0027433750219643116f, 0.0018239867640659213f, 0.002292839577421546f, 0.002470947103574872f, 0.003053831635043025f, 0.003061585361137986f, 0.003401912283152342f, 0.002585839247331023f, 0.0015347322914749384f, 0.001958673121407628f, 0.0018910487415269017f, 0.001585808233357966f, 0.0021820450201630592f);
static const ai_layer_format_type _backbone_stage2_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _backbone_stage2_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 _backbone_stage2_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 _backbone_stage2_cv1_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage2_cv1_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage2_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage2_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage2_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage2_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage2_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001411399687640369f, 0.001755557139404118f, 0.002345251850783825f, 0.0018355629872530699f, 0.0019281621789559722f, 0.0017418306088075042f, 0.001808058936148882f, 0.0027815490029752254f, 0.0015992032131180167f, 0.0020133068319410086f, 0.0027210877742618322f, 0.0027938492130488157f, 0.002156171714887023f, 0.0024237865582108498f, 0.002915917430073023f, 0.0019190098391845822f, 0.0024366576690226793f, 0.0020552126225084066f, 0.0016697307582944632f, 0.0022610377054661512f, 0.0018314758781343699f, 0.0029227393679320812f, 0.002029913477599621f, 0.002457997528836131f, 0.0017132599605247378f, 0.0022451048716902733f, 0.002278445288538933f, 0.0018789648311212659f, 0.0018178600585088134f, 0.0025441013276576996f, 0.002032747259363532f, 0.0016002809861674905f, 0.0018727886490523815f, 0.0021679734345525503f, 0.0018760486273095012f, 0.0017460057279095054f, 0.0018977917497977614f, 0.001964062685146928f, 0.0018919974099844694f, 0.0025284092407673597f, 0.0023320496547967196f, 0.0020800551865249872f, 0.0018463757587596774f, 0.0020036858040839434f, 0.001921811024658382f, 0.001930973376147449f, 0.0015746605349704623f, 0.0031767755281180143f);
static const ai_layer_format_type _backbone_stage2_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001512480666860938f, 0.0017232486279681325f, 0.0020543457940220833f, 0.0017328489338979125f, 0.0014880483504384756f, 0.001607190235517919f, 0.002017037244513631f, 0.0015313192270696163f, 0.0018310261657461524f, 0.0012740325182676315f, 0.001598253264091909f, 0.0011753634316846728f, 0.0017151284264400601f, 0.0016605645650997758f, 0.00180761085357517f, 0.0023019148502498865f, 0.0029003845993429422f, 0.001579002127982676f, 0.001925217336975038f, 0.001956599997356534f, 0.001741721061989665f, 0.0019321198342368007f, 0.0018651974387466908f, 0.00301619921810925f, 0.0017825292889028788f, 0.001796568976715207f, 0.0019756206311285496f, 0.0017109352629631758f, 0.0015243332600221038f, 0.0023161638528108597f, 0.0018379073590040207f, 0.0021528161596506834f, 0.001949648605659604f, 0.0017348281107842922f, 0.0015290240989997983f, 0.002211038488894701f, 0.001849676831625402f, 0.0020306319929659367f, 0.002315982012078166f, 0.0015352488262578845f, 0.0018303890246897936f, 0.0019734581001102924f, 0.001456395722925663f, 0.0016041682101786137f, 0.0017823473317548633f, 0.0015329860616475344f, 0.0017828266136348248f, 0.0023645844776183367f);
static const ai_layer_format_type _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u16 = 36;


static const ai_i8 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0014408213319256902f, 0.001465833862312138f, 0.0011231936514377594f, 0.0015011493815109134f, 0.0013139731017872691f, 0.001210319227539003f, 0.0013752138474956155f, 0.0013090751599520445f, 0.00163797487039119f, 0.0012310027377679944f, 0.0013728541089221835f, 0.0014015918131917715f, 0.0016917074099183083f, 0.0012894842075183988f, 0.0011809600982815027f, 0.0014854328474029899f, 0.001440619002096355f, 0.0021656122989952564f, 0.0013553854078054428f, 0.0015070912195369601f, 0.001139543135650456f, 0.0013614469207823277f, 0.001629760954529047f, 0.0010808565421029925f, 0.0014860308729112148f, 0.001408504555001855f, 0.0012695350451394916f, 0.0012856746325269341f, 0.0014912326587364078f, 0.0013227324234321713f, 0.0016220126999542117f, 0.0015506392810493708f, 0.0014526805607602f, 0.0012268350692465901f, 0.0011755141895264387f, 0.0013887396780773997f, 0.0016583306714892387f, 0.001603945274837315f, 0.0011456324718892574f, 0.0017432193271815777f, 0.0014988139737397432f, 0.0011417061323300004f, 0.0011262441985309124f, 0.0015564147615805268f, 0.0015919519355520606f, 0.001597046386450529f, 0.001497136428952217f, 0.0013758440036326647f);
static const ai_layer_format_type _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_h_const_u16 = 36;



static const ai_u16 _backbone_stage2_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 _backbone_stage2_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 _backbone_stage2_cv3_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage2_cv3_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage2_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage2_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 _backbone_stage2_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage2_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage2_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage2_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002366031287238002f, 0.0020026175770908594f, 0.0016683937283232808f, 0.0019896922167390585f, 0.0018409064505249262f, 0.0018638811307027936f, 0.0019823922775685787f, 0.0016705936286598444f, 0.0016917793545871973f, 0.0019868058152496815f, 0.0015589836984872818f, 0.001756315934471786f, 0.0018208390101790428f, 0.0018494767136871815f, 0.0022916963789612055f, 0.001941029098816216f, 0.0016417044680565596f, 0.0019661509431898594f, 0.002172390464693308f, 0.0015698341885581613f, 0.0022251910995692015f, 0.001683726441115141f, 0.002009936608374119f, 0.0019492132123559713f, 0.00200995453633368f, 0.0025091874413192272f, 0.0018470864742994308f, 0.0010465129744261503f, 0.0023500677198171616f, 0.0023150767665356398f, 0.0023953889030963182f, 0.001787408720701933f, 0.001648510922677815f, 0.0017967696767300367f, 0.0013253050856292248f, 0.002088682260364294f, 0.0018214028095826507f, 0.0017646937631070614f, 0.002513315062969923f, 0.0016867972444742918f, 0.002507022814825177f, 0.0013622291153296828f, 0.002444514771923423f, 0.001959378132596612f, 0.001405233284458518f, 0.002235557185485959f, 0.00228513334877789f, 0.0017135604284703732f, 0.0023254856932908297f, 0.0021925857290625572f, 0.0018050775397568941f, 0.0013647223822772503f, 0.0018840981647372246f, 0.0017548061441630125f, 0.0020302990451455116f, 0.0019425649661570787f, 0.0016426023794338107f, 0.0024976273998618126f, 0.0017160577699542046f, 0.002488416153937578f, 0.0015765819698572159f, 0.002009000163525343f, 0.0018188051180914044f, 0.0022765640169382095f, 0.0017205608310177922f, 0.001294497400522232f, 0.0022349171340465546f, 0.001984341535717249f, 0.0016912705032154918f, 0.002116372110322118f, 0.002055741148069501f, 0.0022809335496276617f, 0.001748507609590888f, 0.002116360468789935f, 0.001652433886192739f, 0.0021466794423758984f, 0.0018888235790655017f, 0.0017349064582958817f, 0.0016856653383001685f, 0.0017254132544621825f, 0.002759804716333747f, 0.0017969828331843019f, 0.002206944627687335f, 0.0016877948073670268f, 0.0022214914206415415f, 0.0020497736986726522f, 0.0019857666920870543f, 0.002480307826772332f, 0.0017173285596072674f, 0.002488000551238656f, 0.001632228377275169f, 0.00238412874750793f, 0.00215190090239048f, 0.0024664979428052902f, 0.0018112348625436425f, 0.0015689346473664045f);
static const ai_layer_format_type _backbone_stage2_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_down3_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_l_stride_0_const_u16 = 2;
static const ai_i8 _backbone_down3_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_down3_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_down3_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_down3_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_down3_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0008663308108225465f, 0.0008011845056898892f, 0.0012255340116098523f, 0.0008696780423633754f, 0.0008513018256053329f, 0.0008147878106683493f, 0.0008385454420931637f, 0.0007210362236946821f, 0.0007146471762098372f, 0.0007278122357092798f, 0.0008003118564374745f, 0.0010143104009330273f, 0.0010844741482287645f, 0.000985554768703878f, 0.00060560047859326f, 0.000712266773916781f, 0.0008553038351237774f, 0.000659495301079005f, 0.0008261603070423007f, 0.0007015205337665975f, 0.0008845921838656068f, 0.0010122992098331451f, 0.0006797088426537812f, 0.0009745375718921423f, 0.0007691739592701197f, 0.0009289328590966761f, 0.001035451889038086f, 0.0007712725782766938f, 0.0009836470708251f, 0.0007252852083183825f, 0.0008090836927294731f, 0.001113229081965983f, 0.0005788159905932844f, 0.0007495256722904742f, 0.0006100806058384478f, 0.000774754153098911f, 0.0006888443022035062f, 0.0006726349820382893f, 0.0009507565991953015f, 0.000698035815730691f, 0.0008417126955464482f, 0.0005632202373817563f, 0.001110855140723288f, 0.0007318492862395942f, 0.0006752163171768188f, 0.0008678065496496856f, 0.0008685602224431932f, 0.0008182507008314133f, 0.0007970654405653477f, 0.0007860638434067369f, 0.0008288710960187018f, 0.0007058626506477594f, 0.0004660797130782157f, 0.0010284740710631013f, 0.0009594780276529491f, 0.0008873181068338454f, 0.000854377809446305f, 0.0007824981585144997f, 0.0005952388746663928f, 0.0010060856584459543f, 0.0006959087331779301f, 0.0007848063833080232f, 0.000792436592746526f, 0.0006283029797486961f, 0.00071921810740605f, 0.00071243557613343f, 0.0008674963610246778f, 0.0010636603692546487f, 0.0010749595239758492f, 0.0008707462693564594f, 0.0005757070612162352f, 0.0011340044438838959f, 0.0007740171859040856f, 0.000734375324100256f, 0.000773423642385751f, 0.0006918881554156542f, 0.0009141055052168667f, 0.00053668167674914f, 0.000738785311114043f, 0.0007719268905930221f, 0.0011864749249070883f, 0.0009375722729600966f, 0.0007891666027717292f, 0.0007657554815523326f, 0.0008777264156378806f, 0.0008668110822327435f, 0.0007189750322140753f, 0.0007480423664674163f, 0.0008412289898842573f, 0.0008111673523671925f, 0.0009918083669617772f, 0.0007788494694977999f, 0.0008057334343902767f, 0.0006743736448697746f, 0.0007908482803031802f, 0.0009535659337416291f);
static const ai_layer_format_type _backbone_down3_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _backbone_down3_block_act_Clip_output_0_t_out_0_shape_h_const_u16 = 18;

static const ai_u16 _backbone_stage3_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 _backbone_stage3_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 _backbone_stage3_cv2_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage3_cv2_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage3_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage3_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage3_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage3_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage3_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0016969459829851985f, 0.001990955090150237f, 0.0016191587783396244f, 0.001946542994119227f, 0.0018285432597622275f, 0.0021895933896303177f, 0.002013077028095722f, 0.0019247506279498339f, 0.002195118460804224f, 0.0020918960217386484f, 0.0021353508345782757f, 0.002418849617242813f, 0.002097044140100479f, 0.0019351138034835458f, 0.00254415743984282f, 0.002021298510953784f, 0.0019266081508249044f, 0.0017086619045585394f, 0.0026610514614731073f, 0.0019246680894866586f, 0.0020086884032934904f, 0.0019891210831701756f, 0.002002335386350751f, 0.0019100114004686475f, 0.0016538869822397828f, 0.0013178118970245123f, 0.002185434103012085f, 0.002620065351948142f, 0.0016693962970748544f, 0.0017488112207502127f, 0.001937083201482892f, 0.002048402326181531f, 0.0016095369355753064f, 0.002137211849913001f, 0.002534726168960333f, 0.0014432243769988418f, 0.0018513565883040428f, 0.0018716660561040044f, 0.0021866140887141228f, 0.0021641929633915424f, 0.0025077741593122482f, 0.0021365026477724314f, 0.0019229025347158313f, 0.0016272051725536585f, 0.0019296088721603155f, 0.0019684976432472467f, 0.001954201376065612f, 0.0019055905286222696f);
static const ai_layer_format_type _backbone_stage3_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _backbone_stage3_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 _backbone_stage3_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 _backbone_stage3_cv1_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage3_cv1_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage3_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage3_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage3_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage3_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage3_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0016676076920703053f, 0.0015669481363147497f, 0.001724878093227744f, 0.001668263808824122f, 0.0015364692080765963f, 0.0022470217663794756f, 0.0015248997369781137f, 0.0024429266341030598f, 0.0017743498319759965f, 0.001876319875009358f, 0.0017984281294047832f, 0.0020218854770064354f, 0.002138484036549926f, 0.002030273200944066f, 0.00159460527356714f, 0.001651101978495717f, 0.0020958282984793186f, 0.0019004314672201872f, 0.002195345237851143f, 0.0017725784564390779f, 0.0017747259698808193f, 0.002347552217543125f, 0.0023462199606001377f, 0.0021109820809215307f, 0.002235815394669771f, 0.0022382556926459074f, 0.0020425471011549234f, 0.0028963543009012938f, 0.001862233504652977f, 0.0022702738642692566f, 0.0013888212852180004f, 0.0018764871638268232f, 0.002449131803587079f, 0.001724069588817656f, 0.0018020027782768011f, 0.0018767290748655796f, 0.002517418470233679f, 0.0017209667712450027f, 0.002461391966789961f, 0.0017027189023792744f, 0.002155846217647195f, 0.0018441814463585615f, 0.0016557051567360759f, 0.001711804186925292f, 0.0018547128420323133f, 0.001517004333436489f, 0.002090592635795474f, 0.001723942463286221f);
static const ai_layer_format_type _backbone_stage3_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 18;

static const ai_u16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u16 = 20;
static const ai_u16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001679674838669598f, 0.0019158722134307027f, 0.0012045112671330571f, 0.0018625467782840133f, 0.001667710137553513f, 0.0018347096629440784f, 0.002100942190736532f, 0.0016275387024506927f, 0.00172419473528862f, 0.0016981650842353702f, 0.0014843677636235952f, 0.001867470215074718f, 0.0018716852646321058f, 0.0014970267657190561f, 0.0018471862422302365f, 0.002393942791968584f, 0.003149670083075762f, 0.00199052132666111f, 0.0013230039039626718f, 0.0012435264652594924f, 0.0016667685704305768f, 0.0024998532608151436f, 0.00312106404453516f, 0.0014184085885062814f, 0.0017600927967578173f, 0.001850535860285163f, 0.0014659727457910776f, 0.0018300089286640286f, 0.00206142570823431f, 0.0021449727937579155f, 0.001925329677760601f, 0.0017444919794797897f, 0.002202423056587577f, 0.0016670959303155541f, 0.001418162602931261f, 0.0017527142772451043f, 0.0019012039992958307f, 0.002018977655097842f, 0.0015823068097233772f, 0.0015779093373566866f, 0.0014065252617001534f, 0.0017900115344673395f, 0.001802270533517003f, 0.001837947522290051f, 0.0019003389170393348f, 0.0018154983408749104f, 0.0017227418720722198f, 0.0018008329207077622f);
static const ai_layer_format_type _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u16 = 18;


static const ai_i8 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 18;

static const ai_u16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_h_const_u16 = 20;
static const ai_u16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0013463327195495367f, 0.001645378302782774f, 0.0011563816806301475f, 0.0013339357683435082f, 0.001143132452853024f, 0.0015780014218762517f, 0.0012360685504972935f, 0.001375748310238123f, 0.0018209784757345915f, 0.0013017237652093172f, 0.0013819073792546988f, 0.0009757027146406472f, 0.0007439380860887468f, 0.001099437940865755f, 0.00126167805865407f, 0.0013380059972405434f, 0.001605581259354949f, 0.0008641935419291258f, 0.001098644221201539f, 0.0012670911382883787f, 0.0012630400015041232f, 0.0009035594994202256f, 0.0013364608166739345f, 0.0011879721423611045f, 0.0010409951210021973f, 0.0014133196091279387f, 0.0020727235823869705f, 0.001519388286396861f, 0.0009858390549197793f, 0.0010153709445148706f, 0.0011426251148805022f, 0.0012522275792434812f, 0.0012375012738630176f, 0.0017699790187180042f, 0.0018688213312998414f, 0.0011120978742837906f, 0.001110614975914359f, 0.0016071428544819355f, 0.0015421438729390502f, 0.0020835071336477995f, 0.001085874391719699f, 0.0016108491690829396f, 0.001201335689984262f, 0.0018268137937411666f, 0.0010327177587896585f, 0.0012118135346099734f, 0.0011500156251713634f, 0.0012078293366357684f);
static const ai_layer_format_type _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_h_const_u16 = 18;



static const ai_u16 _backbone_stage3_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 _backbone_stage3_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 _backbone_stage3_cv3_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage3_cv3_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage3_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage3_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 _backbone_stage3_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage3_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage3_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage3_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001996132545173168f, 0.002426744205877185f, 0.001850718050263822f, 0.0014681161846965551f, 0.0013697819085791707f, 0.0017993991496041417f, 0.0016444202046841383f, 0.0014211667003110051f, 0.0020624094177037477f, 0.0019480175105854869f, 0.0017003261018544436f, 0.0014354671584442258f, 0.0019661271944642067f, 0.0016448625829070807f, 0.001918024499900639f, 0.0018858155235648155f, 0.001459721359424293f, 0.002169307554140687f, 0.0019934021402150393f, 0.0017648398643359542f, 0.002485771430656314f, 0.0018086811760440469f, 0.002027468755841255f, 0.0019590880256146193f, 0.0016119309002533555f, 0.00231693172827363f, 0.0022731758654117584f, 0.0017842687666416168f, 0.0027020389679819345f, 0.0018496881239116192f, 0.0017405555117875338f, 0.001966457348316908f, 0.001583407400175929f, 0.0017641236772760749f, 0.001956693362444639f, 0.002101774327456951f, 0.0018909976352006197f, 0.0016883438220247626f, 0.0021709627471864223f, 0.0018359549576416612f, 0.001603499287739396f, 0.0019339813152328134f, 0.001644554198719561f, 0.0024318413343280554f, 0.0018465806497260928f, 0.0021808345336467028f, 0.0022282053250819445f, 0.0020679247099906206f, 0.0014832771848887205f, 0.0019027342787012458f, 0.0015894427197054029f, 0.0018951756646856666f, 0.002487354911863804f, 0.0021008788608014584f, 0.0016956442268565297f, 0.0021163637284189463f, 0.0018485013861209154f, 0.0014163341838866472f, 0.0024489983916282654f, 0.0011152300285175443f, 0.0011873681796714664f, 0.0021652074065059423f, 0.0017891527386382222f, 0.0020154020749032497f, 0.0014562727883458138f, 0.002266886178404093f, 0.001706529874354601f, 0.0016908750403672457f, 0.0018788084853440523f, 0.0019291223725304008f, 0.0015945672057569027f, 0.002143355319276452f, 0.0014345156960189342f, 0.0018508350476622581f, 0.0016372737009078264f, 0.0017095516668632627f, 0.0015273966128006577f, 0.0019465392688289285f, 0.0015642635989934206f, 0.0020768933463841677f, 0.001599390059709549f, 0.0021146093495190144f, 0.0017123724101111293f, 0.0016170209273695946f, 0.0015642911894246936f, 0.0016729526687413454f, 0.0016843228368088603f, 0.001717121573165059f, 0.0014335063751786947f, 0.0014542336575686932f, 0.0018117619911208749f, 0.002019030274823308f, 0.001536293188109994f, 0.002171282423660159f, 0.0019112478476017714f, 0.0014214059337973595f);
static const ai_layer_format_type _backbone_stage3_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_down4_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 18;

static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_in_0_shape_h_const_u16 = 20;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_l_stride_0_const_u16 = 2;
static const ai_i8 _backbone_down4_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_down4_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_down4_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_down4_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_down4_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0006474071415141225f, 0.0009783171117305756f, 0.0007269857451319695f, 0.0006846085889264941f, 0.0007032311405055225f, 0.0006071997922845185f, 0.0005581147270277143f, 0.0006121443584561348f, 0.0006481190794147551f, 0.0007612394401803613f, 0.0005459309322759509f, 0.0005503392894752324f, 0.0006165856611914933f, 0.0005146634648554027f, 0.0006221784860827029f, 0.0006466510822065175f, 0.0006880321307107806f, 0.0008189246873371303f, 0.0005887984298169613f, 0.000887459609657526f, 0.0004904422676190734f, 0.0006038894061930478f, 0.0006825775490142405f, 0.0006604403024539351f, 0.0007172053446993232f, 0.0007101934170350432f, 0.0008302204660139978f, 0.0007430881960317492f, 0.0005402769893407822f, 0.0008114934316836298f, 0.0005674296407960355f, 0.0007910453714430332f, 0.0006000820430926979f, 0.0005701679619960487f, 0.0005707431701011956f, 0.0007043853984214365f, 0.0006839209818281233f, 0.0006622797809541225f, 0.0008001045789569616f, 0.0005210863309912384f, 0.0005750839482061565f, 0.0006843117298558354f, 0.0005779004422947764f, 0.00079884979641065f, 0.0006748138112016022f, 0.0006217353511601686f, 0.0005387694691307843f, 0.0007315176189877093f, 0.0006143944920040667f, 0.0007883398211561143f, 0.0005562695441767573f, 0.000590972660575062f, 0.0006518361624330282f, 0.0007075624889694154f, 0.0005283611244522035f, 0.0007970812148414552f, 0.0005669851088896394f, 0.0007875431329011917f, 0.0006322184926830232f, 0.0007265493040904403f, 0.0008569550118409097f, 0.00055200036149472f, 0.0005996872787363827f, 0.0008746295934543014f, 0.0007032002904452384f, 0.0008119064732454717f, 0.0006081004976294935f, 0.0007641335250809789f, 0.000693035835865885f, 0.000790005549788475f, 0.0007665145676583052f, 0.0005258649471215904f, 0.0006267133285291493f, 0.0007886734092608094f, 0.0006275775958783925f, 0.0007398041780106723f, 0.0007941152434796095f, 0.0007782587199471891f, 0.0007210877374745905f, 0.0006861091242171824f, 0.0006922273896634579f, 0.0006452961242757738f, 0.0008952438947744668f, 0.000937853823415935f, 0.000536021136213094f, 0.0007178000523708761f, 0.0005594966933131218f, 0.0005382858798839152f, 0.0006165937520563602f, 0.0006623591179959476f, 0.0007750344229862094f, 0.0006243476527743042f, 0.0006539199966937304f, 0.000835220969747752f, 0.0006821764400228858f, 0.0008465432329103351f);
static const ai_layer_format_type _backbone_down4_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 _backbone_down4_block_act_Clip_output_0_t_out_0_shape_h_const_u16 = 9;

static const ai_u16 _backbone_stage4_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 _backbone_stage4_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 _backbone_stage4_cv2_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage4_cv2_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage4_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage4_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage4_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage4_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage4_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage4_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.019807321950793266f;
static const ai_float _backbone_stage4_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0022196441423147917f, 0.0015445046592503786f, 0.001692076213657856f, 0.0022168965078890324f, 0.001702174893580377f, 0.001664279610849917f, 0.0019915462471544743f, 0.0022890777327120304f, 0.0019641374237835407f, 0.001976004568859935f, 0.0018576858565211296f, 0.002380837919190526f, 0.0015665122773498297f, 0.002237582579255104f, 0.0015955439303070307f, 0.0018850569613277912f, 0.001849277294240892f, 0.0027739647775888443f, 0.001879165880382061f, 0.0021525658667087555f, 0.001960444264113903f, 0.0015122952172532678f, 0.0024694567546248436f, 0.001507459906861186f, 0.0017870329320430756f, 0.002301661530509591f, 0.0016784420004114509f, 0.0026471547316759825f, 0.002015853300690651f, 0.002432913752272725f, 0.002849576761946082f, 0.0017043737461790442f, 0.0015222408110275865f, 0.001975943800061941f, 0.001763241714797914f, 0.001857679570093751f, 0.0023023071698844433f, 0.002079433063045144f, 0.0018989121308550239f, 0.0019520903006196022f, 0.0016053167637437582f, 0.0019709693733602762f, 0.0017846818082034588f, 0.002187095582485199f, 0.00155284209176898f, 0.001467852620407939f, 0.0021410470362752676f, 0.0026328631211072206f);
static const ai_layer_format_type _backbone_stage4_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 _backbone_stage4_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 _backbone_stage4_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 _backbone_stage4_cv1_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage4_cv1_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage4_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage4_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _backbone_stage4_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage4_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage4_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage4_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage4_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017874863697215915f, 0.0016858945600688457f, 0.002189611317589879f, 0.0020110346376895905f, 0.0018190175760537386f, 0.0020320250187069178f, 0.0020255455747246742f, 0.001976817147806287f, 0.0016735345125198364f, 0.001374811865389347f, 0.002178549999371171f, 0.001987254014238715f, 0.0016917729517444968f, 0.002236191648989916f, 0.0018164472421631217f, 0.0017008886206895113f, 0.0016313452506437898f, 0.0015684070531278849f, 0.0022452138364315033f, 0.0014898298541083932f, 0.001636187662370503f, 0.0020104164723306894f, 0.0015352369518950582f, 0.0017841266235336661f, 0.0019644801504909992f, 0.002094321884214878f, 0.001440808642655611f, 0.002136963652446866f, 0.0013792309910058975f, 0.0016566019039601088f, 0.0016599539667367935f, 0.001942358328960836f, 0.0020059864036738873f, 0.002080516191199422f, 0.002251699101179838f, 0.0021625873632729053f, 0.0014370274730026722f, 0.0018160982290282845f, 0.002498169196769595f, 0.0017024505650624633f, 0.001531266956590116f, 0.0015406078891828656f, 0.0021464370656758547f, 0.0018387471791356802f, 0.002483666641637683f, 0.0013954754685983062f, 0.0017781212227419019f, 0.0015249857679009438f);
static const ai_layer_format_type _backbone_stage4_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 9;

static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12480;
static const ai_float _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_i8 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u32 = 48;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u32 = 48;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u32 = 13;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u32 = 16;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u32 = 9;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_pad_W_0_const_s32 = 0;
static const ai_i32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_pad_H_0_const_s32 = 0;
static const ai_u16 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_dilation_W_const_u16 = 2;
static const ai_u16 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_dilation_H_const_u16 = 2;
static const ai_size _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_v_n_groups_const_size = 1;

static const ai_u32 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 6912;
static const ai_float _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_i8 _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_fmt_zero_const_s8 = -128;



static const ai_u16 _backbone_stage4_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 _backbone_stage4_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 _backbone_stage4_cv3_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _backbone_stage4_cv3_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _backbone_stage4_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _backbone_stage4_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 _backbone_stage4_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _backbone_stage4_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _backbone_stage4_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage4_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _backbone_stage4_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002585822716355324f, 0.001644138596020639f, 0.0016226472798734903f, 0.002388928784057498f, 0.002301202155649662f, 0.0014789203414693475f, 0.0020550501067191362f, 0.0019395553972572088f, 0.0022292127832770348f, 0.0018292960012331605f, 0.002145356498658657f, 0.002472331514582038f, 0.002502135932445526f, 0.0018808585591614246f, 0.0024835357908159494f, 0.0017875314224511385f, 0.0024615933652967215f, 0.0014867635909467936f, 0.002855411497876048f, 0.002026679925620556f, 0.0023332624696195126f, 0.0021610185503959656f, 0.0018219088669866323f, 0.0015668646665289998f, 0.0025372100062668324f, 0.001988464966416359f, 0.0018125994829460979f, 0.0019605462439358234f, 0.0025145483668893576f, 0.0017091897316277027f, 0.0026296370197087526f, 0.0018977966392412782f, 0.001715439255349338f, 0.001836760900914669f, 0.002562031615525484f, 0.0017500168178230524f, 0.00321861426346004f, 0.002039756393060088f, 0.0027238228358328342f, 0.0016813570400699973f, 0.001787395915016532f, 0.0023755659349262714f, 0.0016923303483054042f, 0.0017594428500160575f, 0.002376463497057557f, 0.0020144011359661818f, 0.0021532271057367325f, 0.0020886866841465235f, 0.0021619643084704876f, 0.001830337569117546f, 0.0016548543935641646f, 0.0023940219543874264f, 0.0022406650241464376f, 0.0022367690689861774f, 0.0017723652999848127f, 0.0023910982999950647f, 0.002258502645418048f, 0.0022482911590486765f, 0.0015437738038599491f, 0.0023778583854436874f, 0.0027024855371564627f, 0.002433049725368619f, 0.0020140698179602623f, 0.002042303327471018f, 0.002110208384692669f, 0.00345251290127635f, 0.002001690212637186f, 0.0018451903015375137f, 0.002083582105115056f, 0.0019951120484620333f, 0.0014852279564365745f, 0.00255428534001112f, 0.0019193589687347412f, 0.0017289527459070086f, 0.0019904065411537886f, 0.0019911895506083965f, 0.001960994675755501f, 0.0030093775130808353f, 0.0021991683170199394f, 0.0016603022813796997f, 0.0017176162218675017f, 0.0025983895175158978f, 0.0021418409887701273f, 0.0019903217907994986f, 0.0017464959528297186f, 0.002388341585174203f, 0.0016415870049968362f, 0.0013457343447953463f, 0.0015835448866710067f, 0.002096177078783512f, 0.0021220988128334284f, 0.0026129186153411865f, 0.0018868475453928113f, 0.0020226070191711187f, 0.0022744459565728903f, 0.001888867816887796f);
static const ai_layer_format_type _backbone_stage4_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 _neck_lat5_Conv_output_0_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 _neck_lat5_Conv_output_0_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 _neck_lat5_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _neck_lat5_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _neck_lat5_Conv_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _neck_lat5_Conv_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _neck_lat5_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _neck_lat5_Conv_output_0_t_out_0_fmt_zero_const_s8 = 6;
static const ai_float _neck_lat5_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_lat5_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.08638046681880951f;
static const ai_float _neck_lat5_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0020285777281969786f, 0.0023932408075779676f, 0.0021231365390121937f, 0.0021691573783755302f, 0.0018066421616822481f, 0.002165839308872819f, 0.002121138619259f, 0.0036486622411757708f, 0.0025305890012532473f, 0.0021947654895484447f, 0.00260941032320261f, 0.002090520691126585f, 0.002215699991211295f, 0.003403704147785902f, 0.0025535125751048326f, 0.002383179496973753f, 0.0024850505869835615f, 0.0018706072587519884f, 0.002873258199542761f, 0.0022865966893732548f, 0.003194444114342332f, 0.0013913890579715371f, 0.0025060134939849377f, 0.0021430396009236574f, 0.001802454236894846f, 0.00248661614023149f, 0.0031674441415816545f, 0.0020606634207069874f, 0.0026626340113580227f, 0.0016601351089775562f, 0.0022842518519610167f, 0.002050562761723995f, 0.0017823700327426195f, 0.0028473646380007267f, 0.0018253177404403687f, 0.0024940846487879753f, 0.003536299802362919f, 0.002250278601422906f, 0.002204055432230234f, 0.0019641490653157234f, 0.0037947020027786493f, 0.002908427268266678f, 0.002633993746712804f, 0.003268023720011115f, 0.002193568041548133f, 0.002724885242059827f, 0.0025171260349452496f, 0.002920277649536729f, 0.0034076219890266657f, 0.002606367226690054f, 0.001978903776034713f, 0.0028442286420613527f, 0.0023073304910212755f, 0.0017801164649426937f, 0.0031715095974504948f, 0.00228873617015779f, 0.002131966408342123f, 0.0021037638653069735f, 0.002384545747190714f, 0.002827464835718274f, 0.002777891233563423f, 0.0032698328141123056f, 0.0033183149062097073f, 0.0037223107647150755f);
static const ai_layer_format_type _neck_lat5_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _neck_out5_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(6);
static const ai_i16 _neck_out5_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _neck_out5_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 9;

static const ai_u16 _neck_out5_act_Clip_output_0_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 _neck_out5_act_Clip_output_0_t_in_0_shape_h_const_u16 = 11;
static const ai_u16 _neck_out5_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _neck_out5_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _neck_out5_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = 6;
static const ai_i8 _neck_out5_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _neck_out5_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.08638046681880951f;
static const ai_float _neck_out5_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_out5_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0005615652771666646f, 0.0004366486391518265f, 0.0005839349469169974f, 0.00037772086216136813f, 0.0006243826355785131f, 0.0002993803645949811f, 0.0005969238118268549f, 0.00029996063676662743f, 0.0005135133396834135f, 0.0006558398599736392f, 0.0003332050982862711f, 0.0003180192725267261f, 0.0005101170390844345f, 0.00026351166889071465f, 0.0004855134757235646f, 0.0005977281834930182f, 0.0003804373263847083f, 0.0006588496617041528f, 0.00047234553494490683f, 0.0004809499077964574f, 0.0003095372812822461f, 0.00040960140177048743f, 0.0007113111205399036f, 0.00041383481584489346f, 0.0003929286322090775f, 0.0005518855177797377f, 0.0005467942683026195f, 0.0006123390630818903f, 0.0003237687051296234f, 0.0006160584744066f, 0.0012801268603652716f, 0.0005887820152565837f, 0.0006502026226371527f, 0.0005440485547296703f, 0.0005133954109624028f, 0.0003998852916993201f, 0.0008598394924774766f, 0.0005965200834907591f, 0.0005444523994810879f, 0.000351595226675272f, 0.00036235348670743406f, 0.00044988468289375305f, 0.0004203548887744546f, 0.0004929266287945211f, 0.0005587278283201158f, 0.0004105033876840025f, 0.0002735575253609568f, 0.0007516908808611333f, 0.0004510282597038895f, 0.0006223349482752383f, 0.0004774271510541439f, 0.0005327222170308232f, 0.0003150172997266054f, 0.0005213554250076413f, 0.0006446462357416749f, 0.00034774435334838927f, 0.0003717046638485044f, 0.0007706974283792078f, 0.0006294531049206853f, 0.0006817011744715273f, 0.00036648192326538265f, 0.00039116141851991415f, 0.000514306069817394f, 0.0005360589129850268f);
static const ai_layer_format_type _neck_out5_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _neck_out5_act_Clip_output_0_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 _neck_out5_act_Clip_output_0_t_out_0_shape_h_const_u16 = 9;

static const ai_u16 _stems_2_act_Clip_output_0_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 _stems_2_act_Clip_output_0_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 _stems_2_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _stems_2_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _stems_2_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _stems_2_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _stems_2_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _stems_2_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _stems_2_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _stems_2_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _stems_2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.003124010283499956f, 0.002416218165308237f, 0.0018666351679712534f, 0.0025293000508099794f, 0.002166138030588627f, 0.00188719411380589f, 0.003010037587955594f, 0.0020253993570804596f, 0.00200972449965775f, 0.0020837869960814714f, 0.0016345459735020995f, 0.0016593161271885037f, 0.00176352565176785f, 0.0019693258218467236f, 0.0015290126902982593f, 0.0019375882111489773f, 0.001657671295106411f, 0.003380672773346305f, 0.0014643481699749827f, 0.0028843071777373552f, 0.0034057132434099913f, 0.002050690818578005f, 0.00223130127415061f, 0.00248368619941175f, 0.0020255474373698235f, 0.0016470809932798147f, 0.0016973739257082343f, 0.0017565188463777304f, 0.001562983845360577f, 0.0014053438790142536f, 0.0033827652223408222f, 0.002506469376385212f, 0.0018308797152712941f, 0.0019562123343348503f, 0.00217398046515882f, 0.0014809166314080358f, 0.0021974241826683283f, 0.001873900881037116f, 0.0022981339134275913f, 0.0023738769814372063f, 0.002345458837226033f, 0.0022180608939379454f, 0.0017015938647091389f, 0.002285622525960207f, 0.0018876469694077969f, 0.002622842090204358f, 0.0018526689382269979f, 0.0024075889959931374f, 0.0028977508191019297f, 0.0018681453075259924f, 0.002550567965954542f, 0.002202583709731698f, 0.001953938277438283f, 0.0021400440018624067f, 0.002375044859945774f, 0.001627280144020915f, 0.0022936395835131407f, 0.0014930767938494682f, 0.0022884986829012632f, 0.0015388779575005174f, 0.0017420889344066381f, 0.0023308752570301294f, 0.0023628247436136007f, 0.0024982821196317673f);
static const ai_layer_format_type _stems_2_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 9;

static const ai_u16 _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_h_const_u16 = 11;
static const ai_u16 _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.020054398104548454f;
static const ai_float _cls_branch_blocks_0_act_2_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007299873395822942f, 0.001480312435887754f, 0.0016225016443058848f, 0.0008302798378281295f, 0.0006655911565758288f, 0.0010303322924301028f, 0.0007750816293992102f, 0.0015458567067980766f, 0.0019151433371007442f, 0.000928387395106256f, 0.0007533625466749072f, 0.0006533655105158687f, 0.0007295698160305619f, 0.00243418593890965f, 0.000985011924058199f, 0.0009472515084780753f, 0.00028123814263381064f, 0.0015055686235427856f, 0.0007834479911252856f, 0.000888346170540899f, 0.000843132147565484f, 0.0007984155672602355f, 0.0005487180897034705f, 0.001152811455540359f, 0.0009412224753759801f, 0.0014997529797255993f, 0.0008529644110240042f, 0.0009233607561327517f, 0.0009818486869335175f, 0.0009685280965641141f, 0.0006385194719769061f, 0.0008024152484722435f, 0.001136220176704228f, 0.0009984661592170596f, 0.001748797483742237f, 0.0008703849744051695f, 0.000661068013869226f, 0.0008707179804332554f, 0.00030498739215545356f, 0.001052050618454814f, 0.0006171558052301407f, 0.0006750821485184133f, 0.0008851780439727008f, 0.0004783830663654953f, 0.0008176799165084958f, 0.0010832472471520305f, 0.0011806567199528217f, 0.0008477528463117778f, 0.0008618335123173892f, 0.0006653855089098215f, 0.0010218966053798795f, 0.0013674399815499783f, 0.0011462949914857745f, 0.0015945294871926308f, 0.000950582092627883f, 0.0006372599746100605f, 0.0007076019537635148f, 0.0009072477114386857f, 0.0008368404232896864f, 0.0006725957500748336f, 0.00048426887951791286f, 0.0009766105795279145f, 0.000863716471940279f, 0.0008298319880850613f);
static const ai_layer_format_type _cls_branch_blocks_0_act_2_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_h_const_u16 = 9;


static const ai_i8 _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 9;

static const ai_u16 _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_h_const_u16 = 11;
static const ai_u16 _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.015236550010740757f;
static const ai_float _reg_branch_blocks_0_act_2_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0011388330021873116f, 0.0009248494170606136f, 0.0012842108262702823f, 0.0009093553526327014f, 0.0013888947432860732f, 0.0007593228365294635f, 0.0008554809610359371f, 0.002609389368444681f, 0.00169592525344342f, 0.001509650843217969f, 0.0009928712388500571f, 0.001135832630097866f, 0.0013879359466955066f, 0.0010006909724324942f, 0.0009995161090046167f, 0.0011988289188593626f, 0.0010540471412241459f, 0.0006796632660552859f, 0.0013532544253394008f, 0.0008424762054346502f, 0.0012920746812596917f, 0.0014866419369354844f, 0.0020438607316464186f, 0.0024921970907598734f, 0.0007729919743724167f, 0.0009375459048897028f, 0.0023394424933940172f, 0.0019081411883234978f, 0.0014200753066688776f, 0.0008195077534765005f, 0.0007445140508934855f, 0.00522446446120739f, 0.0013276945101097226f, 0.0006866511539556086f, 0.0007412057602778077f, 0.0004927624249830842f, 0.0009522880427539349f, 0.0009486612398177385f, 0.0007596739451400936f, 0.0007072736625559628f, 0.0011685184435918927f, 0.000645577791146934f, 0.0009683776297606528f, 0.0020060206297785044f, 0.0017014889745041728f, 0.0008875878411345184f, 0.0008860898669809103f, 0.0019667483866214752f, 0.002503051422536373f, 0.0009139751200564206f, 0.001538081793114543f, 0.0014684776542708278f, 0.0009491646196693182f, 0.0009079938172362745f, 0.002138453535735607f, 0.0021059492137283087f, 0.0012182536302134395f, 0.0017842453671619296f, 0.0005675714346580207f, 0.0010531750740483403f, 0.0004949239082634449f, 0.0010357138235121965f, 0.0016388946678489447f, 0.0010600571986287832f);
static const ai_layer_format_type _reg_branch_blocks_0_act_2_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_h_const_u16 = 9;

static const ai_u16 size32_QuantizeLinear_Input_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 size32_QuantizeLinear_Input_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 size32_QuantizeLinear_Input_l_stride_1_const_u16 = 1;
static const ai_u16 size32_QuantizeLinear_Input_l_stride_0_const_u16 = 1;
static const ai_u16 size32_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 size32_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 2;
static const ai_i8 size32_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 size32_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = 127;
static const ai_float size32_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.015236550010740757f;
static const ai_float size32_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.015480712987482548f;
static const ai_float size32_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001027219812385738f, 0.0011550035560503602f);
static const ai_layer_format_type size32_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 off32_QuantizeLinear_Input_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 off32_QuantizeLinear_Input_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 off32_QuantizeLinear_Input_l_stride_1_const_u16 = 1;
static const ai_u16 off32_QuantizeLinear_Input_l_stride_0_const_u16 = 1;
static const ai_u16 off32_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 off32_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 2;
static const ai_i8 off32_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 off32_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = 43;
static const ai_float off32_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.015236550010740757f;
static const ai_float off32_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.03284941241145134f;
static const ai_float off32_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0011731471167877316f, 0.0038625739980489016f);
static const ai_layer_format_type off32_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 _neck_lat4_Conv_output_0_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 _neck_lat4_Conv_output_0_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 _neck_lat4_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _neck_lat4_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _neck_lat4_Conv_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _neck_lat4_Conv_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _neck_lat4_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _neck_lat4_Conv_output_0_t_out_0_fmt_zero_const_s8 = 18;
static const ai_float _neck_lat4_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_lat4_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.09602516144514084f;
static const ai_float _neck_lat4_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00215055700391531f, 0.0023249986115843058f, 0.002440560609102249f, 0.0015432294458150864f, 0.0017378587508574128f, 0.001934099243953824f, 0.0017296939622610807f, 0.0020464311819523573f, 0.0026991444174200296f, 0.0016208235174417496f, 0.002350209280848503f, 0.0024483862798660994f, 0.001680696732364595f, 0.003024876117706299f, 0.0022301250137388706f, 0.0022030926775187254f, 0.00242694397456944f, 0.0036311300937086344f, 0.0018116000574082136f, 0.0016655047656968236f, 0.0027478316333144903f, 0.0022589010186493397f, 0.0020456372294574976f, 0.0018766779685392976f, 0.001773685566149652f, 0.0031032427214086056f, 0.0022711711935698986f, 0.0023238989524543285f, 0.0023682231549173594f, 0.002043353859335184f, 0.0017542173154652119f, 0.002370334230363369f, 0.0022603494580835104f, 0.002408555941656232f, 0.00271520740352571f, 0.0026051350869238377f, 0.0016041211783885956f, 0.00201077270321548f, 0.002095012692734599f, 0.00208075437694788f, 0.002598684513941407f, 0.0024616417940706015f, 0.0014159982092678547f, 0.0024108942598104477f, 0.0021428996697068214f, 0.002482272684574127f, 0.0029057017527520657f, 0.0025448398664593697f, 0.001751260249875486f, 0.0023517594672739506f, 0.0025322693400084972f, 0.0025649413000792265f, 0.0022471367847174406f, 0.0020519851241260767f, 0.0026663532480597496f, 0.002439441392198205f, 0.0023470318410545588f, 0.002441035583615303f, 0.002194320084527135f, 0.0018412359058856964f, 0.0021283701062202454f, 0.0026709281373769045f, 0.0029142487328499556f, 0.002321928273886442f);
static const ai_layer_format_type _neck_lat4_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 _neck_lat3_Conv_output_0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 _neck_lat3_Conv_output_0_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 _neck_lat3_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _neck_lat3_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _neck_lat3_Conv_output_0_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 _neck_lat3_Conv_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _neck_lat3_Conv_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _neck_lat3_Conv_output_0_t_out_0_fmt_zero_const_s8 = 0;
static const ai_float _neck_lat3_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_lat3_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.119174525141716f;
static const ai_float _neck_lat3_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0029078968800604343f, 0.0027329851873219013f, 0.003694086568430066f, 0.0027935842517763376f, 0.0038116606883704662f, 0.003546714084222913f, 0.0034666545689105988f, 0.0017180810682475567f, 0.00232465541921556f, 0.004043678753077984f, 0.0024126574862748384f, 0.003224150277674198f, 0.0043326131999492645f, 0.002139856806024909f, 0.0032797730527818203f, 0.0027002450078725815f, 0.00353010231629014f, 0.004953418858349323f, 0.003726054448634386f, 0.002457505324855447f, 0.0024571616668254137f, 0.002367093926295638f, 0.00518635381013155f, 0.002307043643668294f, 0.0038711971137672663f, 0.0027016974054276943f, 0.003627289552241564f, 0.0025208620354533195f, 0.004357799421995878f, 0.0027889241464436054f, 0.0019869699608534575f, 0.002559283282607794f, 0.00418423255905509f, 0.003598815528675914f, 0.0029691653326153755f, 0.0015484822215512395f, 0.002239691326394677f, 0.0029545037541538477f, 0.004392258357256651f, 0.002465670695528388f, 0.002604819368571043f, 0.0026726527139544487f, 0.0035215187817811966f, 0.0024060236755758524f, 0.0024484973400831223f, 0.0018473101081326604f, 0.0023184192832559347f, 0.0035060483496636152f, 0.0029113595373928547f, 0.0024610343389213085f, 0.0021291838493198156f, 0.002746849786490202f, 0.0033929089549928904f, 0.0031537276227027178f, 0.0024828885216265917f, 0.003818694967776537f, 0.002736512804403901f, 0.003946819808334112f, 0.003789521288126707f, 0.002625489141792059f, 0.0035674446262419224f, 0.0037700317334383726f, 0.0023927970323711634f, 0.00427155289798975f);
static const ai_layer_format_type _neck_lat3_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_i8 _neck_pan3to4_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(0);
static const ai_i16 _neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_l_stride_1_const_u16 = 2;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_l_stride_0_const_u16 = 2;
static const ai_i8 _neck_pan3to4_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 _neck_pan3to4_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _neck_pan3to4_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0470588244497776f;
static const ai_float _neck_pan3to4_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_pan3to4_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00043539877515286207f, 0.0006934655830264091f, 0.0004366450011730194f, 0.0005229712696745992f, 0.00048067583702504635f, 0.0005353874294087291f, 0.0005957484827376902f, 0.0005654619308188558f, 0.0005337020847946405f, 0.0005852838512510061f, 0.0005182385793887079f, 0.00044013233855366707f, 0.0004956388147547841f, 0.0005672734696418047f, 0.0004123995313420892f, 0.0006438544951379299f, 0.0004953527241013944f, 0.0005385871627368033f, 0.0004127838183194399f, 0.00048077566316351295f, 0.00046094975550659f, 0.0004956206539645791f, 0.00036282214568927884f, 0.00030277043697424233f, 0.0005770957213826478f, 0.0004956369521096349f, 0.0004709114145953208f, 0.0004572070902213454f, 0.0005707309464924037f, 0.0005269077955745161f, 0.0004762723983731121f, 0.0004150599706918001f, 0.0006645904504694045f, 0.0005699790199287236f, 0.0005311027052812278f, 0.0007795221172273159f, 0.0005072119529359043f, 0.0006161538185551763f, 0.0005630314699374139f, 0.0005856037023477256f, 0.0005328045808710158f, 0.0005276512238197029f, 0.0006676781340502203f, 0.00045170518569648266f, 0.0006000539287924767f, 0.0004964491818100214f, 0.0006245295517146587f, 0.00041419966146349907f, 0.00032296005520038307f, 0.00044387768139131367f, 0.0004394433635752648f, 0.0005997665575705469f, 0.0005683684139512479f, 0.0005662484909407794f, 0.0005560790887102485f, 0.000550700759049505f, 0.0006216146284714341f, 0.0004035667225252837f, 0.00037251849425956607f, 0.0004058894992340356f, 0.0005950659979134798f, 0.0005126265459693968f, 0.0005585078033618629f, 0.0006228979327715933f);
static const ai_layer_format_type _neck_pan3to4_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _neck_pan3to4_block_act_Clip_output_0_t_out_0_shape_h_const_u16 = 18;


static const ai_i8 _neck_out4_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(0);
static const ai_i16 _neck_out4_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _neck_out4_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 18;

static const ai_u16 _neck_out4_act_Clip_output_0_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 _neck_out4_act_Clip_output_0_t_in_0_shape_h_const_u16 = 20;
static const ai_u16 _neck_out4_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _neck_out4_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _neck_out4_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 _neck_out4_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _neck_out4_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0470588244497776f;
static const ai_float _neck_out4_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_out4_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00029526144498959184f, 0.0005093600484542549f, 0.00028358574490994215f, 0.0002534314407967031f, 0.0003042714961338788f, 0.00031604579999111593f, 0.0006786260055378079f, 0.00022763949527870864f, 0.0003900699957739562f, 0.0003195014433003962f, 0.0003381441056262702f, 0.0004137365613132715f, 0.0002765742829069495f, 0.0005296922172419727f, 0.00035984281566925347f, 0.00037257507210597396f, 0.0003508813970256597f, 0.0003857649862766266f, 0.0003207097470294684f, 0.0003413868835195899f, 0.00045997221604920924f, 0.00031015812419354916f, 0.0003162386710755527f, 0.00045697615132667124f, 0.00034119581687264144f, 0.00046861174632795155f, 0.000503687362652272f, 0.0003507292130962014f, 0.0002876141224987805f, 0.0003828531189355999f, 0.00025527263642288744f, 0.0002500943373888731f, 0.0002752477885223925f, 0.0003765946312341839f, 0.0004302204179111868f, 0.0002649827511049807f, 0.0005948690231889486f, 0.0003195247845724225f, 0.0003486190689727664f, 0.00025129460846073925f, 0.00037398867425508797f, 0.000652656948659569f, 0.00023257461725734174f, 0.00040219409856945276f, 0.00040772155625745654f, 0.00045947087346576154f, 0.0002594764227978885f, 0.0005504926666617393f, 0.0005515820812433958f, 0.0005002595135010779f, 0.0007652759086340666f, 0.00033420545514672995f, 0.0004901775391772389f, 0.00040364888263866305f, 0.0007082435768097639f, 0.00028915892471559346f, 0.0003260942758060992f, 0.00040114912553690374f, 0.0006248855497688055f, 0.0003828739281743765f, 0.00037081216578371823f, 0.00041592094930820167f, 0.0002812469901982695f, 0.0003963436756748706f);
static const ai_layer_format_type _neck_out4_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _neck_out4_act_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _neck_out4_act_Clip_output_0_t_out_0_shape_h_const_u16 = 18;

static const ai_u16 _stems_1_act_Clip_output_0_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 _stems_1_act_Clip_output_0_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 _stems_1_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _stems_1_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _stems_1_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _stems_1_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _stems_1_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _stems_1_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _stems_1_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _stems_1_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _stems_1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.004660178907215595f, 0.0022027981467545033f, 0.003547517117112875f, 0.003505303990095854f, 0.0022622470278292894f, 0.0022975332103669643f, 0.0022778373677283525f, 0.0020493087358772755f, 0.0030418250244110823f, 0.0026260227896273136f, 0.002607791218906641f, 0.0019567403942346573f, 0.0018654207233339548f, 0.0020481746178120375f, 0.0021477246191352606f, 0.0023211203515529633f, 0.0016430116957053542f, 0.0021780263632535934f, 0.001260161167010665f, 0.0029796368908137083f, 0.0022646626457571983f, 0.0017140328418463469f, 0.002281894674524665f, 0.002286698669195175f, 0.0036330244038254023f, 0.0021208792459219694f, 0.0018085940973833203f, 0.003673659637570381f, 0.004054578486829996f, 0.002019743202254176f, 0.001773571944795549f, 0.002159178489819169f, 0.004132908768951893f, 0.0019993740133941174f, 0.0019474143628031015f, 0.0018633282743394375f, 0.00637091277167201f, 0.003378382185474038f, 0.00161231798119843f, 0.002316331723704934f, 0.0015094102127477527f, 0.004857353866100311f, 0.00229805544950068f, 0.00197649491019547f, 0.00174294738098979f, 0.0019027512753382325f, 0.0016810140805318952f, 0.003236994845792651f, 0.002704648533836007f, 0.0033047902397811413f, 0.002916914876550436f, 0.0014309878461062908f, 0.006836783140897751f, 0.00262751174159348f, 0.004960446618497372f, 0.0020763541106134653f, 0.0013689774787053466f, 0.0030520230066031218f, 0.0040900371968746185f, 0.003217858262360096f, 0.0023489396553486586f, 0.0019784201867878437f, 0.0021083024330437183f, 0.0012845223536714911f);
static const ai_layer_format_type _stems_1_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 18;

static const ai_u16 _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_h_const_u16 = 20;
static const ai_u16 _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _cls_branch_blocks_0_act_1_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00037660336238332093f, 0.002897990634664893f, 0.002776344306766987f, 0.0003578120667953044f, 0.00036107978667132556f, 0.0020214116666465998f, 0.0003361016570124775f, 0.0020248456858098507f, 0.001851619454100728f, 0.00035538265365175903f, 0.0003047022328246385f, 0.00035811314592137933f, 0.00032063384423963726f, 0.0032925342675298452f, 0.0004092180170118809f, 0.0016407237853854895f, 0.0011207395000383258f, 0.0014254148118197918f, 0.0021007717587053776f, 0.001331464620307088f, 0.00033887894824147224f, 0.000331216404447332f, 0.00035147866583429277f, 0.002194270258769393f, 0.0017359070479869843f, 0.0008304077782668173f, 0.00035225358442403376f, 0.0014643807662650943f, 0.0015517950523644686f, 0.0004720380820799619f, 0.00037155215977691114f, 0.00035467828274704516f, 0.001700966153293848f, 0.0006478652358055115f, 0.002490923972800374f, 0.00040016634739004076f, 0.0003573810390662402f, 0.0003781043051276356f, 0.0015074469847604632f, 0.0004259576671756804f, 0.0015108046354725957f, 0.0003444727335590869f, 0.0019720098935067654f, 0.00028861413011327386f, 0.00048574793618172407f, 0.0015266777481883764f, 0.0018292699242010713f, 0.0015850497875362635f, 0.0013572807656601071f, 0.0005182193126529455f, 0.0011807320406660438f, 0.0017044979613274336f, 0.0012334290659055114f, 0.0026278244331479073f, 0.000659684999845922f, 0.0003262718382757157f, 0.0003493183758109808f, 0.002401124918833375f, 0.0003759792889468372f, 0.0008219140581786633f, 0.00035042251693084836f, 0.0004425666411407292f, 0.00040620233630761504f, 0.0003218495985493064f);
static const ai_layer_format_type _cls_branch_blocks_0_act_1_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_h_const_u16 = 18;


static const ai_i8 _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 18;

static const ai_u16 _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_h_const_u16 = 20;
static const ai_u16 _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _reg_branch_blocks_0_act_1_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017935862997546792f, 0.002223706804215908f, 0.0008834358886815608f, 0.0015861920546740294f, 0.0005520891863852739f, 0.0008343454683199525f, 0.0005306805833242834f, 0.0016423005145043135f, 0.0030405796132981777f, 0.0017437017522752285f, 0.0016736503457650542f, 0.0004912573494948447f, 0.0009707476128824055f, 0.0007455893792212009f, 0.0002916739322245121f, 0.0006504837074317038f, 0.0006230666185729206f, 0.000446554011432454f, 0.0004810809623450041f, 0.0006257587228901684f, 0.001150980475358665f, 0.0031125508248806f, 0.00093957589706406f, 0.002369997324422002f, 0.00037063268246129155f, 0.0008983333827927709f, 0.0008361684158444405f, 0.001462419517338276f, 0.0003684939001686871f, 0.0010308022610843182f, 0.0004088277055416256f, 0.0027577579021453857f, 0.001371642341837287f, 0.0003947220684494823f, 0.00044198735849931836f, 0.0004099975631106645f, 0.0014041927643120289f, 0.0010416225995868444f, 0.0005110934725962579f, 0.00028458519955165684f, 0.0014064883580431342f, 0.00043153794831596315f, 0.0010012651327997446f, 0.001055683707818389f, 0.0007236043456941843f, 0.0004821878101211041f, 0.0007305741310119629f, 0.001252001617103815f, 0.0012472261441871524f, 0.0005996408872306347f, 0.001010024338029325f, 0.0007586096180602908f, 0.0007531711016781628f, 0.0004595421487465501f, 0.0006144161452539265f, 0.0011900271056219935f, 0.0007264988962560892f, 0.0008435015333816409f, 0.0005865206476300955f, 0.0005280898185446858f, 0.0003796801611315459f, 0.00041021264041773975f, 0.000852645724080503f, 0.0006342330016195774f);
static const ai_layer_format_type _reg_branch_blocks_0_act_1_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_h_const_u16 = 18;

static const ai_u16 size16_QuantizeLinear_Input_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 size16_QuantizeLinear_Input_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 size16_QuantizeLinear_Input_l_stride_1_const_u16 = 1;
static const ai_u16 size16_QuantizeLinear_Input_l_stride_0_const_u16 = 1;
static const ai_u16 size16_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 size16_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 2;
static const ai_i8 size16_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 size16_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = 124;
static const ai_float size16_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float size16_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.016024518758058548f;
static const ai_float size16_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.000998588395304978f, 0.001585989142768085f);
static const ai_layer_format_type size16_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 off16_QuantizeLinear_Input_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 off16_QuantizeLinear_Input_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 off16_QuantizeLinear_Input_l_stride_1_const_u16 = 1;
static const ai_u16 off16_QuantizeLinear_Input_l_stride_0_const_u16 = 1;
static const ai_u16 off16_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 off16_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 2;
static const ai_i8 off16_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 off16_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = -4;
static const ai_float off16_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float off16_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.14666779339313507f;
static const ai_float off16_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006975782103836536f, 0.016988616436719894f);
static const ai_layer_format_type off16_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_i8 _neck_out3_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(0);
static const ai_i16 _neck_out3_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _neck_out3_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _neck_out3_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _neck_out3_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _neck_out3_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _neck_out3_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 _neck_out3_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 _neck_out3_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _neck_out3_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0470588244497776f;
static const ai_float _neck_out3_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _neck_out3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00022608744620811194f, 0.0002569179341662675f, 0.0004172174958512187f, 0.00043289121822454035f, 0.0004210432234685868f, 0.00030038977274671197f, 0.0003634492459241301f, 0.0004391009861137718f, 0.00017685318016447127f, 0.0005680712056346238f, 0.00039039901457726955f, 0.0003200920473318547f, 0.00024641017080284655f, 0.0002666745276656002f, 0.00046024256153032184f, 0.0003170072450302541f, 0.00031276268418878317f, 0.0002048747701337561f, 0.00030773336766287684f, 0.00029465940315276384f, 0.000314230564981699f, 0.00024074500834103674f, 0.00037334635271690786f, 0.0002502010902389884f, 0.00038028223207220435f, 0.0002750545972958207f, 0.0006340522086247802f, 0.00027327146381139755f, 0.0003440291911829263f, 0.0003637912159319967f, 0.00035670478246174753f, 0.0003330835606902838f, 0.00032086801365949214f, 0.0003785229055210948f, 0.0005717474268749356f, 0.0002868149022106081f, 0.00025047166855074465f, 0.0001657813845667988f, 0.0003410848439671099f, 0.00026510650059208274f, 0.0002626328496262431f, 0.0003485567285679281f, 0.0004445604281499982f, 0.00020927742298226804f, 0.00024632213171571493f, 0.0002822279930114746f, 0.00030878977850079536f, 0.0004833568527828902f);
static const ai_layer_format_type _neck_out3_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _neck_out3_act_Clip_output_0_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 _neck_out3_act_Clip_output_0_t_out_0_shape_h_const_u16 = 36;

static const ai_u16 _stems_0_act_Clip_output_0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 _stems_0_act_Clip_output_0_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 _stems_0_act_Clip_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _stems_0_act_Clip_output_0_l_stride_0_const_u16 = 1;
static const ai_u16 _stems_0_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 _stems_0_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _stems_0_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _stems_0_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _stems_0_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _stems_0_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _stems_0_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0028276133816689253f, 0.0039664036594331264f, 0.0020673952531069517f, 0.003207125701010227f, 0.004970111418515444f, 0.002795042470097542f, 0.0033963534515351057f, 0.0037777952384203672f, 0.003263025777414441f, 0.004319970030337572f, 0.003602894488722086f, 0.005034829024225473f, 0.0033108345232903957f, 0.00248773954808712f, 0.003065753960981965f, 0.0033932700753211975f, 0.004766982514411211f, 0.003907926380634308f, 0.002186185447499156f, 0.004458499141037464f, 0.004498048219829798f, 0.0020609498023986816f, 0.003815095406025648f, 0.00452228169888258f, 0.003296626964583993f, 0.003949227277189493f, 0.003439802210777998f, 0.00470368005335331f, 0.0030417696107178926f, 0.004623197950422764f, 0.003907347563654184f, 0.004638389218598604f, 0.00300018722191453f, 0.0022850336972624063f, 0.004483855329453945f, 0.002421822864562273f, 0.003948187921196222f, 0.0039020124822854996f, 0.0052274977788329124f, 0.0038364706560969353f, 0.010644444264471531f, 0.0030382415279746056f, 0.002168934792280197f, 0.002121324185281992f, 0.0034083209466189146f, 0.003927927929908037f, 0.005902540870010853f, 0.0036257179453969f, 0.003449423238635063f, 0.003468085778877139f, 0.0038580058608204126f, 0.0031395999249070883f, 0.003469320246949792f, 0.0031652827747166157f, 0.004293017089366913f, 0.0038031504955142736f, 0.004054275341331959f, 0.005059742834419012f, 0.0026528022717684507f, 0.0033502434380352497f, 0.0023403670638799667f, 0.0036829959135502577f, 0.004025666508823633f, 0.003429252887144685f);
static const ai_layer_format_type _stems_0_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 _cls_branch_blocks_0_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _cls_branch_blocks_0_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _cls_branch_blocks_0_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _cls_branch_blocks_0_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _cls_branch_blocks_0_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _cls_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _cls_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _cls_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _cls_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _cls_branch_blocks_0_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0001928364799823612f, 0.001354596926830709f, 0.0005942431162111461f, 0.0002443334087729454f, 0.0002556343097239733f, 0.0005969777703285217f, 0.00024252370349131525f, 0.0023574393708258867f, 0.0018363271374255419f, 0.00023083093401510268f, 0.00022361607989296317f, 0.00020381958165671676f, 0.00020370437414385378f, 0.0020955337677150965f, 0.0002700219629332423f, 0.0016122577944770455f, 0.00020201283041387796f, 0.0019528655102476478f, 0.0009742376278154552f, 0.00155715923756361f, 0.00017252810357604176f, 0.0002222625771537423f, 0.00017964252037927508f, 0.00092500657774508f, 0.0003026793710887432f, 0.0002968019980471581f, 0.0002305515226908028f, 0.000368010310921818f, 0.0016527104889973998f, 0.0008269202662631869f, 0.00025020650355145335f, 0.0002422811958240345f, 0.0008903121342882514f, 0.0002233084087492898f, 0.000550577649846673f, 0.00026342322234995663f, 0.00023740924370940775f, 0.0002593753160908818f, 0.0003672533202916384f, 0.00021094834664836526f, 0.0003002049052156508f, 0.00023939560924191028f, 0.00035664308234117925f, 0.00018616837041918188f, 0.000192945240996778f, 0.0021727418061345816f, 0.0003900208685081452f, 0.0003353332285769284f, 0.0010537082562223077f, 0.00027220090851187706f, 0.0002818717039190233f, 0.0018038112903013825f, 0.002067683031782508f, 0.002912472700700164f, 0.00025699628167785704f, 0.00020455304183997214f, 0.00023368012625724077f, 0.0011781033826991916f, 0.00019737714319489896f, 0.0012647672556340694f, 0.00023744904319755733f, 0.00026650773361325264f, 0.0002777857589535415f, 0.0002283314970554784f);
static const ai_layer_format_type _cls_branch_blocks_0_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _cls_branch_blocks_0_act_Clip_output_0_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 _cls_branch_blocks_0_act_Clip_output_0_t_out_0_shape_h_const_u16 = 36;


static const ai_i8 _reg_branch_blocks_0_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32 = 36;

static const ai_u16 _reg_branch_blocks_0_act_Clip_output_0_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 _reg_branch_blocks_0_act_Clip_output_0_t_in_0_shape_h_const_u16 = 38;
static const ai_u16 _reg_branch_blocks_0_act_Clip_output_0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 _reg_branch_blocks_0_act_Clip_output_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 _reg_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 _reg_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float _reg_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _reg_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float _reg_branch_blocks_0_act_Clip_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0018296879716217518f, 0.0019396818242967129f, 0.0005331105785444379f, 0.0008685335051268339f, 0.0004419194010552019f, 0.0007595039787702262f, 0.0004053407465107739f, 0.0019624412525445223f, 0.0011621395824477077f, 0.0009367603342980146f, 0.002266921103000641f, 0.0005152766243554652f, 0.0008343016379512846f, 0.00048374771722592413f, 0.0004619863466359675f, 0.0017438189825043082f, 0.0008610071381554008f, 0.0005903725977987051f, 0.000952875823713839f, 0.00032118064700625837f, 0.0008115695090964437f, 0.0012781750410795212f, 0.0013235363876447082f, 0.001027011894620955f, 0.00023145231534726918f, 0.0009462394518777728f, 0.001169532653875649f, 0.00188862020149827f, 0.0007479139603674412f, 0.0009678491041995585f, 0.0003049240622203797f, 0.0025325336027890444f, 0.0006544175557792187f, 0.00031391388620249927f, 0.0004139697994105518f, 0.00018821927369572222f, 0.00035593603388406336f, 0.0004410678520798683f, 0.0008730646222829819f, 0.0003103950584772974f, 0.0005759104969911277f, 0.0004448728577699512f, 0.0008502195705659688f, 0.000736104731913656f, 0.0008104445296339691f, 0.000743888143915683f, 0.0005305196973495185f, 0.0013136741472408175f, 0.0010843866039067507f, 0.0011572896037250757f, 0.0007161168614402413f, 0.0011025911662727594f, 0.0017204541945829988f, 0.00048209019587375224f, 0.0007194095524027944f, 0.0018717912025749683f, 0.0011172100203111768f, 0.0006270311423577368f, 0.0014629889046773314f, 0.000550118216779083f, 0.00029442255618050694f, 0.0003529098175931722f, 0.0030019700061529875f, 0.0015077021671459079f);
static const ai_layer_format_type _reg_branch_blocks_0_act_Clip_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _reg_branch_blocks_0_act_Clip_output_0_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 _reg_branch_blocks_0_act_Clip_output_0_t_out_0_shape_h_const_u16 = 36;

static const ai_u16 size8_QuantizeLinear_Input_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 size8_QuantizeLinear_Input_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 size8_QuantizeLinear_Input_l_stride_1_const_u16 = 1;
static const ai_u16 size8_QuantizeLinear_Input_l_stride_0_const_u16 = 1;
static const ai_u16 size8_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 size8_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 2;
static const ai_i8 size8_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 size8_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = 119;
static const ai_float size8_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float size8_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.017869222909212112f;
static const ai_float size8_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0014439370715990663f, 0.0015557719161733985f);
static const ai_layer_format_type size8_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 off8_QuantizeLinear_Input_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 off8_QuantizeLinear_Input_t_in_0_shape_h_const_u16 = 36;
static const ai_u16 off8_QuantizeLinear_Input_l_stride_1_const_u16 = 1;
static const ai_u16 off8_QuantizeLinear_Input_l_stride_0_const_u16 = 1;
static const ai_u16 off8_QuantizeLinear_Input_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 off8_QuantizeLinear_Input_t_out_0_shape_ch_const_u16 = 2;
static const ai_i8 off8_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 off8_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8 = 2;
static const ai_float off8_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float off8_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32 = 0.18284200131893158f;
static const ai_float off8_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.015217792242765427f, 0.02094792015850544f);
static const ai_layer_format_type off8_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

STAI_API_ENTRY
stai_return_code stai_nirdet_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN _edge_conv_Conv_output_0 */
  {
      const ai_i8* _edge_conv_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* _edge_conv_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4);
    const ai_i32* _edge_conv_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 40);
    ai_i8* _edge_conv_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 661248);
    ai_i16* _edge_conv_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 956160);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(105, 1, {(stai_ptr) _edge_conv_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_edge_conv_Conv_output_0_t_in_0_ptr_const_s8, _edge_conv_Conv_output_0_t_in_0_shape_w_const_u16, _edge_conv_Conv_output_0_t_in_0_shape_h_const_u16, _edge_conv_Conv_output_0_t_in_0_shape_ch_const_u16, _edge_conv_Conv_output_0_t_weight_0_ptr_const_s8, _edge_conv_Conv_output_0_t_out_0_shape_ch_const_u16, _edge_conv_Conv_output_0_t_weight_0_shape_w_const_u16, _edge_conv_Conv_output_0_t_weight_0_shape_h_const_u16, _edge_conv_Conv_output_0_l_stride_1_const_u16, _edge_conv_Conv_output_0_l_stride_0_const_u16, _edge_conv_Conv_output_0_l_pad_W_0_const_s32, _edge_conv_Conv_output_0_l_pad_H_0_const_s32, _edge_conv_Conv_output_0_t_weight_1_ptr_const_s32, _edge_conv_Conv_output_0_t_in_0_fmt_zero_const_s8, _edge_conv_Conv_output_0_t_out_0_fmt_zero_const_s8, _edge_conv_Conv_output_0_t_in_0_fmt_scale_const_f32, _edge_conv_Conv_output_0_t_out_0_fmt_scale_const_f32, _edge_conv_Conv_output_0_t_weight_0_fmt_scale_const_f32, _edge_conv_Conv_output_0_l_out_ch_format_const_layer_format_type, _edge_conv_Conv_output_0_t_out_0_ptr_s8, _edge_conv_Conv_output_0_t_out_0_shape_w_const_u16, _edge_conv_Conv_output_0_t_out_0_shape_h_const_u16, 1, 96, _edge_conv_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(105, 1, {(stai_ptr) _edge_conv_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _edge_conv_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion */
  {
      const ai_i8* _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 661248);
    ai_float* _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 218880);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(105, 1, {(stai_ptr) _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_ptr_const_s8, _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_out_0_ptr_f32, _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_fmt_scale_const_f32, _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(105, 1, {(stai_ptr) _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _edge_conv_Conv_output_0_0_0__Abs_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _Abs_output_0 */
  {
      ai_handle _Abs_output_0_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 218880);
    const ai_handle _Abs_output_0_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 218880);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(111, 1, {(stai_ptr) _Abs_output_0_t_in_0_ptr_const_handle});
    
  forward_lite_nl_abs_if32of32(_Abs_output_0_t_out_0_ptr_handle, _Abs_output_0_t_in_0_ptr_const_handle, _Abs_output_0_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(111, 1, {(stai_ptr) _Abs_output_0_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END _Abs_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _AveragePool_output_0 */
  {
    
  forward_lite_ap__AveragePool_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _AveragePool_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _AveragePool_1_output_0 */
  {
    
  forward_lite_ap__AveragePool_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _AveragePool_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion */
  {
      const ai_float* _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 366336);
    ai_i8* _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 218880);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(115, 1, {(stai_ptr) _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_in_0_ptr_const_f32, _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_ptr_s8, _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_fmt_scale_const_f32, _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(115, 1, {(stai_ptr) _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _AveragePool_1_output_0_0_0__proj_Conv_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _AveragePool_2_output_0 */
  {
    
  forward_lite_ap_integer_INT8__AveragePool_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _AveragePool_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _AveragePool_3_output_0 */
  {
    
  forward_lite_ap_integer_INT8__AveragePool_3_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _AveragePool_3_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _proj_2_Conv_output_0 */
  {
    
  forward_lite_conv2d_integer_SSSA__proj_2_Conv_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _proj_2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Sigmoid_2_output_0 */
  {
    
  forward_lite_nl_integer__Sigmoid_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Sigmoid_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Add_2_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__Add_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Add_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _proj_1_Conv_output_0 */
  {
    
  forward_lite_conv2d_integer_SSSA__proj_1_Conv_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _proj_1_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Sigmoid_1_output_0 */
  {
    
  forward_lite_nl_integer__Sigmoid_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Sigmoid_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Add_1_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__Add_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Add_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _proj_Conv_output_0 */
  {
    
  forward_lite_conv2d_integer_SSSA__proj_Conv_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _proj_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Sigmoid_output_0 */
  {
    
  forward_lite_nl_integer__Sigmoid_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Sigmoid_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Add_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__Add_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Add_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stem_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stem_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* _backbone_stem_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 64);
    const ai_i32* _backbone_stem_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 208);
    ai_i8* _backbone_stem_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 362224);
    ai_i16* _backbone_stem_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 230352);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(106, 1, {(stai_ptr) _backbone_stem_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_backbone_stem_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stem_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stem_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stem_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stem_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stem_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stem_act_Clip_output_0_t_weight_0_shape_w_const_u16, _backbone_stem_act_Clip_output_0_t_weight_0_shape_h_const_u16, _backbone_stem_act_Clip_output_0_l_stride_1_const_u16, _backbone_stem_act_Clip_output_0_l_stride_0_const_u16, _backbone_stem_act_Clip_output_0_l_pad_W_0_const_s32, _backbone_stem_act_Clip_output_0_l_pad_H_0_const_s32, _backbone_stem_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stem_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stem_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stem_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stem_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stem_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stem_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stem_act_Clip_output_0_t_out_0_ptr_s8, _backbone_stem_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_stem_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 192, _backbone_stem_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(106, 1, {(stai_ptr) _backbone_stem_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stem_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_widen1_block_act_Clip_output_0 */
  {
      const ai_i8* _backbone_widen1_block_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 362224);
    const ai_i8* _backbone_widen1_block_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 272);
    const ai_i32* _backbone_widen1_block_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 7184);
    ai_i8* _backbone_widen1_block_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 356032);
    ai_i16* _backbone_widen1_block_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 230688);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(112, 1, {(stai_ptr) _backbone_widen1_block_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_backbone_widen1_block_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_widen1_block_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_widen1_block_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_widen1_block_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_widen1_block_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_widen1_block_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_widen1_block_act_Clip_output_0_t_weight_0_shape_w_const_u16, _backbone_widen1_block_act_Clip_output_0_t_weight_0_shape_h_const_u16, _backbone_widen1_block_act_Clip_output_0_l_stride_1_const_u16, _backbone_widen1_block_act_Clip_output_0_l_stride_0_const_u16, _backbone_widen1_block_act_Clip_output_0_l_pad_W_0_const_s32, _backbone_widen1_block_act_Clip_output_0_l_pad_H_0_const_s32, _backbone_widen1_block_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_widen1_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_widen1_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_widen1_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_widen1_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_widen1_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_widen1_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_widen1_block_act_Clip_output_0_t_out_0_ptr_s8, _backbone_widen1_block_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_widen1_block_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 960, _backbone_widen1_block_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(112, 1, {(stai_ptr) _backbone_widen1_block_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_widen1_block_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_cv2_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage1_cv2_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 356032);
    const ai_i8* _backbone_stage1_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 7376);
    const ai_i32* _backbone_stage1_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 8528);
    ai_i8* _backbone_stage1_cv2_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 798400);
    ai_i16* _backbone_stage1_cv2_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 355840);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(119, 1, {(stai_ptr) _backbone_stage1_cv2_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage1_cv2_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage1_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage1_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage1_cv2_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage1_cv2_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage1_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage1_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage1_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage1_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage1_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage1_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage1_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage1_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage1_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage1_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage1_cv2_act_Clip_output_0_t_out_0_ptr_s8, 1, 192, _backbone_stage1_cv2_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(119, 1, {(stai_ptr) _backbone_stage1_cv2_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_cv2_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_cv1_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage1_cv1_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 356032);
    const ai_i8* _backbone_stage1_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 8624);
    const ai_i32* _backbone_stage1_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 9776);
    ai_i8* _backbone_stage1_cv1_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* _backbone_stage1_cv1_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 355840);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(118, 1, {(stai_ptr) _backbone_stage1_cv1_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage1_cv1_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage1_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage1_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage1_cv1_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage1_cv1_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage1_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage1_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage1_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage1_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage1_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage1_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage1_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage1_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage1_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage1_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage1_cv1_act_Clip_output_0_t_out_0_ptr_s8, 1, 192, _backbone_stage1_cv1_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(118, 1, {(stai_ptr) _backbone_stage1_cv1_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_cv1_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
    ai_ptr _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 230688);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(129, 1, {(stai_ptr) _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(3072), (ai_i32)(3120), (ai_i32)(3120), (ai_i32)(24), (ai_i32)(24));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(129, 1, {(stai_ptr) _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 230688);
    const ai_i8* _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 9872);
    const ai_i32* _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 15056);
    ai_i8* _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 461568);
    ai_i16* _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 223488);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(129, 1, {(stai_ptr) _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 1088, _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(129, 1, {(stai_ptr) _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_blocks_blocks_0_conv_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_blocks_blocks_0_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__backbone_stage1_blocks_blocks_0_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_blocks_blocks_0_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_Concat_output_0 */
  {
    
  forward_lite_concat__backbone_stage1_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage1_cv3_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage1_cv3_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 356032);
    const ai_i8* _backbone_stage1_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 15152);
    const ai_i32* _backbone_stage1_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 17456);
    ai_i8* _backbone_stage1_cv3_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 349888);
    ai_i16* _backbone_stage1_cv3_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(162, 1, {(stai_ptr) _backbone_stage1_cv3_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage1_cv3_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage1_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage1_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage1_cv3_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage1_cv3_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage1_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage1_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage1_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage1_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage1_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage1_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage1_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage1_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage1_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage1_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage1_cv3_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage1_cv3_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(162, 1, {(stai_ptr) _backbone_stage1_cv3_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage1_cv3_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_down2_block_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 349888);
    ai_ptr _backbone_down2_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 330496);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(169, 1, {(stai_ptr) _backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_down2_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_down2_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_down2_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(6144), (ai_i32)(6240), (ai_i32)(6240), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(169, 1, {(stai_ptr) _backbone_down2_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_down2_block_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_down2_block_act_Clip_output_0 */
  {
      const ai_i8* _backbone_down2_block_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 330496);
    const ai_i8* _backbone_down2_block_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 17648);
    const ai_i32* _backbone_down2_block_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 59120);
    ai_i8* _backbone_down2_block_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 792256);
    ai_i16* _backbone_down2_block_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(169, 1, {(stai_ptr) _backbone_down2_block_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(_backbone_down2_block_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_down2_block_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_down2_block_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_down2_block_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_down2_block_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_down2_block_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_down2_block_act_Clip_output_0_t_weight_0_shape_w_const_u16, _backbone_down2_block_act_Clip_output_0_t_weight_0_shape_h_const_u16, _backbone_down2_block_act_Clip_output_0_l_stride_1_const_u16, _backbone_down2_block_act_Clip_output_0_l_stride_0_const_u16, _backbone_down2_block_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_down2_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_down2_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_down2_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_down2_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_down2_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_down2_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_down2_block_act_Clip_output_0_t_out_0_ptr_s8, _backbone_down2_block_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_down2_block_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 1, 2496, _backbone_down2_block_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(169, 1, {(stai_ptr) _backbone_down2_block_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_down2_block_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_cv2_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage2_cv2_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 792256);
    const ai_i8* _backbone_stage2_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 59504);
    const ai_i32* _backbone_stage2_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 64112);
    ai_i8* _backbone_stage2_cv2_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 384);
    ai_i16* _backbone_stage2_cv2_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(174, 1, {(stai_ptr) _backbone_stage2_cv2_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage2_cv2_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage2_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage2_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage2_cv2_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage2_cv2_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage2_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage2_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage2_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage2_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage2_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage2_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage2_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage2_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage2_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage2_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage2_cv2_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage2_cv2_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(174, 1, {(stai_ptr) _backbone_stage2_cv2_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_cv2_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_cv1_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage2_cv1_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 792256);
    const ai_i8* _backbone_stage2_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 64304);
    const ai_i32* _backbone_stage2_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 68912);
    ai_i8* _backbone_stage2_cv1_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 230688);
    ai_i16* _backbone_stage2_cv1_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(173, 1, {(stai_ptr) _backbone_stage2_cv1_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage2_cv1_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage2_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage2_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage2_cv1_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage2_cv1_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage2_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage2_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage2_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage2_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage2_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage2_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage2_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage2_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage2_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage2_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage2_cv1_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage2_cv1_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(173, 1, {(stai_ptr) _backbone_stage2_cv1_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_cv1_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 230688);
    ai_ptr _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 341280);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(179, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(3072), (ai_i32)(3168), (ai_i32)(3168), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(179, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 341280);
    const ai_i8* _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 69104);
    const ai_i32* _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 89840);
    ai_i8* _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 461664);
    ai_i16* _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 110976);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(179, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2112, _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(179, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_blocks_blocks_0_conv_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_blocks_blocks_0_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__backbone_stage2_blocks_blocks_0_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_blocks_blocks_0_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 341280);
    ai_ptr _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 451872);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(185, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(3072), (ai_i32)(3168), (ai_i32)(3168), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(185, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 451872);
    const ai_i8* _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 90032);
    const ai_i32* _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 110768);
    ai_i8* _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 230688);
    ai_i16* _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 110976);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(185, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_ptr_s8, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2112, _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(185, 1, {(stai_ptr) _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_blocks_blocks_1_conv_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_blocks_blocks_1_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__backbone_stage2_blocks_blocks_1_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_blocks_blocks_1_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_Concat_output_0 */
  {
    
  forward_lite_concat__backbone_stage2_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage2_cv3_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage2_cv3_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 230688);
    const ai_i8* _backbone_stage2_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 110960);
    const ai_i32* _backbone_stage2_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 120176);
    ai_i8* _backbone_stage2_cv3_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 451872);
    ai_i16* _backbone_stage2_cv3_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(194, 1, {(stai_ptr) _backbone_stage2_cv3_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage2_cv3_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage2_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage2_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage2_cv3_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage2_cv3_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage2_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage2_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage2_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage2_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage2_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage2_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage2_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage2_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage2_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage2_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage2_cv3_act_Clip_output_0_t_out_0_ptr_s8, 1, 768, _backbone_stage2_cv3_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(194, 1, {(stai_ptr) _backbone_stage2_cv3_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage2_cv3_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_down3_block_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 451872);
    ai_ptr _backbone_down3_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 673056);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(197, 1, {(stai_ptr) _backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_down3_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_down3_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_down3_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(6144), (ai_i32)(6336), (ai_i32)(6336), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(197, 1, {(stai_ptr) _backbone_down3_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_down3_block_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_down3_block_act_Clip_output_0 */
  {
      const ai_i8* _backbone_down3_block_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 673056);
    const ai_i8* _backbone_down3_block_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 120560);
    const ai_i32* _backbone_down3_block_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 203504);
    ai_i8* _backbone_down3_block_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 4224);
    ai_i16* _backbone_down3_block_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(197, 1, {(stai_ptr) _backbone_down3_block_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(_backbone_down3_block_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_down3_block_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_down3_block_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_down3_block_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_down3_block_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_down3_block_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_down3_block_act_Clip_output_0_t_weight_0_shape_w_const_u16, _backbone_down3_block_act_Clip_output_0_t_weight_0_shape_h_const_u16, _backbone_down3_block_act_Clip_output_0_l_stride_1_const_u16, _backbone_down3_block_act_Clip_output_0_l_stride_0_const_u16, _backbone_down3_block_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_down3_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_down3_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_down3_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_down3_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_down3_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_down3_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_down3_block_act_Clip_output_0_t_out_0_ptr_s8, _backbone_down3_block_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_down3_block_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 1, 4224, _backbone_down3_block_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(197, 1, {(stai_ptr) _backbone_down3_block_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_down3_block_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_cv2_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage3_cv2_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4224);
    const ai_i8* _backbone_stage3_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 203888);
    const ai_i32* _backbone_stage3_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 208496);
    ai_i8* _backbone_stage3_cv2_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 59520);
    ai_i16* _backbone_stage3_cv2_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(204, 1, {(stai_ptr) _backbone_stage3_cv2_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage3_cv2_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage3_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage3_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage3_cv2_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage3_cv2_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage3_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage3_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage3_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage3_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage3_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage3_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage3_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage3_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage3_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage3_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage3_cv2_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage3_cv2_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(204, 1, {(stai_ptr) _backbone_stage3_cv2_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_cv2_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_cv1_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage3_cv1_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4224);
    const ai_i8* _backbone_stage3_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 208688);
    const ai_i32* _backbone_stage3_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 213296);
    ai_i8* _backbone_stage3_cv1_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 87168);
    ai_i16* _backbone_stage3_cv1_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(203, 1, {(stai_ptr) _backbone_stage3_cv1_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage3_cv1_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage3_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage3_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage3_cv1_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage3_cv1_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage3_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage3_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage3_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage3_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage3_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage3_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage3_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage3_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage3_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage3_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage3_cv1_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage3_cv1_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(203, 1, {(stai_ptr) _backbone_stage3_cv1_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_cv1_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 87168);
    ai_ptr _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(212, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1632), (ai_i32)(1632), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(212, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 213488);
    const ai_i32* _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 234224);
    ai_i8* _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 114816);
    ai_i16* _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 32640);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(212, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2112, _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(212, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_blocks_blocks_0_conv_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_blocks_blocks_0_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__backbone_stage3_blocks_blocks_0_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_blocks_blocks_0_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
    ai_ptr _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 87168);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(218, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1632), (ai_i32)(1632), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(218, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 87168);
    const ai_i8* _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 234416);
    const ai_i32* _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 255152);
    ai_i8* _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 29760);
    ai_i16* _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 27648);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(218, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_ptr_s8, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2112, _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(218, 1, {(stai_ptr) _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_blocks_blocks_1_conv_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_blocks_blocks_1_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__backbone_stage3_blocks_blocks_1_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_blocks_blocks_1_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_Concat_output_0 */
  {
    
  forward_lite_concat__backbone_stage3_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage3_cv3_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage3_cv3_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* _backbone_stage3_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 255344);
    const ai_i32* _backbone_stage3_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 264560);
    ai_i8* _backbone_stage3_cv3_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 56064);
    ai_i16* _backbone_stage3_cv3_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 55296);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(227, 1, {(stai_ptr) _backbone_stage3_cv3_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage3_cv3_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage3_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage3_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage3_cv3_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage3_cv3_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage3_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage3_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage3_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage3_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage3_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage3_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage3_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage3_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage3_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage3_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage3_cv3_act_Clip_output_0_t_out_0_ptr_s8, 1, 768, _backbone_stage3_cv3_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(227, 1, {(stai_ptr) _backbone_stage3_cv3_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage3_cv3_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_down4_block_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 56064);
    ai_ptr _backbone_down4_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 111360);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(230, 1, {(stai_ptr) _backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_down4_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_down4_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_down4_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(3072), (ai_i32)(3264), (ai_i32)(3264), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(230, 1, {(stai_ptr) _backbone_down4_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_down4_block_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_down4_block_act_Clip_output_0 */
  {
      const ai_i8* _backbone_down4_block_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 111360);
    const ai_i8* _backbone_down4_block_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 264944);
    const ai_i32* _backbone_down4_block_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 347888);
    ai_i8* _backbone_down4_block_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 4224);
    ai_i16* _backbone_down4_block_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(230, 1, {(stai_ptr) _backbone_down4_block_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(_backbone_down4_block_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_down4_block_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_down4_block_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_down4_block_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_down4_block_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_down4_block_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_down4_block_act_Clip_output_0_t_weight_0_shape_w_const_u16, _backbone_down4_block_act_Clip_output_0_t_weight_0_shape_h_const_u16, _backbone_down4_block_act_Clip_output_0_l_stride_1_const_u16, _backbone_down4_block_act_Clip_output_0_l_stride_0_const_u16, _backbone_down4_block_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_down4_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_down4_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_down4_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_down4_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_down4_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_down4_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_down4_block_act_Clip_output_0_t_out_0_ptr_s8, _backbone_down4_block_act_Clip_output_0_t_out_0_shape_w_const_u16, _backbone_down4_block_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 1, 4224, _backbone_down4_block_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(230, 1, {(stai_ptr) _backbone_down4_block_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_down4_block_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_cv2_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage4_cv2_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4224);
    const ai_i8* _backbone_stage4_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 348272);
    const ai_i32* _backbone_stage4_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 352880);
    ai_i8* _backbone_stage4_cv2_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 18048);
    ai_i16* _backbone_stage4_cv2_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(237, 1, {(stai_ptr) _backbone_stage4_cv2_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage4_cv2_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage4_cv2_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage4_cv2_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage4_cv2_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage4_cv2_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage4_cv2_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage4_cv2_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage4_cv2_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage4_cv2_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage4_cv2_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage4_cv2_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage4_cv2_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage4_cv2_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage4_cv2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage4_cv2_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage4_cv2_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage4_cv2_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(237, 1, {(stai_ptr) _backbone_stage4_cv2_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_cv2_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_cv1_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage4_cv1_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4224);
    const ai_i8* _backbone_stage4_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 353072);
    const ai_i32* _backbone_stage4_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 357680);
    ai_i8* _backbone_stage4_cv1_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 24960);
    ai_i16* _backbone_stage4_cv1_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(236, 1, {(stai_ptr) _backbone_stage4_cv1_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage4_cv1_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage4_cv1_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage4_cv1_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage4_cv1_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage4_cv1_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage4_cv1_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage4_cv1_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage4_cv1_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage4_cv1_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage4_cv1_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage4_cv1_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage4_cv1_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage4_cv1_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage4_cv1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage4_cv1_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage4_cv1_act_Clip_output_0_t_out_0_ptr_s8, 1, 384, _backbone_stage4_cv1_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(236, 1, {(stai_ptr) _backbone_stage4_cv1_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_cv1_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  {
      const ai_ptr _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 24960);
    ai_ptr _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(1920), (ai_i32)(1920), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion */
  {
      const ai_i8* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_float* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 111360);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_ptr_const_s8, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_out_0_ptr_f32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_fmt_scale_const_f32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_pad_before_0_0__backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0 */
  {
      const ai_float* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 111360);
    ai_float* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 161280);
    const ai_u8* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 357872);
    const ai_u8* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 440816);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_ptr_const_f32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_f32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_ptr_const_u8, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_1_ptr_const_u8, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_ch_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_ch_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_w_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_in_0_shape_h_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_w_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_shape_h_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_shape_w_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_weight_0_shape_h_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_pad_W_0_const_s32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_pad_H_0_const_s32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_stride_0_const_u16, 5, 5, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_dilation_W_const_u16, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_l_dilation_H_const_u16, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_v_n_groups_const_size);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion */
  {
      const ai_float* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 161280);
    ai_i8* _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(_backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_in_0_ptr_const_f32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_ptr_s8, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_fmt_scale_const_f32, _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(245, 1, {(stai_ptr) _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_blocks_blocks_0_conv_act_Clip_output_0_0_1__backbone_stage4_blocks_blocks_0_Clip_output_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_blocks_blocks_0_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__backbone_stage4_blocks_blocks_0_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_blocks_blocks_0_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_Concat_output_0 */
  {
    
  forward_lite_concat__backbone_stage4_Concat_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_Concat_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _backbone_stage4_cv3_act_Clip_output_0 */
  {
      const ai_i8* _backbone_stage4_cv3_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 24960);
    const ai_i8* _backbone_stage4_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 441008);
    const ai_i32* _backbone_stage4_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 450224);
    ai_i8* _backbone_stage4_cv3_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 768);
    ai_i16* _backbone_stage4_cv3_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(254, 1, {(stai_ptr) _backbone_stage4_cv3_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_backbone_stage4_cv3_act_Clip_output_0_t_in_0_ptr_const_s8, _backbone_stage4_cv3_act_Clip_output_0_t_in_0_shape_w_const_u16, _backbone_stage4_cv3_act_Clip_output_0_t_in_0_shape_h_const_u16, _backbone_stage4_cv3_act_Clip_output_0_l_stride_1_const_u16, _backbone_stage4_cv3_act_Clip_output_0_l_stride_0_const_u16, _backbone_stage4_cv3_act_Clip_output_0_t_in_0_shape_ch_const_u16, _backbone_stage4_cv3_act_Clip_output_0_t_weight_0_ptr_const_s8, _backbone_stage4_cv3_act_Clip_output_0_t_out_0_shape_ch_const_u16, _backbone_stage4_cv3_act_Clip_output_0_t_weight_1_ptr_const_s32, _backbone_stage4_cv3_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _backbone_stage4_cv3_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _backbone_stage4_cv3_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _backbone_stage4_cv3_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _backbone_stage4_cv3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _backbone_stage4_cv3_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _backbone_stage4_cv3_act_Clip_output_0_t_out_0_ptr_s8, 1, 768, _backbone_stage4_cv3_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(254, 1, {(stai_ptr) _backbone_stage4_cv3_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _backbone_stage4_cv3_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Clip_2_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__Clip_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Clip_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_lat5_Conv_output_0 */
  {
      const ai_i8* _neck_lat5_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14592);
    const ai_i8* _neck_lat5_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 450608);
    const ai_i32* _neck_lat5_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 456752);
    ai_i8* _neck_lat5_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 512);
    ai_i16* _neck_lat5_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(260, 1, {(stai_ptr) _neck_lat5_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_neck_lat5_Conv_output_0_t_in_0_ptr_const_s8, _neck_lat5_Conv_output_0_t_in_0_shape_w_const_u16, _neck_lat5_Conv_output_0_t_in_0_shape_h_const_u16, _neck_lat5_Conv_output_0_l_stride_1_const_u16, _neck_lat5_Conv_output_0_l_stride_0_const_u16, _neck_lat5_Conv_output_0_t_in_0_shape_ch_const_u16, _neck_lat5_Conv_output_0_t_weight_0_ptr_const_s8, _neck_lat5_Conv_output_0_t_out_0_shape_ch_const_u16, _neck_lat5_Conv_output_0_t_weight_1_ptr_const_s32, _neck_lat5_Conv_output_0_t_in_0_fmt_zero_const_s8, _neck_lat5_Conv_output_0_t_out_0_fmt_zero_const_s8, _neck_lat5_Conv_output_0_t_in_0_fmt_scale_const_f32, _neck_lat5_Conv_output_0_t_out_0_fmt_scale_const_f32, _neck_lat5_Conv_output_0_t_weight_0_fmt_scale_const_f32, _neck_lat5_Conv_output_0_l_out_ch_format_const_layer_format_type, _neck_lat5_Conv_output_0_t_out_0_ptr_s8, 1, 512, _neck_lat5_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(260, 1, {(stai_ptr) _neck_lat5_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_lat5_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_out5_act_Clip_output_0_pad_before */
  {
      const ai_ptr _neck_out5_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 512);
    ai_ptr _neck_out5_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 9728);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(264, 1, {(stai_ptr) _neck_out5_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_neck_out5_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _neck_out5_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_neck_out5_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _neck_out5_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _neck_out5_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1024), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(264, 1, {(stai_ptr) _neck_out5_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _neck_out5_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _neck_out5_act_Clip_output_0 */
  {
      const ai_i8* _neck_out5_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 9728);
    const ai_i8* _neck_out5_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 457008);
    const ai_i32* _neck_out5_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 493872);
    ai_i8* _neck_out5_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 25216);
    ai_i16* _neck_out5_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 22400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(264, 1, {(stai_ptr) _neck_out5_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_neck_out5_act_Clip_output_0_t_in_0_ptr_const_s8, _neck_out5_act_Clip_output_0_t_in_0_shape_w_const_u16, _neck_out5_act_Clip_output_0_t_in_0_shape_h_const_u16, _neck_out5_act_Clip_output_0_t_in_0_shape_ch_const_u16, _neck_out5_act_Clip_output_0_t_weight_0_ptr_const_s8, _neck_out5_act_Clip_output_0_t_out_0_shape_ch_const_u16, _neck_out5_act_Clip_output_0_t_weight_1_ptr_const_s32, _neck_out5_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _neck_out5_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _neck_out5_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _neck_out5_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _neck_out5_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _neck_out5_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _neck_out5_act_Clip_output_0_t_out_0_ptr_s8, _neck_out5_act_Clip_output_0_t_out_0_shape_w_const_u16, _neck_out5_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _neck_out5_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(264, 1, {(stai_ptr) _neck_out5_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_out5_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _stems_2_act_Clip_output_0 */
  {
      const ai_i8* _stems_2_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 25216);
    const ai_i8* _stems_2_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 494128);
    const ai_i32* _stems_2_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 498224);
    ai_i8* _stems_2_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9728);
    ai_i16* _stems_2_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(270, 1, {(stai_ptr) _stems_2_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_stems_2_act_Clip_output_0_t_in_0_ptr_const_s8, _stems_2_act_Clip_output_0_t_in_0_shape_w_const_u16, _stems_2_act_Clip_output_0_t_in_0_shape_h_const_u16, _stems_2_act_Clip_output_0_l_stride_1_const_u16, _stems_2_act_Clip_output_0_l_stride_0_const_u16, _stems_2_act_Clip_output_0_t_in_0_shape_ch_const_u16, _stems_2_act_Clip_output_0_t_weight_0_ptr_const_s8, _stems_2_act_Clip_output_0_t_out_0_shape_ch_const_u16, _stems_2_act_Clip_output_0_t_weight_1_ptr_const_s32, _stems_2_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _stems_2_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _stems_2_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _stems_2_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _stems_2_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _stems_2_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _stems_2_act_Clip_output_0_t_out_0_ptr_s8, 1, 512, _stems_2_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(270, 1, {(stai_ptr) _stems_2_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _stems_2_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _cls_branch_blocks_0_act_2_Clip_output_0_pad_before */
  {
      const ai_ptr _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 9728);
    ai_ptr _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 18944);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(276, 1, {(stai_ptr) _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_cls_branch_blocks_0_act_2_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1024), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(276, 1, {(stai_ptr) _cls_branch_blocks_0_act_2_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _cls_branch_blocks_0_act_2_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _cls_branch_blocks_0_act_2_Clip_output_0 */
  {
      const ai_i8* _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18944);
    const ai_i8* _cls_branch_blocks_0_act_2_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 498480);
    const ai_i32* _cls_branch_blocks_0_act_2_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 535344);
    ai_i8* _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 34432);
    ai_i16* _cls_branch_blocks_0_act_2_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 31616);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(276, 1, {(stai_ptr) _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_ptr_const_s8, _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_w_const_u16, _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_h_const_u16, _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_ch_const_u16, _cls_branch_blocks_0_act_2_Clip_output_0_t_weight_0_ptr_const_s8, _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_ch_const_u16, _cls_branch_blocks_0_act_2_Clip_output_0_t_weight_1_ptr_const_s32, _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_zero_const_s8, _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_zero_const_s8, _cls_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_2_Clip_output_0_t_weight_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_2_Clip_output_0_l_out_ch_format_const_layer_format_type, _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_ptr_s8, _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_w_const_u16, _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _cls_branch_blocks_0_act_2_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(276, 1, {(stai_ptr) _cls_branch_blocks_0_act_2_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _cls_branch_blocks_0_act_2_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN cls32_QuantizeLinear_Input */
  {
    
  forward_lite_conv2d_integer_SSSA_cls32_QuantizeLinear_Input(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cls32_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN _reg_branch_blocks_0_act_2_Clip_output_0_pad_before */
  {
      const ai_ptr _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 9728);
    ai_ptr _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 18944);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(277, 1, {(stai_ptr) _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_reg_branch_blocks_0_act_2_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1024), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(277, 1, {(stai_ptr) _reg_branch_blocks_0_act_2_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _reg_branch_blocks_0_act_2_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _reg_branch_blocks_0_act_2_Clip_output_0 */
  {
      const ai_i8* _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18944);
    const ai_i8* _reg_branch_blocks_0_act_2_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 535668);
    const ai_i32* _reg_branch_blocks_0_act_2_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 572532);
    ai_i8* _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 31616);
    ai_i16* _reg_branch_blocks_0_act_2_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 9728);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(277, 1, {(stai_ptr) _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_ptr_const_s8, _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_w_const_u16, _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_h_const_u16, _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_shape_ch_const_u16, _reg_branch_blocks_0_act_2_Clip_output_0_t_weight_0_ptr_const_s8, _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_ch_const_u16, _reg_branch_blocks_0_act_2_Clip_output_0_t_weight_1_ptr_const_s32, _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_zero_const_s8, _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_zero_const_s8, _reg_branch_blocks_0_act_2_Clip_output_0_t_in_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_2_Clip_output_0_t_weight_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_2_Clip_output_0_l_out_ch_format_const_layer_format_type, _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_ptr_s8, _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_w_const_u16, _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _reg_branch_blocks_0_act_2_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(277, 1, {(stai_ptr) _reg_branch_blocks_0_act_2_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _reg_branch_blocks_0_act_2_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN size32_QuantizeLinear_Input */
  {
      const ai_i8* size32_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 31616);
    const ai_i8* size32_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 572788);
    const ai_i32* size32_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 572916);
    ai_i8* size32_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9728);
    ai_i16* size32_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(287, 1, {(stai_ptr) size32_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(size32_QuantizeLinear_Input_t_in_0_ptr_const_s8, size32_QuantizeLinear_Input_t_in_0_shape_w_const_u16, size32_QuantizeLinear_Input_t_in_0_shape_h_const_u16, size32_QuantizeLinear_Input_l_stride_1_const_u16, size32_QuantizeLinear_Input_l_stride_0_const_u16, size32_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, size32_QuantizeLinear_Input_t_weight_0_ptr_const_s8, size32_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, size32_QuantizeLinear_Input_t_weight_1_ptr_const_s32, size32_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, size32_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, size32_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, size32_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, size32_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, size32_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type, size32_QuantizeLinear_Input_t_out_0_ptr_s8, 1, 16, size32_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(287, 1, {(stai_ptr) size32_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END size32_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN size32_QuantizeLinear_Input_Transpose_8 */
  {
    
  forward_lite_transpose_size32_QuantizeLinear_Input_Transpose_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END size32_QuantizeLinear_Input_Transpose_8 */
  /* LITE_KERNEL_SECTION BEGIN off32_QuantizeLinear_Input */
  {
      const ai_i8* off32_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 31616);
    const ai_i8* off32_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 572924);
    const ai_i32* off32_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 573052);
    ai_i8* off32_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9728);
    ai_i16* off32_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(286, 1, {(stai_ptr) off32_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(off32_QuantizeLinear_Input_t_in_0_ptr_const_s8, off32_QuantizeLinear_Input_t_in_0_shape_w_const_u16, off32_QuantizeLinear_Input_t_in_0_shape_h_const_u16, off32_QuantizeLinear_Input_l_stride_1_const_u16, off32_QuantizeLinear_Input_l_stride_0_const_u16, off32_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, off32_QuantizeLinear_Input_t_weight_0_ptr_const_s8, off32_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, off32_QuantizeLinear_Input_t_weight_1_ptr_const_s32, off32_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, off32_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, off32_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, off32_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, off32_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, off32_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type, off32_QuantizeLinear_Input_t_out_0_ptr_s8, 1, 16, off32_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(286, 1, {(stai_ptr) off32_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END off32_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN off32_QuantizeLinear_Input_Transpose_7 */
  {
    
  forward_lite_transpose_off32_QuantizeLinear_Input_Transpose_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END off32_QuantizeLinear_Input_Transpose_7 */
  /* LITE_KERNEL_SECTION BEGIN _neck_Resize_output_0 */
  {
    
  forward_lite_upsample_nearest__neck_Resize_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _neck_Resize_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Clip_1_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__Clip_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Clip_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_lat4_Conv_output_0 */
  {
      const ai_i8* _neck_lat4_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 111360);
    const ai_i8* _neck_lat4_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 573060);
    const ai_i32* _neck_lat4_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 579204);
    ai_i8* _neck_lat4_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 47456);
    ai_i16* _neck_lat4_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(238, 1, {(stai_ptr) _neck_lat4_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_neck_lat4_Conv_output_0_t_in_0_ptr_const_s8, _neck_lat4_Conv_output_0_t_in_0_shape_w_const_u16, _neck_lat4_Conv_output_0_t_in_0_shape_h_const_u16, _neck_lat4_Conv_output_0_l_stride_1_const_u16, _neck_lat4_Conv_output_0_l_stride_0_const_u16, _neck_lat4_Conv_output_0_t_in_0_shape_ch_const_u16, _neck_lat4_Conv_output_0_t_weight_0_ptr_const_s8, _neck_lat4_Conv_output_0_t_out_0_shape_ch_const_u16, _neck_lat4_Conv_output_0_t_weight_1_ptr_const_s32, _neck_lat4_Conv_output_0_t_in_0_fmt_zero_const_s8, _neck_lat4_Conv_output_0_t_out_0_fmt_zero_const_s8, _neck_lat4_Conv_output_0_t_in_0_fmt_scale_const_f32, _neck_lat4_Conv_output_0_t_out_0_fmt_scale_const_f32, _neck_lat4_Conv_output_0_t_weight_0_fmt_scale_const_f32, _neck_lat4_Conv_output_0_l_out_ch_format_const_layer_format_type, _neck_lat4_Conv_output_0_t_out_0_ptr_s8, 1, 512, _neck_lat4_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(238, 1, {(stai_ptr) _neck_lat4_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_lat4_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__neck_Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _neck_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_Resize_1_output_0 */
  {
    
  forward_lite_upsample_nearest__neck_Resize_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _neck_Resize_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _Clip_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__Clip_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_lat3_Conv_output_0 */
  {
      const ai_i8* _neck_lat3_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 673056);
    const ai_i8* _neck_lat3_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 579460);
    const ai_i32* _neck_lat3_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 585604);
    ai_i8* _neck_lat3_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 370944);
    ai_i16* _neck_lat3_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(205, 1, {(stai_ptr) _neck_lat3_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_neck_lat3_Conv_output_0_t_in_0_ptr_const_s8, _neck_lat3_Conv_output_0_t_in_0_shape_w_const_u16, _neck_lat3_Conv_output_0_t_in_0_shape_h_const_u16, _neck_lat3_Conv_output_0_l_stride_1_const_u16, _neck_lat3_Conv_output_0_l_stride_0_const_u16, _neck_lat3_Conv_output_0_t_in_0_shape_ch_const_u16, _neck_lat3_Conv_output_0_t_weight_0_ptr_const_s8, _neck_lat3_Conv_output_0_t_out_0_shape_ch_const_u16, _neck_lat3_Conv_output_0_t_weight_1_ptr_const_s32, _neck_lat3_Conv_output_0_t_in_0_fmt_zero_const_s8, _neck_lat3_Conv_output_0_t_out_0_fmt_zero_const_s8, _neck_lat3_Conv_output_0_t_in_0_fmt_scale_const_f32, _neck_lat3_Conv_output_0_t_out_0_fmt_scale_const_f32, _neck_lat3_Conv_output_0_t_weight_0_fmt_scale_const_f32, _neck_lat3_Conv_output_0_l_out_ch_format_const_layer_format_type, _neck_lat3_Conv_output_0_t_out_0_ptr_s8, 1, 512, _neck_lat3_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(205, 1, {(stai_ptr) _neck_lat3_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_lat3_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_Clip_1_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__neck_Clip_1_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _neck_Clip_1_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_pan3to4_block_act_Clip_output_0_pad_before */
  {
      const ai_ptr _neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 518400);
    ai_ptr _neck_pan3to4_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 121184);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(296, 1, {(stai_ptr) _neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _neck_pan3to4_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_neck_pan3to4_block_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _neck_pan3to4_block_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(4096), (ai_i32)(4224), (ai_i32)(4224), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(296, 1, {(stai_ptr) _neck_pan3to4_block_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _neck_pan3to4_block_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _neck_pan3to4_block_act_Clip_output_0 */
  {
      const ai_i8* _neck_pan3to4_block_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 121184);
    const ai_i8* _neck_pan3to4_block_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 585860);
    const ai_i32* _neck_pan3to4_block_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 622724);
    ai_i8* _neck_pan3to4_block_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    ai_i16* _neck_pan3to4_block_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(296, 1, {(stai_ptr) _neck_pan3to4_block_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(_neck_pan3to4_block_act_Clip_output_0_t_in_0_ptr_const_s8, _neck_pan3to4_block_act_Clip_output_0_t_in_0_shape_w_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_in_0_shape_h_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_in_0_shape_ch_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_weight_0_ptr_const_s8, _neck_pan3to4_block_act_Clip_output_0_t_out_0_shape_ch_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_weight_0_shape_w_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_weight_0_shape_h_const_u16, _neck_pan3to4_block_act_Clip_output_0_l_stride_1_const_u16, _neck_pan3to4_block_act_Clip_output_0_l_stride_0_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_weight_1_ptr_const_s32, _neck_pan3to4_block_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _neck_pan3to4_block_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _neck_pan3to4_block_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _neck_pan3to4_block_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _neck_pan3to4_block_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _neck_pan3to4_block_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _neck_pan3to4_block_act_Clip_output_0_t_out_0_ptr_s8, _neck_pan3to4_block_act_Clip_output_0_t_out_0_shape_w_const_u16, _neck_pan3to4_block_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 1, 2816, _neck_pan3to4_block_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(296, 1, {(stai_ptr) _neck_pan3to4_block_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_pan3to4_block_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_Clip_2_output_0 */
  {
    
  forward_lite_eltwise_integer_INT8__neck_Clip_2_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _neck_Clip_2_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _neck_out4_act_Clip_output_0_pad_before */
  {
      const ai_ptr _neck_out4_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 47456);
    ai_ptr _neck_out4_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 84320);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(308, 1, {(stai_ptr) _neck_out4_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_neck_out4_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _neck_out4_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_neck_out4_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _neck_out4_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _neck_out4_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2048), (ai_i32)(2176), (ai_i32)(2176), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(308, 1, {(stai_ptr) _neck_out4_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _neck_out4_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _neck_out4_act_Clip_output_0 */
  {
      const ai_i8* _neck_out4_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 84320);
    const ai_i8* _neck_out4_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 622980);
    const ai_i32* _neck_out4_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 659844);
    ai_i8* _neck_out4_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    ai_i16* _neck_out4_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(308, 1, {(stai_ptr) _neck_out4_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_neck_out4_act_Clip_output_0_t_in_0_ptr_const_s8, _neck_out4_act_Clip_output_0_t_in_0_shape_w_const_u16, _neck_out4_act_Clip_output_0_t_in_0_shape_h_const_u16, _neck_out4_act_Clip_output_0_t_in_0_shape_ch_const_u16, _neck_out4_act_Clip_output_0_t_weight_0_ptr_const_s8, _neck_out4_act_Clip_output_0_t_out_0_shape_ch_const_u16, _neck_out4_act_Clip_output_0_t_weight_1_ptr_const_s32, _neck_out4_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _neck_out4_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _neck_out4_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _neck_out4_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _neck_out4_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _neck_out4_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _neck_out4_act_Clip_output_0_t_out_0_ptr_s8, _neck_out4_act_Clip_output_0_t_out_0_shape_w_const_u16, _neck_out4_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _neck_out4_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(308, 1, {(stai_ptr) _neck_out4_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_out4_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _stems_1_act_Clip_output_0 */
  {
      const ai_i8* _stems_1_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    const ai_i8* _stems_1_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 660100);
    const ai_i32* _stems_1_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 664196);
    ai_i8* _stems_1_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 47456);
    ai_i16* _stems_1_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(317, 1, {(stai_ptr) _stems_1_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_stems_1_act_Clip_output_0_t_in_0_ptr_const_s8, _stems_1_act_Clip_output_0_t_in_0_shape_w_const_u16, _stems_1_act_Clip_output_0_t_in_0_shape_h_const_u16, _stems_1_act_Clip_output_0_l_stride_1_const_u16, _stems_1_act_Clip_output_0_l_stride_0_const_u16, _stems_1_act_Clip_output_0_t_in_0_shape_ch_const_u16, _stems_1_act_Clip_output_0_t_weight_0_ptr_const_s8, _stems_1_act_Clip_output_0_t_out_0_shape_ch_const_u16, _stems_1_act_Clip_output_0_t_weight_1_ptr_const_s32, _stems_1_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _stems_1_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _stems_1_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _stems_1_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _stems_1_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _stems_1_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _stems_1_act_Clip_output_0_t_out_0_ptr_s8, 1, 512, _stems_1_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(317, 1, {(stai_ptr) _stems_1_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _stems_1_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _cls_branch_blocks_0_act_1_Clip_output_0_pad_before */
  {
      const ai_ptr _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 47456);
    ai_ptr _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 84320);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(329, 1, {(stai_ptr) _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_cls_branch_blocks_0_act_1_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2048), (ai_i32)(2176), (ai_i32)(2176), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(329, 1, {(stai_ptr) _cls_branch_blocks_0_act_1_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _cls_branch_blocks_0_act_1_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _cls_branch_blocks_0_act_1_Clip_output_0 */
  {
      const ai_i8* _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 84320);
    const ai_i8* _cls_branch_blocks_0_act_1_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 664452);
    const ai_i32* _cls_branch_blocks_0_act_1_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 701316);
    ai_i8* _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    ai_i16* _cls_branch_blocks_0_act_1_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 400);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(329, 1, {(stai_ptr) _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_ptr_const_s8, _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_w_const_u16, _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_h_const_u16, _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_ch_const_u16, _cls_branch_blocks_0_act_1_Clip_output_0_t_weight_0_ptr_const_s8, _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_ch_const_u16, _cls_branch_blocks_0_act_1_Clip_output_0_t_weight_1_ptr_const_s32, _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_zero_const_s8, _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_zero_const_s8, _cls_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_1_Clip_output_0_t_weight_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_1_Clip_output_0_l_out_ch_format_const_layer_format_type, _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_ptr_s8, _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_w_const_u16, _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _cls_branch_blocks_0_act_1_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(329, 1, {(stai_ptr) _cls_branch_blocks_0_act_1_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _cls_branch_blocks_0_act_1_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN cls16_QuantizeLinear_Input */
  {
    
  forward_lite_conv2d_integer_SSSA_cls16_QuantizeLinear_Input(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cls16_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN _reg_branch_blocks_0_act_1_Clip_output_0_pad_before */
  {
      const ai_ptr _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 47456);
    ai_ptr _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 84320);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(330, 1, {(stai_ptr) _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_reg_branch_blocks_0_act_1_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2048), (ai_i32)(2176), (ai_i32)(2176), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(330, 1, {(stai_ptr) _reg_branch_blocks_0_act_1_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _reg_branch_blocks_0_act_1_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _reg_branch_blocks_0_act_1_Clip_output_0 */
  {
      const ai_i8* _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 84320);
    const ai_i8* _reg_branch_blocks_0_act_1_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 701640);
    const ai_i32* _reg_branch_blocks_0_act_1_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 738504);
    ai_i8* _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    ai_i16* _reg_branch_blocks_0_act_1_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 976);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(330, 1, {(stai_ptr) _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_ptr_const_s8, _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_w_const_u16, _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_h_const_u16, _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_shape_ch_const_u16, _reg_branch_blocks_0_act_1_Clip_output_0_t_weight_0_ptr_const_s8, _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_ch_const_u16, _reg_branch_blocks_0_act_1_Clip_output_0_t_weight_1_ptr_const_s32, _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_zero_const_s8, _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_zero_const_s8, _reg_branch_blocks_0_act_1_Clip_output_0_t_in_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_1_Clip_output_0_t_weight_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_1_Clip_output_0_l_out_ch_format_const_layer_format_type, _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_ptr_s8, _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_w_const_u16, _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _reg_branch_blocks_0_act_1_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(330, 1, {(stai_ptr) _reg_branch_blocks_0_act_1_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _reg_branch_blocks_0_act_1_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN size16_QuantizeLinear_Input */
  {
      const ai_i8* size16_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    const ai_i8* size16_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 738760);
    const ai_i32* size16_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 738888);
    ai_i8* size16_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 976);
    ai_i16* size16_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(337, 1, {(stai_ptr) size16_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(size16_QuantizeLinear_Input_t_in_0_ptr_const_s8, size16_QuantizeLinear_Input_t_in_0_shape_w_const_u16, size16_QuantizeLinear_Input_t_in_0_shape_h_const_u16, size16_QuantizeLinear_Input_l_stride_1_const_u16, size16_QuantizeLinear_Input_l_stride_0_const_u16, size16_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, size16_QuantizeLinear_Input_t_weight_0_ptr_const_s8, size16_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, size16_QuantizeLinear_Input_t_weight_1_ptr_const_s32, size16_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, size16_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, size16_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, size16_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, size16_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, size16_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type, size16_QuantizeLinear_Input_t_out_0_ptr_s8, 1, 16, size16_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(337, 1, {(stai_ptr) size16_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END size16_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN size16_QuantizeLinear_Input_Transpose_5 */
  {
    
  forward_lite_transpose_size16_QuantizeLinear_Input_Transpose_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END size16_QuantizeLinear_Input_Transpose_5 */
  /* LITE_KERNEL_SECTION BEGIN off16_QuantizeLinear_Input */
  {
      const ai_i8* off16_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    const ai_i8* off16_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 738896);
    const ai_i32* off16_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 739024);
    ai_i8* off16_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 976);
    ai_i16* off16_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(336, 1, {(stai_ptr) off16_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(off16_QuantizeLinear_Input_t_in_0_ptr_const_s8, off16_QuantizeLinear_Input_t_in_0_shape_w_const_u16, off16_QuantizeLinear_Input_t_in_0_shape_h_const_u16, off16_QuantizeLinear_Input_l_stride_1_const_u16, off16_QuantizeLinear_Input_l_stride_0_const_u16, off16_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, off16_QuantizeLinear_Input_t_weight_0_ptr_const_s8, off16_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, off16_QuantizeLinear_Input_t_weight_1_ptr_const_s32, off16_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, off16_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, off16_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, off16_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, off16_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, off16_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type, off16_QuantizeLinear_Input_t_out_0_ptr_s8, 1, 16, off16_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(336, 1, {(stai_ptr) off16_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END off16_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN off16_QuantizeLinear_Input_Transpose_4 */
  {
    
  forward_lite_transpose_off16_QuantizeLinear_Input_Transpose_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END off16_QuantizeLinear_Input_Transpose_4 */
  /* LITE_KERNEL_SECTION BEGIN _neck_out3_act_Clip_output_0_pad_before */
  {
      const ai_ptr _neck_out3_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 518400);
    ai_ptr _neck_out3_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 10592);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(297, 1, {(stai_ptr) _neck_out3_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_neck_out3_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _neck_out3_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_neck_out3_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _neck_out3_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _neck_out3_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(4096), (ai_i32)(4224), (ai_i32)(4224), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(297, 1, {(stai_ptr) _neck_out3_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _neck_out3_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _neck_out3_act_Clip_output_0 */
  {
      const ai_i8* _neck_out3_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    const ai_i8* _neck_out3_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 739032);
    const ai_i32* _neck_out3_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 766680);
    ai_i8* _neck_out3_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 171104);
    ai_i16* _neck_out3_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 4432);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(297, 1, {(stai_ptr) _neck_out3_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_neck_out3_act_Clip_output_0_t_in_0_ptr_const_s8, _neck_out3_act_Clip_output_0_t_in_0_shape_w_const_u16, _neck_out3_act_Clip_output_0_t_in_0_shape_h_const_u16, _neck_out3_act_Clip_output_0_t_in_0_shape_ch_const_u16, _neck_out3_act_Clip_output_0_t_weight_0_ptr_const_s8, _neck_out3_act_Clip_output_0_t_out_0_shape_ch_const_u16, _neck_out3_act_Clip_output_0_t_weight_1_ptr_const_s32, _neck_out3_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _neck_out3_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _neck_out3_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _neck_out3_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _neck_out3_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _neck_out3_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _neck_out3_act_Clip_output_0_t_out_0_ptr_s8, _neck_out3_act_Clip_output_0_t_out_0_shape_w_const_u16, _neck_out3_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2688, _neck_out3_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(297, 1, {(stai_ptr) _neck_out3_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _neck_out3_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _stems_0_act_Clip_output_0 */
  {
      const ai_i8* _stems_0_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 171104);
    const ai_i8* _stems_0_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 766872);
    const ai_i32* _stems_0_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 769944);
    ai_i8* _stems_0_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    ai_i16* _stems_0_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 976);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(303, 1, {(stai_ptr) _stems_0_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(_stems_0_act_Clip_output_0_t_in_0_ptr_const_s8, _stems_0_act_Clip_output_0_t_in_0_shape_w_const_u16, _stems_0_act_Clip_output_0_t_in_0_shape_h_const_u16, _stems_0_act_Clip_output_0_l_stride_1_const_u16, _stems_0_act_Clip_output_0_l_stride_0_const_u16, _stems_0_act_Clip_output_0_t_in_0_shape_ch_const_u16, _stems_0_act_Clip_output_0_t_weight_0_ptr_const_s8, _stems_0_act_Clip_output_0_t_out_0_shape_ch_const_u16, _stems_0_act_Clip_output_0_t_weight_1_ptr_const_s32, _stems_0_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _stems_0_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _stems_0_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _stems_0_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _stems_0_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _stems_0_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _stems_0_act_Clip_output_0_t_out_0_ptr_s8, 1, 512, _stems_0_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(303, 1, {(stai_ptr) _stems_0_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _stems_0_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _cls_branch_blocks_0_act_Clip_output_0_pad_before */
  {
      const ai_ptr _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 10592);
    ai_ptr _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 158048);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(309, 1, {(stai_ptr) _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_cls_branch_blocks_0_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(4096), (ai_i32)(4224), (ai_i32)(4224), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(309, 1, {(stai_ptr) _cls_branch_blocks_0_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _cls_branch_blocks_0_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _cls_branch_blocks_0_act_Clip_output_0 */
  {
      const ai_i8* _cls_branch_blocks_0_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 158048);
    const ai_i8* _cls_branch_blocks_0_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 770200);
    const ai_i32* _cls_branch_blocks_0_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 807064);
    ai_i8* _cls_branch_blocks_0_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 318560);
    ai_i16* _cls_branch_blocks_0_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 4432);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(309, 1, {(stai_ptr) _cls_branch_blocks_0_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_cls_branch_blocks_0_act_Clip_output_0_t_in_0_ptr_const_s8, _cls_branch_blocks_0_act_Clip_output_0_t_in_0_shape_w_const_u16, _cls_branch_blocks_0_act_Clip_output_0_t_in_0_shape_h_const_u16, _cls_branch_blocks_0_act_Clip_output_0_t_in_0_shape_ch_const_u16, _cls_branch_blocks_0_act_Clip_output_0_t_weight_0_ptr_const_s8, _cls_branch_blocks_0_act_Clip_output_0_t_out_0_shape_ch_const_u16, _cls_branch_blocks_0_act_Clip_output_0_t_weight_1_ptr_const_s32, _cls_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _cls_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _cls_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _cls_branch_blocks_0_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _cls_branch_blocks_0_act_Clip_output_0_t_out_0_ptr_s8, _cls_branch_blocks_0_act_Clip_output_0_t_out_0_shape_w_const_u16, _cls_branch_blocks_0_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _cls_branch_blocks_0_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(309, 1, {(stai_ptr) _cls_branch_blocks_0_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _cls_branch_blocks_0_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN cls8_QuantizeLinear_Input */
  {
    
  forward_lite_conv2d_integer_SSSA_cls8_QuantizeLinear_Input(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cls8_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN _reg_branch_blocks_0_act_Clip_output_0_pad_before */
  {
      const ai_ptr _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 10592);
    ai_ptr _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 158048);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(310, 1, {(stai_ptr) _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_ptr_const_ptr, _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_reg_branch_blocks_0_act_Clip_output_0_pad_before_v_pad_constant_value_const_s8), _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(4096), (ai_i32)(4224), (ai_i32)(4224), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(310, 1, {(stai_ptr) _reg_branch_blocks_0_act_Clip_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _reg_branch_blocks_0_act_Clip_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _reg_branch_blocks_0_act_Clip_output_0 */
  {
      const ai_i8* _reg_branch_blocks_0_act_Clip_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 158048);
    const ai_i8* _reg_branch_blocks_0_act_Clip_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 807388);
    const ai_i32* _reg_branch_blocks_0_act_Clip_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 844252);
    ai_i8* _reg_branch_blocks_0_act_Clip_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    ai_i16* _reg_branch_blocks_0_act_Clip_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6736);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(310, 1, {(stai_ptr) _reg_branch_blocks_0_act_Clip_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(_reg_branch_blocks_0_act_Clip_output_0_t_in_0_ptr_const_s8, _reg_branch_blocks_0_act_Clip_output_0_t_in_0_shape_w_const_u16, _reg_branch_blocks_0_act_Clip_output_0_t_in_0_shape_h_const_u16, _reg_branch_blocks_0_act_Clip_output_0_t_in_0_shape_ch_const_u16, _reg_branch_blocks_0_act_Clip_output_0_t_weight_0_ptr_const_s8, _reg_branch_blocks_0_act_Clip_output_0_t_out_0_shape_ch_const_u16, _reg_branch_blocks_0_act_Clip_output_0_t_weight_1_ptr_const_s32, _reg_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_zero_const_s8, _reg_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_zero_const_s8, _reg_branch_blocks_0_act_Clip_output_0_t_in_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_Clip_output_0_t_out_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_Clip_output_0_t_weight_0_fmt_scale_const_f32, _reg_branch_blocks_0_act_Clip_output_0_l_out_ch_format_const_layer_format_type, _reg_branch_blocks_0_act_Clip_output_0_t_out_0_ptr_s8, _reg_branch_blocks_0_act_Clip_output_0_t_out_0_shape_w_const_u16, _reg_branch_blocks_0_act_Clip_output_0_t_out_0_shape_h_const_u16, 1, 2816, _reg_branch_blocks_0_act_Clip_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(310, 1, {(stai_ptr) _reg_branch_blocks_0_act_Clip_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _reg_branch_blocks_0_act_Clip_output_0 */
  /* LITE_KERNEL_SECTION BEGIN size8_QuantizeLinear_Input */
  {
      const ai_i8* size8_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    const ai_i8* size8_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 844508);
    const ai_i32* size8_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 844636);
    ai_i8* size8_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 158048);
    ai_i16* size8_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(320, 1, {(stai_ptr) size8_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(size8_QuantizeLinear_Input_t_in_0_ptr_const_s8, size8_QuantizeLinear_Input_t_in_0_shape_w_const_u16, size8_QuantizeLinear_Input_t_in_0_shape_h_const_u16, size8_QuantizeLinear_Input_l_stride_1_const_u16, size8_QuantizeLinear_Input_l_stride_0_const_u16, size8_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, size8_QuantizeLinear_Input_t_weight_0_ptr_const_s8, size8_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, size8_QuantizeLinear_Input_t_weight_1_ptr_const_s32, size8_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, size8_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, size8_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, size8_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, size8_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, size8_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type, size8_QuantizeLinear_Input_t_out_0_ptr_s8, 1, 16, size8_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(320, 1, {(stai_ptr) size8_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END size8_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN size8_QuantizeLinear_Input_Transpose_2 */
  {
    
  forward_lite_transpose_size8_QuantizeLinear_Input_Transpose_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END size8_QuantizeLinear_Input_Transpose_2 */
  /* LITE_KERNEL_SECTION BEGIN off8_QuantizeLinear_Input */
  {
      const ai_i8* off8_QuantizeLinear_Input_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10592);
    const ai_i8* off8_QuantizeLinear_Input_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 844644);
    const ai_i32* off8_QuantizeLinear_Input_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 844772);
    ai_i8* off8_QuantizeLinear_Input_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 158048);
    ai_i16* off8_QuantizeLinear_Input_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NIRDET_EVENT_NODE_START_CB(319, 1, {(stai_ptr) off8_QuantizeLinear_Input_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(off8_QuantizeLinear_Input_t_in_0_ptr_const_s8, off8_QuantizeLinear_Input_t_in_0_shape_w_const_u16, off8_QuantizeLinear_Input_t_in_0_shape_h_const_u16, off8_QuantizeLinear_Input_l_stride_1_const_u16, off8_QuantizeLinear_Input_l_stride_0_const_u16, off8_QuantizeLinear_Input_t_in_0_shape_ch_const_u16, off8_QuantizeLinear_Input_t_weight_0_ptr_const_s8, off8_QuantizeLinear_Input_t_out_0_shape_ch_const_u16, off8_QuantizeLinear_Input_t_weight_1_ptr_const_s32, off8_QuantizeLinear_Input_t_in_0_fmt_zero_const_s8, off8_QuantizeLinear_Input_t_out_0_fmt_zero_const_s8, off8_QuantizeLinear_Input_t_in_0_fmt_scale_const_f32, off8_QuantizeLinear_Input_t_out_0_fmt_scale_const_f32, off8_QuantizeLinear_Input_t_weight_0_fmt_scale_const_f32, off8_QuantizeLinear_Input_l_out_ch_format_const_layer_format_type, off8_QuantizeLinear_Input_t_out_0_ptr_s8, 1, 16, off8_QuantizeLinear_Input_t_scratch_0_ptr_s16);
    
  _STAI_NIRDET_EVENT_NODE_STOP_CB(319, 1, {(stai_ptr) off8_QuantizeLinear_Input_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END off8_QuantizeLinear_Input */
  /* LITE_KERNEL_SECTION BEGIN off8_QuantizeLinear_Input_Transpose_1 */
  {
    
  forward_lite_transpose_off8_QuantizeLinear_Input_Transpose_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END off8_QuantizeLinear_Input_Transpose_1 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_nirdet_get_context_size()
{
  return (stai_size)STAI_NIRDET_CONTEXT_SIZE;
}

#if defined(HAVE_NIRDET_INFO)
STAI_API_ENTRY
stai_return_code stai_nirdet_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_nirdet_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_nirdet_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NIRDET_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NIRDET_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NIRDET_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NIRDET_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NIRDET_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NIRDET_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NIRDET_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NIRDET_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NIRDET_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_nirdet_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NIRDET_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_nirdet_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NIRDET_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NIRDET_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 808704;

  net_ctx->_outputs[0] = activations[0] + 4432;

  net_ctx->_outputs[1] = activations[0] + 10592;

  net_ctx->_outputs[2] = activations[0] + 162656;

  net_ctx->_outputs[3] = activations[0] + 400;

  net_ctx->_outputs[4] = activations[0] + 3280;

  net_ctx->_outputs[5] = activations[0] + 2128;

  net_ctx->_outputs[6] = activations[0] + 256;

  net_ctx->_outputs[7] = activations[0] + 10304;

  net_ctx->_outputs[8] = activations[0] + 10016;
_stai_nirdet_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NIRDET_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NIRDET_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NIRDET_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_nirdet_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NIRDET_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NIRDET_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NIRDET_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_nirdet_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NIRDET_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NIRDET_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_nirdet_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_nirdet_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_nirdet_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_nirdet_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NIRDET_EVENT_NODE_START_CB
#undef _STAI_NIRDET_EVENT_NODE_STOP_CB
#undef _STAI_NIRDET_MODEL_SIGNATURE
#undef _STAI_NIRDET_DATETIME
#undef _STAI_NIRDET_COMPILE_DATETIME

