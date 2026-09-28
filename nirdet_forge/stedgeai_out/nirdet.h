/**
  ******************************************************************************
  * @file    nirdet.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-27T13:59:54+0530
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
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
#ifndef STAI_NIRDET_H
#define STAI_NIRDET_H

#include "stai.h"
#include "stai_debug.h"

/*****************************************************************************/
/*  Original model name and signature  */
#define STAI_NIRDET_ORIGIN_MODEL_NAME         "nirdetint8qdq"
#define STAI_NIRDET_ORIGIN_MODEL_SIGNATURE    "0x8003b4db0638478ea8b8c5154e9dca25"

/*  Generated model name and signature  */
#define STAI_NIRDET_MODEL_NAME                "nirdet"
#define STAI_NIRDET_MODEL_SIGNATURE           (0xd541b674e0189ab3)


/*****************************************************************************/
/*  Macros to declare network context size and required alignment  */
#define STAI_NIRDET_CONTEXT_ALIGNMENT   8
#define STAI_NIRDET_CONTEXT_SIZE        (sizeof(_stai_nirdet_context))

/*  Macro to declare network flags  */
#define STAI_NIRDET_FLAGS               (STAI_FLAG_INPUTS|STAI_FLAG_OUTPUTS|STAI_FLAG_WEIGHTS)

/*  Macro to declare number of network C nodes generated  */
#define STAI_NIRDET_NODES_NUM           (110)

/*  Macro to declare number macc for generated network model  */
#define STAI_NIRDET_MACC_NUM            (859904270)

/*  Macros to declare input buffers characteristics  */
#define STAI_NIRDET_IN_NUM              (1)
#define STAI_NIRDET_IN_SIZE_BYTES       (147460)

#define STAI_NIRDET_IN_ALIGNMENTS \
  { 4}
#define STAI_NIRDET_IN_NAMES \
  { NULL}
#define STAI_NIRDET_IN_FORMATS \
  { STAI_FORMAT_S8}
#define STAI_NIRDET_IN_SIZES \
  {147456}
#define STAI_NIRDET_IN_SIZES_BYTES \
  {147456}

#define STAI_NIRDET_IN_1_ALIGNMENT   4
#define STAI_NIRDET_IN_1_NAME        (NULL)
#define STAI_NIRDET_IN_1_FLAGS       (STAI_FLAG_INPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_IN_1_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_IN_1_SHAPE       {1,1,288,512}
#define STAI_NIRDET_IN_1_BATCH       (1)
#define STAI_NIRDET_IN_1_CHANNEL     (1)
#define STAI_NIRDET_IN_1_HEIGHT      (288)
#define STAI_NIRDET_IN_1_WIDTH       (512)
#define STAI_NIRDET_IN_1_RANK        (3)
#define STAI_NIRDET_IN_1_SIZE        (147456)
#define STAI_NIRDET_IN_1_SIZE_BYTES  (147456)
#define STAI_NIRDET_IN_1_SCALE       (0.003921568859368563f)
#define STAI_NIRDET_IN_1_ZERO_POINT  (-128)

/*****************************************************************************/
/*  Macros to declare output buffers characteristics  */
#define STAI_NIRDET_OUT_NUM             (9)
#define STAI_NIRDET_OUT_SIZE_BYTES      (15124)
#define STAI_NIRDET_OUT_ALIGNMENTS \
  { 4, 4, 4, 4, 4, 4, 4, 4, 4}
#define STAI_NIRDET_OUT_NAMES \
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}
#define STAI_NIRDET_OUT_FORMATS \
  { STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8, STAI_FORMAT_S8}
#define STAI_NIRDET_OUT_SIZES \
  { 2304, 4608, 4608, 576, 1152, 1152, 144, 288, 288}
#define STAI_NIRDET_OUT_SIZES_BYTES \
  {2304,4608,4608,576,1152,1152,144,288,288}

#define STAI_NIRDET_OUT_1_ALIGNMENT   4
#define STAI_NIRDET_OUT_1_NAME        (NULL)
#define STAI_NIRDET_OUT_1_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_1_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_1_SHAPE       {1,1,36,64}
#define STAI_NIRDET_OUT_1_BATCH       (1)
#define STAI_NIRDET_OUT_1_CHANNEL     (1)
#define STAI_NIRDET_OUT_1_HEIGHT      (36)
#define STAI_NIRDET_OUT_1_WIDTH       (64)
#define STAI_NIRDET_OUT_1_RANK        (3)
#define STAI_NIRDET_OUT_1_SIZE        ((36 * 64 * 1))
#define STAI_NIRDET_OUT_1_SIZE_BYTES  (2304)
#define STAI_NIRDET_OUT_1_SCALE       (0.055070631206035614f)
#define STAI_NIRDET_OUT_1_ZERO_POINT  (81)

#define STAI_NIRDET_OUT_2_ALIGNMENT   4
#define STAI_NIRDET_OUT_2_NAME        (NULL)
#define STAI_NIRDET_OUT_2_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_2_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_2_SHAPE       {1,2,36,64}
#define STAI_NIRDET_OUT_2_BATCH       (1)
#define STAI_NIRDET_OUT_2_CHANNEL     (2)
#define STAI_NIRDET_OUT_2_HEIGHT      (36)
#define STAI_NIRDET_OUT_2_WIDTH       (64)
#define STAI_NIRDET_OUT_2_RANK        (3)
#define STAI_NIRDET_OUT_2_SIZE        ((64 * 2 * 36))
#define STAI_NIRDET_OUT_2_SIZE_BYTES  (4608)
#define STAI_NIRDET_OUT_2_SCALE       (0.18284200131893158f)
#define STAI_NIRDET_OUT_2_ZERO_POINT  (2)

#define STAI_NIRDET_OUT_3_ALIGNMENT   4
#define STAI_NIRDET_OUT_3_NAME        (NULL)
#define STAI_NIRDET_OUT_3_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_3_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_3_SHAPE       {1,2,36,64}
#define STAI_NIRDET_OUT_3_BATCH       (1)
#define STAI_NIRDET_OUT_3_CHANNEL     (2)
#define STAI_NIRDET_OUT_3_HEIGHT      (36)
#define STAI_NIRDET_OUT_3_WIDTH       (64)
#define STAI_NIRDET_OUT_3_RANK        (3)
#define STAI_NIRDET_OUT_3_SIZE        ((64 * 2 * 36))
#define STAI_NIRDET_OUT_3_SIZE_BYTES  (4608)
#define STAI_NIRDET_OUT_3_SCALE       (0.017869222909212112f)
#define STAI_NIRDET_OUT_3_ZERO_POINT  (119)

#define STAI_NIRDET_OUT_4_ALIGNMENT   4
#define STAI_NIRDET_OUT_4_NAME        (NULL)
#define STAI_NIRDET_OUT_4_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_4_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_4_SHAPE       {1,1,18,32}
#define STAI_NIRDET_OUT_4_BATCH       (1)
#define STAI_NIRDET_OUT_4_CHANNEL     (1)
#define STAI_NIRDET_OUT_4_HEIGHT      (18)
#define STAI_NIRDET_OUT_4_WIDTH       (32)
#define STAI_NIRDET_OUT_4_RANK        (3)
#define STAI_NIRDET_OUT_4_SIZE        ((18 * 32 * 1))
#define STAI_NIRDET_OUT_4_SIZE_BYTES  (576)
#define STAI_NIRDET_OUT_4_SCALE       (0.044113121926784515f)
#define STAI_NIRDET_OUT_4_ZERO_POINT  (75)

#define STAI_NIRDET_OUT_5_ALIGNMENT   4
#define STAI_NIRDET_OUT_5_NAME        (NULL)
#define STAI_NIRDET_OUT_5_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_5_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_5_SHAPE       {1,2,18,32}
#define STAI_NIRDET_OUT_5_BATCH       (1)
#define STAI_NIRDET_OUT_5_CHANNEL     (2)
#define STAI_NIRDET_OUT_5_HEIGHT      (18)
#define STAI_NIRDET_OUT_5_WIDTH       (32)
#define STAI_NIRDET_OUT_5_RANK        (3)
#define STAI_NIRDET_OUT_5_SIZE        ((32 * 2 * 18))
#define STAI_NIRDET_OUT_5_SIZE_BYTES  (1152)
#define STAI_NIRDET_OUT_5_SCALE       (0.14666779339313507f)
#define STAI_NIRDET_OUT_5_ZERO_POINT  (-4)

#define STAI_NIRDET_OUT_6_ALIGNMENT   4
#define STAI_NIRDET_OUT_6_NAME        (NULL)
#define STAI_NIRDET_OUT_6_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_6_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_6_SHAPE       {1,2,18,32}
#define STAI_NIRDET_OUT_6_BATCH       (1)
#define STAI_NIRDET_OUT_6_CHANNEL     (2)
#define STAI_NIRDET_OUT_6_HEIGHT      (18)
#define STAI_NIRDET_OUT_6_WIDTH       (32)
#define STAI_NIRDET_OUT_6_RANK        (3)
#define STAI_NIRDET_OUT_6_SIZE        ((32 * 2 * 18))
#define STAI_NIRDET_OUT_6_SIZE_BYTES  (1152)
#define STAI_NIRDET_OUT_6_SCALE       (0.016024518758058548f)
#define STAI_NIRDET_OUT_6_ZERO_POINT  (124)

#define STAI_NIRDET_OUT_7_ALIGNMENT   4
#define STAI_NIRDET_OUT_7_NAME        (NULL)
#define STAI_NIRDET_OUT_7_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_7_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_7_SHAPE       {1,1,9,16}
#define STAI_NIRDET_OUT_7_BATCH       (1)
#define STAI_NIRDET_OUT_7_CHANNEL     (1)
#define STAI_NIRDET_OUT_7_HEIGHT      (9)
#define STAI_NIRDET_OUT_7_WIDTH       (16)
#define STAI_NIRDET_OUT_7_RANK        (3)
#define STAI_NIRDET_OUT_7_SIZE        ((9 * 16 * 1))
#define STAI_NIRDET_OUT_7_SIZE_BYTES  (144)
#define STAI_NIRDET_OUT_7_SCALE       (0.02662220038473606f)
#define STAI_NIRDET_OUT_7_ZERO_POINT  (127)

#define STAI_NIRDET_OUT_8_ALIGNMENT   4
#define STAI_NIRDET_OUT_8_NAME        (NULL)
#define STAI_NIRDET_OUT_8_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_8_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_8_SHAPE       {1,2,9,16}
#define STAI_NIRDET_OUT_8_BATCH       (1)
#define STAI_NIRDET_OUT_8_CHANNEL     (2)
#define STAI_NIRDET_OUT_8_HEIGHT      (9)
#define STAI_NIRDET_OUT_8_WIDTH       (16)
#define STAI_NIRDET_OUT_8_RANK        (3)
#define STAI_NIRDET_OUT_8_SIZE        ((16 * 2 * 9))
#define STAI_NIRDET_OUT_8_SIZE_BYTES  (288)
#define STAI_NIRDET_OUT_8_SCALE       (0.03284941241145134f)
#define STAI_NIRDET_OUT_8_ZERO_POINT  (43)

#define STAI_NIRDET_OUT_9_ALIGNMENT   4
#define STAI_NIRDET_OUT_9_NAME        (NULL)
#define STAI_NIRDET_OUT_9_FLAGS       (STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_HAS_BATCH)
#define STAI_NIRDET_OUT_9_FORMAT      (STAI_FORMAT_S8)
#define STAI_NIRDET_OUT_9_SHAPE       {1,2,9,16}
#define STAI_NIRDET_OUT_9_BATCH       (1)
#define STAI_NIRDET_OUT_9_CHANNEL     (2)
#define STAI_NIRDET_OUT_9_HEIGHT      (9)
#define STAI_NIRDET_OUT_9_WIDTH       (16)
#define STAI_NIRDET_OUT_9_RANK        (3)
#define STAI_NIRDET_OUT_9_SIZE        ((16 * 2 * 9))
#define STAI_NIRDET_OUT_9_SIZE_BYTES  (288)
#define STAI_NIRDET_OUT_9_SCALE       (0.015480712987482548f)
#define STAI_NIRDET_OUT_9_ZERO_POINT  (127)

/*****************************************************************************/
/*  Macros to declare activations buffers characteristics  */
#define STAI_NIRDET_ACTIVATIONS_SIZE              (1019584)
#define STAI_NIRDET_ACTIVATIONS_SIZE_BYTES        (1019584)
#define STAI_NIRDET_ACTIVATIONS_ALIGNMENTS \
  { 4}
#define STAI_NIRDET_ACTIVATIONS_SIZES \
  { 1019584}
#define STAI_NIRDET_ACTIVATIONS_NUM               (1)

#define STAI_NIRDET_ACTIVATION_1_FLAGS      (STAI_FLAG_ACTIVATIONS)
#define STAI_NIRDET_ACTIVATION_1_SIZE       (1019584)
#define STAI_NIRDET_ACTIVATION_1_SIZE_BYTES (1019584)
#define STAI_NIRDET_ACTIVATION_1_ALIGNMENT  4
/*  Macros to declare weights buffers characteristics  */

#define STAI_NIRDET_WEIGHTS_NUM                  (1)
#define STAI_NIRDET_WEIGHTS_SIZE                  (844780)
#define STAI_NIRDET_WEIGHTS_SIZE_BYTES            (844780)
#define STAI_NIRDET_WEIGHTS_ALIGNMENTS \
  { 4}
#define STAI_NIRDET_WEIGHTS_SIZES \
  { 844780}
#define STAI_NIRDET_WEIGHTS_NUM        (1)
#define STAI_NIRDET_WEIGHT_1_FLAGS       ((STAI_FLAG_PREALLOCATED))
#define STAI_NIRDET_WEIGHT_1_SIZE        (844780)
#define STAI_NIRDET_WEIGHT_1_SIZE_BYTES  (844780)


/*****************************************************************************/
/*  Macros to declare states buffers characteristics  */
#define STAI_NIRDET_STATES_SIZE              (0)
#define STAI_NIRDET_STATES_SIZE_BYTES        (0)
#define STAI_NIRDET_STATES_ALIGNMENTS        {}
#define STAI_NIRDET_STATES_SIZES             { 0, }
#define STAI_NIRDET_STATES_NUM               (0)

#define STAI_NIRDET_STATE_1_FLAGS            (0x0)
#define STAI_NIRDET_STATE_1_ALIGNMENT        4
#define STAI_NIRDET_STATE_1_SIZE             (0)
#define STAI_NIRDET_STATE_1_SIZE_BYTES       (0)
/*****************************************************************************/
STAI_API_DECLARE_BEGIN

/*****************************************************************************/
/** Network private context: DO NOT EDIT NOR access this struct directly    **/
/*****************************************************************************/
/**
 * @brief Private internal context C struct definition
 * @details this is the C struct defining the internal context of the generated network.
 *          this struct must be allocated by applications as a byte array of a given size,
 *          but MUST not be accessed directly by application. It must be manipulated / queried only
 *          using st.ai APIs.
 */
typedef struct {
  uint32_t                     _magic;
  uint64_t                     _signature;
  stai_flags                   _flags;
  stai_return_code             _return_code;
  stai_event_cb                _callback;
  void*                        _callback_cookie;
  stai_ptr                     _activations[STAI_NIRDET_ACTIVATIONS_NUM];
  stai_ptr                     _weights[STAI_NIRDET_WEIGHTS_NUM];
  stai_ptr                     _inputs[STAI_NIRDET_IN_NUM];
  stai_ptr                     _outputs[STAI_NIRDET_OUT_NUM];
} _stai_nirdet_context;


/*****************************************************************************/
/*  Network run management APIs Section  */

/**
 * @brief Initialize the network internal context
 * @details The application must provide the memory area where to initialize the internal network context as an opaque
 *          handler named network.
 *          Application must never access directly the directly the context but it must be accessed only with
 *          associates stai APIs. Application knows network context size by using macro
 *          @ref STAI_NIRDET_CONTEXT_SIZE or @ref stai_nirdet_get_context_size(void) API
 * @param[in] network network context opaque pointer
 * @return STAI_SUCCESS on correct initialization, else 1st error return code from @ref stai_return_code enums.
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_init(
  stai_network* network);


/**
 * @brief Deinitialize the network internal context
 * @details The application must provide the memory area where to initialize the internal network context as an opaque
 *          handler named netwrok.
 *          Application must never access directly the directly the context but it must be accessed only with
 *          associates stai APIs.
 * @param[in] network network context opaque pointer
 * @return STAI_SUCCESS on correct de-initialization, else 1st error return code from @ref stai_return_code enums.
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_deinit(
  stai_network* network);


/**
 * @brief Runs the network model inference on a single input sample
 * @details This API runs a newtwork inference. Network input data is expected to be available in expected format
 *          on the input buffers whose memory adresses are retrived using @ref stai_nirdet_get_inputs() API.
 *          Alternatively the application could allocate the inputs and set the addresses using the
 *          @ref stai_nirdet_set_inputs() API. Similar behaviour is expected for the network outputs handling.
 *          The features of each input / output buffer (format, byte size, etc.)may be queried using
 *          @ref stai_nirdet_get_info() API or the set of specific macros available in this header file.
 *          When mode is MODE_SYNC the call to this API is blocking untile the end of the inference. In ASYNC_MODE the
 *          API will exit before the completion of the inference by returning the current state of the inference. It
 *          could be also used to query wether or not the inference has been completed.
 *          After completions of run API the results of the network inference may be retrieved at addresses returned by
 *          @ref stai_nirdet_get_outputs() API.
 * @param[in] network network context opaque pointer
 * @param[in] mode the modality used to run the network. It may be synchronous (MODE_SYNC) or asyncronous (MODE_ASYNC).
 *            see @ref stai_run_mode enum
 * @return STAI_SUCCESS on correct run, else error 1st return code from @ref stai_return_code enums.
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_run(
  stai_network* network,
  const stai_run_mode mode);


/*****************************************************************************/
/*  Getters APIs Section  */

/**
 * @brief Returns the newtwork context size (in bytes)
 * @details This API is used to return the size in byte of the network internal context. The st.ai APIs could support
 *          multiple network contexts, thus tt is expected that the network context is always allocated by application
 *          but MUST be handled / queried / managed only by using st.ai APIs
 * @return the network context size in bytes
 */
STAI_API_ENTRY
stai_size stai_nirdet_get_context_size(void);


/**
 * @brief Get the relevant info for the C generated network model.
 * @details See @ref stai_network_info C struct.
 * @param[in] network network context opaque pointer
 * @param[out] info a pointer to a @ref stai_network_info data struct
 * @return STAI_SUCCESS on correct run, else error 1st return code from @ref stai_return_code enums.
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_info(
  stai_network* network,
  stai_network_info* info);


/**
 * @brief Get the activations buffers addresses.
 * @details as an array of pointers of size n_activations
 * @param[in] network network context opaque pointer
 * @param[out] activations an array of pointers reporting the addresses of all the activation buffers
 * @param[out] n_activations the number of the network activations (i.e. the size of the activations array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note activations array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_activations(
  stai_network* network,
  stai_ptr* activations,
  stai_size* n_activations);


/**
 * @brief Get the weights buffers addresses.
 * @details as an array of pointers of size n_weights
 * @param[in] network network context opaque pointer
 * @param[out] weights an array of pointers reporting the addresses of all the weights buffers
 * @param[out] n_weights the number of the network weights (i.e. the size of the weights array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note weights array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_weights(
  stai_network* network,
  stai_ptr* weights,
  stai_size* n_weights);


/**
 * @brief Get the network inputs buffers addresses.
 * @details as an array of pointers of size n_inputs
 * @param[in] network network context opaque pointer
 * @param[out] inputs an array of pointers reporting the addresses of all the network inputs buffers
 * @param[out] n_inputs the number of the network inputs (i.e. the size of the inputs array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note inputs array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_inputs(
  stai_network* network,
  stai_ptr* inputs,
  stai_size* n_inputs);


/**
 * @brief Get the network outputs buffers addresses.
 * @details as an array of pointers of size n_outputs
 * @param[in] network network context opaque pointer
 * @param[out] outputs an array of pointers reporting the addresses of all the network outputs buffers
 * @param[out] n_outputs the number of the network outputs (i.e. the size of the outputs array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note outputs array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_outputs(
  stai_network* network,
  stai_ptr* outputs,
  stai_size* n_outputs);


/**
 * @brief Get the network internal states buffers addresses.
 * @details as an array of pointers of size n_states. Optional API for stateful newtorks. It gets the
 *          addresses of persistent internal states of the network. This APIs allows an application to manage
 *          stateful buffers (e.g. to save them). State buffers may be set back using
 *          @ref stai_nirdet_set_states() dual API
 * @param[in] network network context opaque pointer
 * @param[out] states an array of pointers reporting the addresses of all the network internal states buffers
 * @param[out] n_states the number of the network internal states (i.e. the size of the states array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note outputs array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_states(
  stai_network* network,
  stai_ptr* states,
  stai_size* n_states);


/**
 * @brief Get the newtork error state.
 * @details In particular it return STAI_SUCCESS if no errors encountered during APIs orchestration, otherwise this
 *          API return the 1st error generated.
 * @param[in] network network context opaque pointer
 * @return STAI_SUCCESS if no errors, else the 1st error generated.
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_get_error(
  stai_network* network);


/*****************************************************************************/
/*  Setters APIS Section  */

/**
 * @brief Set the activations buffers addresses.
 * @details activations buffers are provides as as an array of pointers of size n_activations
 * @param[in] network network context opaque pointer
 * @param[in] activations an array of pointers reporting the addresses of all the activation buffers
 * @param[in] n_activations the number of the network activations (i.e. the size of the activations array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note activations array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations);


/**
 * @brief Set the weights buffers addresses.
 * @details weights buffers are provides as as an array of pointers of size n_weights
 * @param[in] network network context opaque pointer
 * @param[in] weights an array of pointers reporting the addresses of all the weights buffers
 * @param[in] n_weights the number of the network weights (i.e. the size of the weights array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note weights array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights);


/**
 * @brief Set the network inputs buffers addresses.
 * @details network inputs buffers are provides as as an array of pointers of size n_inputs
 * @param[in] network network context opaque pointer
 * @param[in] inputs an array of pointers reporting the addresses of all the network inputs buffers
 * @param[in] n_inputs the number of the network inputs (i.e. the size of the inputs array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note inputs array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs);


/**
 * @brief Set the network outputs buffers addresses.
 * @details network outputs buffers are provides as as an array of pointers of size n_inputs
 * @param[in] network network context opaque pointer
 * @param[in] outputs an array of pointers reporting the addresses of all the network outputs buffers
 * @param[in] n_outputs the number of the network outputs (i.e. the size of the outputs array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note outputs array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs);


/**
 * @brief Set the network internal states buffers addresses.
 * @details network states buffers are provides as as an array of pointers of size n_states
 * @param[in] network network context opaque pointer
 * @param[in] states an array of pointers reporting the addresses of all the network internal states buffers
 * @param[in] n_states the number of the network states (i.e. the size of the states array)
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 * @note states array pointers are passed by copy
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states);


/**
 * @brief Set a callback function to inspect network activation buffers.
 * @details this is an API to retrieve the content on intermediate activations buffers while executing the
 *          @ref stai_nirdet_run() API. It is a helper routine that could be used by an application for debug
 *          and / or benchmarking intermadiate results of the network.
 * @param[in] network network context opaque pointer
 * @param[in] cb a function pointer to the inspection callback routine. See @ref stai_event_cb function pointer for
 *            a definition of the callback API.
 * @param[in] cb_cookie an opaque pointer to an application memory buffer and / or variable.
 * @return STAI_SUCCESS if successfull, else error 1st return code from @ref stai_return_code enums.
 */
STAI_API_ENTRY
stai_return_code stai_nirdet_set_callback(
  stai_network* network,
  const stai_event_cb cb,
  void* cb_cookie);



STAI_API_DECLARE_END

#endif    /* STAI_NIRDET_H */

