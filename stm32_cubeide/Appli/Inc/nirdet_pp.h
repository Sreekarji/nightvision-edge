/*
 * nirdet_pp.h — NIRDet-Forge post-processor public interface
 * ===========================================================
 * Contract stamp: ed9bd4eb8e4f1922
 *
 * Include this header in any translation unit that calls nirdet_postprocess().
 * The implementation lives entirely in nirdet_pp.c — no inline functions here.
 */

#ifndef NIRDET_PP_H
#define NIRDET_PP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------- *
 * Compile-time limits — must match nirdet_pp.c
 * ---------------------------------------------------------------------- */
#define NIRDET_NUM_CLASSES   1
#define NIRDET_MAX_LEVELS    3
#define NIRDET_MAX_DET       300

/* ---------------------------------------------------------------------- *
 * NirdetBox — one detected person in canvas pixels
 * ---------------------------------------------------------------------- */
typedef struct {
    float x1, y1, x2, y2;   /* canvas pixels, clamped to image bounds */
    float score;             /* sigmoid(logit) in [0, 1] */
} NirdetBox;

/* ---------------------------------------------------------------------- *
 * NirdetLevelCfg — per-FPN-level quantisation and grid parameters
 *
 * cls  : INT8 CHW blob, 1  channel,  grid_h * grid_w elements
 * off  : INT8 CHW blob, 2  channels, 2 * grid_h * grid_w elements
 *         channel 0 = t_cx, channel 1 = t_cy
 * size : INT8 CHW blob, 2  channels, 2 * grid_h * grid_w elements
 *         channel 0 = t_w,  channel 1 = t_h
 *
 * Channel index 1 is at blob_ptr + grid_h * grid_w (no padding).
 * Buffers must be valid for the lifetime of nirdet_postprocess().
 * ---------------------------------------------------------------------- */
typedef struct {
    const int8_t *cls;
    const int8_t *off;
    const int8_t *size;

    float   cls_scale;   int32_t cls_zp;
    float   off_scale;   int32_t off_zp;
    float   size_scale;  int32_t size_zp;

    int32_t grid_h;
    int32_t grid_w;
    int32_t stride;
} NirdetLevelCfg;

/* ---------------------------------------------------------------------- *
 * NirdetCfg — top-level post-processor configuration
 * ---------------------------------------------------------------------- */
typedef struct {
    NirdetLevelCfg levels[NIRDET_MAX_LEVELS];
    int32_t n_levels;        /* 1 .. NIRDET_MAX_LEVELS         */
    int32_t img_h;           /* canvas height in pixels        */
    int32_t img_w;           /* canvas width  in pixels        */
    float   score_thresh;    /* pre-NMS score gate             */
    float   iou_thresh;      /* NMS IoU threshold              */
    int32_t max_det;         /* post-NMS cap, <= NIRDET_MAX_DET */
} NirdetCfg;

/* ---------------------------------------------------------------------- *
 * nirdet_postprocess — decode INT8 blobs, run NMS, return box count
 *
 * cfg      : fully populated NirdetCfg (all level blobs must be valid)
 * out      : caller-allocated array of NirdetBox, capacity >= NIRDET_MAX_DET
 * out_cap  : must be >= NIRDET_MAX_DET (NMS runs over full candidate pool)
 *
 * Returns : number of detections written to out[] (0 .. cfg->max_det)
 *           or -1 on any argument error.
 *
 * NOT REENTRANT: uses two internal static buffers (heap + NMS suppression).
 * ---------------------------------------------------------------------- */
int32_t nirdet_postprocess(const NirdetCfg *cfg, NirdetBox *out,
                           int32_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* NIRDET_PP_H */
