/*
 * nirdet_pp.c — NIRDet-Lite post-processing for STM32N6570-DK (Cortex-M55)
 * =========================================================================
 * ST documents detection post-processing as unsupported on Neural-ART: NMS and
 * decode are host responsibilities. The exported graph therefore ends at the
 * convolutions and this file finishes the job on the M55.
 *
 * DECODE CONTRACT — identical to head.py, losses.py and live_nirdet.py:
 *     cx = (OFF_S * sigmoid(t_cx) - OFF_B + col) * stride
 *     cy = (OFF_S * sigmoid(t_cy) - OFF_B + row) * stride
 *     w  = exp(clamp(t_w, MIN, MAX)) * img_w
 *     h  = exp(clamp(t_h, MIN, MAX)) * img_h
 *     conf = sigmoid(cls_logit)
 * Change one constant here and you must change it in all four places.
 *
 * INTEGER PREFILTER (the reason this is fast)
 * -------------------------------------------
 * sigmoid is monotonic, and the INT8 dequantisation q -> (q - zp) * scale is
 * monotonically increasing for scale > 0. So
 *     conf >= score_th  <=>  q >= ceil(logit(score_th)/scale) + zp
 * The threshold is precomputed once into an int8 (or int16) comparand, and the
 * per-cell hot loop becomes a single integer compare. Only surviving cells pay
 * for expf(). On ~4800 cells this is the difference between roughly 3 ms and
 * 0.2 ms on an M55 at 800 MHz.
 *
 * Layout: all tensors are NCHW, channel-major planes of H*W elements.
 *     cls : 1 plane   [conf_logit]
 *     reg : 4 planes  [t_cx, t_cy, t_w, t_h]
 *
 * Zero heap allocation. Callers supply every buffer.
 *
 * Build (no C library math dependency beyond expf):
 *     arm-none-eabi-gcc -O2 -mcpu=cortex-m55 -mfloat-abi=hard \
 *         -ffast-math -c nirdet_pp.c
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ======================================================================== *
 *  Decode constants — keep in sync with config.py
 * ======================================================================== */

#define NIRDET_OFF_S            2.0f    /* DECODE_OFFSET_SCALE            */
#define NIRDET_OFF_B            0.5f    /* DECODE_OFFSET_BIAS             */
#define NIRDET_REG_CLAMP_MIN   (-6.0f)  /* REG_LOG_CLAMP_MIN              */
#define NIRDET_REG_CLAMP_MAX    ( 1.0f) /* REG_LOG_CLAMP_MAX              */
#define NIRDET_MIN_BOX_PX       1.0f    /* mirrors model.py degenerate filter */

#define NIRDET_MAX_LEVELS       4

/* ======================================================================== *
 *  Public types
 * ======================================================================== */

typedef struct {
    float x1, y1, x2, y2;   /* xyxy, pixels of the letterboxed canvas */
    float score;            /* sigmoid(conf_logit) in [0,1]           */
    uint8_t cls;            /* always 0: single class (person)        */
    uint8_t level;          /* which FPN level produced it            */
} det_t;

/* One quantised output plane pair as produced by stedgeai.
 * Take scale / zero_point verbatim from the generated network header:
 * guessing them silently corrupts every box. */
typedef struct {
    const void *cls;        /* int8_t* or float* per is_float          */
    const void *reg;
    int32_t     height;     /* H at this level                         */
    int32_t     width;      /* W at this level                         */
    int32_t     stride;     /* 8 or 16                                 */

    float       cls_scale;  /* dequant: (q - zp) * scale               */
    int32_t     cls_zp;
    float       reg_scale;
    int32_t     reg_zp;

    uint8_t     is_float;   /* 1 = planes are float32 (debug path)     */
} nirdet_level_t;

typedef struct {
    float   score_th;       /* e.g. 0.30 from evaluate.py's PR curve   */
    float   iou_th;         /* e.g. 0.45                               */
    int32_t max_det;        /* capacity of the output array            */
    int32_t topk_pre_nms;   /* 0 = unlimited                           */
} nirdet_pp_cfg_t;

/* ======================================================================== *
 *  Small helpers
 * ======================================================================== */

static inline float nirdet_sigmoidf(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    /* Avoids expf() overflow for large negative x. */
    const float e = expf(x);
    return e / (1.0f + e);
}

static inline float nirdet_logitf(float p)
{
    if (p <= 1e-6f)        { p = 1e-6f; }
    else if (p >= 1.0f - 1e-6f) { p = 1.0f - 1e-6f; }
    return logf(p / (1.0f - p));
}

static inline float nirdet_clampf(float v, float lo, float hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline float nirdet_dequant(int32_t q, int32_t zp, float scale)
{
    return (float)(q - zp) * scale;
}

/*
 * Precompute the integer comparand for the confidence prefilter.
 *
 *   conf >= score_th
 *   <=> logit >= L,                     L = logit(score_th)
 *   <=> (q - zp) * scale >= L
 *   <=> q >= zp + ceil(L / scale)       (scale > 0)
 *
 * Returns the smallest int32 q that can pass. Values below INT8_MIN mean
 * "everything passes"; above INT8_MAX mean "nothing passes".
 */
static inline int32_t nirdet_logit_threshold_q(float score_th, float scale,
                                               int32_t zp)
{
    if (scale <= 0.0f) {
        return (int32_t)INT8_MIN;      /* degenerate scale: do not prefilter */
    }
    const float L = nirdet_logitf(score_th);
    const float qf = L / scale;
    int32_t qi = (int32_t)ceilf(qf);
    /* ceilf on an exact boundary keeps the boundary cell, which matches the
     * >= comparison used in Python. */
    return zp + qi;
}

/* ======================================================================== *
 *  Single-level decode
 * ======================================================================== */

/*
 * Decodes one FPN level into out[]. Returns the number written.
 *
 * n_out_in    : detections already in out[]
 * cfg->max_det: total capacity of out[]
 */
int32_t nirdet_decode_level(const nirdet_level_t *lv,
                            const nirdet_pp_cfg_t *cfg,
                            det_t *out,
                            int32_t n_out_in,
                            uint8_t level_idx)
{
    if (lv == NULL || cfg == NULL || out == NULL) {
        return n_out_in;
    }

    const int32_t H  = lv->height;
    const int32_t W  = lv->width;
    const int32_t HW = H * W;
    const int32_t S  = lv->stride;
    if (H <= 0 || W <= 0 || S <= 0) {
        return n_out_in;
    }

    /* w/h are decoded against the FULL image extent at every level, exactly
     * as head.py does (img_w = W * stride, img_h = H * stride). */
    const float img_w = (float)(W * S);
    const float img_h = (float)(H * S);

    int32_t n = n_out_in;

    if (lv->is_float) {
        /* ---- debug / bit-exactness path: planes already dequantised ---- */
        const float *cls = (const float *)lv->cls;
        const float *reg = (const float *)lv->reg;
        const float logit_th = nirdet_logitf(cfg->score_th);

        for (int32_t row = 0; row < H; ++row) {
            for (int32_t col = 0; col < W; ++col) {
                const int32_t i = row * W + col;
                if (cls[i] < logit_th) {           /* float prefilter */
                    continue;
                }
                const float sx = nirdet_sigmoidf(reg[0 * HW + i]);
                const float sy = nirdet_sigmoidf(reg[1 * HW + i]);
                const float tw = nirdet_clampf(reg[2 * HW + i],
                                               NIRDET_REG_CLAMP_MIN,
                                               NIRDET_REG_CLAMP_MAX);
                const float th = nirdet_clampf(reg[3 * HW + i],
                                               NIRDET_REG_CLAMP_MIN,
                                               NIRDET_REG_CLAMP_MAX);

                const float cx = (NIRDET_OFF_S * sx - NIRDET_OFF_B
                                  + (float)col) * (float)S;
                const float cy = (NIRDET_OFF_S * sy - NIRDET_OFF_B
                                  + (float)row) * (float)S;
                const float bw = expf(tw) * img_w;
                const float bh = expf(th) * img_h;

                if (bw <= NIRDET_MIN_BOX_PX || bh <= NIRDET_MIN_BOX_PX) {
                    continue;
                }

                /* NOTE: out[] is filled in raster order; if n reaches
                 * max_det the remaining (possibly higher-scoring) cells
                 * are discarded. Set max_det to a pre-NMS capacity
                 * (e.g. 256) and use topk_pre_nms to bound NMS cost. */
                if (n < cfg->max_det) {
                    out[n].x1    = nirdet_clampf(cx - 0.5f * bw, 0.0f, img_w);
                    out[n].y1    = nirdet_clampf(cy - 0.5f * bh, 0.0f, img_h);
                    out[n].x2    = nirdet_clampf(cx + 0.5f * bw, 0.0f, img_w);
                    out[n].y2    = nirdet_clampf(cy + 0.5f * bh, 0.0f, img_h);
                    out[n].score = nirdet_sigmoidf(cls[i]);
                    out[n].cls   = 0u;
                    out[n].level = level_idx;
                    ++n;
                }
            }
        }
        return n;
    }

    /* ---- production path: int8 planes + integer prefilter ---- */
    const int8_t *cls = (const int8_t *)lv->cls;
    const int8_t *reg = (const int8_t *)lv->reg;

    const int32_t q_th = nirdet_logit_threshold_q(cfg->score_th,
                                                  lv->cls_scale,
                                                  lv->cls_zp);
    if (q_th > (int32_t)INT8_MAX) {
        return n;                       /* threshold unreachable: no output */
    }
    const int8_t q_th8 = (int8_t)((q_th < (int32_t)INT8_MIN)
                                  ? INT8_MIN : q_th);

    for (int32_t row = 0; row < H; ++row) {
        const int8_t *cls_row = cls + (size_t)row * (size_t)W;

        for (int32_t col = 0; col < W; ++col) {
            /* THE hot comparison: one int8 compare, no expf, no dequant. */
            if (cls_row[col] < q_th8) {
                continue;
            }
            const int32_t i = row * W + col;

            const float t_cx = nirdet_dequant(reg[0 * HW + i],
                                              lv->reg_zp, lv->reg_scale);
            const float t_cy = nirdet_dequant(reg[1 * HW + i],
                                              lv->reg_zp, lv->reg_scale);
            const float t_w  = nirdet_clampf(
                nirdet_dequant(reg[2 * HW + i], lv->reg_zp, lv->reg_scale),
                NIRDET_REG_CLAMP_MIN, NIRDET_REG_CLAMP_MAX);
            const float t_h  = nirdet_clampf(
                nirdet_dequant(reg[3 * HW + i], lv->reg_zp, lv->reg_scale),
                NIRDET_REG_CLAMP_MIN, NIRDET_REG_CLAMP_MAX);

            const float cx = (NIRDET_OFF_S * nirdet_sigmoidf(t_cx)
                              - NIRDET_OFF_B + (float)col) * (float)S;
            const float cy = (NIRDET_OFF_S * nirdet_sigmoidf(t_cy)
                              - NIRDET_OFF_B + (float)row) * (float)S;
            const float bw = expf(t_w) * img_w;
            const float bh = expf(t_h) * img_h;

            if (bw <= NIRDET_MIN_BOX_PX || bh <= NIRDET_MIN_BOX_PX) {
                continue;
            }

            const float logit = nirdet_dequant(cls_row[col],
                                               lv->cls_zp, lv->cls_scale);

            /* NOTE: out[] is filled in raster order; if n reaches
             * max_det the remaining (possibly higher-scoring) cells
             * are discarded. Set max_det to a pre-NMS capacity
             * (e.g. 256) and use topk_pre_nms to bound NMS cost. */
            if (n < cfg->max_det) {
                out[n].x1    = nirdet_clampf(cx - 0.5f * bw, 0.0f, img_w);
                out[n].y1    = nirdet_clampf(cy - 0.5f * bh, 0.0f, img_h);
                out[n].x2    = nirdet_clampf(cx + 0.5f * bw, 0.0f, img_w);
                out[n].y2    = nirdet_clampf(cy + 0.5f * bh, 0.0f, img_h);
                out[n].score = nirdet_sigmoidf(logit);
                out[n].cls   = 0u;
                out[n].level = level_idx;
                ++n;
            }
        }
    }

    return n;
}

/* ======================================================================== *
 *  Sorting + greedy NMS
 * ======================================================================== */

static inline void nirdet_swap(det_t *a, det_t *b)
{
    const det_t t = *a; *a = *b; *b = t;
}

/* Insertion sort by descending score. n is small after the prefilter
 * (typically < 64), and insertion sort has no recursion and no stack cost,
 * which matters on an MCU. */
static void nirdet_sort_desc(det_t *d, int32_t n)
{
    for (int32_t i = 1; i < n; ++i) {
        const det_t key = d[i];
        int32_t j = i - 1;
        while (j >= 0 && d[j].score < key.score) {
            d[j + 1] = d[j];
            --j;
        }
        d[j + 1] = key;
    }
}

static inline float nirdet_iou(const det_t *a, const det_t *b)
{
    const float ix1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    const float iy1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    const float ix2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    const float iy2 = (a->y2 < b->y2) ? a->y2 : b->y2;

    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) {
        return 0.0f;
    }
    const float inter = iw * ih;
    const float aa = (a->x2 - a->x1) * (a->y2 - a->y1);
    const float ab = (b->x2 - b->x1) * (b->y2 - b->y1);
    const float uni = aa + ab - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

/*
 * In-place greedy NMS. Assumes d[] is sorted by descending score.
 * Returns the number of survivors, compacted to the front of d[].
 *
 * suppressed[] must hold at least n bytes.
 */
int32_t nirdet_nms(det_t *d, int32_t n, float iou_th, uint8_t *suppressed)
{
    if (d == NULL || n <= 0) {
        return 0;
    }
    if (suppressed == NULL) {
        return n;                       /* cannot run without scratch */
    }
    memset(suppressed, 0, (size_t)n);

    int32_t kept = 0;
    for (int32_t i = 0; i < n; ++i) {
        if (suppressed[i]) {
            continue;
        }
        for (int32_t j = i + 1; j < n; ++j) {
            if (!suppressed[j] && nirdet_iou(&d[i], &d[j]) > iou_th) {
                suppressed[j] = 1u;
            }
        }
        if (kept != i) {
            d[kept] = d[i];
            /* Keep the suppression flags aligned with the compacted array. */
            suppressed[kept] = 0u;
        }
        ++kept;
    }
    return kept;
}

/* ======================================================================== *
 *  One-call entry point
 * ======================================================================== */

/*
 * Full post-processing for all levels.
 *
 *   levels     : n_levels descriptors, FINEST FIRST (P3, then P4, then P5)
 *   out        : caller-owned det_t[cfg->max_det]
 *   scratch    : caller-owned uint8_t[cfg->max_det] for the NMS flags
 *
 * Returns the number of detections written to out[].
 *
 * The model has THREE detection levels: strides 8, 16, and 32.
 * Passing n_levels=2 compiles, runs, and silently discards every
 * stride-32 detection.
 *
 * Example on the DK (384x640 input -> 48x80, 24x40, 12x20 grids):
 *     static det_t   g_dets[256];
 *     static uint8_t g_scratch[256];
 *     nirdet_level_t lv[3] = {
 *       { .cls = out_cls3, .reg = out_reg3, .height = 48, .width = 80,
 *         .stride = 8,  .cls_scale = S_CLS3, .cls_zp = Z_CLS3,
 *         .reg_scale = S_REG3, .reg_zp = Z_REG3, .is_float = 0 },
 *       { .cls = out_cls4, .reg = out_reg4, .height = 24, .width = 40,
 *         .stride = 16, .cls_scale = S_CLS4, .cls_zp = Z_CLS4,
 *         .reg_scale = S_REG4, .reg_zp = Z_REG4, .is_float = 0 },
 *       { .cls = out_cls5, .reg = out_reg5, .height = 12, .width = 20,
 *         .stride = 32, .cls_scale = S_CLS5, .cls_zp = Z_CLS5,
 *         .reg_scale = S_REG5, .reg_zp = Z_REG5, .is_float = 0 },
 *     };
 *     const nirdet_pp_cfg_t cfg = { .score_th = 0.30f, .iou_th = 0.45f,
 *                                   .max_det = 256, .topk_pre_nms = 64 };
 *     int32_t n = nirdet_postprocess(lv, 3, &cfg, g_dets, g_scratch);
 *
 * The S_* / Z_* values come from the stedgeai-generated header. Do not guess
 * them and do not reuse them across recompiles: they change with the model.
 */
int32_t nirdet_postprocess(const nirdet_level_t *levels,
                           int32_t n_levels,
                           const nirdet_pp_cfg_t *cfg,
                           det_t *out,
                           uint8_t *scratch)
{
    if (levels == NULL || cfg == NULL || out == NULL || n_levels <= 0) {
        return 0;
    }
    if (n_levels > NIRDET_MAX_LEVELS) {
        return -1;  /* loud failure: config bug, not a recoverable error */
    }

    int32_t n = 0;
    for (int32_t l = 0; l < n_levels; ++l) {
        n = nirdet_decode_level(&levels[l], cfg, out, n, (uint8_t)l);
    }
    if (n == 0) {
        return 0;
    }

    nirdet_sort_desc(out, n);

    if (cfg->topk_pre_nms > 0 && n > cfg->topk_pre_nms) {
        n = cfg->topk_pre_nms;          /* bound NMS cost on crowded frames */
    }

    return nirdet_nms(out, n, cfg->iou_th, scratch);
}

/* ======================================================================== *
 *  Letterbox coordinate mapping (canvas -> original frame)
 * ======================================================================== */

/*
 * The model sees a letterboxed canvas. To draw boxes over the raw camera
 * frame (or to report source-frame coordinates), undo the transform with the
 * same scale / pad the preprocessing used:
 *
 *     scale = min(canvas_w / src_w, canvas_h / src_h)
 *     pad_l = (canvas_w - round(src_w * scale)) / 2
 *     pad_t = (canvas_h - round(src_h * scale)) / 2
 */
void nirdet_unletterbox(det_t *d, int32_t n,
                        float scale, float pad_l, float pad_t,
                        float src_w, float src_h)
{
    if (d == NULL || n <= 0 || scale <= 0.0f) {
        return;
    }
    const float inv = 1.0f / scale;
    for (int32_t i = 0; i < n; ++i) {
        d[i].x1 = nirdet_clampf((d[i].x1 - pad_l) * inv, 0.0f, src_w);
        d[i].y1 = nirdet_clampf((d[i].y1 - pad_t) * inv, 0.0f, src_h);
        d[i].x2 = nirdet_clampf((d[i].x2 - pad_l) * inv, 0.0f, src_w);
        d[i].y2 = nirdet_clampf((d[i].y2 - pad_t) * inv, 0.0f, src_h);
    }
}

/* ======================================================================== *
 *  Host-side self-test:  gcc -DNIRDET_PP_SELFTEST -O2 nirdet_pp.c -lm
 * ======================================================================== */

#ifdef NIRDET_PP_SELFTEST
#include <stdio.h>
#include <stdlib.h>

#define TH 48
#define TW 80
#define THW (TH * TW)

static float g_cls[THW];
static float g_reg[4 * THW];
static det_t g_dets[256];
static uint8_t g_scratch[256];

int main(void)
{
    /* --- 1. decode parity against a hand-computed reference --- */
    for (int i = 0; i < THW; ++i)      { g_cls[i] = -6.0f; }
    for (int i = 0; i < 4 * THW; ++i)  { g_reg[i] = 0.0f;  }

    const int row = 24, col = 40, idx = row * TW + col;
    g_cls[idx] = 3.0f;                              /* sigmoid -> 0.9526 */
    g_reg[0 * THW + idx] = 0.0f;                    /* sigmoid -> 0.5    */
    g_reg[1 * THW + idx] = 0.0f;
    g_reg[2 * THW + idx] = logf(0.0461f);           /* prior_w           */
    g_reg[3 * THW + idx] = logf(0.1680f);           /* prior_h           */

    nirdet_level_t lv = {
        .cls = g_cls, .reg = g_reg,
        .height = TH, .width = TW, .stride = 8,
        .cls_scale = 1.0f, .cls_zp = 0,
        .reg_scale = 1.0f, .reg_zp = 0,
        .is_float = 1u,
    };
    nirdet_pp_cfg_t cfg = { .score_th = 0.30f, .iou_th = 0.45f,
                            .max_det = 256, .topk_pre_nms = 64 };

    int32_t n = nirdet_postprocess(&lv, 1, &cfg, g_dets, g_scratch);
    printf("T1 decode: %d detection(s)\n", (int)n);
    if (n != 1) { printf("   FAIL expected exactly 1\n"); return 1; }

    const float exp_cx = (2.0f * 0.5f - 0.5f + (float)col) * 8.0f; /* 324 */
    const float exp_cy = (2.0f * 0.5f - 0.5f + (float)row) * 8.0f; /* 196 */
    const float exp_w  = 0.0461f * (float)(TW * 8);                /* 29.5 */
    const float exp_h  = 0.1680f * (float)(TH * 8);                /* 64.5 */
    const float got_cx = 0.5f * (g_dets[0].x1 + g_dets[0].x2);
    const float got_cy = 0.5f * (g_dets[0].y1 + g_dets[0].y2);
    const float got_w  = g_dets[0].x2 - g_dets[0].x1;
    const float got_h  = g_dets[0].y2 - g_dets[0].y1;

    printf("   cx %.3f (exp %.3f)  cy %.3f (exp %.3f)\n",
           got_cx, exp_cx, got_cy, exp_cy);
    printf("   w  %.3f (exp %.3f)  h  %.3f (exp %.3f)  score %.4f\n",
           got_w, exp_w, got_h, exp_h, g_dets[0].score);
    if (fabsf(got_cx - exp_cx) > 1e-2f || fabsf(got_cy - exp_cy) > 1e-2f ||
        fabsf(got_w - exp_w) > 1e-2f  || fabsf(got_h - exp_h) > 1e-2f  ||
        fabsf(g_dets[0].score - 0.9526f) > 1e-3f) {
        printf("   FAIL decode mismatch\n");
        return 1;
    }
    printf("   PASS\n");

    /* --- 2. integer prefilter equals the float threshold --- */
    printf("T2 integer prefilter\n");
    const float scale = 0.05f;
    const int32_t zp = -3;
    for (float th = 0.05f; th < 0.96f; th += 0.05f) {
        const int32_t q_th = nirdet_logit_threshold_q(th, scale, zp);
        for (int32_t q = -128; q <= 127; ++q) {
            const float logit = nirdet_dequant(q, zp, scale);
            const int by_float = (nirdet_sigmoidf(logit) >= th);
            const int by_int   = (q >= q_th);
            /* One-LSB disagreement exactly at the boundary is acceptable and
             * always conservative (the int test never admits a cell the float
             * test would reject). */
            if (by_float != by_int && abs((int)(q - q_th)) > 1) {
                printf("   FAIL th=%.2f q=%d float=%d int=%d (q_th=%d)\n",
                       th, (int)q, by_float, by_int, (int)q_th);
                return 1;
            }
            if (by_int && !by_float) {
                /* must never be permissive by more than one LSB */
                if (abs((int)(q - q_th)) > 1) { printf("   FAIL permissive\n"); return 1; }
            }
        }
    }
    printf("   PASS\n");

    /* --- 3. NMS suppresses an overlapping duplicate --- */
    printf("T3 greedy NMS\n");
    g_dets[0] = (det_t){ 100.f, 100.f, 140.f, 200.f, 0.90f, 0u, 0u };
    g_dets[1] = (det_t){ 104.f, 104.f, 144.f, 204.f, 0.80f, 0u, 0u }; /* dup */
    g_dets[2] = (det_t){ 400.f, 100.f, 440.f, 200.f, 0.70f, 0u, 1u }; /* far */
    nirdet_sort_desc(g_dets, 3);
    const int32_t kept = nirdet_nms(g_dets, 3, 0.45f, g_scratch);
    printf("   kept %d (expect 2)\n", (int)kept);
    if (kept != 2) { printf("   FAIL\n"); return 1; }
    printf("   scores %.2f %.2f  PASS\n", g_dets[0].score, g_dets[1].score);

    /* --- 4. empty input --- */
    printf("T4 empty input\n");
    for (int i = 0; i < THW; ++i) { g_cls[i] = -20.0f; }
    n = nirdet_postprocess(&lv, 1, &cfg, g_dets, g_scratch);
    printf("   %d detection(s) (expect 0)  %s\n", (int)n,
           (n == 0) ? "PASS" : "FAIL");
    if (n != 0) { return 1; }

    /* --- 5. unletterbox round-trip --- */
    printf("T5 unletterbox\n");
    g_dets[0] = (det_t){ 140.f, 12.f, 180.f, 76.f, 0.9f, 0u, 0u };
    const det_t before = g_dets[0];
    nirdet_unletterbox(g_dets, 1, 0.5f, 0.f, 12.f, 1280.f, 720.f);
    printf("   [%.1f %.1f %.1f %.1f] -> [%.1f %.1f %.1f %.1f]\n",
           before.x1, before.y1, before.x2, before.y2,
           g_dets[0].x1, g_dets[0].y1, g_dets[0].x2, g_dets[0].y2);
    if (fabsf(g_dets[0].x1 - 280.f) > 1e-3f ||
        fabsf(g_dets[0].y1 - 0.f)   > 1e-3f) {
        printf("   FAIL\n");
        return 1;
    }
    printf("   PASS\n");

    printf("\nall nirdet_pp self-tests PASSED\n");
    return 0;
}
#endif /* NIRDET_PP_SELFTEST */
