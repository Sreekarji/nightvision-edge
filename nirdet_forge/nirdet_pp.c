/*
 * nirdet_pp.c — NIRDet-Lite post-processor for STM32N6 / Cortex-M55
 * ==================================================================
 * Consumes the INT8 NPU output (THREE blobs per level: cls 1ch, off 2ch,
 * size 2ch, each with its own scale / zero-point) and produces final
 * person boxes in canvas pixels. No dynamic allocation anywhere.
 *
 * THREE BLOBS, THREE SCALES
 * -------------------------
 * The head emits off and size as separate convolutions precisely so the
 * quantiser gives them independent scale / zero-point pairs: the offsets are
 * unbounded while the sizes are clamped to [-6, 1], and a shared scale
 * spanning [-6, 6] costs 4.7% of box width per LSB instead of 2.7%.
 *
 * MIN-HEAP, NOT RASTER-ORDER TRUNCATION
 * -------------------------------------
 * A previous implementation filled out[] in raster order and stopped at
 * max_det. Because the grid is scanned top-left to bottom-right, that
 * discarded every candidate in the lower-right of the frame once the buffer
 * filled — a systematic spatial bias, and the discarded cells were often the
 * highest-scoring ones (near-field pedestrians appear low in the frame).
 *
 * This version keeps a fixed-capacity MIN-HEAP keyed on the RAW INT8 SCORE
 * LOGIT. Sigmoid is monotonic and the per-level dequantisation
 * (q - zp) * scale is affine with positive scale, so ordering by the
 * dequantised logit is identical to ordering by confidence — but only WITHIN
 * a level. Across levels the scales differ, so the heap key is the
 * dequantised logit in float, computed once per surviving candidate. The
 * threshold test itself is done in the INTEGER domain (a per-level int8
 * threshold derived once from the score threshold), so the overwhelming
 * majority of cells are rejected with a single int8 compare and never touch
 * the FPU at all.
 *
 * Decode formula — identical to config.py, losses.AnchorGeometry.decode,
 * head.PedestrianHead.forward, live_nirdet.decode_level and
 * evaluate_onnx.decode_onnx_outputs.
 *
 * Build the selftest with:
 *   cc -DNIRDET_PP_SELFTEST -O2 nirdet_pp.c -lm -o nirdet_pp_test && ./nirdet_pp_test
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------- *
 * DECODE CONTRACT — must match config.py verbatim.
 * ---------------------------------------------------------------------- */
#define DECODE_OFFSET_SCALE  2.0f
#define DECODE_OFFSET_BIAS   0.5f
#define REG_LOG_CLAMP_MIN   -6.0f
#define REG_LOG_CLAMP_MAX    1.0f

#define NIRDET_NUM_CLASSES   1
#define NIRDET_MAX_LEVELS    3
#define NIRDET_MAX_DET     300

/* ---------------------------------------------------------------------- */

typedef struct {
    float x1, y1, x2, y2;   /* canvas pixels */
    float score;            /* sigmoid(logit) */
} NirdetBox;

typedef struct {
    /* cls: 1 channel, off: 2 channels (t_cx, t_cy),
       size: 2 channels (t_w, t_h). Each blob is CHW int8. */
    const int8_t *cls;
    const int8_t *off;
    const int8_t *size;

    float cls_scale;   int32_t cls_zp;
    float off_scale;   int32_t off_zp;
    float size_scale;  int32_t size_zp;

    int32_t grid_h;
    int32_t grid_w;
    int32_t stride;
} NirdetLevelCfg;

typedef struct {
    NirdetLevelCfg levels[NIRDET_MAX_LEVELS];
    int32_t n_levels;        /* runtime, not compile-time */
    int32_t img_h;           /* canvas height, e.g. 288 */
    int32_t img_w;           /* canvas width,  e.g. 512 */
    float   score_thresh;    /* from the dataset profile */
    float   iou_thresh;      /* e.g. 0.45 */
    int32_t max_det;         /* <= NIRDET_MAX_DET */
} NirdetCfg;

/* ---------------------------------------------------------------------- *
 * scalar helpers
 * ---------------------------------------------------------------------- */

static inline float nirdet_sigmoid(float x)
{
    /* Exact two-branch logistic, bit-comparable with numpy's _sigmoid and
     * torch.sigmoid. Clamping the ARGUMENT (as the previous revision did)
     * shifts every decoded centre by up to
     *   DECODE_OFFSET_SCALE * 4.5e-5 * stride
     * and is invisible to test_decode_contract, whose reference clamps too. */
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

static inline float nirdet_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float nirdet_deq(int8_t q, int32_t zp, float scale)
{
    return ((float)((int32_t)q - zp)) * scale;
}

/*
 * Integer-domain score gate.
 *
 * sigmoid(logit) >= t  <=>  logit >= logit(t)  <=>  q >= zp + logit(t)/scale
 * Computed ONCE per level, then every cell is a single int8 compare. At
 * stride 8 there are 2304 cells per frame and almost all of them fail this
 * test, so keeping it out of the FPU is most of the post-processing budget.
 */
static int32_t nirdet_int_thresh(float score_thresh, float scale, int32_t zp)
{
    float t = nirdet_clampf(score_thresh, 1e-6f, 1.0f - 1e-6f);
    float logit = logf(t / (1.0f - t));
    float q = (float)zp + logit / (scale > 0.f ? scale : 1e-9f);
    float c = ceilf(q);
    if (c < -128.f) c = -128.f;
    if (c >  127.f) c =  127.f;
    return (int32_t)c;
}

/* ---------------------------------------------------------------------- *
 * fixed-capacity min-heap on .score
 * ---------------------------------------------------------------------- */

static inline void heap_swap(NirdetBox *a, NirdetBox *b)
{
    NirdetBox t = *a; *a = *b; *b = t;
}

static inline void heap_sift_down(NirdetBox *h, int32_t n, int32_t i)
{
    for (;;) {
        int32_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].score < h[m].score) m = l;
        if (r < n && h[r].score < h[m].score) m = r;
        if (m == i) return;
        heap_swap(&h[i], &h[m]);
        i = m;
    }
}

static inline void heap_sift_up(NirdetBox *h, int32_t i)
{
    while (i > 0) {
        int32_t p = (i - 1) / 2;
        if (h[p].score <= h[i].score) return;
        heap_swap(&h[p], &h[i]);
        i = p;
    }
}

/* Insert unconditionally; caller guarantees *n < cap. */
static inline void heap_push(NirdetBox *h, int32_t *n, const NirdetBox *b)
{
    h[*n] = *b;
    heap_sift_up(h, *n);
    (*n)++;
}

/* Remove the minimum (the root). */
static inline void heap_pop(NirdetBox *h, int32_t *n)
{
    if (*n <= 0) return;
    (*n)--;
    h[0] = h[*n];
    heap_sift_down(h, *n, 0);
}

/*
 * Offer a candidate to a capacity-limited heap.
 *   heap not full            -> push
 *   heap full, score > root  -> replace the weakest candidate
 *   heap full, score <= root -> discard
 * The frame's max_det strongest candidates survive regardless of where in
 * the raster scan they appeared.
 */
static inline void heap_offer(NirdetBox *h, int32_t *n, int32_t cap,
                              const NirdetBox *b)
{
    if (*n < cap) {
        heap_push(h, n, b);
    } else if (cap > 0 && b->score > h[0].score) {
        h[0] = *b;
        heap_sift_down(h, *n, 0);
    }
}

/* ---------------------------------------------------------------------- *
 * IoU + greedy NMS (no allocation)
 * ---------------------------------------------------------------------- */

static float nirdet_iou(const NirdetBox *a, const NirdetBox *b)
{
    float x1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float y1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float x2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float y2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float iw = x2 - x1, ih = y2 - y1;
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    float inter = iw * ih;
    float aa = (a->x2 - a->x1) * (a->y2 - a->y1);
    float ab = (b->x2 - b->x1) * (b->y2 - b->y1);
    float uni = aa + ab - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

/* Descending insertion sort; n <= max_det (<= 300) and the array is already
 * partially ordered by the heap, so this beats a qsort call on the M55. */
static void nirdet_sort_desc(NirdetBox *b, int32_t n)
{
    for (int32_t i = 1; i < n; ++i) {
        NirdetBox k = b[i];
        int32_t j = i - 1;
        while (j >= 0 && b[j].score < k.score) {
            b[j + 1] = b[j];
            --j;
        }
        b[j + 1] = k;
    }
}

static int32_t nirdet_nms(NirdetBox *boxes, int32_t n, float iou_thresh,
                          int32_t max_out)
{
    static uint8_t suppressed[NIRDET_MAX_DET];
    if (n <= 0) return 0;
    if (n > NIRDET_MAX_DET) n = NIRDET_MAX_DET;
    memset(suppressed, 0, (size_t)n);

    nirdet_sort_desc(boxes, n);

    int32_t kept = 0;
    for (int32_t i = 0; i < n && kept < max_out; ++i) {
        if (suppressed[i]) continue;
        for (int32_t j = i + 1; j < n; ++j) {
            if (suppressed[j]) continue;
            if (nirdet_iou(&boxes[i], &boxes[j]) > iou_thresh)
                suppressed[j] = 1u;
        }
        boxes[kept++] = boxes[i];
    }
    return kept;
}

/* ---------------------------------------------------------------------- *
 * one level -> heap
 * ---------------------------------------------------------------------- */

static void nirdet_decode_level(const NirdetLevelCfg *lv, const NirdetCfg *cfg,
                                NirdetBox *heap, int32_t *heap_n, int32_t cap)
{
    const int32_t H = lv->grid_h, W = lv->grid_w;
    const int32_t plane = H * W;
    const int32_t q_thr = nirdet_int_thresh(cfg->score_thresh,
                                            lv->cls_scale, lv->cls_zp);
    /* Level-independent: img_w = W * stride, img_h = H * stride. */
    const float img_w = (float)cfg->img_w;
    const float img_h = (float)cfg->img_h;
    const float st = (float)lv->stride;

    for (int32_t row = 0; row < H; ++row) {
        for (int32_t col = 0; col < W; ++col) {
            const int32_t i = row * W + col;
            const int8_t qc = lv->cls[i];
            if ((int32_t)qc < q_thr) continue;      /* integer reject */

            const float logit = nirdet_deq(qc, lv->cls_zp, lv->cls_scale);
            const float score = nirdet_sigmoid(logit);
            if (score < cfg->score_thresh) continue;

            const float t_cx = nirdet_deq(lv->off[i],             lv->off_zp,  lv->off_scale);
            const float t_cy = nirdet_deq(lv->off[plane + i],     lv->off_zp,  lv->off_scale);
            const float t_w  = nirdet_deq(lv->size[i],            lv->size_zp, lv->size_scale);
            const float t_h  = nirdet_deq(lv->size[plane + i],    lv->size_zp, lv->size_scale);

            const float cx = (DECODE_OFFSET_SCALE * nirdet_sigmoid(t_cx)
                              - DECODE_OFFSET_BIAS + (float)col) * st;
            const float cy = (DECODE_OFFSET_SCALE * nirdet_sigmoid(t_cy)
                              - DECODE_OFFSET_BIAS + (float)row) * st;
            const float bw = expf(nirdet_clampf(t_w, REG_LOG_CLAMP_MIN,
                                                REG_LOG_CLAMP_MAX)) * img_w;
            const float bh = expf(nirdet_clampf(t_h, REG_LOG_CLAMP_MIN,
                                                REG_LOG_CLAMP_MAX)) * img_h;
            if (bw <= 1.f || bh <= 1.f) continue;   /* degenerate exp box */

            NirdetBox b;
            b.x1 = cx - 0.5f * bw;
            b.y1 = cy - 0.5f * bh;
            b.x2 = cx + 0.5f * bw;
            b.y2 = cy + 0.5f * bh;
            if (b.x1 < 0.f) b.x1 = 0.f;
            if (b.y1 < 0.f) b.y1 = 0.f;
            if (b.x2 > img_w) b.x2 = img_w;
            if (b.y2 > img_h) b.y2 = img_h;
            b.score = score;

            heap_offer(heap, heap_n, cap, &b);
        }
    }
}

/* ---------------------------------------------------------------------- *
 * public entry point
 * ---------------------------------------------------------------------- */

/*
 * Returns the number of boxes written to out[], or -1 on a configuration
 * error. out_cap must be >= cfg->max_det.
 */
int32_t nirdet_postprocess(const NirdetCfg *cfg, NirdetBox *out,
                           int32_t out_cap)
{
    static NirdetBox heap[NIRDET_MAX_DET];
    int32_t heap_n = 0;

    if (cfg == NULL || out == NULL) return -1;
    if (cfg->n_levels <= 0 || cfg->n_levels > NIRDET_MAX_LEVELS) return -1;
    if (cfg->max_det <= 0 || cfg->max_det > NIRDET_MAX_DET) return -1;
    if (out_cap < cfg->max_det) return -1;
    if (cfg->img_h <= 0 || cfg->img_w <= 0) return -1;

    const int32_t cap = cfg->max_det;

    /* n_levels is a RUNTIME loop bound: the --p5-ablate configuration ships
     * two levels, and a compile-time 3 would read a null pointer. */
    for (int32_t l = 0; l < cfg->n_levels; ++l) {
        const NirdetLevelCfg *lv = &cfg->levels[l];
        if (lv->cls == NULL || lv->off == NULL || lv->size == NULL) return -1;
        if (lv->grid_h <= 0 || lv->grid_w <= 0 || lv->stride <= 0) return -1;
        nirdet_decode_level(lv, cfg, heap, &heap_n, cap);
    }

    memcpy(out, heap, (size_t)heap_n * sizeof(NirdetBox));
    return nirdet_nms(out, heap_n, cfg->iou_thresh, cap);
}

/* ---------------------------------------------------------------------- *
 * selftest
 * ---------------------------------------------------------------------- */

#ifdef NIRDET_PP_SELFTEST

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define ST_H 9
#define ST_W 16
#define ST_STRIDE 32
#define ST_PLANE (ST_H * ST_W)

static int8_t g_cls[ST_PLANE];
static int8_t g_off[2 * ST_PLANE];
static int8_t g_size[2 * ST_PLANE];

static int8_t q_of(float v, float scale, int32_t zp)
{
    float q = v / scale + (float)zp;
    q = q < -128.f ? -128.f : (q > 127.f ? 127.f : q);
    return (int8_t)lrintf(q);
}

/* Write one synthetic candidate into the level buffers. */
static void put_cell(int32_t row, int32_t col, float logit,
                     float t_cx, float t_cy, float t_w, float t_h,
                     float cls_s, int32_t cls_z, float off_s, int32_t off_z,
                     float size_s, int32_t size_z)
{
    int32_t i = row * ST_W + col;
    g_cls[i] = q_of(logit, cls_s, cls_z);
    g_off[i] = q_of(t_cx, off_s, off_z);
    g_off[ST_PLANE + i] = q_of(t_cy, off_s, off_z);
    g_size[i] = q_of(t_w, size_s, size_z);
    g_size[ST_PLANE + i] = q_of(t_h, size_s, size_z);
}

int main(void)
{
    const float cls_s = 0.08f;   const int32_t cls_z = 0;
    const float off_s = 0.05f;   const int32_t off_z = 0;
    /* size range [-6, 1] -> 7/255; asymmetric, so a non-zero zp. */
    const float size_s = 7.0f / 255.0f; const int32_t size_z = -45;

    const int32_t img_h = ST_H * ST_STRIDE;   /* 288 */
    const int32_t img_w = ST_W * ST_STRIDE;   /* 512 */

    /* Background: a strongly negative logit everywhere, and sizes at the
     * lower clamp so any leaked cell is rejected as degenerate. */
    for (int32_t i = 0; i < ST_PLANE; ++i) {
        g_cls[i] = q_of(-8.0f, cls_s, cls_z);
        g_off[i] = q_of(0.0f, off_s, off_z);
        g_off[ST_PLANE + i] = q_of(0.0f, off_s, off_z);
        g_size[i] = q_of(REG_LOG_CLAMP_MIN, size_s, size_z);
        g_size[ST_PLANE + i] = q_of(REG_LOG_CLAMP_MIN, size_s, size_z);
    }

    /* log sizes for a ~51 x 115 px box on the 512x288 canvas */
    const float lw = logf(100.0f / (float)img_w);   /* ~ -1.63 */
    const float lh = logf(115.0f / (float)img_h);   /* ~ -0.92 */

    /* A: the strongest candidate, cell (4, 8) */
    put_cell(4, 8, 4.0f, 0.0f, 0.0f, lw, lh,
             cls_s, cls_z, off_s, off_z, size_s, size_z);
    /* B: the neighbouring cell, weaker, heavily overlapping A (same size,
     *    centre one stride away out of ~100 px width -> IoU > 0.45) */
    put_cell(4, 9, 2.0f, 0.0f, 0.0f, lw, lh,
             cls_s, cls_z, off_s, off_z, size_s, size_z);
    /* C: far away, weaker still, must survive */
    put_cell(1, 1, 1.0f, 0.0f, 0.0f, lw, lh,
             cls_s, cls_z, off_s, off_z, size_s, size_z);

    NirdetCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_levels = 1;
    cfg.img_h = img_h;
    cfg.img_w = img_w;
    cfg.score_thresh = 0.30f;
    cfg.iou_thresh = 0.45f;
    cfg.max_det = 64;
    cfg.levels[0].cls = g_cls;
    cfg.levels[0].off = g_off;
    cfg.levels[0].size = g_size;
    cfg.levels[0].cls_scale = cls_s;   cfg.levels[0].cls_zp = cls_z;
    cfg.levels[0].off_scale = off_s;   cfg.levels[0].off_zp = off_z;
    cfg.levels[0].size_scale = size_s; cfg.levels[0].size_zp = size_z;
    cfg.levels[0].grid_h = ST_H;
    cfg.levels[0].grid_w = ST_W;
    cfg.levels[0].stride = ST_STRIDE;

    NirdetBox out[NIRDET_MAX_DET];
    int32_t n = nirdet_postprocess(&cfg, out, NIRDET_MAX_DET);
    printf("[selftest] nirdet_postprocess -> %d box(es)\n", (int)n);
    assert(n >= 1);
    for (int32_t i = 0; i < n; ++i) {
        printf("  box %d: [%.1f %.1f %.1f %.1f] score %.4f\n", (int)i,
               out[i].x1, out[i].y1, out[i].x2, out[i].y2, out[i].score);
    }

    /* (a) the highest-scoring box survives, and is first */
    const float exp_cx_a = (DECODE_OFFSET_SCALE * 0.5f - DECODE_OFFSET_BIAS
                            + 8.0f) * (float)ST_STRIDE;
    float cx0 = 0.5f * (out[0].x1 + out[0].x2);
    printf("[selftest] strongest cx %.2f (expect ~%.2f), score %.4f\n",
           cx0, exp_cx_a, out[0].score);
    assert(fabsf(cx0 - exp_cx_a) < 2.0f);
    assert(out[0].score > 0.95f);
    for (int32_t i = 1; i < n; ++i) assert(out[i].score <= out[0].score);
    printf("[selftest] (a) highest-scoring box survived NMS  PASS\n");

    /* (b) the IoU > 0.45 neighbour was suppressed, (c) the far box survived */
    const float exp_cx_b = (DECODE_OFFSET_SCALE * 0.5f - DECODE_OFFSET_BIAS
                            + 9.0f) * (float)ST_STRIDE;
    const float exp_cx_c = (DECODE_OFFSET_SCALE * 0.5f - DECODE_OFFSET_BIAS
                            + 1.0f) * (float)ST_STRIDE;
    int found_b = 0, found_c = 0;
    for (int32_t i = 0; i < n; ++i) {
        float cx = 0.5f * (out[i].x1 + out[i].x2);
        if (fabsf(cx - exp_cx_b) < 2.0f) found_b = 1;
        if (fabsf(cx - exp_cx_c) < 2.0f) found_c = 1;
    }
    {
        /* prove the overlap really is above the threshold */
        NirdetBox a = out[0], b;
        b.x1 = exp_cx_b - 50.f; b.x2 = exp_cx_b + 50.f;
        b.y1 = a.y1; b.y2 = a.y2; b.score = 0.f;
        printf("[selftest] IoU(A, B) = %.3f (threshold %.2f)\n",
               nirdet_iou(&a, &b), cfg.iou_thresh);
        assert(nirdet_iou(&a, &b) > cfg.iou_thresh);
    }
    assert(!found_b);
    printf("[selftest] (b) overlapping neighbour suppressed  PASS\n");
    assert(found_c);
    printf("[selftest] (c) distant box survived              PASS\n");

    /* heap: max_det strongest survive regardless of raster position. Fill
     * the whole grid with increasing scores so the best cells are LAST in
     * raster order — the exact case raster truncation got wrong. */
    for (int32_t i = 0; i < ST_PLANE; ++i) {
        g_cls[i] = q_of(0.5f + 0.02f * (float)i, cls_s, cls_z);
        g_size[i] = q_of(lw, size_s, size_z);
        g_size[ST_PLANE + i] = q_of(lh, size_s, size_z);
    }
    cfg.max_det = 8;
    cfg.iou_thresh = 1.01f;          /* disable suppression for this check */
    n = nirdet_postprocess(&cfg, out, NIRDET_MAX_DET);
    printf("[selftest] heap cap 8 -> %d box(es), top score %.4f\n",
           (int)n, out[0].score);
    assert(n > 0 && n <= 8);
    {
        /* the strongest cell is the last one in raster order */
        float best_logit = 0.5f + 0.02f * (float)(ST_PLANE - 1);
        float best = nirdet_sigmoid(nirdet_deq(
            q_of(best_logit, cls_s, cls_z), cls_z, cls_s));
        printf("[selftest] expected strongest score %.4f\n", best);
        assert(fabsf(out[0].score - best) < 1e-3f);
    }
    printf("[selftest] (d) min-heap kept the strongest cells, not the "
           "raster-first ones  PASS\n");

    /* configuration guards */
    cfg.n_levels = NIRDET_MAX_LEVELS + 1;
    assert(nirdet_postprocess(&cfg, out, NIRDET_MAX_DET) == -1);
    cfg.n_levels = 1;
    assert(nirdet_postprocess(&cfg, out, 2) == -1);
    printf("[selftest] (e) n_levels and out_cap guards return -1  PASS\n");

    printf("[selftest] ALL PASSED\n");
    return 0;
}

#endif /* NIRDET_PP_SELFTEST */
