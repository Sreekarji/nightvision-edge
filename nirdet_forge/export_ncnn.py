"""
export_ncnn.py — ONNX -> NCNN (fp32 + INT8) export for the Pi 5
================================================================
    python export_ncnn.py --onnx nirdet-sim.onnx --profile datasets/<n>.yaml
    python export_ncnn.py --onnx nirdet-int8-qdq.onnx --profile <y> --skip-int8

The NCNN toolchain (onnx2ncnn, ncnnoptimize, ncnn2table, ncnn2int8) is invoked
via subprocess; nothing from NCNN is imported.

CALIBRATION PARITY: ncnn2table calibrates on PNGs produced by
dataset.preprocess_frame with the EXACT cfg.aug parameters — the same tensors
quantize_qdq.py fed the ORT calibrater. live_nirdet.infer() builds its Mat
from uint8 pixels and sets norm=1/255, so the runtime input distribution is
float [0,1]; calibration therefore writes canvas*255 uint8 PNGs and passes
mean=[0] norm=[1/255].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

import numpy as np

from config import Config, get_config
from dataset import (list_images, load_flat_field, preprocess_frame,
                     resolve_split_dirs)
from export_onnx import write_contract_sidecar

try:
    import cv2
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit("export_ncnn.py needs opencv: "
                     "pip install opencv-python") from exc


def _mb(path: str) -> str:
    if not os.path.isfile(path):
        return "(missing)"
    return f"{os.path.getsize(path) / (1024 * 1024):.2f} MB"


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"{name} not found. Install NCNN: "
            f"https://github.com/Tencent/ncnn/releases\n"
            f"  (build with -DNCNN_BUILD_TOOLS=ON, or download a prebuilt "
            f"release and add its tools dir to PATH)")
    return path



# ---------------------------------------------------------------------------
# WSL tool resolver
# ncnnoptimize 20250916 Windows crashes. Linux build via WSL is stable.
# Windows paths are translated to /mnt/<drive>/... before WSL calls.
# ---------------------------------------------------------------------------

_WSL_TOOL_PATHS = {
    "onnx2ncnn":    "/home/sreekar/ncnn/build/tools/onnx/onnx2ncnn",
    "ncnnoptimize": "/home/sreekar/ncnn/build/tools/ncnnoptimize",
    "ncnn2table":   "/home/sreekar/ncnn/build/tools/quantize/ncnn2table",
    "ncnn2int8":    "/home/sreekar/ncnn/build/tools/quantize/ncnn2int8",
}


def _wsl_exe() -> str:
    w = shutil.which("wsl.exe") or shutil.which("wsl")
    if w is None:
        raise SystemExit("wsl.exe not found — add WSL or native NCNN tools to PATH.")
    return w


def _wsl_available() -> bool:
    return bool(shutil.which("wsl.exe") or shutil.which("wsl"))


def _win_to_wsl(path: str) -> str:
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        rest  = path[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return path


def _tool_cmd(name: str) -> list:
    if _wsl_available() and name in _WSL_TOOL_PATHS:
        return [_wsl_exe(), _WSL_TOOL_PATHS[name]]
    native = shutil.which(name)
    if native:
        return [native]
    raise SystemExit(
        f"{name} not found. Add NCNN tools to PATH or install WSL with NCNN "
        f"built at {_WSL_TOOL_PATHS.get(name, '(unknown)')}. ")


def _is_wsl(cmd: list) -> bool:
    return len(cmd) >= 2 and ("wsl" in os.path.basename(cmd[0]).lower())

def _run(cmd: List[str], what: str, fail_hint: str) -> str:
    print(f"[ncnn-export] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr or "")
        raise SystemExit(f"{what} failed — {fail_hint}\n"
                         f"stderr above is the tool's own words.")
    return r.stdout or ""


# ===========================================================================
# steps
# ===========================================================================

def step_onnx2ncnn(onnx_path: str, param: str, bin_path: str) -> None:
    tc = _tool_cmd("onnx2ncnn")
    _args = ([_win_to_wsl(p) for p in [onnx_path, param, bin_path]]
             if _is_wsl(tc) else [onnx_path, param, bin_path])
    out = _run(tc + _args, "onnx2ncnn",
               "check that the ONNX was exported with static shapes and "
               "opset 12. An unsupported op is named in the stderr above.")
    for ln in (out.strip().splitlines()[-6:] if out.strip() else []):
        print(f"[onnx2ncnn] | {ln}")
    for p in (param, bin_path):
        if not os.path.isfile(p):
            raise SystemExit(f"onnx2ncnn exited 0 but {p} was not produced")


def step_ncnnoptimize(param: str, bin_path: str, o_param: str, o_bin: str,
                      flag: str = "65536", skip_int8: bool = False
                      ) -> Tuple[str, str]:
    """
    flag 65536 = fp16 storage + winograd; flag 0 = fp32 storage, still fused.

    Use 0 when an INT8 pass will follow (F43): calibrating and quantising from
    fp16-rounded weights bakes an extra rounding stage into the INT8 model for
    no benefit — the INT8 .bin does not keep the fp16 weights.

    F51: missing ncnnoptimize is FATAL for the INT8 path — the INT8 table
    calibrates against the FUSED graph; skipping fusion produces a table keyed
    to different blob names and mismatched activation ranges.
    """
    tc_opt = _tool_cmd("ncnnoptimize")
    if not tc_opt:
        if not skip_int8:
            raise SystemExit(
                "ncnnoptimize not on PATH. The INT8 path calibrates against "
                "the FUSED graph; skipping fusion produces a table keyed to "
                "different blob names and mismatched activation ranges. "
                "Install NCNN tools or pass --skip-int8.")
        print("[ncnn-export] ncnnoptimize not on PATH — SKIPPED "
              "(fp32 passthrough only)")
        return param, bin_path
    wsl_p = ([_win_to_wsl(p) for p in [param, bin_path, o_param, o_bin]]
             if _is_wsl(tc_opt) else [param, bin_path, o_param, o_bin])
    cmd = tc_opt + wsl_p + [flag]
    print(f"[ncnn-export] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        # Some ncnnoptimize builds (e.g. 20250916) exit non-zero when
        # shape_inference is skipped (no static input shape in the .param),
        # but still write valid fused output files. Treat non-zero as a
        # warning rather than a hard failure if both output files were produced.
        valid_output = (os.path.isfile(o_param) and os.path.getsize(o_param) > 0
                        and os.path.isfile(o_bin) and os.path.getsize(o_bin) > 0)
        if valid_output:
            print(f"[ncnn-export] ncnnoptimize exited {r.returncode} but "
                  f"produced {o_param} and {o_bin} — treating as success "
                  f"(shape_inference skipped is harmless for fixed-canvas export)")
        elif not skip_int8:
            # SAFETY NET: step_fuse_param should never be reached when the WSL
            # build is configured — _tool_cmd() routes ncnnoptimize through the
            # stable Linux binary. If we are here it means:
            #   (a) wsl.exe is not on PATH (WSL not installed / not in PATH), OR
            #   (b) the WSL binary itself crashed (investigate immediately), OR
            #   (c) the WSL binary path in _WSL_TOOL_PATHS is wrong/stale.
            # step_fuse_param only fuses Conv+Clip and Conv+Sigmoid pairs — any
            # new op types added to the model that also need fusion will be
            # silently missed. Treat this as an error condition to investigate,
            # not a normal operating mode.
            print(
                f"\n[ncnn-export] *** UNEXPECTED FALLBACK TO step_fuse_param ***\n"
                f"[ncnn-export] ncnnoptimize crashed (exit {r.returncode}, "
                f"output files are 0 bytes or missing).\n"
                f"[ncnn-export] This should not happen when WSL tools are "
                f"configured. Check that:\n"
                f"[ncnn-export]   1. wsl.exe is on PATH (run: where wsl)\n"
                f"[ncnn-export]   2. the WSL binary exists: "
                f"{_WSL_TOOL_PATHS.get('ncnnoptimize', '(not set)')}\n"
                f"[ncnn-export]   3. no model architecture changes introduced "
                f"new op types needing fusion beyond Conv+Clip/Sigmoid.\n"
                f"[ncnn-export] Proceeding with Python fallback — verify the "
                f"exported INT8 model carefully.\n"
            )
            step_fuse_param(param, bin_path, o_param, o_bin)
        else:
            raise SystemExit(
                f"ncnnoptimize failed (exit {r.returncode}) and produced no "
                f"valid output files (files may exist but are 0 bytes). "
                f"Check stderr above.\n"
                f"flag={flag}; try a different flag or pass --skip-int8.")
    return o_param, o_bin


def step_fuse_param(param: str, bin_path: str, o_param: str, o_bin: str) -> None:
    """
    Pure-Python fallback for ncnnoptimize Conv+Clip fusion.

    ncnnoptimize 20250916 crashes with STATUS_ACCESS_VIOLATION (0xC0000005) on
    this graph regardless of flag, producing 0-byte output files.  The only
    thing we need from it for the INT8 calibration path is Conv+Activation
    fusion — specifically fusing each Convolution node with the Clip (ReLU6)
    that immediately follows it into a single ConvolutionDepthWise or
    Convolution layer with activation_type=1 and activation_params set.

    What the .param format looks like before fusion:

        Convolution  /some/Conv  1 1 in_blob out_blob  0=64 1=3 ...
        Clip         /some/Clip  1 1 out_blob fused_blob  0=0 1=6

    After fusion the Clip line is removed and the Convolution line gains:
        9=1        (activation_type = ReLU6 = Clip)
        10=2       (activation_params count)
        1010=0.0   (clip min)
        1011=6.0   (clip max)

    and its output blob name is rewritten to the Clip's output blob.

    The .bin is unchanged — weight data is not touched.

    We also handle the Sigmoid-fused proj convolutions (EAA projection):
        Convolution  /proj/Conv   1 1 in_blob out_blob  0=1 ...
        Sigmoid      /Sigmoid     1 1 out_blob sig_blob
    These become activation_type=4 (Sigmoid), no activation_params needed.

    Every other optimisation ncnnoptimize performs (winograd, fp16 storage,
    shape inference) is either harmful for INT8 (fp16), not applicable (no
    static shapes), or not required for ncnn2table to work.
    """
    with open(param, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Parse the header (first two lines: magic + layer_count blob_count).
    # We will recount layers after removing Clip/Sigmoid lines.
    if len(lines) < 2:
        raise SystemExit(f"step_fuse_param: {param} has fewer than 2 lines")
    magic = lines[0].rstrip("\r\n")
    header_parts = lines[1].split()
    if len(header_parts) != 2:
        raise SystemExit(f"step_fuse_param: unexpected header '{lines[1]}'")

    # Build a list of parsed layer dicts so we can do a forward pass.
    # We only need: type, name, n_inputs, n_outputs, input_blobs, output_blobs,
    # and the raw attribute string.
    layers = []
    for raw in lines[2:]:
        raw = raw.rstrip("\r\n")
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 4:
            layers.append({"raw": raw, "type": parts[0] if parts else ""})
            continue
        ltype = parts[0]
        lname = parts[1]
        try:
            n_in = int(parts[2])
            n_out = int(parts[3])
        except ValueError:
            layers.append({"raw": raw, "type": ltype})
            continue
        in_blobs = parts[4: 4 + n_in]
        out_blobs = parts[4 + n_in: 4 + n_in + n_out]
        attrs = parts[4 + n_in + n_out:]
        layers.append({
            "raw": raw,
            "type": ltype,
            "name": lname,
            "n_in": n_in,
            "n_out": n_out,
            "in_blobs": in_blobs,
            "out_blobs": out_blobs,
            "attrs": attrs,
            "fused_into": False,   # True = this layer is consumed by fusion
        })

    # Build a map from INPUT blob -> [layer indices] so we can find consumers.
    # (blob_to_producer would go the other way; we need the consumer direction.)
    in_blob_to_layers: dict = {}
    for idx, layer in enumerate(layers):
        for b in layer.get("in_blobs", []):
            in_blob_to_layers.setdefault(b, []).append(idx)

    # --- forward pass: fuse Conv -> Clip and Conv -> Sigmoid ---
    FUSE_ACTS = {
        "Clip":    (1, ["0", "1"]),   # activation_type=1, params min/max
        "Sigmoid": (4, []),           # activation_type=4, no params
    }
    n_fused = 0
    for idx, layer in enumerate(layers):
        if layer.get("fused_into"):
            continue
        if layer.get("type") not in ("Convolution", "ConvolutionDepthWise"):
            continue
        if not layer.get("out_blobs"):
            continue
        # The Conv must have exactly one output blob, and that blob must be
        # consumed by exactly one layer which is a fuseable activation.
        conv_out = layer["out_blobs"]
        if len(conv_out) != 1:
            continue
        consumers = in_blob_to_layers.get(conv_out[0], [])
        if len(consumers) != 1:
            continue   # fan-out: blob feeds multiple layers, cannot fuse
        act_idx = consumers[0]
        act = layers[act_idx]
        if act.get("fused_into") or act.get("type") not in FUSE_ACTS:
            continue
        # The activation must consume ONLY the conv's output.
        if act.get("n_in") != 1 or act.get("n_out") != 1:
            continue

        act_type_id, _ = FUSE_ACTS[act["type"]]

        # Build the fused attr list: existing conv attrs + activation_type.
        existing_attrs = list(layer["attrs"])
        # Remove any existing activation_type attr (9=...) to avoid duplicates.
        existing_attrs = [a for a in existing_attrs if not a.startswith("9=")]
        fused_attrs = existing_attrs + [f"9={act_type_id}"]
        # Do NOT emit 10=/1010=/1011= activation_params. Those use attribute
        # indices >= 1010 which exceed NCNN_MAX_PARAM_COUNT=32 in the 20250916
        # prebuilt, causing "id < NCNN_MAX_PARAM_COUNT failed" at load time.
        # For activation_type=1 (ReLU6) NCNN hard-codes [0, 6] internally, so
        # the params are not needed. For activation_type=4 (Sigmoid) there are
        # no params. Both are correct without the extra attrs.

        # Rewrite the Conv layer: new out_blob = act's out_blob, add attrs.
        layer["out_blobs"] = list(act["out_blobs"])
        layer["attrs"] = fused_attrs
        act["fused_into"] = True
        n_fused += 1

    # --- reconstruct the .param text ---
    out_lines = []
    for layer in layers:
        if layer.get("fused_into"):
            continue
        if "name" not in layer:
            # header/blank lines we didn't fully parse — kept verbatim
            out_lines.append(layer["raw"])
            continue
        parts = ([layer["type"], layer["name"],
                  str(layer["n_in"]), str(layer["n_out"])]
                 + layer["in_blobs"]
                 + layer["out_blobs"]
                 + layer["attrs"])
        out_lines.append(" ".join(parts))

    # Fix the layer count in the header.
    new_layer_count = len(out_lines)
    old_blob_count = int(header_parts[1])
    # blob count stays the same — fusion just removes layers, not blobs.

    with open(o_param, "w", encoding="utf-8") as fh:
        fh.write(magic + "\n")
        fh.write(f"{new_layer_count} {old_blob_count}\n")
        for ln in out_lines:
            fh.write(ln + "\n")

    # .bin is unchanged — copy it unless it's already the destination.
    if os.path.abspath(bin_path) != os.path.abspath(o_bin):
        shutil.copy2(bin_path, o_bin)

    print(f"[ncnn-export] step_fuse_param: fused {n_fused} Conv+Act pairs "
          f"in {param} -> {o_param}  ({new_layer_count} layers, bin unchanged)")
    if n_fused == 0:
        print("[ncnn-export] WARNING: step_fuse_param found 0 fuseable pairs — "
              "check that onnx2ncnn produced Conv + Clip/Sigmoid sequences")


def build_calibration_images(cfg: Config, n_images: int, tmp_dir: str
                             ) -> List[str]:
    """preprocess_frame the train split exactly like quantize_qdq.py's reader,
    and write each canvas as the uint8 image live_nirdet.infer will feed.

    F49: the SAME seeded selection (config.calibration_paths, seed
    cfg.export.calib_seed) as the ORT reader, so the two INT8 paths calibrate
    on identical images."""
    from config import calibration_paths
    paths = calibration_paths(cfg.data.root, "train",
                              n=None, seed=cfg.export.calib_seed)
    if not paths:
        raise SystemExit("no images for calibration — check the train split")
    n = min(int(n_images), len(paths))
    flat_field = load_flat_field(cfg.aug.flat_field_path)
    written: List[str] = []
    checked = False
    for i, p in enumerate(paths[:n]):
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        canvas, _s, _px, _py = preprocess_frame(
            raw, cfg.data.img_h, cfg.data.img_w,
            clahe_enabled=cfg.aug.clahe_enabled,
            clahe_clip=cfg.aug.clahe_clip, clahe_grid=cfg.aug.clahe_grid,
            flat_field=flat_field)
        u8 = np.clip(np.rint(canvas * 255.0), 0.0, 255.0).astype(np.uint8)
        if not checked:
            # ORT calibrates on the float32 canvas; ncnn2table calibrates on
            # this uint8 round trip. The parity is currently exact because the
            # chain is uint8-valued up to the final /255 — but that is
            # accidental, not enforced (F45).
            back = u8.astype(np.float32) / 255.0
            if float(np.abs(back - canvas).max()) >= 1.0 / 255.0:
                raise SystemExit(
                    "preprocess_frame no longer round-trips through uint8; "
                    "ncnn2table would calibrate on PNGs while ORT calibrates "
                    "on float32, so the two INT8 activation ranges would "
                    "diverge. Keep the chain uint8-valued up to the /255.")
            checked = True
        dst = os.path.join(tmp_dir, f"calib_{i:04d}.png")
        cv2.imwrite(dst, u8)
        written.append(dst)
    return written


def step_ncnn2table(cfg: Config, param: str, bin_path: str, n_images: int,
                    table_path: str, threads: int) -> None:
    tc2t = _tool_cmd("ncnn2table")
    tmp_dir = tempfile.mkdtemp(prefix="ncnn_calib_")
    try:
        pngs = build_calibration_images(cfg, n_images, tmp_dir)
        if not pngs:
            raise SystemExit("calibration produced zero images — is the "
                             "train split readable?")
        list_file = os.path.join(tmp_dir, "calibration.txt")
        with open(list_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(pngs) + "\n")

        ch = int(cfg.data.num_channels)
        pixel = "GRAY" if ch == 1 else "RGB"
        norm = "0.0039216"          # = 1/255; see the NORM NOTE in the header
        # F52: stream the long-running calibration output live instead of
        # capturing it silently.
        if _is_wsl(tc2t):
            wsl_list = os.path.join(tmp_dir, "calibration_wsl.txt")
            with open(wsl_list, "w", encoding="utf-8") as _fh:
                _fh.write("\n".join(_win_to_wsl(p) for p in pngs) + "\n")
            cmd = tc2t + [
                _win_to_wsl(param), _win_to_wsl(bin_path),
                _win_to_wsl(wsl_list), _win_to_wsl(table_path),
                "mean=[0]", f"norm=[{norm}]",
                f"shape=[{cfg.data.img_w},{cfg.data.img_h},{ch}]",
                f"pixel={pixel}", f"thread={threads}", "method=kl"]
        else:
            cmd = tc2t + [param, bin_path, list_file, table_path,
                "mean=[0]", f"norm=[{norm}]",
                f"shape=[{cfg.data.img_w},{cfg.data.img_h},{ch}]",
                f"pixel={pixel}", f"thread={threads}", "method=kl"]
        print(f"[ncnn-export] $ {' '.join(cmd)}")
        r = subprocess.run(cmd)   # no capture_output: output streams live
        if r.returncode != 0:
            raise SystemExit("ncnn2table failed — check the output above")
        if not os.path.isfile(table_path) or os.path.getsize(table_path) == 0:
            raise SystemExit("ncnn2table produced no table")
        print(f"[ncnn-export] calibration parity: {len(pngs)} images | "
              f"seed={cfg.export.calib_seed} | "
              f"clahe_enabled={cfg.aug.clahe_enabled} "
              f"clahe_clip={cfg.aug.clahe_clip} "
              f"clahe_grid={cfg.aug.clahe_grid} "
              f"flat_field={cfg.aug.flat_field_path!r} "
              f"mean=[0] norm=[{norm}] shape=[{cfg.data.img_w},"
              f"{cfg.data.img_h},{ch}] pixel={pixel} method=kl")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def step_ncnn2int8(param: str, bin_path: str, table_path: str,
                   o_param: str, o_bin: str) -> None:
    tc2i = _tool_cmd("ncnn2int8")
    n2i_a = ([_win_to_wsl(p) for p in [param, bin_path, o_param, o_bin, table_path]]
             if _is_wsl(tc2i) else [param, bin_path, o_param, o_bin, table_path])
    _run(tc2i + n2i_a, "ncnn2int8",
         "the table must have been generated for THIS param/bin pair")


def step_verify_blobs(param: str, cfg: Config,
                      input_blob: str = "images") -> None:
    """
    Verify the runtime's blob names exist HERE, on the build machine (F44).

    onnx2ncnn and ncnnoptimize can rename blobs, and live_nirdet.NCNNEngine
    addresses them by exact name — a rename otherwise surfaces as
    "NCNN extract failed for blob 'cls8'" on the Pi, after the whole export
    pipeline reported success. A .param is a whitespace table, so every blob
    name appears as a token.
    """
    with open(param, "r", encoding="utf-8", errors="replace") as fh:
        tokens = set(fh.read().split())
    want = [input_blob] + [f"{p}{s}" for s in cfg.model.strides
                           for p in ("cls", "off", "size")]
    missing = [b for b in want if b not in tokens]
    if missing:
        raise SystemExit(
            f"{param} does not declare blob(s) {missing}. live_nirdet.py and "
            f"nirdet_pp.c address blobs by these exact names. onnx2ncnn/"
            f"ncnnoptimize renamed them; re-export with matching ONNX "
            f"output_names, or update NCNNEngine.blob_names.")
    print(f"[ncnn-export] blob names verified in {param}: {want}")


def step_bench(param: str, bin_path: str, threads: int,
               budget_ms: float = 50.0) -> None:
    """
    Latency via live_nirdet.py --bench.

    F50: only an aarch64 machine (the Pi 5) produces a meaningful number.

    benchncnn is deliberately NOT used (F42): upstream's CLI is positional —
    benchncnn [loop] [threads] [powersave] [gpu] [cooling] — and it benchmarks
    a built-in model list found in the working directory. It does not take a
    .param/.bin pair, so the old invocation either mis-parsed the param path
    as a loop count or timed the wrong model. live_nirdet --bench uses the
    same Net options, the same Mat construction and the same blob extraction
    as deployment.
    """
    import platform as _platform
    if _platform.machine() not in ("aarch64", "arm64"):
        print(f"[bench] SKIPPED: this machine is {_platform.machine()}, "
              f"not the Pi 5 (aarch64). Latency measured here is meaningless. "
              f"Run on the Pi 5:\n"
              f"  python live_nirdet.py --bench --param {param} "
              f"--bin {bin_path} --threads {threads}")
        return
    cmd = [sys.executable, "live_nirdet.py", "--bench",
           "--param", param, "--bin", bin_path, "--threads", str(threads)]
    print(f"[bench] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout or "")
    if r.returncode != 0:
        sys.stderr.write(r.stderr or "")
        print("[bench] live_nirdet --bench failed; run it directly on the Pi 5")
        return
    m = re.search(r"avg\s+([0-9.]+)\s*ms", r.stdout or "")
    if not m:
        tail_lines = (r.stdout or "").strip().splitlines()[-6:]
        tail = "\n  ".join(tail_lines) if tail_lines else "(no output)"
        print("[bench] WARNING: could not parse 'avg ... ms' from "
              "live_nirdet output; last lines:\n  " + tail)
        return
    avg = float(m.group(1))
    print(f"[bench] {'PASS' if avg <= budget_ms else 'FAIL'} — {avg:.2f} ms "
          f"vs {budget_ms:.0f} ms budget (20 FPS, leaving headroom for "
          f"capture + post-processing)")


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ONNX -> NCNN fp32/INT8 export for NIRDet-Forge")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--profile", default=None,
                    help="dataset profile YAML — REQUIRED on every path (F12)")
    ap.add_argument("--out-prefix", default="nirdet")
    ap.add_argument("--skip-int8", action="store_true",
                    help="input ONNX is already QDQ INT8")
    ap.add_argument("--calib-images", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bench", action="store_true",
                    help="time the final model via live_nirdet.py --bench")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        raise SystemExit(f"ONNX not found: {args.onnx}")

    cfg = get_config()
    # F12 (audit fix): --profile is required on EVERY path, including
    # --skip-int8. This script writes a .contract.json beside the artefacts,
    # and that hash covers the CLAHE parameters and the flat-field digest —
    # generating it from a default config stamps the model with preprocessing
    # it was never trained with, and live_nirdet.py's contract check then
    # verifies the artefact against itself.
    if not args.profile:
        raise SystemExit(
            "--profile is required, including with --skip-int8. See the "
            "contract sidecar this script emits: its hash covers clahe_* and "
            "the flat-field digest.\n"
            "  python export_ncnn.py --onnx <graph> --profile datasets/<n>.yaml")
    from dataset_profiles import DatasetProfile
    # verify=True on the INT8 path: the build machine hosts the dataset it is
    # about to calibrate on, so a stale profile must be caught here. On the
    # QDQ-passthrough path the dataset is not needed for anything, so allow
    # the Pi-style verify=False application.
    DatasetProfile.load(args.profile).apply(cfg, verify=not args.skip_int8)
    if not args.skip_int8 and not cfg.data.root:
        raise SystemExit("--profile must resolve a readable dataset root for "
                         "INT8 calibration")

    p = args.out_prefix
    raw_param, raw_bin = f"{p}.param", f"{p}.bin"
    opt_param, opt_bin = f"{p}-opt.param", f"{p}-opt.bin"
    int8_param, int8_bin = f"{p}-int8.param", f"{p}-int8.bin"
    table = f"{p}.table"
    n_cal = args.calib_images or cfg.export.calib_images

    print("=" * 60)
    print("  ONNX -> NCNN export")
    print(f"  input      : {args.onnx}")
    print(f"  mode       : "
          f"{'QDQ INT8 passthrough (--skip-int8)' if args.skip_int8 else 'fp32 -> ncnn -> calibrate -> ncnn2int8'}")
    print(f"  canvas     : {cfg.data.img_h}x{cfg.data.img_w} "
          f"x{cfg.data.num_channels}ch")
    print("=" * 60)

    step_onnx2ncnn(args.onnx, raw_param, raw_bin)
    # F43: fp32 storage (flag=0) when INT8 follows — calibrating from
    # fp16-rounded weights bakes an extra rounding stage into the INT8 model
    # for no benefit, and winograd (included in flag=65536) has been observed
    # to crash ncnnoptimize on this graph with STATUS_ACCESS_VIOLATION,
    # producing 0-byte output files. fp16 + winograd only for --skip-int8.
    flag = "65536" if args.skip_int8 else "0"
    use_param, use_bin = step_ncnnoptimize(raw_param, raw_bin,
                                           opt_param, opt_bin, flag,
                                           skip_int8=bool(args.skip_int8))

    if args.skip_int8:
        final_param, final_bin = use_param, use_bin
        print("[ncnn-export] --skip-int8: the QDQ graph's int8 storage lives "
              "in the converted param/bin above")
    else:
        step_ncnn2table(cfg, use_param, use_bin, n_cal, table, args.threads)
        step_ncnn2int8(use_param, use_bin, table, int8_param, int8_bin)
        final_param, final_bin = int8_param, int8_bin

    step_verify_blobs(final_param, cfg)
    sidecar = write_contract_sidecar(cfg, final_param)

    if args.bench:
        step_bench(final_param, final_bin, args.threads)

    with open(sidecar, encoding="utf-8") as fh:
        chash = json.load(fh).get("hash", "?")

    print("=" * 60)
    print("  NCNN export complete"
          + ("" if not args.skip_int8 else " (QDQ passthrough)"))
    print(f"  source ONNX : {args.onnx} ({_mb(args.onnx)})")
    print(f"  param       : {final_param} ({_mb(final_param)})")
    print(f"  bin         : {final_bin} ({_mb(final_bin)})")
    if not args.skip_int8:
        print(f"  table       : {table} ({n_cal} calib images — see the "
              f"parity line above)")
    print(f"  contract    : {sidecar} (hash {chash})")
    print("  next (on the Pi):")
    print(f"    python live_nirdet.py --param {final_param} "
          f"--bin {final_bin} --contract {os.path.basename(sidecar)}"
          + (f" --profile {os.path.basename(args.profile)}"
             if args.profile else ""))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
