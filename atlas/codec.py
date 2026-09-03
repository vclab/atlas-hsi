"""Bitstream (de)serialization for a trained ATLAS model — the real compressed file.

A trained ATLAS model *is* the compressed representation of one cube. This module writes
that representation to a compact, self-describing binary container (no pickle) and reads it
back, reconstructing the cube via the same forward pass used in training/eval.

Container layout (little-endian)::

    magic    b"ATLS"      (4 bytes)
    version  uint8        (1 byte)
    hlen     uint32       (JSON header length in bytes)
    header   utf-8 JSON   {"meta": {...}, "arrays": [{name,dtype,shape,nbytes}, ...]}
    payload  raw bytes of each array, concatenated in header order

Parsing/inspection (``read_bitstream``, ``payload_bpppb``) is pure-numpy and needs no CUDA;
``reconstruct`` / ``decompress`` run the CUDA kernel.
"""
import json
import struct

import numpy as np

from .quantize import (
    compress_matrix_flatten_categorical,
    decompress_matrix_flatten_categorical,
    get_np_size,
)

MAGIC = b"ATLS"
VERSION = 1


def _pack(meta, arrays):
    """Pack a json-able ``meta`` dict and an ordered ``{name: np.ndarray}`` map into bytes."""
    specs, payload = [], bytearray()
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        specs.append({"name": name, "dtype": arr.dtype.str,
                      "shape": list(arr.shape), "nbytes": len(raw)})
        payload += raw
    header = json.dumps({"meta": meta, "arrays": specs}, separators=(",", ":")).encode("utf-8")
    return b"".join([MAGIC, struct.pack("<BI", VERSION, len(header)), header, bytes(payload)])


def _unpack(blob):
    """Inverse of :func:`_pack` → ``(meta, {name: np.ndarray})``."""
    if blob[:4] != MAGIC:
        raise ValueError("not an ATLAS bitstream (bad magic bytes)")
    version, hlen = struct.unpack("<BI", blob[4:9])
    if version != VERSION:
        raise ValueError(f"unsupported ATLAS bitstream version {version}")
    header = json.loads(blob[9:9 + hlen].decode("utf-8"))
    arrays, off = {}, 9 + hlen
    for spec in header["arrays"]:
        n = spec["nbytes"]
        arrays[spec["name"]] = np.frombuffer(
            blob[off:off + n], dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
        off += n
    return header["meta"], arrays


def encode_model(model):
    """Serialize a quantization-trained (QAT) ATLAS model to ATLAS-bitstream bytes.

    Positions and the endmember are stored FP16; the cholesky covariance and abundances are
    stored as their LSQ+ integer codes (ANS entropy-coded) plus per-channel scale/offset.
    """
    import torch
    if not getattr(model, "qat_enabled", False):
        raise ValueError(
            "encode_model requires a quantization-trained model (train with qat_start > 0); "
            "an FP32 model has no quantized bitstream.")
    sc, sf = model.cholesky_quantizer, model.features_dc_quantizer
    codes_c = sc.compress(model._cholesky)[0].round().to(torch.int64).cpu().numpy()
    codes_f = sf.compress(model._features_dc)[0].round().to(torch.int64).cpu().numpy()
    comp_c, cnt_c, uniq_c = compress_matrix_flatten_categorical(codes_c.flatten().tolist())
    comp_f, cnt_f, uniq_f = compress_matrix_flatten_categorical(codes_f.flatten().tolist())
    meta = {
        "H": int(model.H), "W": int(model.W), "C": int(model.C), "rank": int(model.rank),
        "N": int(model._xyz.shape[0]),
        "BLOCK_H": int(model.BLOCK_H), "BLOCK_W": int(model.BLOCK_W),
        "abundance_bits": int(model.abundance_bits), "cholesky_bits": int(model.cholesky_bits),
        "coef": float(model.coef.item()),
        "shape_c": list(codes_c.shape), "n_c": int(codes_c.size),
        "shape_f": list(codes_f.shape), "n_f": int(codes_f.size),
    }
    arrays = {
        "xyz": model._xyz.detach().half().cpu().numpy(),
        "endmember": model.endmember_activated.detach().half().cpu().numpy(),
        "cholesky_bound": model.cholesky_bound.detach().cpu().numpy().astype(np.float32),
        "scale_c": sc.scale.detach().cpu().numpy().astype(np.float32),
        "beta_c": sc.beta.detach().cpu().numpy().astype(np.float32),
        "scale_f": sf.scale.detach().cpu().numpy().astype(np.float32),
        "beta_f": sf.beta.detach().cpu().numpy().astype(np.float32),
        "comp_c": np.asarray(comp_c), "cnt_c": np.asarray(cnt_c), "uniq_c": np.asarray(uniq_c),
        "comp_f": np.asarray(comp_f), "cnt_f": np.asarray(cnt_f), "uniq_f": np.asarray(uniq_f),
    }
    return _pack(meta, arrays)


def reconstruct(meta, arrays, return_abundance=False):
    """Reconstruct the ``(H, W, C)`` cube from decoded bitstream arrays. Needs the CUDA kernel.

    With ``return_abundance=True``, stop before the endmember mixing matmul and return the
    splatted ``(H, W, rank)`` abundance map instead: the compressed-domain representation
    (partial decoding; the H*W*C cube is never materialized)."""
    import torch
    from atlas_gsplat.project_gaussians_2d import project_gaussians_2d
    from atlas_gsplat.rasterize_sum import rasterize_gaussians_sum_general
    if not torch.cuda.is_available():
        raise RuntimeError("reconstruct() needs a CUDA GPU (the atlas-gsplat kernel is CUDA-only)")

    dev = "cuda"
    H, W, C, rank, N = meta["H"], meta["W"], meta["C"], meta["rank"], meta["N"]
    BH, BW = meta["BLOCK_H"], meta["BLOCK_W"]
    tile_bounds = ((W + BW - 1) // BW, (H + BH - 1) // BH, 1)

    codes_c = decompress_matrix_flatten_categorical(
        arrays["comp_c"], arrays["cnt_c"], arrays["uniq_c"], meta["n_c"], tuple(meta["shape_c"]))
    codes_f = decompress_matrix_flatten_categorical(
        arrays["comp_f"], arrays["cnt_f"], arrays["uniq_f"], meta["n_f"], tuple(meta["shape_f"]))

    def to_dev(a):
        # np.array(copy) → writable (the unpacked arrays are read-only views into the blob).
        return torch.from_numpy(np.array(a)).to(dev)

    means = torch.tanh(to_dev(arrays["xyz"]).float())
    chol = (to_dev(codes_c).float() * to_dev(arrays["scale_c"]) + to_dev(arrays["beta_c"])
            + to_dev(arrays["cholesky_bound"]))
    feat = to_dev(codes_f).float() * to_dev(arrays["scale_f"]) + to_dev(arrays["beta_f"])
    E = to_dev(arrays["endmember"]).float()

    with torch.no_grad():
        xys, depths, radii, conics, nth = project_gaussians_2d(means, chol, H, W, tile_bounds)
        out = rasterize_gaussians_sum_general(
            xys, depths, radii, conics, nth, feat, torch.ones(N, 1, device=dev),
            H, W, BH, BW, return_alpha=False)
        out = torch.clamp(out, 0, 1).view(H * W, rank) * torch.exp(
            torch.tensor(float(meta["coef"]), device=dev))
        if return_abundance:
            return out.view(H, W, rank).detach().cpu().numpy()
        cube = (out @ E).view(H, W, C)
    return cube.detach().cpu().numpy()


def write_bitstream(model, path):
    """Encode ``model`` to an ATLAS bitstream file; returns the file size in bytes."""
    blob = encode_model(model)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def read_bitstream(path):
    """Read an ATLAS bitstream file → ``(meta, arrays)`` (pure-numpy, no CUDA)."""
    with open(path, "rb") as fh:
        return _unpack(fh.read())


def decompress(path, abundance=False):
    """Read an ATLAS bitstream file and reconstruct the ``(H, W, C)`` cube (needs CUDA).

    ``abundance=True`` returns the ``(H, W, rank)`` splatted abundance map instead
    (partial decoding, see :func:`reconstruct`)."""
    meta, arrays = read_bitstream(path)
    return reconstruct(meta, arrays, return_abundance=abundance)


def payload_bits(meta, arrays):
    """Entropy-payload bits, matching the model's reported bpppb formula.

    Counts positions (FP16) + endmember (FP16) + the two ANS streams (compressed bytes,
    symbol tables, and per-channel scale/offset, plus the 64-bit/stream overhead the trainer
    accounts for). Excludes the small self-describing container header.
    """
    def stream(comp, cnt, uniq, scale, beta):
        return (get_np_size(comp) * 8 + 64 + get_np_size(cnt) * 8 + get_np_size(uniq) * 8
                + scale.size * 32 + beta.size * 32)
    a = arrays
    m = a["xyz"].nbytes * 8
    e = a["endmember"].nbytes * 8
    s = stream(a["comp_c"], a["cnt_c"], a["uniq_c"], a["scale_c"], a["beta_c"])
    c = stream(a["comp_f"], a["cnt_f"], a["uniq_f"], a["scale_f"], a["beta_f"])
    return m + e + s + c


def payload_bpppb(meta, arrays):
    """Entropy-payload bits-per-pixel-per-band (matches the trainer's reported bpppb)."""
    return payload_bits(meta, arrays) / (meta["H"] * meta["W"] * meta["C"])
