"""Isolation on a GPU through Vulkan, for the Vulkan image.

The CUDA image runs the Voice Isolator on NVIDIA GPUs. This runs it on any
GPU with a Vulkan driver, such as an AMD GPU with Mesa's RADV driver, through
wgpu. ONNX Runtime's own WebGPU provider was measured first and is no use
here: its LSTM takes 1.45 s for a 5 s command on an RTX 4000, where the CPU
takes 0.6 s.

The model is cut in three when it loads. The first part (the spectrogram and
its bands) and the last (the masks applied and the way back to audio) are
light and stay in ONNX Runtime on the CPU. The middle is nearly all of the
work: six blocks, each a two-way LSTM along time and one along the bands,
and a mask network for each band. It runs as compute shaders.

Three things keep it from taking the service down:

* **Short jobs.** A GPU driver gives a job a couple of seconds and then
  resets the GPU for the process that sent it: 2 s with Linux's AMD driver.
  The work is sent in jobs of about ``JOB_S`` each, sized from how fast this
  GPU has been so far, so a slow or busy GPU gets more and smaller jobs.
* **A process of its own.** Everything that touches the GPU runs in a helper
  process (``python -m ground_control.vulkan``). A driver that hangs while
  it compiles a shader, or a GPU that is lost anyway, ends that process and
  not Ground Control. The command it was working on goes on as Trimming left
  it and the helper starts again.
* **A check at the start.** A second of noise goes through the GPU and
  through the whole model on the CPU. A GPU whose audio differs is refused,
  and so is one that is slower than this machine's CPU.

The features stay in one buffer, ``F[band][channel][frame]``. Along time a
band is a sequence of frames and the bands are the batch. Along the bands a
frame is a sequence of bands and the frames are the batch. Both read ``F``
through strides, so the model's transposes need no pass of their own. What
the heavy shaders read is laid out with four batch rows in one ``vec4``, and
a weight ``vec4`` holds the four gates (or four outputs) of one input, so one
loop turn loads a weight once for every row it works on.

How many rows an invocation takes is the one setting that differs between
GPUs: a large GPU wants many small invocations along time, a small one few
large ones. So a few shapes are timed at the start and the fastest is kept.
"""

from __future__ import annotations

import json
import logging
import os
import select
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

_LOG = logging.getLogger(__name__)

BANDS = 32
N = 128
"""Channels of the features."""
H = 256
"""Hidden size of each LSTM direction."""
D = 512
"""Width of the mask networks."""
BLOCKS = (1, 3, 5, 7, 9, 11)
"""The separator's layers that are blocks. The ones between scale the channels by the voiceprint."""
ROW = 256
"""Bytes between two sets of shader parameters."""

JOB_S = 0.2
"""How long a GPU job is meant to take. Linux's AMD driver resets the GPU for
a process whose job takes 2 s."""
CHECK_SAMPLES = 16000
CHECK_DB = -30.0
"""The GPU's audio has to be this far under the CPU's, as the difference of
the two against the CPU's. A GPU that computes the model is 60 dB under."""

# (vec4s of batch rows an invocation takes, workgroup size, inputs a loop turn takes)
LSTM_SHAPES = ((2, 8, 8), (4, 16, 4), (8, 16, 4))
"""Timed at the start for each direction. The first suits a large GPU along
time (0.7 TFLOPS more on an RTX 4000 than the last), the last a small one
(a Radeon 610M is 1.7 times faster with it)."""
SMALL_SHAPES = ((2, 8, 8),)
"""For a driver that didn't finish compiling the larger shaders."""
DENSE_SHAPE = (4, 16, 2)

READY_S = 60.0
"""How long the helper process gets to start: the shaders compiled, the shapes timed and the check done."""
RUN_S = 60.0
"""How long it gets for one run before it is taken for hung."""


class Unavailable(RuntimeError):
    """VULKAN is on, but Isolation can't run on a GPU through it."""


class Slow(Unavailable):
    """The GPU computes the model, but slower than this machine's CPU."""


class Failed(RuntimeError):
    """A run didn't come back from the GPU. The helper process is started again."""


# --------------------------------------------------------------------------
# The model, cut in three
# --------------------------------------------------------------------------

class Layer(NamedTuple):
    """One direction of a block: its normalization, LSTM and projection."""
    gamma: np.ndarray
    beta: np.ndarray
    w: np.ndarray
    r: np.ndarray
    b: np.ndarray
    proj: np.ndarray
    proj_bias: np.ndarray


class Parts(NamedTuple):
    head: bytes
    """ONNX model: audio and voiceprint to the features, the voiceprint as the scales take it and what the tail needs."""
    tail: bytes
    """ONNX model: the mask networks' outputs and what the head gave to the audio."""
    features: str
    voiceprint: str
    mask_names: list[str]
    mask_rows: list[int]
    blocks: list[dict[str, Layer]]
    """For each block its layer along time (``t``) and along the bands (``b``)."""
    scales: list[tuple[np.ndarray, np.ndarray]]
    masks: dict[str, np.ndarray]
    epsilon: float


def split(path: Path) -> Parts:
    """Cut the Voice Isolator at ``path`` into the parts for ONNX Runtime and the weights for the shaders."""
    try:
        import onnx
        from onnx import helper, numpy_helper, shape_inference
        from onnx.utils import Extractor
    except ImportError:
        raise Unavailable("the onnx package is not installed") from None

    model = onnx.load(str(path))
    graph = model.graph
    nodes = {n.name: n for n in graph.node}
    weights = {t.name: t for t in graph.initializer}
    weights.update({n.output[0]: n.attribute[0].t for n in graph.node
                    if n.op_type == "Constant" and n.attribute and n.attribute[0].HasField("t")})

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise Unavailable(f"this Voice Isolator is not the model the Vulkan shaders were written for: {what}")

    def node(name: str, op: str) -> Any:
        n = nodes.get(name)
        need(n is not None and n.op_type == op, f"no {op} named {name}")
        return n

    def weight(name: str, op: str, *shape: int, at: int | None = None) -> np.ndarray:
        """The weight a node takes: its input ``at``, or the only weight among its inputs."""
        n = node(name, op)
        names = [n.input[at]] if at is not None else [i for i in n.input if i in weights]
        need(len(names) == 1 and names[0] in weights, f"{name} has no single weight")
        a = numpy_helper.to_array(weights[names[0]]).astype(np.float32)
        need(a.shape == shape, f"the weight of {name} is {a.shape}, not {shape}")
        return a

    def attrs(n: Any) -> dict[str, Any]:
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    epsilons = set()
    sep = "/separator/separation"
    blocks = []
    for k in BLOCKS:
        layers = {}
        for mode, part in (("t", "band_rnn"), ("b", "band_comm")):
            base = f"{sep}.{k}/{part}"
            epsilons.add(attrs(node(f"{base}/norm/InstanceNormalization", "InstanceNormalization"))["epsilon"])
            lstm = node(f"{base}/rnn/LSTM", "LSTM")
            a = attrs(lstm)
            need(a.get("direction") == b"bidirectional" and a.get("hidden_size") == H and len(lstm.input) == 7
                 and not lstm.input[4], f"{lstm.name} is not a two-way LSTM of {H}")
            for state in lstm.input[5:7]:
                made = next((n for n in graph.node if state in n.output), None)
                need(made is not None and made.op_type == "ConstantOfShape"
                     and not ("value" in attrs(made) and numpy_helper.to_array(attrs(made)["value"]).any()),
                     f"{lstm.name} doesn't start from zeros")
            g = attrs(node(f"{base}/proj/Gemm", "Gemm"))
            need(g.get("transB") == 1 and g.get("alpha", 1.0) == 1.0 and g.get("beta", 1.0) == 1.0, f"{base}/proj/Gemm")
            layers[mode] = Layer(
                weight(f"{base}/norm/Mul", "Mul", N, 1).reshape(N), weight(f"{base}/norm/Add", "Add", N, 1).reshape(N),
                weight(lstm.name, "LSTM", 2, 4 * H, N, at=1), weight(lstm.name, "LSTM", 2, 4 * H, H, at=2),
                weight(lstm.name, "LSTM", 2, 8 * H, at=3),
                weight(f"{base}/proj/Gemm", "Gemm", N, 2 * H, at=1), weight(f"{base}/proj/Gemm", "Gemm", N, at=2))
        blocks.append(layers)
    scales = []
    for k in range(BLOCKS[0] + 1, BLOCKS[-1], 2):
        w = node(f"{sep}.{k}/fc/linear/MatMul", "MatMul")
        a = numpy_helper.to_array(weights[next(i for i in w.input if i in weights)]).astype(np.float32)
        need(a.ndim == 2 and a.shape[1] == N, f"the scale of layer {k} is {a.shape}")
        scales.append((a, weight(f"{sep}.{k}/fc/linear/Add", "Add", N)))
    masks: dict[str, list[np.ndarray]] = defaultdict(list)
    mask_names, mask_rows = [], []
    for k in range(BANDS):
        base = f"/mask.{k}/mask.{k}"
        epsilons.add(attrs(node(f"{base}.0/InstanceNormalization", "InstanceNormalization"))["epsilon"])
        masks["gamma"].append(weight(f"{base}.0/Mul", "Mul", N, 1).reshape(N))
        masks["beta"].append(weight(f"{base}.0/Add", "Add", N, 1).reshape(N))
        masks["w1"].append(weight(f"{base}.1/Conv_as_MatMul", "MatMul", D, N))
        masks["b1"].append(weight(f"{base}.1/Conv_bias", "Add", D, 1).reshape(D))
        masks["w2"].append(weight(f"{base}.3/Conv_as_MatMul", "MatMul", D, D))
        masks["b2"].append(weight(f"{base}.3/Conv_bias", "Add", D, 1).reshape(D))
        last = node(f"{base}.5/Conv_as_MatMul", "MatMul")
        w3 = numpy_helper.to_array(weights[next(i for i in last.input if i in weights)]).astype(np.float32)
        need(w3.ndim == 2 and w3.shape[1] == D and len(w3) % 4 == 0, f"the last mask layer of band {k} is {w3.shape}")
        masks["w3"].append(w3)
        masks["b3"].append(weight(f"{base}.5/Conv_bias", "Add", len(w3), 1).reshape(len(w3)))
        mask_names.append(node(f"{base}.5/Conv_bias", "Add").output[0])
        mask_rows.append(len(w3))
    need(len(epsilons) == 1, "the normalizations differ")
    features = node(f"{sep}.{BLOCKS[0]}/Reshape", "Reshape").input[0]
    voiceprint = node(f"{sep}.{BLOCKS[0] + 1}/Expand", "Expand").input[0]

    # The tail is everything that depends on a mask network's output. What it
    # takes from elsewhere, apart from constants, the head has to hand over.
    made_by = {o: n for n in graph.node for o in n.output}
    takers: dict[str, list[Any]] = defaultdict(list)
    for n in graph.node:
        for i in n.input:
            takers[i].append(n)
    tail, todo = {}, list(mask_names)
    while todo:
        for n in takers.get(todo.pop(), ()):
            if n.name not in tail:
                tail[n.name] = n
                todo.extend(n.output)
    handed: list[str] = []
    for n in tail.values():
        for i in n.input:
            maker = made_by.get(i)
            if (i and i not in mask_names and i not in weights and i not in handed
                    and (maker is None or maker.name not in tail)):
                handed.append(i)
    inputs = [i.name for i in graph.input if i.name not in weights]
    need(not any(i in inputs for i in handed), "the last part reads the model's inputs")
    # The cut tensors need their types, which the model only has for its inputs and outputs.
    extractor = Extractor(shape_inference.infer_shapes(model))

    def part(takes: list[str], gives: list[str]) -> bytes:
        try:
            sub = extractor.extract_model(takes, gives)
        except Exception as err:  # noqa: BLE001 - whatever stops the cut is reported as the model not fitting
            raise Unavailable(f"the Voice Isolator can't be cut for the Vulkan shaders: {err}") from None
        need(not any(n.op_type == "LSTM" for n in sub.graph.node), "a part for the CPU has an LSTM in it")
        # The model's version, copyright and license go with each part.
        del sub.metadata_props[:]
        sub.metadata_props.extend(model.metadata_props)
        return sub.SerializeToString()

    return Parts(head=part(inputs, list(dict.fromkeys([features, voiceprint, *handed]))),
                 tail=part([*mask_names, *handed], [o.name for o in graph.output]),
                 features=features, voiceprint=voiceprint, mask_names=mask_names, mask_rows=mask_rows,
                 blocks=blocks, scales=scales,
                 masks={k: (np.concatenate(v) if k in ("w3", "b3") else np.stack(v)) for k, v in masks.items()},
                 epsilon=float(epsilons.pop()))


# --------------------------------------------------------------------------
# The shaders
# --------------------------------------------------------------------------
# Every shader takes one set of parameters, P:
#   step, steps  the first step of this dispatch and how many the sequence has
#   rows, bq     the batch: its rows, and its vec4s of rows with the padding
#   ss, si, tt   F[row * ss + channel * tt + step * si] is the feature of a
#                batch row at a step. tt is also the number of frames.
# Along time the steps are the frames and the batch is the bands. Along the
# bands the steps are the bands and the batch is the frames. The mask
# networks take the second set: their batch is the frames too.

_P = """
struct P { step: u32, steps: u32, rows: u32, bq: u32, ss: u32, si: u32, tt: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
"""

_STATS_A = (("F", "r", "array<f32>"), ("Part", "w", "array<vec2<f32>>")), """
// A channel of a batch row over its steps: its mean and the squares around it.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let base = id.y * p.ss + id.x * p.tt;
  var sum = 0.0;
  for (var s = 0u; s < p.steps; s++) { sum += F[base + s * p.si]; }
  let mean = sum / f32(p.steps);
  var squares = 0.0;
  for (var s = 0u; s < p.steps; s++) { let d = F[base + s * p.si] - mean; squares += d * d; }
  Part[id.y * 128u + id.x] = vec2<f32>(mean, squares);
}
"""


def _stats_b(epsilon: float) -> tuple:
    return (("Part", "r", "array<vec2<f32>>"), ("Stats", "w", "array<vec2<f32>>")), f"""
// A batch row from its channels: its mean and one over its deviation.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let row = id.x;
  if (row >= p.rows) {{ return; }}
  var mean = 0.0;
  for (var n = 0u; n < 128u; n++) {{ mean += Part[row * 128u + n].x; }}
  mean = mean / 128.0;
  var squares = 0.0;
  for (var n = 0u; n < 128u; n++) {{
    let c = Part[row * 128u + n];
    squares += c.y + f32(p.steps) * (c.x - mean) * (c.x - mean);
  }}
  Stats[row] = vec2<f32>(mean, inverseSqrt(squares / (128.0 * f32(p.steps)) + {epsilon!r}));
}}
"""


_NORM = (("F", "r", "array<f32>"), ("Stats", "r", "array<vec2<f32>>"), ("V", "r", "array<f32>"),
         ("X", "w", "array<vec4<f32>>")), """
// A block's input, normalized and laid out for its LSTM: X[step][channel][four rows].
fn one(row: u32, n: u32, step: u32) -> f32 {
  if (row >= p.rows) { return 0.0; }
  let s = Stats[row];
  return (F[row * p.ss + n * p.tt + step * p.si] - s.x) * s.y * V[n] + V[128u + n];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = id.x; let step = p.step + id.z; let row = id.y * 4u;
  X[(step * 128u + n) * p.bq + id.y] = vec4<f32>(one(row, n, step), one(row + 1u, n, step), one(row + 2u, n, step), one(row + 3u, n, step));
}
"""

_MASK_NORM = (("F", "r", "array<f32>"), ("Stats", "r", "array<vec2<f32>>"), ("V", "r", "array<f32>"),
              ("X", "w", "array<vec4<f32>>")), """
// A band over its channels and frames, normalized for its mask network: X[band][channel][four frames].
fn one(band: u32, n: u32, frame: u32) -> f32 {
  if (frame >= p.tt) { return 0.0; }
  let s = Stats[band];
  return (F[(band * 128u + n) * p.tt + frame] - s.x) * s.y * V[band * 256u + n] + V[band * 256u + 128u + n];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = id.x; let band = p.step + id.z; let frame = id.y * 4u;
  X[(band * 128u + n) * p.bq + id.y] = vec4<f32>(one(band, n, frame), one(band, n, frame + 1u), one(band, n, frame + 2u), one(band, n, frame + 3u));
}
"""


def _sums(count: str, weights: str, first: str, source: str, at: str, quads: int, turn: int, pad: str) -> str:
    """A loop that adds ``count`` inputs into the sums a0, a1, ...: one for each
    batch row, each a vec4 over the four gates or outputs. A turn loads
    ``turn`` weights and the rows' values first and adds after, which a GPU
    runs side by side."""
    lines = [f"{pad}for (var k = 0u; k < {count}; k += {turn}u) {{"]
    for u in range(turn):
        lines.append(f"{pad}  let w{u} = vec4<f32>({weights}[{first} + k + {u}u]);")
        lines += [f"{pad}  let v{u}_{m} = {source}[({at} + k + {u}u) * p.bq + q + {m}u];" for m in range(quads)]
    for u in range(turn):
        lines += [f"{pad}  a{4 * m + c} += w{u} * v{u}_{m}.{'xyzw'[c]};" for m in range(quads) for c in range(4)]
    return "\n".join(lines + [f"{pad}}}"])


def _across(m: int, part: str) -> str:
    """One gate or output of the four rows of a vec4 of rows."""
    return "vec4<f32>(" + ", ".join(f"a{4 * m + row}.{part}" for row in range(4)) + ")"


def _lstm(half: bool, quads: int, group: int, turn: int) -> tuple:
    """One step of both directions. G[direction][unit][input] has the four
    gates' weights for an input: the layer's inputs first, then its own last
    output. Y[step][direction][unit][four rows] is the output, which the next
    step reads back."""
    start = "\n".join(f"  var a{i} = bias;" for i in range(4 * quads))
    out = "\n".join(f"""  {{
    var c = vec4<f32>(0.0);
    if (p.step > 0u) {{ c = C[cell + {m}u]; }}
    c = sig({_across(m, 'z')}) * c + sig({_across(m, 'x')}) * tanh({_across(m, 'w')});
    C[cell + {m}u] = c;
    Y[here + {m}u] = sig({_across(m, 'y')}) * tanh(c);
  }}""" for m in range(quads))
    return (("X", "r", "array<vec4<f32>>"), ("G", "r", f"array<vec4<{'f16' if half else 'f32'}>>"),
            ("Bias", "r", "array<vec4<f32>>"), ("Y", "w", "array<vec4<f32>>"), ("C", "w", "array<vec4<f32>>")), f"""
fn sig(v: vec4<f32>) -> vec4<f32> {{ return 1.0 / (1.0 + exp(-v)); }}
@compute @workgroup_size({group})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let unit = id.x; let q = id.y * {quads}u; let d = id.z;
  let t = select(p.step, p.steps - 1u - p.step, d == 1u);
  let bias = Bias[d * 256u + unit];
{start}
  let g = (d * 256u + unit) * 384u;
{_sums("128u", "G", "g", "X", "t * 128u", quads, turn, "  ")}
  if (p.step > 0u) {{
    let last = select(t - 1u, t + 1u, d == 1u);
{_sums("256u", "G", "g + 128u", "Y", "(last * 2u + d) * 256u", quads, turn, "    ")}
  }}
  let cell = (d * 256u + unit) * p.bq + q;
  let here = ((t * 2u + d) * 256u + unit) * p.bq + q;
{out}
}}
"""


def _proj(quads: int, group: int, turn: int) -> tuple:
    """Both directions' outputs back to the channels, added to F. V has the
    projection's bias and the scale by the voiceprint that follows the block.
    An invocation takes four channels."""
    start = "\n".join(f"  var a{i} = vec4<f32>(0.0);" for i in range(4 * quads))
    puts = "\n".join(f"  put(a{i}, (q + {i // 4}u) * 4u + {i % 4}u, id.x * 4u, step);" for i in range(4 * quads))
    return (("Y", "r", "array<vec4<f32>>"), ("W", "r", "array<vec4<f32>>"), ("V", "r", "array<f32>"),
            ("F", "w", "array<f32>")), f"""
fn put(a: vec4<f32>, row: u32, n: u32, step: u32) {{
  if (row >= p.rows) {{ return; }}
  let i = row * p.ss + n * p.tt + step * p.si;
  F[i] = (F[i] + a.x + V[n]) * V[128u + n];
  F[i + p.tt] = (F[i + p.tt] + a.y + V[n + 1u]) * V[129u + n];
  F[i + 2u * p.tt] = (F[i + 2u * p.tt] + a.z + V[n + 2u]) * V[130u + n];
  F[i + 3u * p.tt] = (F[i + 3u * p.tt] + a.w + V[n + 3u]) * V[131u + n];
}}
@compute @workgroup_size({group})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let q = id.y * {quads}u; let step = p.step + id.z;
{start}
{_sums("512u", "W", "id.x * 512u", "Y", "step * 512u", quads, turn, "  ")}
{puts}
}}
"""


def _dense(layer: str, quads: int, group: int, turn: int) -> tuple:
    """One layer of the mask networks, four outputs for an invocation.
    ``first`` and ``mid`` run every band's own layer, with tanh. ``last`` has
    the outputs of all bands in one list, and Band says whose each four are."""
    start = "\n".join(f"  var a{i} = bias;" for i in range(4 * quads))
    act = "tanh" if layer != "last" else ""
    puts = "\n".join(f"  O[(to + {c}u) * p.bq + q + {m}u] = {act}({_across(m, 'xyzw'[c])});"
                     for m in range(quads) for c in range(4))
    count = "128u" if layer == "first" else "512u"
    if layer == "last":
        head = f"""  if (id.x >= arrayLength(&Band)) {{ return; }}
  let band = Band[id.x]; let j = id.x; let to = id.x * 4u;
  let w = id.x * 512u; let src = band * 512u;"""
    else:
        head = f"""  let band = p.step + id.z; let j = band * 128u + id.x; let to = band * 512u + id.x * 4u;
  let w = j * {count}; let src = band * {count};"""
    bindings = [("A", "r", "array<vec4<f32>>"), ("W", "r", "array<vec4<f32>>"), ("K", "r", "array<vec4<f32>>"),
                ("O", "w", "array<vec4<f32>>")] + ([("Band", "r", "array<u32>")] if layer == "last" else [])
    return tuple(bindings), f"""
@compute @workgroup_size({group})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let q = id.y * {quads}u;
{head}
  let bias = K[j];
{start}
{_sums(count, "W", "w", "A", "src", quads, turn, "  ")}
{puts}
}}
"""


class _Program:
    """A compute shader with its bind group layout, both from one list of
    buffers: (name, ``r`` or ``w``, type). Binding 0 is the parameters."""

    def __init__(self, device: Any, source: tuple, half: bool = False) -> None:
        import wgpu

        bindings, code = source
        stage = wgpu.ShaderStage.COMPUTE
        entries = [{"binding": 0, "visibility": stage,
                    "buffer": {"type": "uniform", "has_dynamic_offset": True, "min_binding_size": 32}}]
        lines = ["enable f16;" if half else "", _P]
        for i, (name, access, kind) in enumerate(bindings, 1):
            entries.append({"binding": i, "visibility": stage,
                            "buffer": {"type": "read-only-storage" if access == "r" else "storage"}})
            lines.append(f"@group(0) @binding({i}) var<storage, {'read' if access == 'r' else 'read_write'}> {name}: {kind};")
        self._device = device
        self._layout = device.create_bind_group_layout(entries=entries)
        self.pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            compute={"module": device.create_shader_module(code="\n".join(lines) + code), "entry_point": "main"})

    def bind(self, params: Any, *buffers: Any) -> Any:
        return self._device.create_bind_group(layout=self._layout, entries=[
            {"binding": 0, "resource": {"buffer": params, "offset": 0, "size": 32}},
            *({"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(buffers, 1))])


def _fours(a: np.ndarray) -> np.ndarray:
    """Weights [.., outputs, inputs] as [.., outputs / 4, inputs, 4]: a vec4 is one input's weight for four outputs."""
    return np.ascontiguousarray(a.reshape(*a.shape[:-2], a.shape[-2] // 4, 4, a.shape[-1]).swapaxes(-1, -2))


def _quads(rows: int) -> int:
    """A batch in vec4s of rows, padded to what the largest invocation takes."""
    return -(-rows // 32) * 8


# --------------------------------------------------------------------------
# The GPU
# --------------------------------------------------------------------------

def adapters() -> list[Any]:
    """The Vulkan devices of this machine, a real GPU first."""
    import wgpu

    order = {"DiscreteGPU": 0, "IntegratedGPU": 1}
    found = [a for a in wgpu.gpu.enumerate_adapters_sync() if a.info["backend_type"] == "Vulkan"]
    return sorted(found, key=lambda a: order.get(a.info["adapter_type"], 2 if a.info["adapter_type"] != "CPU" else 3))


def label(adapter: Any) -> str:
    info = adapter.info
    driver = str(info.get("description", "")).strip()
    return f"{info['device']}{' (' + driver + ')' if driver else ''}"


def choose(want: str = "", software: bool = False) -> Any:
    """The GPU to run on: the first real one, or the one with ``want`` in its
    name (``VULKAN_DEVICE``). ``software``: also take a renderer that runs on
    the CPU, such as Mesa's llvmpipe, which only the tests use."""
    try:
        found = adapters()
    except ImportError:
        raise Unavailable("the wgpu package is not installed") from None
    except Exception as err:  # noqa: BLE001 - every way of finding no GPU is reported the same
        raise Unavailable(f"Vulkan can't start: {str(err).strip().splitlines()[-1][:300]}") from None
    names = ", ".join(label(a) for a in found) or "none"
    if want:
        found = [a for a in found if want.lower() in label(a).lower()]
        if not found:
            raise Unavailable(f"Vulkan found no GPU with {want!r} in its name. It found: {names}")
    real = [a for a in found if a.info["adapter_type"] != "CPU"]
    if real:
        return real[0]
    if found and software:
        return found[0]
    if found:
        raise Unavailable(f"Vulkan found no GPU, only {label(found[0])}, which runs on the CPU. In a container, "
                          "pass the GPU with --device /dev/dri, or the devices section of compose.vulkan.yaml. "
                          "Without a GPU, use the CPU image (ghcr.io/jxlarrea/kiosk-satellite-ground-control:latest)")
    raise Unavailable("Vulkan found no GPU. In a container, pass the GPU with --device /dev/dri, "
                      "or the devices section of compose.vulkan.yaml")


class _Mode(NamedTuple):
    """A direction of a block, for one number of frames."""
    steps: int
    rows: int
    bq: int
    offset: int
    """Where its parameters start in the parameter buffer, one set for each step."""


class _Jobs:
    """The dispatches of a run, sent to the GPU in jobs of about JOB_S."""

    def __init__(self, engine: "Engine") -> None:
        self._e = engine
        self._encoder = self._pass = None
        self._work = 0.0
        self.count = 0

    def _open(self) -> Any:
        if self._pass is None:
            self._encoder = self._e.device.create_command_encoder()
            self._pass = self._encoder.begin_compute_pass()
            self._pipeline = None
        return self._pass

    def add(self, program: _Program, group: Any, offset: int, size: tuple, work: float) -> None:
        """One dispatch. ``work`` is how many multiply-adds it is."""
        if self._work and self._work + work > self._e.rate * JOB_S:
            self.send()
        p = self._open()
        if self._pipeline is not program:
            p.set_pipeline(program.pipeline)
            self._pipeline = program
        p.set_bind_group(0, group, [offset], 0, 1)
        p.dispatch_workgroups(*size)
        self._work += work

    def slices(self, program: _Program, group: Any, offset: int, size: tuple, count: int, work: float) -> None:
        """A dispatch over ``count`` steps or bands that don't depend on each
        other, in as many pieces as the job length asks for. ``work`` is for one."""
        at = 0
        while at < count:
            room = self._e.rate * JOB_S - self._work
            n = int(min(count - at, max(1, room // work)))
            self.add(program, group, offset + at * ROW, (*size, n), work * n)
            at += n

    def send(self) -> None:
        if self._pass is None:
            return
        self._pass.end()
        started = time.perf_counter()
        self._e.device.queue.submit([self._encoder.finish()])
        self._e.device.queue.on_submitted_work_done_sync()
        took = max(time.perf_counter() - started, 1e-6)
        # Slower than thought counts at once, faster only bit by bit: a job
        # that runs too long loses the GPU, one that is too short costs nothing.
        rate = self._work / took
        self._e.rate = rate if rate < self._e.rate else 0.5 * (self._e.rate + rate)
        self._encoder = self._pass = None
        self._work = 0.0
        self.count += 1


class Engine:
    """The blocks and the mask networks on the GPU."""

    def __init__(self, adapter: Any, parts: Parts, shapes: tuple = LSTM_SHAPES) -> None:
        import wgpu

        self.half = "shader-f16" in adapter.features
        """The GPU loads half precision numbers, which the LSTM weights are then
        stored in: a quarter less time for an LSTM on an AMD GPU. Everything
        is computed in single precision. Over 95 test commands the audio was
        83 dB under the CPU's at the median and 28 dB at the worst, where the
        model made for the CPU is about 36 dB from the full one on every
        command. With full precision weights the worst was 80 dB."""
        self.device = d = adapter.request_device_sync(required_features=["shader-f16"] if self.half else [])
        self.name = label(adapter)
        self.rate = 2e8
        """Multiply-adds a second this GPU has done, which sizes the jobs. It
        starts far too low, so the first jobs are tiny, and is measured from then on."""
        self._usage = u = wgpu.BufferUsage
        store = lambda a, kind=np.float32: d.create_buffer_with_data(  # noqa: E731
            data=np.ascontiguousarray(a, kind), usage=u.STORAGE | u.COPY_DST)
        self._stats = _Program(d, _STATS_A), _Program(d, _stats_b(parts.epsilon))
        self._norm = _Program(d, _NORM)
        self._shapes = shapes
        self._lstms = {s: _Program(d, _lstm(self.half, *s), self.half) for s in shapes}
        self.shape = {"t": shapes[0], "b": shapes[-1]}
        """The LSTM shape of each direction, until ``tune`` has timed them."""
        self._proj = _Program(d, _proj(*DENSE_SHAPE))
        self._blocks = []
        for layers in parts.blocks:
            block = {}
            for m, layer in layers.items():
                gates = np.concatenate([layer.w.reshape(2, 4, H, N).transpose(0, 2, 3, 1),
                                        layer.r.reshape(2, 4, H, H).transpose(0, 2, 3, 1)], axis=2)
                bias = (layer.b[:, :4 * H] + layer.b[:, 4 * H:]).reshape(2, 4, H).transpose(0, 2, 1)
                block[m] = (store(gates, np.float16 if self.half else np.float32), store(bias),
                            store(_fours(layer.proj)), store(np.concatenate([layer.gamma, layer.beta])),
                            store(np.concatenate([layer.proj_bias, np.ones(N, np.float32)])))
            self._blocks.append(block)
        self._scales = parts.scales
        mask = parts.masks
        self._rows = len(mask["w3"])
        self._mask_norm = _Program(d, _MASK_NORM), store(np.concatenate([mask["gamma"], mask["beta"]], axis=1))
        self._dense = ((_Program(d, _dense("first", *DENSE_SHAPE)), store(_fours(mask["w1"])), store(mask["b1"])),
                       (_Program(d, _dense("mid", *DENSE_SHAPE)), store(_fours(mask["w2"])), store(mask["b2"])),
                       (_Program(d, _dense("last", *DENSE_SHAPE)), store(_fours(mask["w3"])), store(mask["b3"])))
        self._band = store(np.repeat(np.arange(BANDS), np.array(parts.mask_rows) // 4), np.uint32)
        self._capacity = self._frames = 0
        self.jobs = 0
        """GPU jobs the last run was sent as."""

    def reserve(self, frames: int) -> None:
        """Buffers for up to ``frames`` frames. Fewer frames use the start of each."""
        d, u = self.device, self._usage
        bq = _quads(frames)
        buf = lambda size, extra=0: d.create_buffer(size=size, usage=u.STORAGE | extra)  # noqa: E731
        self._features = buf(BANDS * N * frames * 4, u.COPY_DST)
        self._out = buf(self._rows * bq * 16, u.COPY_SRC)
        self._params = d.create_buffer(size=(frames + BANDS) * ROW, usage=u.UNIFORM | u.COPY_DST)
        part, stats = buf(max(BANDS, frames) * N * 8), buf(max(BANDS, frames) * 8)
        # Along time and along the bands take the same amount, apart from the padding.
        x = buf(max(frames * N * 8, BANDS * N * bq) * 16)
        y = buf(max(frames * 2 * H * 8, BANDS * 2 * H * bq) * 16)
        cell = buf(2 * H * max(8, bq) * 16)
        hidden = buf(BANDS * D * bq * 16), buf(BANDS * D * bq * 16)
        self._stat_groups = (self._stats[0].bind(self._params, self._features, part),
                             self._stats[1].bind(self._params, part, stats))
        self._groups = [{m: (self._norm.bind(self._params, self._features, stats, vectors, x),
                             {s: prog.bind(self._params, x, gates, bias, y, cell) for s, prog in self._lstms.items()},
                             self._proj.bind(self._params, y, proj, after, self._features))
                         for m, (gates, bias, proj, vectors, after) in block.items()} for block in self._blocks]
        norm, vectors = self._mask_norm
        (first, w1, k1), (mid, w2, k2), (last, w3, k3) = self._dense
        self._mask_groups = (norm.bind(self._params, self._features, stats, vectors, x),
                             first.bind(self._params, x, w1, k1, hidden[0]),
                             mid.bind(self._params, hidden[0], w2, k2, hidden[1]),
                             last.bind(self._params, hidden[1], w3, k3, self._out, self._band))
        self._capacity, self._frames = frames, 0

    def _set(self, frames: int) -> None:
        """The parameters of both directions for this many frames: a set for each step."""
        along_time = _Mode(frames, BANDS, BANDS // 4, 0)
        along_bands = _Mode(BANDS, frames, _quads(frames), frames * ROW)
        params = np.zeros((frames + BANDS, ROW // 4), np.uint32)
        for mode, strides in ((along_time, (N * frames, 1)), (along_bands, (1, N * frames))):
            rows = params[mode.offset // ROW:][:mode.steps]
            rows[:, 0] = np.arange(mode.steps)
            rows[:, 1:7] = (mode.steps, mode.rows, mode.bq, *strides, frames)
        self.device.queue.write_buffer(self._params, 0, params)
        self._modes, self._frames = {"t": along_time, "b": along_bands}, frames

    def _stats_of(self, jobs: _Jobs, mode: _Mode) -> None:
        work = mode.rows * N * mode.steps * 2.0
        jobs.add(self._stats[0], self._stat_groups[0], mode.offset, (N // 64, mode.rows, 1), work)
        jobs.add(self._stats[1], self._stat_groups[1], mode.offset, (-(-mode.rows // 64), 1, 1), mode.rows * N)

    def _lstm_steps(self, jobs: _Jobs, mode: _Mode, groups: dict, shape: tuple, steps: int | None = None) -> None:
        quads, group, _ = shape
        work = 2.0 * H * 4 * mode.bq * 4 * (N + H)
        for step in range(mode.steps if steps is None else steps):
            jobs.add(self._lstms[shape], groups[shape], mode.offset + step * ROW, (H // group, mode.bq // quads, 2), work)

    def run(self, features: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        """The outputs of the mask networks, [rows of all bands, frames], from
        the head's features [1, bands * channels, frames] and voiceprint."""
        frames = features.shape[-1]
        if frames > self._capacity:
            self.reserve(frames)
        if frames != self._frames:
            self._set(frames)
        queue = self.device.queue
        queue.write_buffer(self._features, 0, np.ascontiguousarray(features, np.float32))
        v = voiceprint.reshape(-1)
        for (w, b), block in zip(self._scales, self._blocks):
            queue.write_buffer(block["b"][4], N * 4, (v @ w + b).astype(np.float32))
        jobs = _Jobs(self)
        quads, group, _ = DENSE_SHAPE
        for groups in self._groups:
            for m, (norm, lstm, proj) in groups.items():
                mode = self._modes[m]
                self._stats_of(jobs, mode)
                jobs.slices(self._norm, norm, mode.offset, (N // 64, mode.bq), mode.steps, mode.bq * 4.0 * N)
                self._lstm_steps(jobs, mode, lstm, self.shape[m])
                jobs.slices(self._proj, proj, mode.offset, (N // 4 // group, mode.bq // quads), mode.steps,
                            mode.bq * 4.0 * N * 2 * H)
        # A mask network normalizes its band over the channels and frames, as a block does along time.
        self._stats_of(jobs, self._modes["t"])
        mode = self._modes["b"]
        norm, first, mid, last = self._mask_groups
        rows = mode.bq * 4.0
        jobs.slices(self._mask_norm[0], norm, mode.offset, (N // 64, mode.bq), BANDS, rows * N)
        jobs.slices(self._dense[0][0], first, mode.offset, (D // 4 // group, mode.bq // quads), BANDS, rows * D * N)
        jobs.slices(self._dense[1][0], mid, mode.offset, (D // 4 // group, mode.bq // quads), BANDS, rows * D * D)
        jobs.add(self._dense[2][0], last, mode.offset, (-(-self._rows // 4 // group), mode.bq // quads, 1), rows * self._rows * D)
        jobs.send()
        self.jobs = jobs.count
        out = np.frombuffer(queue.read_buffer(self._out, 0, self._rows * mode.bq * 16), np.float32)
        return np.ascontiguousarray(out.reshape(self._rows, mode.bq * 4)[:, :frames])

    def tune(self, frames: int) -> dict[str, tuple]:
        """Time each LSTM shape in both directions on ``frames`` frames and keep
        the fastest of each. What the buffers hold is used as it is: the time
        doesn't depend on the numbers."""
        if frames > self._capacity:
            self.reserve(frames)
        self._set(frames)
        took: dict[str, dict[tuple, float]] = {"t": {}, "b": {}}
        for m, mode in self._modes.items():
            steps = min(mode.steps, 24)
            for shape in self._shapes:
                best = float("inf")
                for _ in range(3):
                    jobs = _Jobs(self)
                    started = time.perf_counter()
                    self._lstm_steps(jobs, mode, self._groups[0][m][1], shape, steps)
                    jobs.send()
                    best = min(best, time.perf_counter() - started)
                took[m][shape] = best
            self.shape[m] = min(took[m], key=took[m].get)
        self._frames = 0
        return self.shape


class Model:
    """The Voice Isolator with its middle on the GPU. It takes and gives what the ONNX model does."""

    def __init__(self, adapter: Any, path: Path, threads: int, longest: int, shapes: tuple = LSTM_SHAPES) -> None:
        import onnxruntime as ort

        parts = split(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.log_severity_level = 3
        cpu = ["CPUExecutionProvider"]
        self._options = so
        self._head = ort.InferenceSession(parts.head, so, providers=cpu)
        self._tail = ort.InferenceSession(parts.tail, so, providers=cpu)
        self.meta = dict(self._head.get_modelmeta().custom_metadata_map)
        self.dim = self._head.get_inputs()[1].shape[-1]
        self._head_outputs = [o.name for o in self._head.get_outputs()]
        self._tail_inputs = [i.name for i in self._tail.get_inputs()]
        self._features, self._voiceprint = parts.features, parts.voiceprint
        self._masks = list(zip(parts.mask_names, parts.mask_rows))
        self.engine = Engine(adapter, parts, shapes)
        # Buffers for the longest run now, so the first long command doesn't make them.
        silence = {"audio": np.zeros((1, longest), np.float32), "voiceprint": np.zeros((1, self.dim), np.float32)}
        self._longest = self._head.run([parts.features], silence)[0].shape[-1]
        self.engine.reserve(self._longest)

    def run(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        """``audio`` [1, samples] and ``voiceprint`` [1, dim] to the cleaned audio [1, samples]."""
        values = dict(zip(self._head_outputs, self._head.run(None, {"audio": audio, "voiceprint": voiceprint})))
        masks = self.engine.run(values[self._features], values[self._voiceprint])
        row = 0
        for name, rows in self._masks:
            values[name] = masks[None, row:row + rows]
            row += rows
        return self._tail.run(None, {name: values[name] for name in self._tail_inputs})[0]

    def check(self, path: Path, cpu_path: Path | None = None, software: bool = False) -> dict[str, Any]:
        """Time the LSTM shapes, then run a second of noise here and through the
        whole model on the CPU. Raises Unavailable when the audio differs and
        Slow when the CPU is faster. ``cpu_path``: the model made for the CPU,
        which is what Isolation would run without the GPU. ``software``: a
        renderer on the CPU was asked for (the tests), so slower is fine."""
        import onnxruntime as ort

        shapes = self.engine.tune(self._longest)
        rng = np.random.default_rng(0)
        audio = (0.1 * rng.standard_normal((1, CHECK_SAMPLES))).astype(np.float32)
        voiceprint = rng.standard_normal((1, self.dim)).astype(np.float32)
        voiceprint /= np.linalg.norm(voiceprint)
        feed = {"audio": audio, "voiceprint": voiceprint}
        self.run(audio, voiceprint)
        gpu_s = float("inf")
        for _ in range(3):
            started = time.perf_counter()
            there = self.run(audio, voiceprint)
            gpu_s = min(gpu_s, time.perf_counter() - started)
        whole = ort.InferenceSession(str(path), self._options, providers=["CPUExecutionProvider"])
        here = whole.run(None, feed)[0]
        off = float(10 * np.log10(np.mean((there - here) ** 2) / np.mean(here ** 2) + 1e-20))
        if not off < CHECK_DB:
            raise Unavailable(f"{self.engine.name} doesn't compute the Voice Isolator as the CPU does: its audio is "
                              f"{off:.0f} dB from the CPU's and has to be under {CHECK_DB:.0f} dB. Its Vulkan driver "
                              "may be at fault")
        if cpu_path is not None and Path(cpu_path).is_file():
            whole = ort.InferenceSession(str(cpu_path), self._options, providers=["CPUExecutionProvider"])
        whole.run(None, feed)
        cpu_s = float("inf")
        for _ in range(2):
            started = time.perf_counter()
            whole.run(None, feed)
            cpu_s = min(cpu_s, time.perf_counter() - started)
        result = {"db": round(off, 1), "gpu_ms": round(gpu_s * 1000, 1), "cpu_ms": round(cpu_s * 1000, 1),
                  "shapes": {m: list(s) for m, s in shapes.items()}, "half": self.engine.half}
        if gpu_s > cpu_s and not software:
            raise Slow(f"{self.engine.name} takes {gpu_s * 1000:.0f} ms for a second of audio and this machine's "
                       f"CPU {cpu_s * 1000:.0f} ms, so Isolation runs on the CPU")
        return result


# --------------------------------------------------------------------------
# Messages between the service and a helper process
# --------------------------------------------------------------------------
# A message is a 4 byte length, a JSON header of that length and then as many
# bytes of data as the header's "data" says.

def _write(stream: Any, header: dict[str, Any], *data: bytes) -> None:
    body = b"".join(data)
    head = json.dumps({**header, "data": len(body)}).encode()
    stream.write(struct.pack(">I", len(head)) + head + body)
    stream.flush()


def _read(fd: int, deadline: float | None = None) -> tuple[dict[str, Any], bytes]:
    """The next message from ``fd``. Raises EOFError when the other end is
    gone and TimeoutError when ``deadline`` (a ``time.monotonic``) passes."""
    def take(count: int) -> bytes:
        parts = []
        while count:
            if deadline is not None:
                wait = deadline - time.monotonic()
                if wait <= 0 or not select.select([fd], [], [], wait)[0]:
                    raise TimeoutError
            piece = os.read(fd, min(count, 1 << 20))
            if not piece:
                raise EOFError
            parts.append(piece)
            count -= len(piece)
        return b"".join(parts)

    header = json.loads(take(struct.unpack(">I", take(4))[0]))
    return header, take(header.get("data", 0))


# ---- The other models: ONNX Runtime's WebGPU provider ----

CHECK_COSINE = 0.998
"""Each output of a model on the GPU has to be this close to the CPU's."""
GAIN = 0.9
"""The GPU is used for a model when it takes less than this much of the CPU's time."""


def rewrite(model: Any) -> dict[str, int]:
    """Change what ONNX Runtime's WebGPU provider (1.27) gets wrong, in place:

    * A grouped convolution with several outputs for each group is computed
      wrong, and slowly: the Accurate speaker model's voiceprints came out at
      a cosine of 0.91 from the CPU's. It becomes a plain convolution with
      zeros where the groups didn't connect, which is the same layer.
    * PRelu has no kernel there, so the face recognition models went back to
      the CPU 34 times in a run. It becomes relu(x) - slope * relu(-x).
    """
    from onnx import helper, numpy_helper

    graph = model.graph
    weights = {t.name: t for t in graph.initializer}
    done: dict[str, int] = defaultdict(int)
    nodes = []
    for n in graph.node:
        if n.op_type == "Conv" and n.input[1] in weights:
            group = next((a for a in n.attribute if a.name == "group"), None)
            w = numpy_helper.to_array(weights[n.input[1]])
            outputs, each_in = w.shape[:2]
            if group is not None and group.i > 1 and not (each_in == 1 and outputs == group.i):
                plain = np.zeros((outputs, group.i * each_in, *w.shape[2:]), w.dtype)
                each_out = outputs // group.i
                for o in range(outputs):
                    plain[o, (o // each_out) * each_in:(o // each_out + 1) * each_in] = w[o]
                weights[n.input[1]].CopyFrom(numpy_helper.from_array(plain, n.input[1]))
                group.i = 1
                done["grouped convolutions"] += 1
            nodes.append(n)
        elif n.op_type == "PRelu":
            x, slope, y = n.input[0], n.input[1], n.output[0]
            base = n.name or y
            nodes += [helper.make_node("Relu", [x], [base + "/above"], name=base + "/above"),
                      helper.make_node("Neg", [x], [base + "/negated"], name=base + "/negated"),
                      helper.make_node("Relu", [base + "/negated"], [base + "/below"], name=base + "/below"),
                      helper.make_node("Mul", [base + "/below", slope], [base + "/sloped"], name=base + "/sloped"),
                      helper.make_node("Sub", [base + "/above", base + "/sloped"], [y], name=base + "/joined")]
            done["PRelu"] += 1
        else:
            nodes.append(n)
    del graph.node[:]
    graph.node.extend(nodes)
    return dict(done)


def _sample(session: Any) -> dict[str, np.ndarray]:
    """An input to check and time a model with: a second of audio or a 640 by 480 frame of noise."""
    rng = np.random.default_rng(0)
    feed = {}
    for i in session.get_inputs():
        dims = list(i.shape)
        if len(dims) == 4:
            fill = (1, 3, 480, 640)
        else:
            fill = (1,) * (len(dims) - 1) + (16000,)
        shape = [d if isinstance(d, int) and d > 0 else fill[k] for k, d in enumerate(dims)]
        feed[i.name] = (rng.standard_normal(shape) * (1.0 if len(dims) == 4 else 0.05)).astype(np.float32)
    return feed


def _open_model(path: Path, threads: int, software: bool) -> tuple[Any, dict[str, Any]]:
    """The model at ``path`` on the GPU through ONNX Runtime, checked against
    the CPU and timed. Raises Unavailable when it differs and Slow when the
    CPU is about as fast."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as err:
        raise Unavailable(f"the {err.name} package is not installed") from None
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        raise Unavailable("this ONNX Runtime has no WebGPU provider")
    model = onnx.load(str(path))
    rewritten = rewrite(model)
    quiet = ort.SessionOptions()
    quiet.log_severity_level = 3
    # One of ONNX Runtime's graph changes gives the speaker models a Gemm whose sizes don't fit.
    gpu = ort.InferenceSession(model.SerializeToString(), quiet, disabled_optimizers=["GemmTransposeFusion"],
                               providers=["WebGpuExecutionProvider", "CPUExecutionProvider"])
    del model
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.log_severity_level = 3
    cpu = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = _sample(gpu)

    def timed(session: Any, runs: int) -> tuple[list[np.ndarray], float]:
        out = session.run(None, feed)
        took = []
        for _ in range(runs):
            started = time.perf_counter()
            out = session.run(None, feed)
            took.append(time.perf_counter() - started)
        return out, float(min(took))

    there, gpu_s = timed(gpu, 5)
    here, cpu_s = timed(cpu, 3)
    del cpu
    cosine = 1.0
    for a, b in zip(there, here):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        if a.shape != b.shape:
            cosine = 0.0
        elif np.linalg.norm(b) > 0:
            cosine = min(cosine, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
    if not cosine >= CHECK_COSINE:
        raise Unavailable(f"the GPU doesn't compute {path.name} as the CPU does (cosine {cosine:.3f})")
    if gpu_s > GAIN * cpu_s and not software:
        raise Slow(f"{path.name} takes {gpu_s * 1000:.1f} ms on the GPU and {cpu_s * 1000:.1f} ms on the CPU")
    return gpu, {"inputs": [i.name for i in gpu.get_inputs()], "ranks": [len(i.shape) for i in gpu.get_inputs()],
                 "outputs": len(gpu.get_outputs()),
                 "meta": dict(gpu.get_modelmeta().custom_metadata_map), "cosine": round(cosine, 5),
                 "gpu_ms": round(gpu_s * 1000, 2), "cpu_ms": round(cpu_s * 1000, 2), "rewritten": rewritten}


# ---- The helper process ----

def _serve(argv: list[str]) -> None:
    """What a helper process runs:

    * ``probe`` names the GPU.
    * ``isolation MODEL THREADS LONGEST`` loads the Voice Isolator on the
      shaders, checks it and then answers runs until its input closes.
    * ``model MODEL THREADS`` does the same for another model through ONNX Runtime.
    """
    # Messages go out on what was standard output. Anything a driver or a
    # library prints goes to the log with standard error.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    flags = {a for a in argv if a.startswith("--") and "=" not in a}
    named = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    plain = [a for a in argv if not a.startswith("--")]
    # Not sent back to the CPU for being slower: a renderer on the CPU that the
    # tests asked for, or a helper that starts again after it was chosen.
    software = "--software" in flags or "--chosen" in flags
    try:
        if plain[0] == "model":
            session, info = _open_model(Path(plain[1]), int(plain[2]), software)
        else:
            adapter = choose(named.get("device", ""), software="--software" in flags)
            if plain[0] == "probe":
                found = adapter.info
                _write(out, {"ok": True, "name": label(adapter),
                             "select": f"{found['vendor_id']:x}:{found['device_id']:x}"})
                return
            path, threads, longest = Path(plain[1]), int(plain[2]), int(plain[3])
            model = Model(adapter, path, threads, longest, SMALL_SHAPES if "--small" in flags else LSTM_SHAPES)
            info = {"name": model.engine.name, "meta": model.meta, "dim": model.dim,
                    **model.check(path, Path(named["cpu"]) if named.get("cpu") else None, software)}
    except Slow as err:
        _write(out, {"ok": False, "slow": True, "error": str(err)})
        return
    except Unavailable as err:
        _write(out, {"ok": False, "error": str(err)})
        return
    except Exception as err:  # noqa: BLE001 - the reason goes to the service, which decides
        _write(out, {"ok": False, "error": f"{type(err).__name__}: {str(err).strip()[:400]}"})
        return
    # The model file as it was read, the CPU sessions of the check and their
    # results are done with: 0.2 to 0.4 GB a helper that goes back to the system.
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    _write(out, {"ok": True, **info})
    while True:
        try:
            header, data = _read(0)
        except EOFError:
            return
        try:
            started = time.perf_counter()
            if plain[0] == "model":
                feed, at = {}, 0
                for name, kind, shape in header["inputs"]:
                    count = int(np.prod(shape))
                    feed[name] = np.frombuffer(data, kind, count, at).reshape(shape)
                    at += count * np.dtype(kind).itemsize
                outputs = [np.ascontiguousarray(o) for o in session.run(None, feed)]
                _write(out, {"ok": True, "ms": round((time.perf_counter() - started) * 1000, 2),
                             "outputs": [[str(o.dtype), list(o.shape)] for o in outputs]}, *(o.tobytes() for o in outputs))
            else:
                samples = header["samples"]
                audio = np.frombuffer(data, np.float32, samples)[None]
                voiceprint = np.frombuffer(data, np.float32, model.dim, samples * 4)[None]
                cleaned = model.run(audio, voiceprint)
                _write(out, {"ok": True, "ms": round((time.perf_counter() - started) * 1000, 1), "jobs": model.engine.jobs},
                       np.ascontiguousarray(cleaned[0], np.float32).tobytes())
        except Exception as err:  # noqa: BLE001 - one bad run doesn't end the helper
            _write(out, {"ok": False, "error": f"{type(err).__name__}: {str(err).strip()[:400]}"})


def _spawn(args: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "ground_control.vulkan", *args],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)


def probe(device: str = "", software: bool = False) -> dict[str, str]:
    """The GPU the Vulkan image would run on, found in a process of its own:
    its ``name`` and what selects it among the machine's Vulkan devices
    (``select``). Raises Unavailable with the reason when there is none."""
    proc = _spawn(["probe", f"--device={device}", *(["--software"] if software else [])])
    try:
        header, _ = _read(proc.stdout.fileno(), time.monotonic() + 60.0)
    except (EOFError, TimeoutError):
        header = {"ok": False, "error": "Vulkan can't start: the GPU's driver didn't answer"}
    finally:
        proc.kill()
        proc.wait()
    if not header.get("ok"):
        raise Unavailable(header.get("error", "unknown error"))
    return {"name": header["name"], "select": header["select"]}


class _Helper:
    """A helper process, seen from the service: started, asked and started again when it is lost."""

    RESTARTS = 3
    """Starts in a row that may fail before the GPU is given up for what this helper runs."""

    def __init__(self, what: str, args: list[str], env: dict[str, str] | None = None,
                 smaller: list[str] | None = None) -> None:
        """Starts the helper and waits until it has checked the GPU. Raises
        Slow when the CPU is faster and Unavailable when the GPU can't be used.
        ``smaller``: arguments to add for a second try when the first doesn't
        finish in time."""
        self.what, self._args, self._env, self._smaller = what, args, env, smaller
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._small = False
        self._failures = 0
        self.down: str | None = None
        """Why the GPU was given up, once it was."""
        self.info = self._start()
        """What the helper found when it started."""

    def _start(self, again: bool = False) -> dict[str, Any]:
        """``again``: after a loss. The GPU was already chosen over the CPU
        for this, when the machine was idle, so it isn't sent to the CPU for
        being slower now, with kiosks talking."""
        tries = [self._smaller] if self._small else [[], *([self._smaller] if self._smaller else [])]
        for extra in tries:
            proc = _spawn([*self._args, *extra, *(["--chosen"] if again else [])], self._env)
            try:
                header, _ = _read(proc.stdout.fileno(), time.monotonic() + READY_S)
            except TimeoutError:
                # A driver that hangs on a large shader, as Intel's did on one with 32 sums.
                proc.kill()
                proc.wait()
                again = bool(self._smaller) and not extra
                _LOG.warning("Vulkan: the GPU's driver didn't finish %s in %d s%s", self.what, READY_S,
                             ", trying again with the smallest shaders" if again else "")
                self._small = bool(self._smaller)
                continue
            except EOFError:
                proc.wait()
                raise Unavailable(f"Vulkan can't start {self.what}: its process ended (exit code {proc.returncode}). "
                                  "The lines above it in the log are the driver's") from None
            if not header.get("ok"):
                proc.kill()
                proc.wait()
                raise (Slow if header.get("slow") else Unavailable)(header.get("error", "unknown error"))
            with self._lock:
                self._proc = proc
            return header
        raise Unavailable(f"Vulkan can't start {self.what}: the GPU's driver doesn't finish")

    def _restart(self) -> None:
        """Start the helper again after it was lost. Calls meanwhile fail at
        once: nothing waits for the GPU to come back."""
        while True:
            try:
                info = self._start(again=True)
            except Unavailable as err:
                self._failures += 1
                if self._failures >= self.RESTARTS:
                    self.down = str(err)
                    _LOG.error("%s is off the GPU until Ground Control restarts: %s", self.what, err)
                    return
                _LOG.warning("%s didn't start on the GPU again (%s), trying once more", self.what, err)
                continue
            self.info = info
            _LOG.info("%s is back on the GPU", self.what)
            return

    @property
    def up(self) -> bool:
        return self._proc is not None

    def call(self, header: dict[str, Any], *data: bytes) -> tuple[dict[str, Any], bytes]:
        """One request and its answer. Raises Failed when the helper doesn't
        answer, and starts it again for the next call."""
        if not self._lock.acquire(timeout=RUN_S):
            raise Failed("the GPU is busy")
        try:
            proc = self._proc
            if proc is None:
                raise Failed(self.down or "the GPU's process is starting again")
            try:
                _write(proc.stdin, header, *data)
                answer, body = _read(proc.stdout.fileno(), time.monotonic() + RUN_S)
            except (EOFError, TimeoutError, OSError) as err:
                proc.kill()
                proc.wait()
                self._proc = None
                why = "took too long" if isinstance(err, TimeoutError) else f"ended (exit code {proc.returncode})"
                _LOG.warning("%s: the GPU's process %s. It starts again", self.what, why)
                threading.Thread(target=self._restart, name="vulkan-restart", daemon=True).start()
                raise Failed(f"the GPU's process {why}") from None
            if not answer.get("ok"):
                raise Failed(answer.get("error", "unknown error"))
            self._failures = 0
            return answer, body
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._proc.kill()
                self._proc.wait()
                self._proc = None


class Worker(_Helper):
    """The helper process that runs Isolation on the GPU."""

    def __init__(self, path: Path, threads: int, longest: int, cpu_path: Path | None = None,
                 device: str = "", software: bool = False, env: dict[str, str] | None = None) -> None:
        super().__init__("Isolation", ["isolation", str(path), str(threads), str(longest), f"--device={device}",
                                       *([f"--cpu={cpu_path}"] if cpu_path else []),
                                       *(["--software"] if software else [])], env, ["--small"])
        self.last: dict[str, Any] = {}

    def run(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        """``audio`` cleaned for ``voiceprint``, both one-dimensional. Raises
        Failed when the helper doesn't answer."""
        self.last, data = self.call({"samples": len(audio)}, np.ascontiguousarray(audio, np.float32).tobytes(),
                                    np.ascontiguousarray(voiceprint, np.float32).tobytes())
        return np.frombuffer(data, np.float32).copy()


class Remote:
    """A model that runs on the GPU in a helper process when the GPU is
    faster for it, and here on the CPU when it isn't or the helper is lost.
    It takes the place of an ONNX Runtime session the way ``ov.Model`` does."""

    def __init__(self, gpu: "Gpu", path: Path) -> None:
        self.path, self._threads = path, gpu.threads
        self._session: Any = None
        self._helper: _Helper | None = None
        self.why_cpu: str | None = None
        """Why the model runs on the CPU, when it does."""
        try:
            self._helper = _Helper(path.name, ["model", str(path), str(gpu.threads),
                                               *(["--software"] if gpu.software else [])], gpu.env(models=True))
            info = self._helper.info
            self.inputs, self.outputs, self._meta = list(info["inputs"]), int(info["outputs"]), dict(info["meta"])
            self.ranks = list(info["ranks"])
            _LOG.debug("%s on the GPU: %s ms against %s ms on the CPU, cosine %s", path.name, info["gpu_ms"],
                       info["cpu_ms"], info["cosine"])
        except Unavailable as err:
            self.why_cpu = str(err)
            (_LOG.info if isinstance(err, Slow) else _LOG.warning)("%s runs on the CPU: %s", path.name, err)
            cpu = self._cpu()
            self.inputs = [i.name for i in cpu.get_inputs()]
            self.outputs = len(cpu.get_outputs())
            self._meta = dict(cpu.get_modelmeta().custom_metadata_map)
            self.ranks = [len(i.shape) for i in cpu.get_inputs()]
        self.closed = False

    def _cpu(self) -> Any:
        if self._session is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, self._threads)
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            self._session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        return self._session

    @property
    def on_gpu(self) -> bool:
        return self._helper is not None and self._helper.down is None

    @property
    def gpu_ms(self) -> float | None:
        """What a run took on the GPU in the check at load, in ms. None on the CPU."""
        return float(self._helper.info["gpu_ms"]) if self.on_gpu else None

    def metadata(self, key: str) -> str | None:
        return self._meta.get(key)

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """The model's outputs, in the ONNX file's order."""
        if self._helper is not None and self._helper.up:
            arrays = {name: np.ascontiguousarray(x) for name, x in feeds.items()}
            try:
                answer, data = self._helper.call(
                    {"inputs": [[name, str(x.dtype), list(x.shape)] for name, x in arrays.items()]},
                    *(x.tobytes() for x in arrays.values()))
            except Failed:
                pass  # logged where it failed: this run is done on the CPU
            else:
                outputs, at = [], 0
                for kind, shape in answer["outputs"]:
                    count = int(np.prod(shape))
                    outputs.append(np.frombuffer(data, kind, count, at).reshape(shape).copy())
                    at += count * np.dtype(kind).itemsize
                return outputs
        return self._cpu().run(None, feeds)

    def close(self) -> None:
        self.closed = True
        if self._helper is not None:
            self._helper.close()


class Gpu:
    """The GPU the Vulkan image runs on, seen from the service."""

    def __init__(self, device: str = "", threads: int = 1, software: bool = False) -> None:
        """Finds the GPU in a process of its own, so a machine without a
        usable one stops at once with the reason (Unavailable)."""
        found = probe(device, software)
        self.name, self._select = found["name"], found["select"]
        self.device, self.threads, self.software = device, threads, software
        self._models: list[Remote] = []

    def env(self, models: bool = False) -> dict[str, str]:
        """The environment of a helper process. Mesa's device selection shows
        it only this GPU, which is how ONNX Runtime is kept to it. ``models``:
        the helper loads ONNX Runtime's WebGPU build from ``VULKAN_ORT``, a
        folder beside the CPU build the service itself runs."""
        env = {**os.environ, "MESA_VK_DEVICE_SELECT": self._select + "!"}
        if models and os.environ.get("VULKAN_ORT"):
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ["VULKAN_ORT"], env.get("PYTHONPATH")]))
        return env

    def model(self, path: Path, shapes: int = 1) -> Remote:
        """``shapes`` is taken for ``ov.Gpu``'s sake: ONNX Runtime keeps what it compiled for every shape."""
        remote = Remote(self, path)
        self._models.append(remote)
        return remote

    def time(self, path: Path) -> float | None:
        """What a run of the model at ``path`` takes on the GPU, in ms, without
        keeping it loaded. None when the model would run on the CPU."""
        remote = Remote(self, path)
        try:
            return remote.gpu_ms
        finally:
            remote.close()

    def isolation(self, path: Path, longest: int, cpu_path: Path | None = None) -> Worker:
        return Worker(path, self.threads, longest, cpu_path, self.device, self.software, self.env())

    def on_cpu(self) -> list[str]:
        """The models that were given to the GPU and run on the CPU."""
        self._models = [m for m in self._models if not m.closed]
        return [m.path.name for m in self._models if not m.on_gpu]


if __name__ == "__main__":
    _serve(sys.argv[1:])
