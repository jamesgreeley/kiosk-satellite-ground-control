# Apple Silicon performance

These measurements use native macOS on an M1 Pro. For setup, see [Run natively for Core ML](installation.md#run-natively-for-core-ml).

## Face detection and recognition

On native macOS, both SCRFD face detectors specialize their input and nine output shapes for 640×480, 640×384 and 480×640 tensors after resizing and padding. Core ML compiles each size on first use and reuses the session. Other sizes use the dynamic Light detector or the Accurate CPU detector. Recognition uses Core ML NeuralNetwork for Light and MLProgram for Accurate. The Accurate recognition compute plan assigned convolution layers to the Neural Engine.

| Vision stage (model-level) | CPU | Core ML | Change |
|---|---:|---:|---:|
| Light Face Detection, 640×480 | 22.6 ms | 4.8 ms | 4.7× faster |
| Light Face Detection, 640×384 | 18.1 ms | 4.0 ms | 4.5× faster |
| Accurate Face Detection, 640×480 | 73.2 ms | 9.3 ms | 7.9× faster |
| Light Face Recognition, one 112×112 crop | 6.1 ms | 0.8 ms | 7.6× faster |
| Accurate Face Recognition, one 112×112 crop | 56.1 ms | 3.2 ms | 17.5× faster |

A repeated thread-count sweep (1, 2, 4, 8 and 16 ONNX Runtime intra-op threads) found no gain from a larger CPU pool on the Core ML paths. Light detection and recognition use one thread; Accurate recognition uses two. For the Light recognition graph, an interleaved eight-run comparison measured 0.817 ms with `CPUAndNeuralEngine` and 0.824 ms with `ALL`, so it keeps the Neural Engine-specific setting.

## Complete camera requests

These timings include JPEG decoding, Light face detection, Accurate face recognition and gestures together. The face models used Core ML in both runs. Only the gesture backend changed. On 27 sample frames made from HaGRID images and the scikit-image astronaut photograph, both gesture backends returned the same label or absence of a hand on every frame.

| Gesture model | Median with CPU gestures | Median with Core ML gestures |
|---|---:|---:|
| Light | 32.0 ms | 24.1 ms |
| Accurate | 440.1 ms | 32.4 ms |

With Accurate gestures, p95 fell from 488.6 to 44.2 ms. A second tuning pass put the 640×480 face detector on Core ML’s CPU/GPU policy when Accurate gestures are configured. Median request time fell from 31.8 to 28.0 ms, and from 30.8 to 29.2 ms with the frame order reversed.

HD camera requests reuse the decoded image for recognition and resize it once when face and hand detection need the same dimensions. Accurate recognition can process up to three aligned faces concurrently through its existing Core ML session. Against the preceding version (`2faf12e`), median complete request latency on 1920×1080 frames with three frontal faces fell from 47.4 to 31.2 ms on the M1 Pro. Reversing the frame and comparison order gave 46.5 to 31.3 ms (p95 58.6 to 40.1 ms). Each comparison used 45 timed three-face frames, including repeats of the same image crops. The earlier and updated versions returned the same boxes and gestures in the reverse-order comparison, and the lowest corresponding embedding cosine was 0.9999999. These timings use Light face and gesture detection plus Accurate face recognition.

## Speaker recognition

Voice Match can use Core ML for half-second increments from 0.5 to 4 seconds. Run the local tuner once after installing the Apple tuning dependencies. It simplifies the speaker graph for each length and saves only configurations with at least 10% lower median latency, equal or lower p95, and embedding cosine similarity of at least 0.999 on its test signals. Restart Ground Control to load the selected windows. Other lengths use the CPU. On the M1 Pro, the Accurate 2 s window fell from 42.3 to 18.6 ms. Voice isolation uses the CPU. Tests of the dynamic Core ML graphs on one-, three- and five-second synthetic speech clips took longer.

```sh
python -m pip install '.[apple-tuning]'
python tools/tune_coreml_voice.py --threads 4
```

Use the service's thread count and optionally supply a representative mono 16 kHz, 16-bit PCM recording with `--wav`. The tuner also checks silence and seeded noise. Loading all eight Accurate windows took 15.9 s on the M1 Pro. Peak resident memory of the standalone speaker process reached 534 MiB.

## Validation and reproduction

On 18 resized and mirrored variations of the scikit-image astronaut photograph, both detectors agreed with CPU on one face in every frame. Median times including preprocessing and decoding fell from 24.1 to 8.0 ms for Light and from 76.0 to 12.6 ms for Accurate. The largest change in a reported box coordinate was 0.0003 of the frame dimension. This checks one subject. The table measures warmed model inference on synthetic inputs. Embedding cosine similarity between CPU and Core ML was 0.99998 for the Light model and 0.99540 for Accurate on those synthetic inputs. Please validate match rates with enrolled faces before relying on that accuracy-sensitive stage.

Ground Control caches compiled recognition graphs under the user's macOS Caches folder, keyed by the ONNX file hash and ONNX Runtime version. Fixed detector sessions compile once per size in each process.

## Benchmark your Mac

From the checkout with the virtual environment active, run these commands one at a time. For the camera test, replace the example paths with JPEGs representative of your kiosks.

```sh
python tools/benchmark_apple_silicon.py --suite face --iterations 100 --threads 4
python tools/benchmark_apple_silicon.py --suite hands --iterations 100 --threads 4
python tools/benchmark_camera.py --hands light --threads 4 --rounds 10 frame1.jpg frame2.jpg
python tools/benchmark_camera.py --hands accurate --threads 4 --rounds 10 frame1.jpg frame2.jpg
```

For comparisons across Macs, use the same commit, JPEGs, model choices and thread count. Record the chip, macOS and ONNX Runtime versions, frame dimensions, median and p95. The camera tool measures processing on the server, from JPEG decoding through model output. Camera capture and network time are outside that interval.

The model benchmark accepts `--tune` to compare conversion formats, `--tune-compute-units` to compare compute policies, and `--profile-compute-plan` to report hardware assignments where available. Use `--suite voice --tune` for synthetic audio inference timings. For the actual speaker windows selected on your hardware, use the [speaker tuner](#speaker-recognition).
