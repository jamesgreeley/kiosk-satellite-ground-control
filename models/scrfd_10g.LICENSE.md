# SCRFD 10G (scrfd_10g.onnx)

Third-party work, shipped with Kiosk Satellite Ground Control under its own license.

Face detector from [InsightFace](https://github.com/deepinsight/insightface): SCRFD with five face points, `det_10g.onnx` from the `buffalo_l` model pack of [release v0.7](https://github.com/deepinsight/insightface/releases/tag/v0.7). Unchanged. Ground Control runs it on Vision Match's frames to find faces and tell whether they look at the kiosk.

InsightFace's license for its models, from its README:

> The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage.
>
> The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.
>
> Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy (which is for non-commercial research purposes only).
