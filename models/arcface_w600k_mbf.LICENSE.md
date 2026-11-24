# ArcFace MobileFaceNet (arcface_w600k_mbf.onnx)

Third-party work, shipped with Kiosk Satellite Ground Control under its own license.

Face recognition model from [InsightFace](https://github.com/deepinsight/insightface): ArcFace with a MobileFaceNet backbone, `w600k_mbf.onnx` from the `buffalo_s` model pack of [release v0.7](https://github.com/deepinsight/insightface/releases/tag/v0.7). Unchanged. Ground Control runs it on the faces in Vision Match's frames to tell who is looking at the kiosk, with the Light vision model.

InsightFace's license for its models, from its README:

> The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage.
>
> The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.
>
> Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy (which is for non-commercial research purposes only).
