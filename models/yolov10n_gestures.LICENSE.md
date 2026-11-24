# YOLOv10n gestures (yolov10n_gestures.onnx)

Third-party work, shipped with Kiosk Satellite Ground Control under its own licenses.

Hand gesture detector from [HaGRIDv2](https://github.com/hukenovs/hagrid) (Kapitanov, Kvanchiani, Nagaev, Kraynov and Makhliarchuk, ai-forever): the `YOLOv10n_gestures.pt` checkpoint, trained on the HaGRIDv2 dataset with [YOLOv10](https://github.com/THU-MIG/yolov10). Converted by `tools/export_vision_models.py` to ONNX at 480x640, opset 17. The weights are otherwise unchanged. Ground Control runs it on Vision Match's frames to find a raised hand and read its gesture.

**HaGRIDv2.** The checkpoint and the dataset it was trained on are provided under HaGRID's "Public license with attribution and conditions reserved", a reworking of Creative Commons Attribution-ShareAlike 4.0 (it is not a Creative Commons license). Full text: <https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf>. Converting the format is a technical change under its section 2(a)(4), not Adapted Material.

**YOLOv10.** The network architecture and the training code are licensed under the GNU Affero General Public License v3.0: <https://github.com/THU-MIG/yolov10/blob/main/LICENSE>. This file is distributed as a separate work next to Ground Control. The original checkpoint is at <https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_gestures.pt>.

The model files are provided as is, without warranty of any kind.
