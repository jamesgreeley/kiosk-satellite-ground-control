# Vision Match

Navigate to **Settings > Vision Match** in the Ground Control app.

Vision Match runs Kiosk Satellite's camera features on Ground Control. The camera and motion detection stay on the kiosk. When something moves, the kiosk sends a camera frame to Ground Control and gets back what is in it.

- **Face Detection:** finds the faces looking at the kiosk, so a face can dismiss or postpone the screensaver.
- **Hand Gestures:** counts the fingers held up on a raised hand for **Show fingers** gestures.
- **Face Recognition:** names the person looking at the kiosk from the faces enrolled on the People page.

A kiosk that is too slow for these models or can't run them, like an Echo Show 5 or an Android 7 tablet, gets the same features as a fast one.

## On the Kiosks

Turn the features on for each kiosk in Kiosk Satellite under **Settings > Ground Control > Vision Match**:

| Kiosk setting | What it does |
| --- | --- |
| Enable Vision Match | Sends the camera's frames to Ground Control for the features below. Needs the camera and Ground Control. |
| Enable Face Detection | Runs Face Detection on Ground Control. With it off, Face Detection runs on the device. |
| Enable Face Recognition | Fills the kiosk's **Face** and **People in view** sensors in Home Assistant. Needs Face Detection on Ground Control. |
| Confirm voices with faces | Accepts a voice under the **Threshold** as the person recognized at the kiosk. Needs Voice Match. See [Confirmed by the Face](voice-match.md#confirmed-by-the-face). |
| Enable Hand Gestures | Runs Hand Gestures on Ground Control. With it off, Hand Gestures runs on the device. |

**Dismiss on face** and **Postpone on face** under **Settings > Screensaver > Face Detection** work as they do without Ground Control. So do **Show fingers** gestures under **Settings > Gestures**. The kiosk's page on the [Kiosks](kiosks.md) page shows which features each kiosk has on. See [Vision Match](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/vision-match.md) in the Kiosk Satellite docs for every kiosk setting.

## Looks

A look is one camera frame and what Ground Control finds in it:

- **A frame is sent only after something moves.** The kiosk waits for each frame's result before it sends the next one.
- **Each frame is a 640x480 JPEG.** Ground Control reads it and runs the models each feature needs.
- **Looks are paced to the machine.** When the models load, Ground Control measures how many face looks and hand looks a second it keeps up with for one kiosk and sends that rate to each kiosk. Kiosks look at most 2 times a second for faces and 4 times a second for hands.
- **A look for faces and hands runs both at the same time.** It takes about as long as the slower of the two and not their sum.

With Face Recognition on, the kiosk also looks for faces whenever the screen is on. It still looks only after something moves.

When Ground Control doesn't answer, the kiosk waits to look until it answers again. The models on the device run only for a feature turned off in Kiosk Satellite.

Each stretch of time someone is in front of a kiosk is saved as a vision session with what the camera saw, what the kiosk did and how long each look took. See [Vision Sessions](sessions.md#vision-sessions).

## Face Detection

Face Detection finds the faces in a look and whether each one looks at the kiosk. A face looks at the kiosk when the head is turned 30 degrees or less from the camera. Only a face looking at the kiosk dismisses or postpones the screensaver. Only those faces are recognized.

On kiosk frames from 1 to 3 m, a head turned away passed as a face looking at the screen in 10% of frames. With the models on the kiosk it was 82%.

## Hand Gestures

Hand Gestures finds a raised hand in a look and counts the fingers held up. A finger reading counts once two hand looks in a row give the same count. A hand that is blurred while it moves can read as any count for one look, so a single look never runs a gesture. The kiosk then runs the gesture's action after its hold time.

On kiosk frames from 1 to 3 m, the finger count was right in 91% of frames with Light and 96% with Accurate. With the models on the kiosk it was 28%.

**Hand sensitivity** under **Settings > Vision Match > Hand Gestures** sets the smallest hand counted, by the longer side of its box. At the default of 50 that is about 9% of the frame. A smaller hand counts as no hand. It opens no vision session and doesn't reach the kiosk's gestures. Lower it if a hand that is only a small part of the frame, such as one cut off at its edge, shows up in vision sessions. Hands that gave a finger count on kiosk frames were 10% of the frame or more.

## Face Recognition

Face Recognition compares each face looking at the kiosk with the faces enrolled on the People page. Enroll each person's face once with a guided capture of about half a minute through a kiosk's camera. See [Face](people.md#face) for the capture.

- **A face is named after two looks in a row above the Threshold** or after one very close match. The best match also has to be clearly ahead of the next person.
- **A recognized face is checked again every 2 seconds.** A face nobody enrolled is checked again once a second, so a guest standing at the kiosk costs little.
- **Up to three faces are recognized at once.** The person at the kiosk is the one with the highest score. Someone else becomes the person at the kiosk only after scoring higher two checks in a row or when the person at the kiosk leaves. This keeps it from flipping between two people side by side.
- **Nobody counts as in front of the kiosk** once no face has looked at it for 3 seconds.

On kiosk frames from 1 to 3 m in bright and dim light, the enrolled person was recognized in every frame where their face looked at the kiosk. In simulated households of six people, nobody was ever taken for someone else.

Ground Control keeps only small pictures of each enrolled face, lined up and 112 pixels wide. Recognition never keeps a frame. The frames kept with vision sessions follow **Keep frames** under [Settings](#settings).

### On the Kiosk and in Home Assistant

- **Sensors:** the kiosk's **Face** sensor shows the person at the kiosk. Its **People in view** sensor lists everyone recognized with the person at the kiosk first.
- **Face event:** the kiosk fires `esphome.kiosk_satellite_face` when someone recognized arrives at the kiosk and when they leave. See [Face](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/events.md#face) in the Kiosk Satellite docs.
- **Recognition cards:** cards on the kiosk screen show who was recognized and what happened because of it, like the kiosk PIN skipped or Lockdown Mode turned off. See [Recognition Cards](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/vision-match.md#recognition-cards).
- **Settings that act on faces:** Kiosk Satellite can let only chosen people dismiss the screensaver (**Detect faces**), skip the kiosk PIN (**Skip the PIN for**) or turn Lockdown Mode off (**Disable Lockdown Mode for**).
- **Voices:** with **Confirm voices with faces** on, a voice close to the voiceprint of the person at the kiosk is accepted for them. See [Confirmed by the Face](voice-match.md#confirmed-by-the-face).

A vision session shows who was recognized. It also shows when the kiosk skipped its PIN or turned Lockdown Mode off or on because of who was recognized.

### Not a Lock

There is no check for a real person, so a photo or a video of an enrolled person held up to the camera can be recognized as them. Twins, siblings and young children can be mistaken for each other. Use the settings that act on faces to skip steps for your family, not to keep anyone out. Don't unlock doors or disarm an alarm with the Face sensor or the face event. See [Not a Lock](people.md#not-a-lock).

## Vision Models

Each feature has its own Light or Accurate model under **Settings > Vision Match > Models**:

| Feature | Light | Accurate | Default |
| --- | --- | --- | --- |
| Face Detection | SCRFD 2.5G | SCRFD 10G | Accurate on an NVIDIA GPU. Light on a CPU. |
| Hand Gestures | YOLOv10n | YOLOv10x | Accurate on an NVIDIA GPU. Light on a CPU. |
| Face Recognition | ArcFace MobileFaceNet | ArcFace R50 | Accurate on an NVIDIA GPU, an x86 CPU or an Apple M-series chip. Light on other ARM CPUs. |

Both hand models are trained on HaGRIDv2 gestures. What each choice changes:

- **Face Detection:** both models found every face looking at the kiosk on the test set and let the same turned heads through. Accurate takes about 3 times as long, so Light is the default on a CPU.
- **Hand Gestures:** Accurate gets the finger count right more often at 3 m and in dim light, 96% of frames compared to 68%. The hand model runs on every hand look and Accurate is slow on a CPU: 1.8 s a look on an Intel N100, 352 ms on an i9-13900H and 256 ms on a MacBook Pro M5. Light is the default on a CPU.
- **Face Recognition:** Accurate recognizes small faces in dim light more often, 93% compared to 71%. A face is checked at most once a second, so Accurate is the default on x86 CPUs and Apple M-series chips. It takes 173 ms a face on an Intel N100. A Raspberry Pi and other ARM CPUs use Light.

Each model's row shows its time on this machine once the models load. Changing the Face Recognition model rebuilds everyone's face from the kept pictures, so nobody has to enroll again. Each Face Recognition model has its own **Threshold**.

### Time per Look

A look with every feature on each machine's default models:

| Machine | Vision models | A look with every feature |
| --- | --- | --- |
| Raspberry Pi 4 | Light | 671 ms |
| Intel N100 | Light, Accurate recognition | 292 ms |
| i9-13900H | Light, Accurate recognition | 64 ms |
| MacBook Pro M5, in Docker | Light, Accurate recognition | 70 ms |
| RTX 3050 | Accurate | 34 ms |
| RTX 4000 SFF Ada | Accurate | 17 ms |
| GB10 | Accurate | 17 ms |

The GPU numbers are with `KEEP_GPU_MEMORY` on. On a Pi 4, Face Detection dismisses the screensaver about a second after a face looks at the kiosk. Hand Gestures feel slow there because a gesture needs more than one look. See [Performance Benchmarks](../README.md#performance-benchmarks) for each feature's time and how it is measured.

### GPU Memory

With every model on Accurate, Vision Match keeps about 0.25 GB of GPU memory on top of Voice Match's or 0.5 GB with `KEEP_GPU_MEMORY`. The models load when the first kiosk asks for a look. Each frame shape a kiosk sends, such as a kiosk in portrait, adds about 0.05 GB. See [GPU Memory](installation.md#gpu-memory).

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| **Models** | | |
| Face Detection | Accurate with an NVIDIA GPU, else Light | Light or Accurate. See [Vision Models](#vision-models). |
| Hand Gestures | Accurate with an NVIDIA GPU, else Light | Light or Accurate. On a CPU, Accurate lowers the looks a second each kiosk gets. |
| Face Recognition | Accurate with an NVIDIA GPU, an x86 CPU or an Apple M-series chip, else Light | Light or Accurate. Faces are rebuilt from their pictures when it changes, so nobody has to enroll again. |
| Runs on | | NVIDIA GPU (CUDA) or CPU. The CUDA option sets it in the same way as the speaker model. |
| Looks a second | | The face looks and hand looks a second one kiosk can get. It is measured on this machine when the models load. |
| **Hand Gestures** | | |
| Hand sensitivity | 50 | From 1 to 100. It sets the smallest hand counted, by the longer side of its box: about 9% of the frame at 50, 13% at 30 and any size at 100. See [Hand Gestures](#hand-gestures). |
| **Frames** | | |
| Keep frames | on | Keeps up to 8 camera frames for each vision session. See [Vision Sessions](sessions.md#vision-sessions). |
| Delete all frames | | Deletes every kept frame right away. The sessions themselves are kept. |
| **Face Recognition** | | |
| Threshold | Accurate 0.3, Light 0.35 | How close a face has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown faces. Each Face Recognition model has its own. |
| Faces enrolled | | People with a face enrolled. |

The **Threshold** is a starting point. If a face is taken for someone else, raise it. If a person is often unknown, enroll their face again on that kiosk or in that light before lowering it a little.
