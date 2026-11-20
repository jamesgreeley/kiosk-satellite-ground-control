<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banners/ks_banner_dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="assets/banners/ks_banner_light.svg" />
    <img alt="Kiosk Satellite for Home Assistant" src="assets/banners/ks_banner_default.svg" width="650" />
  </picture>
</h1>

Ground Control is a companion service for [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite). It runs on a computer or server in your home and handles the voice and vision processing that kiosks don't have the computing power for. Install it as a Home Assistant app, or run it in Docker on a CPU or an NVIDIA GPU.

Kiosk Satellite Ground Control currently has two components:

- **Voice Match:** Recognition, Trimming and Isolation.
- **Vision Match:** Coming soon.

## Voice Match

Voice Match is Ground Control's first set of features: who is speaking, and only their voice in the command.

### Recognition

Who is speaking, from the voiceprints of the people enrolled. Enroll each person once on every kind of device they talk to, from the People page. Recognition then takes about a second into each command.

Each kiosk gets a Speaker sensor in Home Assistant, and a kiosk can answer only the people you choose. An LLM conversation agent can tell who is speaking too: paste the template from **Settings > General > Conversation agents** into the agent's instructions, and "my calendar" or "my list" means the speaker's.

### Trimming

Turn on **Enable Trimming** in Kiosk Satellite and the kiosk sends everything its microphone hears. Ground Control sends back only the speech of the person who said the wake word. A TV or another person before, between and after their words is cut out, long pauses are shortened and the command ends once the user stops talking, whatever the TV does. "Turn on the (TV talking) kitchen lights" reaches the assistant as "turn on the kitchen lights".

Trimming doesn't just end the command when someone else starts talking, which would lose whatever the user says after the other voice. It cuts the other voice out and keeps the user's words on both sides of it.

<video src="assets/videos/voice-match-trimming.mp4" controls width="100%"></video>

https://github.com/user-attachments/assets/985faf93-9b86-48ce-bbda-41d91d0a1b49

### Isolation

Trimming leaves other voices out between the user's words. Isolation also removes them where they talk over the user, using the user's voiceprint. Turn it on under **Settings > Voice Match > Isolation**. It needs the Accurate speaker model. Each kiosk can turn it off with **Enable Isolation** in Kiosk Satellite.

Isolation works on a whole turn at once, so each command waits for it once the user stops talking: about a tenth of a second on an NVIDIA GPU (see [What to run it on](#what-to-run-it-on)).

<video src="assets/videos/voice-match-isolation.mp4" controls width="100%"></video>

https://github.com/user-attachments/assets/4485f1fc-af2a-4d17-a7de-2b0b3ebe7f3f

## Vision Match

Coming soon.

## Performance Benchmarks

How much latency each feature adds to a 5-second command:

| | Raspberry Pi 4, Light | x86 CPU (i9-13900H), Accurate | NVIDIA RTX 4000 SFF Ada, Accurate | NVIDIA GB10 GPU, Accurate |
|---|---|---|---|---|
| Recognition | 0.10 s | 0.04 s | 0.01 s | pending |
| Trimming | 0.41 s | 0.11 s | 0.01 s | pending |
| Isolation | Too slow on a Pi | 1.18 s | 0.09 s | pending |
| CPU, per second of audio | 27% | 10% | 2% | pending |
| **All features** | **0.41 s** (no Isolation) | **1.29 s** | **0.10 s** | pending |

A 5 s command between stretches of TV, sent through Ground Control's own code in real time the way a kiosk sends it, with Ground Control's default threads (2 on the Pi, 4 elsewhere).

- **Recognition** is the time to check a voice. It runs while the user talks and holds nothing up. With a clear wake word, the speaker is known before the user says anything.
- **Trimming** is the time to decide the command is over once the user stops. Whatever the TV does, the command ends there.
- **Isolation** runs once the command ends. On a CPU it uses a version of the model made for the CPU (LSTMs in 8 bits), twice as fast and nearly identical.
- **All features** is what reaches the end of each command: Trimming, then Isolation.
- On a Pi 4, use the Light model. Accurate uses more CPU than a Pi 4 has, so everything runs late, and Isolation (which needs Accurate) takes about 25 s there.

The Accurate model makes the fewest mix-ups between people and with TV voices. The Light model uses about a third of the CPU, and a TV voice cutting in ends the command about a second later. Ground Control picks Accurate on x86 and Light on ARM, and **Settings > Voice Match** changes it without needing to re-enroll users.

For Isolation on a GPU, any NVIDIA GPU from the Turing generation (GTX 16 series, RTX 20 series) or newer with 4 GB works.

## Install

**Home Assistant app:** add this repository under **Settings > Apps > App store > Repositories**, then install **Kiosk Satellite Ground Control**.

**Docker Compose:** [compose.yaml](compose.yaml) for the CPU and [compose.cuda.yaml](compose.cuda.yaml) for an NVIDIA GPU. Set `HA_URL` in the file, then `docker compose up -d` (or `docker compose -f compose.cuda.yaml up -d`). Both use the host network, so kiosks on the same network find the service on their own.

**Docker, CPU:**

```bash
docker run -d --name ground-control -p 10900:10900 \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control
```

**Docker, NVIDIA GPU:**

```bash
docker run -d --name ground-control --gpus all -p 10900:10900 \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control:cuda
```

The GPU needs an NVIDIA driver for CUDA 13 (580 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. If Docker runs inside an LXC container (Proxmox), set `nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place` and pass the devices too, or `nvidia-smi` fails with "Failed to initialize NVML: Unknown Error":

```bash
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
```

`HA_URL` is your Home Assistant. Kiosks sign in with the Home Assistant token they already have, and the service checks it there. Your browser goes to `HA_URL` to log in, so use an address your browser can open too, not one that only works inside Docker. See `ground_control/config.py` for every setting.

To try a build before it's released, see [TESTING.md](TESTING.md).

## Getting started

1. **Open Ground Control.** As a Home Assistant app it's in the sidebar. As a container, open port 10900 and select **Log in with Home Assistant**.
2. **Connect your kiosks.** Kiosks on the same network find Ground Control on their own. For a kiosk on another network, enter the address from **Settings > General > Kiosks** in Kiosk Satellite. Approve each new kiosk on the Kiosks page. Ground Control only accepts kiosks running Kiosk Satellite.
3. **Enroll people.** Add each person on the People page, then enroll their voice on a kiosk of every kind they talk to.
4. **Turn on what you want.** In Kiosk Satellite, under **Voice Satellite > Ground Control Voice Match**, choose who each kiosk answers and turn on **Enable Trimming** and **Enable Isolation**. Turn on Isolation in Ground Control's settings too.

## License and credits

Free for personal, non-commercial use under [CC BY-NC-ND 4.0](LICENSE). Commercial use and redistribution of modified builds are not permitted.

Each model has its license next to it in `models/`, in a file named after the model. ReDimNet2 by Palabra.ai and Silero VAD by the Silero Team keep their MIT licenses. The Voice Isolator model is under CC BY-NC-ND 4.0, like the rest of Ground Control.
