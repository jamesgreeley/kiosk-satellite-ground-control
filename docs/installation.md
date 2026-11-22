# Installation

Kiosk Satellite Ground Control runs as a Home Assistant app or as a Docker container, on a CPU or an NVIDIA GPU. Kiosks running [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite) connect to it over the network.

## Choosing a Machine

Every Voice Match feature runs on the machine Ground Control is on, not on the kiosks. How long each takes depends on that machine:

| Machine | Speaker model | Isolation |
| --- | --- | --- |
| NVIDIA GPU | Accurate | About a tenth of a second per command |
| x86 CPU with more than 4 cores | Accurate | About a second per command |
| Mac with an Apple M-series chip | Accurate | Under a second per command on an M5 |
| Intel N100 and other small x86 CPUs | Light | About 6 seconds per command with Accurate |
| Raspberry Pi 4, Home Assistant Green and other ARM boards | Light | Too slow |

See [Performance Benchmarks](../README.md#performance-benchmarks) for the measured times. Isolation is optional, so a small machine still runs Recognition and Trimming.

The CUDA image works with NVIDIA GPUs on amd64 from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell), and on arm64 with the GB10 and other Blackwell boards. It needs 1 GB of free GPU memory, or 2.5 GB with Isolation.

## Home Assistant App

[![Open your Home Assistant instance and show the add app repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fjxlarrea%2Fkiosk-satellite-ground-control)

Select the button above to add the repository to your Home Assistant, or add it by hand:

1. In Home Assistant, go to **Settings > Apps > App store**, open the menu and choose **Repositories**.
2. Add `https://github.com/jxlarrea/kiosk-satellite-ground-control`.
3. Install **Kiosk Satellite Ground Control** and start it.
4. Open **Ground Control** in the sidebar.

The app runs on the CPU. Home Assistant OS has no NVIDIA driver, so use the Docker image on another machine for a GPU.

The app uses the host network, so kiosks on the same network find it on their own. It signs in to Home Assistant through the Supervisor, so there is no address or token to enter.

### App Options

The app's **Configuration** tab:

| Option | Default | What it does |
| --- | --- | --- |
| Speaker model | auto | `auto` picks Accurate on an x86 CPU with more than 4 cores or an Apple M-series chip, and Light on other ARM CPUs and smaller CPUs. `accurate` or `light` sets it. See [Speaker Models](voice-match.md#speaker-models). |
| Threads | 0 | Threads per check. `0` uses half the CPUs, up to 4. |
| Learn from commands | on | Confident commands fold into each person's voiceprint, so it follows colds and room changes. |
| Days to keep sessions | 7 | How long saved sessions stay on the Sessions page, 1 to 90 days. |
| Keep session audio | on | Keeps each session's audio, so it can be played back or added to a person. |
| mDNS address | empty | Optional. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |

The speaker model, learning and session keeping can also be changed in the app's [Settings](settings.md), which then take precedence over these options.

### Files and Backups

People, voiceprints, kiosks and saved sessions are in the app's own folder, `addon_configs/…_kiosk_satellite_ground_control`, which the Samba share and File editor apps show. A Home Assistant backup that includes the app keeps all of it, so a restore brings back everyone's voiceprint. See [Backup and Moving](backup.md) to back it up or move it to another machine.

## Docker

Docker Compose files are in the repository: [compose.yaml](../compose.yaml) for the CPU and [compose.cuda.yaml](../compose.cuda.yaml) for an NVIDIA GPU.

1. Download the file for your machine.
2. Set `HA_URL` to your Home Assistant.
3. Run `docker compose up -d`, or `docker compose -f compose.cuda.yaml up -d` for the GPU.
4. Open `http://THIS_MACHINE:10900` and choose **Log in with Home Assistant**.

Both files use the host network, so kiosks on the same network find Ground Control on their own. Without the host network, publish port 10900 and enter the machine's address in each kiosk.

Without Compose:

```bash
docker run -d --name ground-control --network host \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control
```

For an NVIDIA GPU, add `--gpus all` and use the `cuda` tag:

```bash
docker run -d --name ground-control --network host --gpus all \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control:cuda
```

`HA_URL` is used two ways: kiosk tokens are checked there, and your browser goes there to log in to the web app. Use an address your browser can open too, not one that only works inside Docker.

### Mac with Apple Silicon

Ground Control runs on a Mac with an Apple M-series chip in [Docker Desktop](https://www.docker.com/products/docker-desktop/) or [OrbStack](https://orbstack.dev), with the CPU image. Docker on a Mac can't use the Mac's GPU, so everything runs on the CPU, and an M-series chip is fast enough for every feature. On a MacBook Pro M5, all features together add about 0.8 s to a 5 s command.

1. Install Docker Desktop or OrbStack.
2. Start Ground Control with its port published:

    ```bash
    docker run -d --name ground-control --restart unless-stopped -p 10900:10900 \
      -e HA_URL=http://homeassistant.local:8123 \
      -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control
    ```

3. Open `http://localhost:10900` and choose **Log in with Home Assistant**.
4. In each kiosk, enter the Mac's address, such as `192.168.1.20:10900`, under **Address** in Kiosk Satellite's Ground Control settings.

Kiosks can't find Ground Control on a Mac on their own. Docker runs it in a small Linux VM, so its mDNS announcement doesn't reach your network, and Docker Desktop's host network option doesn't change that. Give the Mac a fixed address in your router so the kiosks keep finding it.

Ground Control picks the Accurate speaker model on an Apple chip by itself. Keep the Mac from sleeping, under **System Settings > Energy** on a Mac mini or **Battery > Options** on a MacBook. While the Mac sleeps, kiosks go on with their own microphone and without Voice Match.

### NVIDIA GPU

The CUDA image needs:

- An NVIDIA GPU with 1 GB of free memory, or 2.5 GB with Isolation. On amd64, any GPU from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell). On arm64, the GB10 and other Blackwell boards.
- An NVIDIA driver from the 570 series or newer on amd64, or 580 or newer on arm64.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

Ground Control keeps about 0.7 GB on the GPU, or 0.9 GB with Isolation. A command needs up to 0.3 GB more while it runs, or 1.5 GB more with Isolation, and that memory is free again once it is done. Commands that end at the same moment are cleaned one after the other, so it is never more than that. On a GPU shared with an LLM or speech-to-text, leave that much free for Ground Control.

If the GPU has memory to spare, set `KEEP_GPU_MEMORY` to `true`. Isolation then keeps its memory between commands, about 2 GB more, and each command is done sooner: about 0.05 s on a GB10 and 0.01 s on an RTX 4000.

The image is built for amd64 and arm64, so it also runs on an NVIDIA GB10. The CUDA image runs on the GPU or not at all. If it can't use the GPU, it stops with a line in its log that starts with `CUDA can't start` and says why, such as no GPU visible to the container or a driver that is too old. Use the CPU image on a machine without a supported GPU.

> **Info:** If Docker runs inside an LXC container on Proxmox, `nvidia-smi` fails with "Failed to initialize NVML: Unknown Error" until you set `nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place` and pass the devices to the container: `--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools`.

### Environment Variables

| Variable | Default | What it does |
| --- | --- | --- |
| `HA_URL` | `http://homeassistant:8123` | Your Home Assistant. Kiosk tokens are checked there and the web app logs in there. |
| `MODEL` | `auto` | `accurate`, `light` or `auto`. `auto` picks Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with more than 4 cores, and Light otherwise. |
| `THREADS` | half the CPUs, 1 to 4 | Threads per check. |
| `LEARN` | `true` | Fold confident commands into each person's voiceprint. |
| `KEEP_DAYS` | `7` | Days saved sessions are kept. |
| `KEEP_AUDIO` | `true` | Keep each session's audio. |
| `PORT` | `10900` | The port for kiosks and the web app. |
| `MDNS` | `true` | Advertise the service so kiosks find it on their own. |
| `MDNS_ADDRESS` | the main network's | The address kiosks get. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |
| `DATA_DIR` | `/data` | People, voiceprints, kiosks and sessions. |
| `UPDATE_CHECK` | `true` | Look for a new Ground Control version on GitHub every 12 hours and show it on Overview. |
| `KEEP_GPU_MEMORY` | `false` | CUDA image only. Isolation keeps its GPU memory between commands instead of handing it back. See [NVIDIA GPU](#nvidia-gpu). |

## Updating

**Home Assistant app:** update it from the app's page when Home Assistant offers a new version.

**Docker:** run `docker compose pull && docker compose up -d`, or pull the image again and recreate the container. The data volume is kept.

Ground Control checks GitHub for a new version a minute after it starts and every 12 hours after that. When there is one, a card at the top of **Overview** shows the version, with links to what's new and to the repository. The check only asks GitHub for the list of versions. Set `UPDATE_CHECK=false` to turn it off. The Home Assistant app doesn't check, since Home Assistant offers its updates.

To move Ground Control to another machine, or between the app and Docker, see [Backup and Moving](backup.md).

Each Ground Control version works with the Kiosk Satellite version released alongside it. Update the kiosks too when Ground Control changes.

## Next Steps

1. [Connect and approve your kiosks](kiosks.md).
2. [Add people and enroll their voices](people.md).
3. [Turn on Voice Match on each kiosk](voice-match.md#on-the-kiosks).
