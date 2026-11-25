# Installation

Kiosk Satellite Ground Control runs on a CPU or an NVIDIA GPU as a Home Assistant app or a Docker container. Kiosks running [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite) connect to it over the network.

## Choosing a Machine

Every Voice Match feature runs on the Ground Control machine instead of the kiosks. How long each one takes depends on that machine:

| Machine | Speaker model | Isolation |
| --- | --- | --- |
| NVIDIA GPU | Accurate | About a tenth of a second per command |
| x86 CPU with more than 4 cores | Accurate | About a second per command |
| Mac with an Apple M-series chip | Accurate | Under a second per command on an M5 |
| Intel N100 and other small x86 CPUs | Light | About 6 seconds per command with Accurate |
| Raspberry Pi 4, Home Assistant Green and other ARM boards | Light | Too slow |

See [Performance Benchmarks](../README.md#performance-benchmarks) for the measured times. Isolation is optional, so a small machine still runs Recognition, Trimming and the Wake Word Aligner.

On amd64, the CUDA image works with NVIDIA GPUs from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell). On arm64, it works with the GB10 and other Blackwell boards. It needs 1 GB of free GPU memory or 2.5 GB with Isolation.

## Home Assistant App

[![Open your Home Assistant instance and show the add app repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fjxlarrea%2Fkiosk-satellite-ground-control)

Select the button above to add the repository to your Home Assistant or add it by hand:

1. In Home Assistant, go to **Settings > Apps > App store**, open the menu and choose **Repositories**.
2. Add `https://github.com/jxlarrea/kiosk-satellite-ground-control`.
3. Install **Kiosk Satellite Ground Control** and start it.
4. Open **Ground Control** in the sidebar.

The app runs on the CPU. Home Assistant OS has no NVIDIA driver, so use the Docker image on another machine for a GPU.

The app uses the host network, so it is discovered automatically on the same network. It signs in to Home Assistant through the Supervisor, so there is no address or token to enter.

### App Options

These options are on the app's **Configuration** tab:

| Option | Default | What it does |
| --- | --- | --- |
| Speaker model | auto | `auto` picks Accurate on an x86 CPU with more than 4 cores or an Apple M-series chip. It picks Light on other ARM CPUs and smaller CPUs. Use `accurate` or `light` to set it. See [Speaker Models](voice-match.md#speaker-models). |
| Threads | 0 | Threads per check. `0` uses half the CPUs with a maximum of 4. |
| Learn from commands | on | Adds confident commands to each person's voiceprint so it stays accurate through colds and room changes. |
| Days to keep sessions | 7 | How long saved sessions stay on the Sessions page. Set it from 1 to 90 days. |
| Keep session audio | on | Keeps each session's audio, so it can be played back or added to a person. |
| mDNS address | empty | Optional. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |

You can also change the speaker model, learning and session keeping in the app's [Settings](settings.md). Those settings then take precedence over these options.

### Files and Backups

People, voiceprints, kiosks and saved sessions are stored in the app's own folder at `addon_configs/…_kiosk_satellite_ground_control`. You can open it with the Samba share and File editor apps. A Home Assistant backup that includes the app keeps all of it, so a restore brings back everyone's voiceprint. See [Backup and Moving](backup.md) to back it up or move it to another machine.

## Docker

Docker Compose files are in the repository: [compose.yaml](../compose.yaml) for the CPU and [compose.cuda.yaml](../compose.cuda.yaml) for an NVIDIA GPU.

1. Download the file for your machine.
2. Set `HA_URL` to your Home Assistant.
3. Run `docker compose up -d`. For the GPU, run `docker compose -f compose.cuda.yaml up -d`.
4. Open `http://THIS_MACHINE:10900` and choose **Log in with Home Assistant**.

Both files use the host network, so Ground Control is discovered automatically on the same network. If you don't use the host network, publish port 10900 and enter the machine's address in each kiosk.

To run it without Compose:

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

`HA_URL` is used in two ways. Kiosk tokens are checked there and you log in to the web app there from your browser. Use an address your browser can open. Avoid one that only works inside Docker.

### Mac with Apple Silicon

Ground Control runs with the CPU image in [Docker Desktop](https://www.docker.com/products/docker-desktop/) or [OrbStack](https://orbstack.dev) on a Mac with an Apple M-series chip. Docker on a Mac can't use the Mac's GPU, so everything runs on the CPU. An M-series chip is fast enough for every feature. On a MacBook Pro M5, all features together add about 0.8 s to a 5 s command.

1. Install Docker Desktop or OrbStack.
2. Start Ground Control with its port published:

    ```bash
    docker run -d --name ground-control --restart unless-stopped -p 10900:10900 \
      -e HA_URL=http://homeassistant.local:8123 \
      -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control
    ```

3. Open `http://localhost:10900` and choose **Log in with Home Assistant**.
4. In Kiosk Satellite's Ground Control settings on each kiosk, enter the Mac's address (for example `192.168.1.20:10900`) under **Address**.

Ground Control on a Mac isn't discovered automatically. Docker runs it in a small Linux VM, so its mDNS announcement doesn't reach your network. Docker Desktop's host network option doesn't change that. Give the Mac a fixed address in your router so the address in each kiosk keeps working.

Ground Control picks the Accurate speaker model automatically on an Apple chip. Keep the Mac from sleeping. On a Mac mini, the setting is under **System Settings > Energy**. On a MacBook, it is under **Battery > Options**. While the Mac sleeps, the kiosks use their own microphone and Voice Match is unavailable.

### NVIDIA GPU

The CUDA image needs:

- An NVIDIA GPU with 1 GB of free memory or 2.5 GB with Isolation. On amd64, it can be any GPU from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell). On arm64, it can be the GB10 or another Blackwell board.
- An NVIDIA driver from the 570 series or newer on amd64 or 580 or newer on arm64.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

The image is built for amd64 and arm64, so it also runs on an NVIDIA GB10. The CUDA image runs on the GPU or not at all. If it can't use the GPU, it stops and logs a line that starts with `CUDA can't start`. The line gives the reason. For example, no GPU is visible to the container or the driver is too old. Use the CPU image on a machine without a supported GPU.

> **Info:** If Docker runs inside an LXC container on Proxmox, `nvidia-smi` fails with "Failed to initialize NVML: Unknown Error" until you set `nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place` and pass the devices to the container: `--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools`.

### GPU Memory

This table shows how much GPU memory Ground Control uses with the Accurate speaker model. It was measured on an RTX 4000 SFF Ada and a GB10:

| | amd64 image (CUDA 12) | arm64 image (CUDA 13) |
| --- | --- | --- |
| Kept, Recognition and Trimming | 0.6 GB | 0.7 GB |
| Kept, with Isolation | 0.8 GB | 0.9 GB |
| Most during a command | 0.9 GB | 0.9 GB |
| Most during a command, with Isolation | 2.3 GB | 2.4 GB |

Ground Control keeps the first two amounts for as long as it runs. While a command runs, it needs up to 0.3 GB more. With Isolation, a command of 10 seconds or longer needs up to 1.5 GB more. That memory is freed once the command is done. Commands that end at the same moment are cleaned one after the other, so the total never goes above that. If the GPU is shared with an LLM or speech-to-text, leave that much free for Ground Control.

If the GPU has memory to spare, set `KEEP_GPU_MEMORY` to `true`. Isolation then keeps about 2 GB more memory between commands. Each command is done about 0.05 s sooner on a GB10 and 0.01 s sooner on an RTX 4000.

### Environment Variables

| Variable | Default | What it does |
| --- | --- | --- |
| `HA_URL` | `http://homeassistant:8123` | Your Home Assistant address. Kiosk tokens are checked there and you log in to the web app there. |
| `MODEL` | `auto` | `accurate`, `light` or `auto`. `auto` picks Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with more than 4 cores and Light otherwise. |
| `THREADS` | half the CPUs, 1 to 4 | Threads per check. |
| `LEARN` | `true` | Fold confident commands into each person's voiceprint. |
| `KEEP_DAYS` | `7` | Days saved sessions are kept. |
| `KEEP_AUDIO` | `true` | Keep each session's audio. |
| `PORT` | `10900` | The port for kiosks and the web app. |
| `MDNS` | `true` | Advertise the service so it is discovered automatically on the network. |
| `MDNS_ADDRESS` | the main network's | The address kiosks get. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |
| `DATA_DIR` | `/data` | Where people, voiceprints, kiosks and sessions are stored. |
| `UPDATE_CHECK` | `true` | Look for a new Ground Control version on GitHub every 12 hours and show it on Overview. |
| `KEEP_GPU_MEMORY` | `false` | CUDA image only. Isolation keeps its GPU memory between commands instead of handing it back. See [GPU Memory](#gpu-memory). |

## Updating

**Home Assistant app:** update it from the app's page when Home Assistant offers a new version.

**Docker:** run `docker compose pull && docker compose up -d` or pull the image again and recreate the container. The data volume is kept.

Ground Control checks GitHub for a new version a minute after it starts and every 12 hours after that. When there is one, a card at the top of **Overview** shows the version and links to what's new and to the repository. The check only asks GitHub for the list of versions. Set `UPDATE_CHECK=false` to turn it off. The Home Assistant app skips this check because Home Assistant offers its updates.

To move Ground Control to another machine or between the app and Docker, see [Backup and Moving](backup.md).

Each Ground Control version works with the Kiosk Satellite version released alongside it. Update the kiosks too when you update Ground Control.

## Next Steps

1. [Connect and approve your kiosks](kiosks.md).
2. [Add people and enroll their voices](people.md).
3. [Turn on Voice Match on each kiosk](voice-match.md#on-the-kiosks).
