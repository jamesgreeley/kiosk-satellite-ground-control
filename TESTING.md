# Testing Kiosk Satellite Ground Control

How to run a test build of Ground Control with Docker and connect a kiosk to it. Test builds are tagged `edge` for the CPU and `edge-cuda` for an NVIDIA GPU. They are private, so you pull them with your own GitHub account.

## 1. Make a GitHub token

On GitHub, go to **Settings > Developer settings > Personal access tokens > Tokens (classic)** and select **Generate new token**. Give it only the `read:packages` scope and copy it.

## 2. Log in to GitHub's registry

Use your GitHub username, and paste the token when asked for the password:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```

## 3. Get the compose file

Download one from this repository:

- [compose.yaml](compose.yaml) for the CPU.
- [compose.cuda.yaml](compose.cuda.yaml) for an NVIDIA GPU. The host needs an NVIDIA driver for CUDA 13 (580 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## 4. Edit it

- Change the image tag to the test build: `:latest` becomes `:edge`, and `:cuda` becomes `:edge-cuda`.
- Set `HA_URL` to your Home Assistant. Your browser logs in there too, so use an address it can open, such as `http://192.168.1.10:8123`, not one that only works inside Docker.

## 5. Start it

```bash
docker compose up -d
```

For the GPU:

```bash
docker compose -f compose.cuda.yaml up -d
```

`docker logs ground-control` should show "model loaded on the GPU" or "model loaded on the CPU".

The compose files use the host network, so kiosks find Ground Control on their own. Docker Desktop on Mac and Windows only has host networking as an opt-in setting. Without it, replace `network_mode: host` with `ports: ["10900:10900"]` and enter the address in each kiosk by hand (step 7).

## 6. Open the web UI

Go to `http://THIS_MACHINE:10900` and select **Log in with Home Assistant**. Sign in with your Home Assistant account and you're sent back to Ground Control.

## 7. Connect a kiosk

Install the Kiosk Satellite test build you were sent. Under **Settings > Ground Control**, a kiosk on the same network finds the service on its own. Otherwise, enter `THIS_MACHINE:10900` there.

## 8. Enroll

1. On the kiosk, turn on **Allow enrollment here** under **Settings > Voice Satellite > Ground Control Voice Match**.
2. In the web UI, open **People**, add a person and select **Enroll**.
3. Pick the kiosk and read the text aloud while walking slowly around the room. Your phone shows the same text as the kiosk, so you can read from it.

Enroll on each kind of device you talk to, since a voice is recognized best on the kinds of device it was enrolled on.

## Updating

```bash
docker compose pull && docker compose up -d
```

## Reporting a problem

In the web UI, open **Logs** and select **Download**. Send the file along with what you said and what happened. If you can make it happen again, turn on **Detailed logs** first, repeat it and then download. Each session is also saved under **Sessions**, with its audio and every score, for 7 days.
