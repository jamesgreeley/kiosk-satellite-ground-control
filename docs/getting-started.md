# Getting Started

You need:

- Kiosks running [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite) with Voice Satellite's native runtime and wake word detection on.
- Home Assistant, which Ground Control and the kiosks sign in to.
- For Vision Match, kiosks with a camera and **Enable camera** on under **Settings > Camera** in Kiosk Satellite.
- A machine for Ground Control: Home Assistant itself or any computer or server on the same network. See [Choosing a Machine](installation.md#choosing-a-machine).

## 1. Install Ground Control

**As a Home Assistant app:** add `https://github.com/jxlarrea/kiosk-satellite-ground-control` under **Settings > Apps > App store > Repositories**, install **Kiosk Satellite Ground Control** and start it. Open **Ground Control** in the sidebar.

**On an Apple Silicon Mac:** follow the [native macOS install](installation.md#run-natively-for-core-ml) to enable Core ML. Open `http://localhost:10900` on the Mac and choose **Log in with Home Assistant**.

**With Docker:** set `HA_URL` to your Home Assistant in [compose.yaml](../compose.yaml) and run it. On a machine with an NVIDIA GPU, use [compose.cuda.yaml](../compose.cuda.yaml) instead. Open `http://THIS_MACHINE:10900` and choose **Log in with Home Assistant**.

See [Installation](installation.md) for the options.

## 2. Connect the Kiosks

On each kiosk, in Kiosk Satellite:

1. Go to **Settings > Ground Control** and turn on **Enable Ground Control**. Ground Control is discovered automatically on the same network.
2. The status row shows **Waiting for approval**.

In Ground Control, open **Kiosks**, pick the kiosk and choose **Approve**. The kiosk's status row changes to **Connected**.

See [Kiosks](kiosks.md).

## 3. Enroll People

1. In Ground Control, open **People**, choose **Add a person** and enter their name.
2. On their page, choose **Enroll** on a device model they talk to or pick a kiosk under **Enroll on a kiosk** and choose **Enroll voice**.
3. They read the text on the kiosk aloud for about 45 seconds while walking slowly around the room.

Enroll each person once on every device model they talk to. For example, enroll them once on an Echo Show and once on an Echo Dot. See [People](people.md).

## 4. Turn On Voice Match

On each kiosk, in Kiosk Satellite, go to **Settings > Ground Control > Voice Match** and turn on **Enable Voice Match**. **Enable Trimming** and **Enable Isolation** are on by default.

For Isolation, also turn on **Use Isolation** in Ground Control under **Settings > Voice Match**. It needs the Accurate speaker model and is best on an NVIDIA GPU.

To have a kiosk answer only some people, set **Answer** to **Chosen people** and pick them under **People**.

See [Voice Match](voice-match.md).

## 5. Turn On Vision Match

On each kiosk with a camera, in Kiosk Satellite, go to **Settings > Ground Control > Vision Match** and turn on **Enable Vision Match**. **Enable Face Detection**, **Enable Face Recognition** and **Enable Hand Gestures** are on by default. Face Detection then dismisses or postpones the screensaver with **Dismiss on face** and **Postpone on face** under **Settings > Screensaver > Face Detection**. Hand Gestures run the **Show fingers** gestures under **Settings > Gestures**.

To enroll a face:

1. In Ground Control, open **People** and pick the person.
2. In their **Vision Match** card, pick a kiosk under **Enroll on a kiosk** and choose **Enroll face**. You can also start it on the kiosk with **Enroll a face** under **Settings > Ground Control > Vision Match**.
3. They follow the steps on the kiosk for about half a minute: look at the screen, turn a little to each side and take a step back.

Face Recognition is a convenience and not a lock, since a photo of an enrolled person can be recognized as them. See [Vision Match](vision-match.md).

## 6. Try It

Say the wake word and a command with a TV on. On the kiosk, the name of the person talking shows before the command. In Ground Control, the kiosk's page shows each voice check as it happens. The session also shows up on the **Sessions** page with what was sent to the assistant.

With Vision Match on, walk up to the kiosk and look at it. A card on the kiosk shows the person recognized and the kiosk's **Face** sensor in Home Assistant shows their name. The kiosk's page in Ground Control shows each look under **What it sees**. The vision session shows up on the **Sessions** page with the faces and hands seen and what the kiosk did.

If something doesn't work as expected, see [Logs and Issues](logs.md).

## Optional

- **Personal commands:** set each person's to-do list, calendar and playlist on the People page, so "add milk to my list" goes on the list of the person talking. See [People](people.md#home-assistant-and-personal-commands).
- **LLM conversation agents:** paste the template from **Settings > General > Conversation agents** into the agent's instructions so the speaker's name is passed to the agent. See [Settings](settings.md#conversation-agents).
