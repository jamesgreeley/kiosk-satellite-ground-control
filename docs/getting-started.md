# Getting Started

You need:

- Kiosks running [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite) with Voice Satellite's native runtime and wake word detection on.
- Home Assistant, which Ground Control and the kiosks sign in to.
- A machine for Ground Control: Home Assistant itself, or any computer or server on the same network. See [Choosing a Machine](installation.md#choosing-a-machine).

## 1. Install Ground Control

**As a Home Assistant app:** add `https://github.com/jxlarrea/kiosk-satellite-ground-control` under **Settings > Apps > App store > Repositories**, install **Kiosk Satellite Ground Control** and start it. Open **Ground Control** in the sidebar.

**With Docker:** run [compose.yaml](../compose.yaml), or [compose.cuda.yaml](../compose.cuda.yaml) on a machine with an NVIDIA GPU, with `HA_URL` set to your Home Assistant. Open `http://THIS_MACHINE:10900` and choose **Log in with Home Assistant**.

See [Installation](installation.md) for the options.

## 2. Connect the Kiosks

On each kiosk, in Kiosk Satellite:

1. Go to **Settings > Ground Control** and turn on **Enable Ground Control**. Kiosks on the same network find Ground Control on their own.
2. The status row shows **Waiting for approval**.

In Ground Control, open **Kiosks**, pick the kiosk and choose **Approve**. The kiosk's status row changes to **Connected**.

See [Kiosks](kiosks.md).

## 3. Enroll People

1. In Ground Control, open **People**, choose **Add a person** and enter their name.
2. On their page, choose **Enroll** next to a kind of device they talk to, or **Enroll on a kiosk** to pick the kiosk.
3. They read the text on the kiosk aloud for about 45 seconds, walking slowly around the room.

Enroll each person once on every kind of device they talk to, such as once on an Echo Show and once on an Echo Dot. See [People](people.md).

## 4. Turn On Voice Match

On each kiosk, in Kiosk Satellite, go to **Settings > Voice Satellite > Ground Control Voice Match** and turn on **Enable Voice Match**. **Enable Trimming** and **Enable Isolation** are on by default.

For Isolation, also turn on **Use Isolation** in Ground Control under **Settings > Voice Match**. It needs the Accurate speaker model and is best on an NVIDIA GPU.

To have a kiosk answer only some people, set **Answer** to **Chosen people** and pick them under **People**.

See [Voice Match](voice-match.md).

## 5. Try It

Say the wake word and a command with a TV on. On the kiosk, the name of the person talking shows before the command. In Ground Control, the kiosk's page shows each voice check as it happens, and the session shows up on the **Sessions** page with what was sent to the assistant.

If something doesn't work as expected, see [Logs and Issues](logs.md).

## Optional

- **Personal commands:** set each person's to-do list, calendar and playlist on the People page, so "add milk to my list" goes on the list of the person talking. See [People](people.md#home-assistant-and-personal-commands).
- **LLM conversation agents:** paste the template from **Settings > General > Conversation agents** into the agent's instructions, so the agent knows who is speaking. See [Settings](settings.md#conversation-agents).
