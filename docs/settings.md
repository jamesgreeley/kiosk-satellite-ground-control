# Settings

Navigate to **Settings** in the Ground Control app.

Settings has three areas. Pick one from the list on the left:

- **Voice Match:** Recognition, Trimming, the Wake Word Aligner and Isolation. See [Voice Match](voice-match.md#settings).
- **Vision Match:** the vision models, Hand sensitivity, Face Recognition and the frames kept with vision sessions. See [Vision Match](#vision-match).
- **General:** the engine, sessions, conversation agents and kiosks.

Changes are saved right away and take precedence over the app's options or the container's environment variables.

## Vision Match

| Setting | Default | What it does |
| --- | --- | --- |
| **Models** | | |
| Face Detection | Accurate with an NVIDIA GPU, else Light | Light or Accurate. Both found every face looking at the kiosk on the test set and let the same turned heads through. Accurate takes about 3 times as long. |
| Hand Gestures | Accurate with an NVIDIA GPU, else Light | Light or Accurate. Accurate gets the finger count right more often at 3 m and in dim light. The hand model runs on every look, so on a CPU Accurate lowers the looks a second each kiosk gets. |
| Face Recognition | Accurate with an NVIDIA GPU, an x86 CPU or an Apple M-series chip, else Light | Light or Accurate. Accurate recognizes small faces in dim light more often. A recognized face is checked again every 2 seconds and an unknown face once a second, so Accurate costs little even on a CPU. Faces are rebuilt from their pictures when it changes, so nobody has to enroll again. |
| Runs on | | NVIDIA GPU (CUDA) or CPU. The CUDA option sets it. The models load when the first kiosk requests them. Each row above shows its model's time on this machine once they load. |
| Looks a second | | How many looks a second one kiosk can get. It is measured on this machine when the models load. |
| **Hand Gestures** | | |
| Hand sensitivity | 50 | From 1 to 100. It sets the smallest hand counted, by the longer side of its box: about 9% of the frame at 50, 13% at 30 and any size at 100. Lower it if a hand that is only a small part of the frame, such as one cut off at its edge, shows up in vision sessions. Hands that gave a finger count on kiosk frames were 10% of the frame or more. |
| **Face Recognition** | | |
| Threshold | Accurate 0.3, Light 0.35 | How close a face has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown faces. Each Face Recognition model has its own. See [Face](people.md#face). |
| Faces enrolled | | People with a face enrolled. |
| **Frames** | | |
| Keep frames | on | Keeps up to 8 camera frames for each vision session. See [Vision Sessions](sessions.md#vision-sessions). |
| Delete all frames | | Deletes every kept frame right away. The sessions themselves are kept. |

Turn on Vision Match for each kiosk in Kiosk Satellite under **Settings > Ground Control > Vision Match**.

## General

| Setting | Default | What it does |
| --- | --- | --- |
| **Engine** | | |
| Runs on | | NVIDIA GPU (CUDA) or CPU. CUDA is available in the Docker image on a machine with an NVIDIA GPU. |
| Threads | half the CPUs, up to 4 | Threads per check. Set in the app's options or with `THREADS`. |
| **Sessions** | | |
| Keep sessions for | 7 days | Older sessions, their audio and their frames are deleted. See [Sessions](sessions.md#keeping-sessions). |
| Keep audio | on | Needed to replay a session or add it to a person. |
| **Conversation agents** | | |
| Speaker in the agent's instructions | | A template to paste into an LLM agent's instructions. See [Conversation Agents](#conversation-agents). |
| **Kiosks** | | |
| Address | | The address to enter in Kiosk Satellite for a kiosk on another network. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |
| **This browser** | | |
| Signed in with Home Assistant | | **Sign out** ends this browser's Home Assistant sign-in. This only shows when the web app runs on its own instead of as a Home Assistant app. |

## Conversation Agents

You can pass the speaker to an LLM conversation agent in Home Assistant so "my calendar" or "remind me" means the person talking. One template works for every kiosk:

1. In Kiosk Satellite, keep **Speaker sensor** on under **Settings > Ground Control > Voice Match**.
2. In Home Assistant, expose each kiosk's Speaker sensor to Assist under **Settings > Voice assistants > Expose**.
3. In Ground Control, open **Settings > General**, choose **Copy the template** under **Conversation agents** and paste it at the end of the agent's instructions.

The agent needs the Assist API. It reads the kiosk's Speaker sensor with GetLiveContext on each request. That way the current speaker is used even when the conversation continues from an earlier command. A name from earlier in the conversation is never reused. An empty sensor means nobody was identified.
