# Settings

Navigate to **Settings** in the Ground Control app.

Settings has three areas, picked from the list on the left:

- **Voice Match:** Recognition, Trimming, the Wake Word Aligner and Isolation. See [Voice Match](voice-match.md#settings).
- **Vision Match:** the vision models, Face Recognition and the frames kept with vision sessions. See [Vision Match](#vision-match).
- **General:** the engine, sessions, conversation agents and kiosks.

Changes are saved right away and take precedence over the app's options or the container's environment variables.

## Vision Match

| Setting | Default | What it does |
| --- | --- | --- |
| **Models** | | |
| Vision model | Accurate with an NVIDIA GPU, else Light | Light runs on any CPU. Accurate gets the finger count right more often, recognizes faces better in dim light and needs an NVIDIA GPU. |
| Runs on | | NVIDIA GPU (CUDA) or CPU, set by the CUDA option. The models load with the first kiosk that asks. |
| Looks a second | | How many looks a second one kiosk can get, measured on this machine when the models load. |
| **Face Recognition** | | |
| Threshold | Accurate 0.3, Light 0.35 | How close a face has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown faces. Each vision model has its own. See [Face](people.md#face). |
| Faces enrolled | | People with a face enrolled. |
| **Frames** | | |
| Keep frames | on | Up to 8 camera frames for each vision session. See [Vision Sessions](sessions.md#vision-sessions). |
| Delete all frames | | Deletes every kept frame right away. The sessions stay, without their frames. |

Each kiosk turns Vision Match on in Kiosk Satellite, under **Settings > Ground Control > Vision Match**.

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
| Signed in with Home Assistant | | **Sign out** asks Home Assistant to end this browser's sign-in. Shown when the web app runs on its own, not as a Home Assistant app. |

## Conversation Agents

An LLM conversation agent in Home Assistant can tell who is speaking, so "my calendar" or "remind me" means the person talking. One template works for every kiosk:

1. In Kiosk Satellite, keep **Speaker sensor** on under **Settings > Ground Control > Voice Match**.
2. In Home Assistant, expose each kiosk's Speaker sensor to Assist under **Settings > Voice assistants > Expose**.
3. In Ground Control, open **Settings > General**, choose **Copy the template** under **Conversation agents** and paste it at the end of the agent's instructions.

The agent needs the Assist API. It reads the kiosk's Speaker sensor with GetLiveContext on each request, so it knows who is talking even when the conversation continues from an earlier command, and never reuses a name from earlier in it. An empty sensor means nobody was identified.
