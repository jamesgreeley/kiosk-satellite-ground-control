# Settings

Navigate to **Settings** in the Ground Control app.

Settings has three areas. Pick one from the list on the left:

- **Voice Match:** Recognition, Trimming, the Wake Word Aligner and Isolation. See [Voice Match](voice-match.md#settings).
- **Vision Match:** the vision models, Hand sensitivity, Face Recognition and the frames kept with vision sessions. See [Vision Match](vision-match.md#settings).
- **General:** the engine, sessions, conversation agents and kiosks.

Changes are saved right away and take precedence over the app's options or the container's environment variables.

## General

| Setting | Default | What it does |
| --- | --- | --- |
| **Engine** | | |
| Runs on | | Shows where speaker checks run. On native macOS, **Core ML and CPU** means the local tuner selected one or more audio lengths. Other backends include CPU, CUDA and Vulkan. See [speaker tuning](apple-silicon.md#speaker-recognition). |
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
