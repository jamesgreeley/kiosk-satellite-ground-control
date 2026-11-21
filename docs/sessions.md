# Sessions

Navigate to **Sessions** in the Ground Control app.

Every voice command a kiosk sends to Ground Control is saved as a session: who talked, every voice check, why it ended and its audio. A conversation with follow-ups is one session with all its turns. Sessions are the place to see why a command went the way it did.

## The Sessions List

Each row shows when the session started, the kiosk, the speaker, the mode, the outcome and its length. The newest are first.

- **Search** finds kiosks and people by name.
- **Kiosk** and **Person** narrow the list to one kiosk or one person. **Unknown voices** shows the sessions where nobody enrolled was recognized.
- **Show** narrows it to sessions that were **Cut short** or **Rejected**.

## A Session's Page

| Section | What it shows |
| --- | --- |
| Timeline | The waveform of the session colored by who was talking, the voice scores over time against the thresholds and where speech started and stopped. With Trimming, it marks the parts sent to the assistant and the other voices left out. |
| Steps | Each step in order: the wake word and who said it, the user starting to talk, each recognition, other voices, Isolation and why the session ended. **Result** sums it up with the number of checks, the CPU time and how long Isolation took. |
| Audio | The audio as heard by the kiosk's microphone, the audio sent to the assistant and the wake word. |
| Use this voice | Adds the user's speech from this session to a person. |

### Use This Voice

Pick a person and choose **Add**. Only the user's speech is added, without the voices left out. When another voice talked, only the speech before it is added. This needs the session's audio, so it isn't available when **Keep audio** was off.

## Downloading a Session

**Download** in the session's header saves the session as one zip, ready to attach to an issue:

- `as-heard.wav`: the audio as the kiosk's microphone heard it.
- `sent-to-assistant.wav`: what was sent to the assistant.
- `wake-word.wav`: the wake word.
- `session.json`: the session's record, with every event and score.
- `ground-control.txt`: Ground Control's details and its log from while the session ran.

Tokens never reach the zip. Without kept audio, the zip has the record and the log only.

## Keeping Sessions

Sessions are kept for 7 days by default, then deleted with their audio. Change it under **Settings > General > Sessions**:

| Setting | Default | What it does |
| --- | --- | --- |
| Keep sessions for | 7 days | Older sessions and their audio are deleted. 1, 7, 30 or 90 days, or any number from 1 to 90 in the app's options. |
| Keep audio | on | Needed to replay a session or add it to a person. Off, sessions keep their record but no audio. |

**Delete session** in a session's header deletes it and its audio right away.
