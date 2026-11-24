# Sessions

Navigate to **Sessions** in the Ground Control app.

Ground Control saves two kinds of session:

- **Voice Match:** every voice command a kiosk sends: who talked, every voice check, why it ended and its audio. A conversation with follow-ups is one session with all its turns.
- **Vision Match:** each stretch of time someone is in front of a kiosk with Vision Match on: what the camera saw, what the kiosk did with it, how long each look took and a few of the frames. See [Vision Sessions](#vision-sessions).

Sessions are the place to see why a command or a gesture went the way it did.

## The Sessions List

Each row shows when the session started, the kiosk, who it was, its type, the outcome and its length. The newest are first. A vision session shows how many looks it took and, as its outcome, what the kiosk did, such as a gesture and the action it ran, or else what was seen.

- **Type** shows only Voice Match or only Vision Match sessions.
- **Search** finds kiosks and people by name.
- **Kiosk** and **Person** narrow the list to one kiosk or one person. **Unknown voices** shows the sessions where nobody enrolled was recognized.
- **Show** narrows it to sessions that were **Cut short** or **Rejected**.

## A Voice Session's Page

The top of the page shows the speaker, the outcome, the mode, who the kiosk answers and the speaker model. With Kiosk Satellite 2026.11.30 or later it also shows the wake word and its engine, whether an Assist pipeline or a realtime assistant answered and whether **Talk right after the wake word** was on. With Kiosk Satellite 2026.11.31 or later it shows how the turn started when it wasn't a wake word: **Ask Question**, **Start Conversation**, a follow-up or by hand. The two kinds of question reach a kiosk the same way. The kiosk tells them apart when Home Assistant's pipeline starts, so a session where it never started shows neither.

| Section | What it shows |
| --- | --- |
| Timeline | The waveform of the session colored by who was talking, the voice scores over time against the thresholds and where speech started and stopped. With Trimming, it marks the parts sent to the assistant and the other voices left out. |
| Steps | Each step in order: the wake word and who said it, where the wake word ended when its end was left out, the user starting to talk, each recognition, other voices, Isolation and why the session ended. **Result** sums it up with the number of checks, the CPU time and how long Isolation took. |
| Audio | The audio as heard by the kiosk's microphone, the audio sent to the assistant and the wake word. |
| Use this voice | Adds the user's speech from this session to a person. |

### Use This Voice

Pick a person and choose **Add**. Only the user's speech is added, without the voices left out. When another voice talked, only the speech before it is added. This needs the session's audio, so it isn't available when **Keep audio** was off.

## Vision Sessions

A vision session starts with the first look that finds a face or a hand. It ends 10 seconds after the last look that found one, or when the kiosk's camera stops. A stretch longer than 30 minutes goes on in a new session. Looks that find nothing, like a pet walking by or a light changing, open no session. They are counted on the [kiosk's page](kiosks.md#a-kiosks-page) instead, so the list doesn't fill up with empty stretches.

The top of a vision session's page shows what the kiosk did, its length and the vision model.

| Section | What it shows |
| --- | --- |
| Timeline | One lane each: **Kiosk** shows what the kiosk did, the screensaver dismissed or postponed and each finger gesture with its action. **Face** marks a face looking at the kiosk or turned away, **Hand** marks a hand with its finger count and **Time** shows how long Ground Control took and the round trip from the kiosk. |
| Steps | Each step in order: a face seen, a face looking at the kiosk, a hand, each new finger reading, what the kiosk did and why the session ended. |
| Performance | The median, 95th percentile and slowest time for reading the frame, Face Detection, Hand Gestures and Ground Control's whole look. Also the kiosk's time to encode each frame and the round trip from the kiosk. For each finger gesture, the time from the first look that read its finger count to the kiosk running its action, the gesture's hold time included. It also counts the frames dropped for a newer one and the frames that couldn't be read. |
| Frames | Up to 8 frames with the faces and the hand drawn on them: the first look, the first face looking at the kiosk, each new finger reading, the frame each gesture or screensaver dismissal acted on and the last look. |

## Downloading a Session

**Download** in the session's header saves the session as one zip, ready to attach to an issue:

- `as-heard.wav`: the audio as the kiosk's microphone heard it.
- `sent-to-assistant.wav`: what was sent to the assistant.
- `wake-word.wav`: the wake word.
- `frames/`: a vision session's kept frames.
- `session.json`: the session's record, with every event and score, or every look of a vision session.
- `ground-control.txt`: Ground Control's details and its log from while the session ran.

Tokens never reach the zip. Without kept audio or frames, the zip has the record and the log only.

## Keeping Sessions

Sessions are kept for 7 days by default, then deleted with their audio and frames. Change it under **Settings > General > Sessions**:

| Setting | Default | What it does |
| --- | --- | --- |
| Keep sessions for | 7 days | Older sessions, their audio and their frames are deleted. 1, 7, 30 or 90 days, or any number from 1 to 90 in the app's options. |
| Keep audio | on | Needed to replay a session or add it to a person. Off, sessions keep their record but no audio. |

Frames are kept under **Settings > Vision Match > Frames**:

| Setting | Default | What it does |
| --- | --- | --- |
| Keep frames | on | Up to 8 frames for each vision session. Off, vision sessions keep their record but no frames. |
| Delete all frames | | Deletes every kept frame right away. The sessions stay, without their frames. |

Ground Control keeps up to 2,000 voice sessions and 3,000 vision sessions, so a busy week in front of the kiosks never pushes voice sessions out.

**Delete session** in a session's header deletes it with its audio or frames right away.
