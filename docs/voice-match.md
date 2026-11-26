# Voice Match

Navigate to **Settings > Voice Match** in the Ground Control app.

Voice Match is Ground Control's first set of features. It recognizes who is speaking to a kiosk and keeps only their voice in the command.

- **Recognition:** identifies who is speaking using the voiceprints of the people enrolled.
- **Trimming:** a TV or another person before, between and after the user's words is cut out of the command. The command ends when the user stops.
- **Wake Word Verifier:** checks that the wake word was really said and stops the session when the kiosk woke up to something else.
- **Wake Word Aligner:** the end of the wake word is left out of the command when the kiosk has **Talk right after the wake word** on.
- **Isolation:** other voices talking over the user's words are removed too.

Without Voice Match, a TV in the background can keep the assistant listening for many seconds after the command is over and its words end up in the command.

## On the Kiosks

Turn the features on for each kiosk in Kiosk Satellite under **Settings > Ground Control > Voice Match**:

| Kiosk setting | What it does |
| --- | --- |
| Enable Voice Match | Turns on Recognition, which identifies who is talking. |
| Answer and People | **Chosen people** makes the kiosk answer only the people picked under **People** and drop a command from anyone else. |
| Enable Trimming | Sends everything the microphone picks up to Ground Control and passes on only the user's speech. |
| Enable Wake Word Verifier | Stops the session when the kiosk woke up to something that wasn't its wake word. Needs **Use Wake Word Verifier** on in Ground Control. |
| Trust enrolled voices | A wake word in the voice of someone enrolled always passes the Wake Word Verifier. |
| Wait for wake word verification | Plays the chime and shows the voice session only after the wake word is verified. Off by default. |
| Enable Wake Word Aligner | Leaves the end of the wake word out of the command. Needs **Use Wake Word Aligner** on in Ground Control. |
| Enable Isolation | Removes other voices under the user's words too. Needs **Use Isolation** on in Ground Control. |

The kiosk's page on the [Kiosks](kiosks.md) page shows which of these each kiosk has on. See [Voice Match](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/voice-match.md) in the Kiosk Satellite docs for every kiosk setting.

## Recognition

The user is whoever said the wake word. The kiosk sends the wake word with the command, so the voice is often recognized before the user says anything else. Otherwise it is recognized about a second into the command while the person is still talking.

A voice counts as someone enrolled when its score against their voiceprint is above the **Threshold**. Below it, the voice is unknown. A kiosk set to answer chosen people drops a command from an unknown voice or from someone not chosen.

A short command is recognized from the speech of the wake word and the command together. If that falls short of the **Threshold**, the command is scored once more as it was heard with the room around it. Only then is the voice left unknown or turned away.

In a conversation, a follow-up carries on with the person recognized in its first turn, so a one word answer like "yes" goes through.

### Confirmed by the Face

Turn on **Confirm voices with faces** in Kiosk Satellite under **Settings > Ground Control > Vision Match** to accept a voice under the **Threshold** as the person recognized at the kiosk by [Face Recognition](people.md#face). This only happens when both of these are true:

- That person's voiceprint is the closest to the voice. A face alone is never enough to name a voice. A voice closer to someone else's voiceprint is never taken for the person at the kiosk.
- The score reaches a line a seventh of the way from the other-voice line to the **Threshold**. This is the same line a short answer starts at. It is 0.32 with the Accurate model.

The session's page shows when a voice was accepted this way. The log shows it as `(face 0.41)` after the name. It is off by default on the kiosk. With it on, a guest who talks while someone enrolled stands at the kiosk can now and then be taken for that person.

### Short Answers

When Home Assistant asks a question (`assist_satellite.ask_question`, `assist_satellite.start_conversation`), the reply comes with no wake word to help and "yes" is too short to recognize anyone. A kiosk that answers chosen people still accepts a short reply when it sounds like one of them:

- **Up to 0.75 seconds of speech** needs a score against one of their voiceprints just over the **Other voice threshold**. That line is a seventh of the way to the **Threshold**. It is 0.32 with the Accurate model's defaults.
- **More speech needs a higher score.** The line rises to the **Threshold** at 2 seconds because a voice scores higher against its own voiceprint the more of it there is.
- **The answer is scored with the room around it.** The score uses 1.5 seconds of audio centered on the answer. One word on its own isn't enough to recognize a voice.
- **From 2 seconds on** the answer is recognized like any command.

A reply that reaches the **Threshold** this way is recognized by name like any command. A reply of a second or more often reaches it this way when its speech alone would not. Under the **Threshold**, a short answer is accepted without a name. The Speaker sensor only names who is speaking when the score reaches the **Threshold**. An answer from someone the kiosk doesn't answer is still dropped. A very short one can get through now and then. If a TV is on, turn on Isolation to keep the TV out of the answer.

#### Limits of a Short Answer

A second of speech or less is not enough to recognize a voice reliably. A short answer is a looser check than recognition and has these limits:

- **The person who answered isn't named.** Under the **Threshold** the reply is accepted without a name and the kiosk's Speaker sensor stays empty. To find out who answered in an automation, read the Speaker sensor after the reply. The sensor shows a name only when the voice reached the **Threshold**.
- **Someone else's reply can get through.** In our tests, a short reply from a voice the kiosk doesn't answer was accepted about 1 time in 50 on a kiosk that answers one person and up to 1 time in 5 on a kiosk that answers four people with similar voices. Don't count on a one word reply to keep someone out. If it matters, like when unlocking a door, ask for a reply of two seconds or more such as "yes, unlock the door". A reply that long is recognized like any command.
- **The right person's reply can still be dropped.** A quiet or mumbled "yes" can score under the line. With a TV or other people talking over the reply, about half of the short replies in our tests were dropped. Isolation cleans up the audio sent to the assistant, but the reply is accepted or dropped before Isolation runs.
- **Less than 0.2 seconds of speech is never accepted.**
- **It can't be added to a person.** Adding a session's voice to a person needs at least a second of speech.

These numbers are from the Accurate speaker model. The Light model follows the same rule with its own thresholds.

Which turns can be a short answer depends on how the kiosk reports the turn started:

| Turn | Short answer |
| --- | --- |
| A reply to something Home Assistant asked | Yes |
| A follow-up in a conversation | Yes |
| A wake word | No. A short command is recognized from the wake word and the command together. |
| A conversation started by hand, with no wake word | No |
| Kiosk Satellite before 2026.11.31, which doesn't report how the turn started | Yes, for every turn without a wake word |

Turn off **Take short answers** under **Settings > Voice Match > Recognition** to stop this. A kiosk that answers chosen people then drops a short reply to a question, as it did before Ground Control 2026.11.13.

## Trimming

With Trimming on, the kiosk sends everything its microphone picks up and Ground Control sends back only the speech of the person who said the wake word:

- **Other voices are cut out.** A TV or another person before, between and after the user's words never reaches the assistant. "Turn on the (TV talking) kitchen lights" reaches it as "turn on the kitchen lights".
- **Long pauses are shortened.**
- **The command ends when the user stops talking.** This happens whatever the TV is doing and without waiting for Home Assistant's own silence detection to detect quiet.
- **The user's words on both sides of another voice are kept.** Another voice that cuts in doesn't end the command.

The assistant gets the user's speech about a second late while they talk. That is the time it takes to tell voices apart. This doesn't delay the end of the command.

After a turn, a voice has to sound clearly like the user to start another turn in the same conversation. A TV that briefly sounds a little like them can't interrupt a realtime answer.

The **Other voice threshold** sets when someone else counts as talking. Two checks in a row under it mean another voice.

## Wake Word Verifier

A wake word engine now and then wakes a kiosk up to speech that is nothing like its wake word. Most often it is a TV across the room. The Wake Word Verifier checks that the wake word was really said. If it wasn't, the session is stopped and nothing reaches the assistant.

- **It runs 0.3 seconds after the kiosk wakes up.** A wake word engine recognizes the word before it is over, so the check waits for the end of the word. It listens to the wake word the kiosk sends and to the 0.3 seconds after it.
- **It works with every wake word and every wake word engine.** Only the wake word's text is needed. It is pronounced in English and in the kiosk's language, so a custom wake word works too.
- **A wake word in the voice of someone enrolled always passes,** however unclear it was. A kiosk can turn that off with **Trust enrolled voices** in Kiosk Satellite. That is for someone whose own voice wakes the kiosk up by mistake, like during a conference call.
- **It doesn't delay anything.** The check takes about as long as the Wake Word Aligner. The wake sound has already played and the command's first audio is still held for its voice check. To hold the wake sound until the check is done, see [Wait for Wake Word Verification](#wait-for-wake-word-verification).
- **If the check can't run, the session goes on** as it did before.
- **The kiosk stops listening without a sound** when a session is stopped. A Kiosk Satellite from before the Wake Word Verifier follows its **Voice not chosen** setting instead, which is quiet by default.

It doesn't stop a TV that says the wake word itself. Set the kiosk to answer **Chosen people** for that, so a voice nobody enrolled is dropped.

In our tests on three kiosks with "Hey Jarvis", "Ok Nova" and "Alexa", 1 of 114 real wake words was stopped. Of 672 stretches of other speech from the same kiosks, 1 would have passed. Every session where a kiosk had woken up to a TV with no wake word in it was stopped.

The Wake Word Verifier needs:

1. **Use Wake Word Verifier** is on in Ground Control under **Settings > Voice Match**. It is on by default.
2. **Enable Trimming** and **Enable Wake Word Verifier** on in Kiosk Satellite on each kiosk that uses it. Both are on by default.
3. Kiosk Satellite 2026.11.30 or later. An older Kiosk Satellite than the one with its own Wake Word Verifier switch is checked whenever it is on in Ground Control.

A session stopped this way shows as **False positive** on the Sessions page. Its page shows the check under **Steps**.

### Wait for Wake Word Verification

A kiosk starts the session the moment it wakes up, so a false positive still plays the wake sound and shows the voice overlay for a moment before it is stopped. With **Wait for wake word verification** on in Kiosk Satellite, the wake word goes to Ground Control first and the session starts only when it was said. A false positive then plays no sound and shows nothing on the screen.

- **It can add a small delay before the voice session starts.** The kiosk waits for the 0.3 seconds the check listens to after the wake word and then for the answer. The answer takes a few milliseconds on a PC and about 40 ms on a Raspberry Pi 4.
- **What is said during the wait is kept.** With **Talk right after the wake word** on, the command still starts right after the wake word.
- **Without an answer within 0.3 seconds the session starts** and the wake word is checked in the session instead.
- **A false positive is still saved** as a session of its own, with the wake word as the kiosk heard it.
- **A wake word that was said isn't checked again** in its session. The session's page says the kiosk waited for the check.

It is off by default. It needs Kiosk Satellite 2026.11.33 or later, with **Enable Wake Word Verifier** on.

## Wake Word Aligner

With **Talk right after the wake word** on in Kiosk Satellite, the command's audio starts the moment the wake word is recognized. A wake word engine recognizes the word before it is over, so its end can reach the assistant in front of the command. The "xa" of "Alexa" turns "what time is it" into "So what time is it".

The Wake Word Aligner leaves that end out. It finds where the wake word ends in the wake word and the first 0.8 seconds of the command. The command then starts there:

- **It works with every wake word and every wake word engine.** Only the wake word's text is needed. It is pronounced in English and in the kiosk's language, so a custom wake word works too.
- **A command said in one breath with the wake word gets half a second of silence in front.** Speech-to-text picks up an extra word when the audio starts on a sound.
- **When the end of the wake word isn't clear, nothing is left out** and the command goes on as it did before.
- **It doesn't delay the command.** It runs once while the command's first audio is still held for its voice check.

Even in the CUDA image it always runs on the CPU:

| Machine | Time per command |
| --- | --- |
| Fast x86 CPU | About 4 ms |
| MacBook Pro M5, in Docker | About 10 ms |
| Intel N100 | About 15 ms |
| Raspberry Pi 4 | About 50 ms |

The Wake Word Aligner needs:

1. **Use Wake Word Aligner** is on in Ground Control under **Settings > Voice Match**. It is on by default.
2. **Enable Trimming**, **Talk right after the wake word** and **Enable Wake Word Aligner** on in Kiosk Satellite on each kiosk that uses it.
3. Kiosk Satellite 2026.11.30 or later.

A session's page shows where the wake word ended under **Steps**.

## Isolation

Trimming cuts other voices out between the user's words. Isolation uses the user's voiceprint to also remove them where they talk over the user. It runs on the whole turn once the user stops talking, so it adds a wait to each command:

| Machine | Wait per command |
| --- | --- |
| NVIDIA GPU | Under a tenth of a second |
| Fast x86 CPU | About half a second |
| Intel N100 | About 3 seconds |
| Raspberry Pi 4 | Too slow |

On a CPU, Isolation uses a version of the model made for the CPU. It is twice as fast and gives nearly identical results.

Isolation needs:

1. The Accurate speaker model.
2. **Use Isolation** is on in Ground Control under **Settings > Voice Match**.
3. **Enable Trimming** and **Enable Isolation** on in Kiosk Satellite on each kiosk that uses it. Turn it off on a kiosk where the short wait matters more.

While a turn is held for Isolation, a little silence is sent to keep the audio stream to Home Assistant open. This way speech-to-text that times out on a quiet stream, like Home Assistant Cloud's, still gets the command.

## Speaker Models

| Model | Best for | Notes |
| --- | --- | --- |
| Accurate (ReDimNet2 B3, multilingual) | An NVIDIA GPU or an x86 CPU with more than 4 cores | The fewest mix-ups between people and with TV voices. Needed for Isolation. |
| Light (ReDimNet2 B1) | A Raspberry Pi, Home Assistant Green, an Intel N100 or other small machines | About a third of the CPU. A TV voice cutting in ends the command about a second later. |

Ground Control picks Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with more than 4 cores. It picks Light on other ARM CPUs and smaller x86 CPUs. Changing the model rebuilds everyone's voiceprint from the kept enrollment audio in the background, so nobody records again. Kiosks keep using the current model until the rebuild is done.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| **Recognition** | | |
| Speaker model | auto | Accurate or Light. See [Speaker Models](#speaker-models). |
| Threshold | 0.5 Accurate, 0.4 Light | How close a voice has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown voices. Each model has its own. |
| Learn from commands | on | Commands recognized with high confidence are folded into the speaker's voice, so it adapts to colds and room changes. |
| Take short answers | on | Accepts a short reply like "yes" or "no" to a question from the assistant when the voice vaguely sounds like one of the people the kiosk accepts. A reply this short isn't enough to recognize who is speaking, so with this off it is ignored. Only applies to kiosks that answer chosen people. See [Short Answers](#short-answers). |
| **Trimming** | | |
| Other voice threshold | 0.29 Accurate, 0.26 Light | Two checks in a row under this score mean someone else is talking. |
| **Wake Word Verifier** | | |
| Use Wake Word Verifier | on | Checks that the wake word was really said on kiosks with Trimming and the Wake Word Verifier on in Kiosk Satellite and stops the session when it wasn't. See [Wake Word Verifier](#wake-word-verifier). |
| **Wake Word Aligner** | | |
| Use Wake Word Aligner | on | Leaves the end of the wake word out of the command on kiosks with Trimming and **Talk right after the wake word** on in Kiosk Satellite. See [Wake Word Aligner](#wake-word-aligner). |
| **Isolation** | | |
| Use Isolation | off | Removes other voices under the user's words on kiosks with Trimming and Isolation on in Kiosk Satellite. Needs the Accurate model. |
| Runs on | | NVIDIA GPU (CUDA) or CPU. The image sets which one. |

The thresholds are a starting point. If a TV voice is taken for someone, raise **Threshold**. If a person is often unknown, enroll them on that device model first and then lower it a little.
