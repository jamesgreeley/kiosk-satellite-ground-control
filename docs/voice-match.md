# Voice Match

Navigate to **Settings > Voice Match** in the Ground Control app.

Voice Match is Ground Control's first set of features: who is speaking to a kiosk, and only their voice in the command.

- **Recognition:** who is speaking, from the voiceprints of the people enrolled.
- **Trimming:** a TV or another person before, between and after the user's words is cut out of the command, and the command ends when the user stops.
- **Wake Word Aligner:** the end of the wake word is left out of the command when the kiosk has **Talk right after the wake word** on.
- **Isolation:** other voices talking over the user's words are removed too.

Without Voice Match, a TV in the background can keep the assistant listening for many seconds after the command is over, and its words end up in the command.

## On the Kiosks

Each kiosk turns the features on for itself in Kiosk Satellite, under **Settings > Ground Control > Voice Match**:

| Kiosk setting | What it does |
| --- | --- |
| Enable Voice Match | Recognition. The kiosk knows who is talking. |
| Answer and People | **Chosen people** makes the kiosk answer only the people picked under **People** and drop a command from anyone else. |
| Enable Trimming | Sends everything the microphone hears to Ground Control and passes on only the user's speech. |
| Enable Isolation | Removes other voices under the user's words too. Needs **Use Isolation** on in Ground Control. |

The kiosk's page on the [Kiosks](kiosks.md) page shows which of these each kiosk has on. See [Voice Match](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/voice-match.md) in the Kiosk Satellite docs for every kiosk setting.

## Recognition

The user is whoever said the wake word. The kiosk sends the wake word with the command, so the voice is often recognized before the user says anything else. Otherwise it is recognized about a second into the command, while the person is still talking.

A voice counts as someone enrolled when its score against their voiceprint is above the **Threshold**. Below it, the voice is unknown. A kiosk set to answer chosen people drops a command from an unknown voice or from someone not chosen.

A short command is recognized from the speech of the wake word and the command together. When that falls short of the **Threshold**, the command is scored once more as it was heard, with the room around it, before the voice is left unknown or turned away.

In a conversation, a follow-up carries on with the person recognized in its first turn, so a one word answer like "yes" goes through.

### Confirmed by the Face

With **Confirm voices with faces** on in Kiosk Satellite, under **Settings > Ground Control > Vision Match**, a voice under the **Threshold** is taken for the person [Face Recognition](people.md#face) sees at the kiosk, when two things hold:

- That person's voiceprint is the closest to the voice. A face never names a voice on its own, and never one closer to someone else.
- The score reaches a line a seventh of the way from the other-voice line to the **Threshold**, the same line a short answer starts at: 0.32 with the Accurate model.

The session's page says the voice was taken this way, and the log shows it as `(face 0.41)` after the name. It is off by default on the kiosk: with it on, a guest talking while someone enrolled stands at the kiosk can now and then be taken for them.

### Short Answers

When Home Assistant asks a question (`assist_satellite.ask_question`, `assist_satellite.start_conversation`), the reply comes with no wake word to help and "yes" is too short to recognize anyone. A kiosk that answers chosen people still takes a short reply when it sounds like one of them:

- **Up to 0.75 seconds of speech** needs a score just over the **Other voice threshold** against one of their voiceprints, a seventh of the way to the **Threshold**: 0.32 with the Accurate model's defaults.
- **More speech needs a higher score,** rising to the **Threshold** at 2 seconds, since a voice scores higher against its own voiceprint the more there is of it.
- **The answer is scored with the room around it,** on 1.5 seconds centered on it. One word on its own is too little to tell a voice by.
- **From 2 seconds on** the answer is recognized like any command.

A reply that reaches the **Threshold** this way is recognized by name, as any command is. A reply of a second or more often does, where its speech alone would not. Under the **Threshold**, a short answer is taken without a name: the Speaker sensor only says who is speaking at the **Threshold**. An answer from someone the kiosk doesn't answer is still dropped, though a very short one can get through now and then. With a TV on, turn on Isolation, which keeps the TV out of the answer.

#### Limits of a Short Answer

A second of speech or less is not enough to recognize a voice reliably. A short answer is a looser check than recognition, with these limits:

- **It doesn't say who answered.** Under the **Threshold** the reply is accepted without a name and the kiosk's Speaker sensor stays empty. An automation that needs to know who answered can read the Speaker sensor after the reply: it names someone only when the voice reached the **Threshold**.
- **Someone else's reply can get through.** In our tests, a short reply from a voice the kiosk doesn't answer was accepted about 1 time in 50 on a kiosk that answers one person and up to 1 time in 5 on a kiosk that answers four people with similar voices. Don't count on a one word reply to keep someone out. For a question where that matters, like unlocking a door, ask for a reply of two seconds or more, such as "yes, unlock the door". That one is recognized like any command.
- **The right person's reply can still be dropped.** A quiet or mumbled "yes" can score under the line. With a TV or other people talking over the reply, about half of the short replies in our tests were dropped. Isolation cleans up what the assistant hears, but the reply is accepted or dropped before Isolation runs.
- **Less than 0.2 seconds of speech is never accepted.**
- **It can't be added to a person.** Adding a session's voice to a person needs at least a second of speech.

These numbers are from the Accurate speaker model. The Light model follows the same rule with its own thresholds.

Which turns can be a short answer depends on what the kiosk says about how the turn started:

| Turn | Short answer |
| --- | --- |
| A reply to something Home Assistant asked | Yes |
| A follow-up in a conversation | Yes |
| A wake word | No. A short command is recognized from the wake word and the command together. |
| A conversation started by hand, with no wake word | No |
| Kiosk Satellite before 2026.11.31, which doesn't say how the turn started | Yes, for every turn without a wake word |

**Take short answers** under **Settings > Voice Match > Recognition** turns this off. A short reply to a question is then dropped on a kiosk that answers chosen people, as before Ground Control 2026.11.13.

## Trimming

With Trimming on, the kiosk sends everything its microphone hears and Ground Control sends back only the speech of the person who said the wake word:

- **Other voices are cut out.** A TV or another person before, between and after the user's words never reaches the assistant. "Turn on the (TV talking) kitchen lights" reaches it as "turn on the kitchen lights".
- **Long pauses are shortened.**
- **The command ends when the user stops talking,** whatever the TV does, instead of when Home Assistant's own silence detection hears quiet.
- **The user's words on both sides of another voice are kept.** Another voice cuts in without ending the command.

The assistant hears the user about a second late while they talk, the time it takes to tell voices apart. The end of the command isn't delayed by it.

After a turn, a voice has to sound clearly like the user to start another turn in the same conversation. A TV that briefly sounds a little like them can't interrupt a realtime answer.

The **Other voice threshold** decides when someone else is talking: two checks in a row under it mean another voice.

## Wake Word Aligner

With **Talk right after the wake word** on in Kiosk Satellite, the command's audio starts the moment the wake word is recognized. A wake word engine recognizes the word before it is over, so its end can reach the assistant in front of the command. The "xa" of "Alexa" turns "what time is it" into "So what time is it".

The Wake Word Aligner leaves that end out. The wake word and the first 0.8 seconds of the command are checked for where the wake word ends and the command starts there:

- **It works with every wake word and every wake word engine.** Only the wake word's text is needed, pronounced in English and in the kiosk's language, so a custom wake word works too.
- **A command said in one breath with the wake word gets half a second of silence in front.** Speech-to-text hears an extra word in audio that starts on a sound.
- **When the end of the wake word isn't clear, nothing is left out** and the command goes on as it did before.
- **It doesn't delay the command.** It runs once, while the command's first audio is still held for its voice check.

It always runs on the CPU, also in the CUDA image:

| Machine | Time per command |
| --- | --- |
| Fast x86 CPU | About 4 ms |
| MacBook Pro M5, in Docker | About 10 ms |
| Intel N100 | About 15 ms |
| Raspberry Pi 4 | About 50 ms |

The Wake Word Aligner needs:

1. **Use Wake Word Aligner** on in Ground Control, under **Settings > Voice Match**. It is on by default.
2. **Enable Trimming** and **Talk right after the wake word** on in Kiosk Satellite on each kiosk that uses it.
3. Kiosk Satellite 2026.11.30 or later.

A session's page shows where the wake word ended under **Steps**.

## Isolation

Trimming cuts other voices out between the user's words. Isolation also removes them where they talk over the user, using the user's voiceprint. It runs on the whole turn once the user stops talking, so each command waits for it:

| Machine | Wait per command |
| --- | --- |
| NVIDIA GPU | About a tenth of a second |
| Fast x86 CPU | About a second |
| Intel N100 | About 6 seconds |
| Raspberry Pi 4 | Too slow |

On a CPU, Isolation uses a version of the model made for the CPU, twice as fast and nearly identical.

Isolation needs:

1. The Accurate speaker model.
2. **Use Isolation** on in Ground Control, under **Settings > Voice Match**.
3. **Enable Trimming** and **Enable Isolation** on in Kiosk Satellite on each kiosk that uses it. A kiosk where the short wait matters more can turn it off.

While a turn is held for Isolation, a little silence keeps the audio stream to Home Assistant open, so speech-to-text that gives up on a quiet stream, like Home Assistant Cloud's, still gets the command.

## Speaker Models

| Model | Best for | Notes |
| --- | --- | --- |
| Accurate (ReDimNet2 B3, multilingual) | An NVIDIA GPU or an x86 CPU with more than 4 cores | The fewest mix-ups between people and with TV voices. Needed for Isolation. |
| Light (ReDimNet2 B1) | A Raspberry Pi, Home Assistant Green, an Intel N100 or other small machines | About a third of the CPU. A TV voice cutting in ends the command about a second later. |

Ground Control picks Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with more than 4 cores, and Light on other ARM CPUs and smaller x86 CPUs. Changing the model rebuilds everyone's voiceprint from the kept enrollment audio in the background, so nobody records again. Kiosks keep using the current model until the rebuild is done.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| **Recognition** | | |
| Speaker model | auto | Accurate or Light. See [Speaker Models](#speaker-models). |
| Threshold | 0.5 Accurate, 0.4 Light | How close a voice has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown voices. Each model has its own. |
| Learn from commands | on | Confident commands fold into the speaker's voice, so it follows colds and room changes. |
| Take short answers | on | Accepts a short reply to a question from the assistant, like "yes" or "no", when the voice vaguely sounds like one of the people the kiosk accepts. A reply this short isn't enough to recognize who is speaking, so with this off it is ignored. Only applies to kiosks that answer chosen people. See [Short Answers](#short-answers). |
| **Trimming** | | |
| Other voice threshold | 0.29 Accurate, 0.26 Light | Two checks in a row under this score mean someone else is talking. |
| **Wake Word Aligner** | | |
| Use Wake Word Aligner | on | Leaves the end of the wake word out of the command, for kiosks with Trimming and **Talk right after the wake word** on in Kiosk Satellite. See [Wake Word Aligner](#wake-word-aligner). |
| **Isolation** | | |
| Use Isolation | off | Removes other voices under the user's words, for kiosks with Trimming and Isolation on in Kiosk Satellite. Needs the Accurate model. |
| Runs on | | NVIDIA GPU (CUDA) or CPU, set by the image. |

The thresholds are a starting point. If a TV voice is taken for someone, raise **Threshold**. If a person is often unknown, enroll them on that device model first, then lower it a little.
