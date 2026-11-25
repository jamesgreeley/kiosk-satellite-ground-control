# People

Navigate to **People** in the Ground Control app.

Everyone Voice Match and Face Recognition can recognize is on this page, with their voice, their face, their color and the things "my" means for them. The list on the left shows each person and whether their voice and face are enrolled, with how many device models the voice is enrolled on.

## Adding a Person

1. Choose **Add a person** and enter their name. This is how the kiosks and the assistant call them.
2. Enroll their voice on each device model they talk to. See [Enrollment](#enrollment).

A person without a voice is never recognized, but can still be picked on a kiosk under **Answer > Chosen people**.

## Enrollment

A voice is recognized best on the device models it was enrolled on, since an Echo Show and an Echo Dot hear the same voice differently. Enrollment always records with a kiosk's own microphone, so enroll each person once on every device model they talk to.

1. On the person's page, the **Voice Match** card lists every device model in the house under **Device models**, with the seconds of speech enrolled on each. Choose **Enroll** on a model, or pick a kiosk under **Enroll on a kiosk** and choose **Enroll voice**.
2. The kiosk shows the text to read, full screen in the kiosk's language. The Ground Control app shows the same text, so it can also be read from a phone.
3. Read it aloud for about 45 seconds:
    - Talk normally, in the voice you use for commands, not a reading voice.
    - Start close to the kiosk, then move across the room.
    - Face away from the kiosk now and then.

The sentence to read fills as it is spoken and moves on when it is full or after a pause. When another voice talks, such as a TV, that part is left out and a banner says so. The enrollment ends once 45 seconds of speech are counted, and **Enroll next** offers the next device model the person isn't enrolled on.

What was recorded is kept when the enrollment is stopped early with at least 10 seconds of speech. The wake word doesn't listen during an enrollment, so reading the text never starts a voice command.

Only kiosks with **Allow enrollment here** on, under **Settings > Ground Control > Voice Match** in Kiosk Satellite, are offered. A kiosk busy with a voice command or an alarm can't start an enrollment until it is done.

### Recordings

**Recordings**, at the bottom of the **Voice Match** card, opens the list of each enrollment with the kiosk it was recorded on and how much speech it has. Deleting one rebuilds the person's voice without it.

### Adding a Voice from a Session

On the **Sessions** page, on a session where the person talked, **Use this voice** adds their speech from it to their voice. Only the user's speech goes in, without the voices left out. This is useful for a device model nobody enrolled on yet, or for a voice that has changed. See [Sessions](sessions.md#use-this-voice).

### Learning

With **Learn from commands** on, under **Settings > Voice Match**, commands where a person was recognized with high confidence are folded into their voice. Their voiceprint follows a cold or a new room without enrolling again.

## Face

Face Recognition tells who is looking at a kiosk, from the faces enrolled here. Each kiosk with Vision Match turns it on in Kiosk Satellite, under **Settings > Ground Control > Vision Match**, and shows who it sees in its Face sensor in Home Assistant.

### Enrolling a Face

A face is enrolled through a kiosk's own camera, in a guided capture of about half a minute. Start it on the kiosk, or in the person's **Vision Match** card here: pick the kiosk under **Enroll on a kiosk** and choose **Enroll face**. Only kiosks with Face Recognition on are offered. The kiosk then shows the steps:

1. Look at the screen.
2. Turn your head a little to one side.
3. Now a little to the other side.
4. Take a step back.

Each step needs a few good pictures of the face. The kiosk says what is wrong when one isn't good: no face, more than one face, come closer, turn more or less, too dark. A different face is left out, so someone walking past doesn't end up in the enrollment. The person's page follows the steps as they happen, wherever the capture was started: the kiosk, the step, the pictures taken and what to fix. To stop early, use **Stop** on the kiosk.

What was captured is kept when the capture stops early after the first step. Enrolling on another kiosk, or in other light, helps the face be recognized there.

Only small pictures of the face, lined up and 112 pixels wide, are kept, so a change of vision model rebuilds the faces without anyone enrolling again. The **Vision Match** card shows them under **Face pictures**, and **Captures** opens the list of each capture with the kiosk it was taken on. Deleting a capture rebuilds the person's face without it.

### How a Face Is Recognized

A face looking at the kiosk is compared with every enrolled face. A name needs two looks in a row above the **Threshold** under **Settings > Vision Match > Face Recognition**, or one very close match, and the best match has to be clearly ahead of the next person. Once named, the face is checked again every 2 seconds. With no face looking at the kiosk for 3 seconds, nobody is in front of it.

Up to three faces looking at the kiosk are recognized at once, each followed from look to look by where it is. The person at the kiosk is the one with the highest score. Someone else takes over only after they score higher two checks in a row, or when the person at the kiosk leaves, so two people side by side don't make it flip. The kiosk's **Face** sensor names the person at the kiosk and its **People in view** sensor everyone recognized. The kiosk's page and the vision sessions show the others in view too.

## Color

**Color on the kiosk** is the person's color: the name before a command on the voice overlay, the listening arc on a device with a light ring and the person's marks in the Ground Control app.

## Home Assistant and Personal Commands

Each person can be linked to things in Home Assistant. The fields suggest matching entities, and Music Assistant's playlists, as you type.

| Field | What it does |
| --- | --- |
| To-do list | "Add milk to my list" and "what's on my list" use it when this person says them. |
| Calendar | "What's on my calendar" reads it. |
| Playlist | "Play my playlist" plays it in Music Assistant. |
| Person | This person in Home Assistant. The People page shows their Home Assistant picture, and speaker events carry it for automations. |

The kiosks get these with the voices, so "my" is known on the kiosk with no extra step. Anyone without a list of their own uses the shared list. Kiosk Satellite's blueprints add the sentences to Home Assistant's own conversation agent and to LLM agents. See [Personal Commands](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/voice-match.md#personal-commands) in the Kiosk Satellite docs.

## Renaming and Forgetting

**Rename** changes the name everywhere, on the kiosks too.

**Forget** deletes the person, their voice, their face and all their enrollment audio and face pictures. Kiosks that answered only chosen people stop answering them. A kiosk whose chosen people are all gone answers nobody until people are chosen again, and its page on the Kiosks page says so.
