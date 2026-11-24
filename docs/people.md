# People

Navigate to **People** in the Ground Control app.

Everyone Voice Match can recognize is on this page, with their voice, their color and the things "my" means for them. The list on the left shows each person and how many kinds of device they are enrolled on.

## Adding a Person

1. Choose **Add a person** and enter their name. This is how the kiosks and the assistant call them.
2. Enroll their voice on each kind of device they talk to. See [Enrollment](#enrollment).

A person without a voice is never recognized, but can still be picked on a kiosk under **Answer > Chosen people**.

## Enrollment

A voice is recognized best on the kinds of device it was enrolled on, since an Echo Show and an Echo Dot hear the same voice differently. Enrollment always records with a kiosk's own microphone, so enroll each person once on every kind of device they talk to.

1. On the person's page, the **Devices** section lists every kind of device in the house and whether they are enrolled on it. Choose **Enroll** next to one, or **Enroll on a kiosk** to pick the kiosk.
2. The kiosk shows the text to read, full screen in the kiosk's language. The Ground Control app shows the same text, so it can also be read from a phone.
3. Read it aloud for about 45 seconds:
    - Talk normally, in the voice you use for commands, not a reading voice.
    - Start close to the kiosk, then move across the room.
    - Face away from the kiosk now and then.

The sentence to read fills as it is spoken and moves on when it is full or after a pause. When another voice talks, such as a TV, that part is left out and a banner says so. The enrollment ends once 45 seconds of speech are counted, and **Enroll next** offers the next kind of device the person isn't enrolled on.

What was recorded is kept when the enrollment is stopped early with at least 10 seconds of speech. The wake word doesn't listen during an enrollment, so reading the text never starts a voice command.

Only kiosks with **Allow enrollment here** on, under **Settings > Ground Control > Voice Match** in Kiosk Satellite, are offered. A kiosk busy with a voice command or an alarm can't start an enrollment until it is done.

### Enrollments

The **Voice Match** section lists each enrollment with the kiosk it was recorded on and how much speech it has. Deleting one rebuilds the person's voice without it.

### Adding a Voice from a Session

**Add from a session** opens the Sessions page. On a session where the person talked, **Use this voice** adds their speech from it to their voice. Only the user's speech goes in, without the voices left out. This is useful for a kind of device nobody enrolled on yet, or for a voice that has changed. See [Sessions](sessions.md#use-this-voice).

### Learning

With **Learn from commands** on, under **Settings > Voice Match**, commands where a person was recognized with high confidence are folded into their voice. Their voiceprint follows a cold or a new room without enrolling again.

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

**Forget** deletes the person, their voice and all their enrollment audio. Kiosks that answered only chosen people stop answering them. A kiosk whose chosen people are all gone answers nobody until people are chosen again, and its page on the Kiosks page says so.
