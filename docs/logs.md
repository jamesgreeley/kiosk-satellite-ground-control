# Logs and Issues

Navigate to **Logs** in the Ground Control app.

The Logs page shows what Ground Control has done since it started: kiosks connecting, each session with who talked and why it ended, enrollments, setting changes and errors. New lines show up as they happen.

## Reading the Log

- **Level** shows **Info** by default. **Warnings** and **Errors** show only problems. **All** adds the detailed lines.
- **Search the log** narrows it to lines with the text you type.
- **Copy** copies the lines shown.

Useful lines to look for:

| Line | Meaning |
| --- | --- |
| `kiosk … connected from …` | A kiosk signed in. The line includes its model, Kiosk Satellite version and whether it allows enrollment. |
| `session … ended: …` | One voice command: who talked and their score, why it ended, how much audio and how long it took. |
| `rejected …, not one of the people this kiosk answers` | The voice was recognized but the kiosk is set to answer other people. |
| `answers … Ground Control doesn't have` | The kiosk's chosen people aren't here. See [Chosen People Ground Control Doesn't Have](kiosks.md#chosen-people-ground-control-doesnt-have). |
| `Isolation runs on the CPU` | Isolation adds about a second to each command. |
| `CUDA can't start: …` | The CUDA image can't use the GPU, so it stopped. The line says why and what the image needs. See [NVIDIA GPU](installation.md#nvidia-gpu). |

If you run the Home Assistant app, the same log is also on the app's **Log** tab. With Docker, use `docker logs`.

## Reporting an Issue

1. If you can make the problem happen again, turn on **Detailed logs** first and repeat it. Detailed logs stay on until Ground Control restarts.
2. Choose **Download** on the Logs page. The file holds the log since Ground Control started. The version, the speaker model, the settings and every kiosk are listed at the top. Tokens are never written to it.
3. For a problem with one voice command, also download that session from its page on the [Sessions](sessions.md#downloading-a-session) page. The zip has its audio, its record and the log from while it ran.
4. Attach the files to an issue at [github.com/jxlarrea/kiosk-satellite-ground-control/issues](https://github.com/jxlarrea/kiosk-satellite-ground-control/issues) and include the Kiosk Satellite version and the kind of device.
