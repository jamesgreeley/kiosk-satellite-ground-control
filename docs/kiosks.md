# Kiosks

Navigate to **Kiosks** in the Ground Control app.

Every kiosk running Kiosk Satellite that has connected to Ground Control is listed here, with its connection, its Kiosk Satellite version and what it is doing right now. Only Kiosk Satellite can connect, and each kiosk needs your approval before it can use Ground Control.

## Connecting a Kiosk

1. In Kiosk Satellite, go to **Settings > Ground Control** and turn on **Enable Ground Control**.
2. Kiosks on the same network find Ground Control on their own. The kiosk signs in with the Home Assistant token it already has, so there is nothing else to enter.
3. The kiosk shows up on the Kiosks page with **Waiting for approval**. Open it and choose **Approve**.

Until it is approved, a kiosk gets no names or voices and can't use Voice Match. Its status row in Kiosk Satellite shows **Waiting for approval**. **Decline** keeps it out, and its status row shows **Declined**.

**Remove access** on an approved kiosk takes it back to waiting for approval. It stops using Voice Match until you approve it again.

See [Ground Control](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/ground-control.md) in the Kiosk Satellite docs for the kiosk's side and every status.

## Kiosks on Another Network

Kiosks find Ground Control over mDNS, which stays within one network unless your router relays it. For a kiosk on another network, enter Ground Control's address under **Address** in Kiosk Satellite's Ground Control settings, such as `192.168.1.10:10900`. The address to enter is under **Settings > General > Kiosks** in the Ground Control app, with a copy button.

Ground Control advertises its address on the machine's main network, the one with the default route. A machine with several networks, such as a VLAN for IoT devices, gives kiosks that one address. If the kiosks can't reach it, set the address they get:

- **Home Assistant app:** the **mDNS address** option on the app's Configuration tab, under the optional settings.
- **Docker:** the `MDNS_ADDRESS` environment variable.

Every kiosk that finds Ground Control on its own then gets that address. A kiosk with the address entered by hand doesn't use mDNS at all.

## A Kiosk's Page

| Section | What it shows |
| --- | --- |
| Voice Match | The Voice Match features the kiosk has on, as the kiosk last said: Recognition with who it answers, Trimming, Isolation, the Speaker sensor, Show who is talking and Allow enrollment here. Each is set on the kiosk, in Kiosk Satellite under **Settings > Ground Control > Voice Match**. |
| Vision Match | The Vision Match features the kiosk has on, as the kiosk last said: Vision Match, whether Face Detection and Hand Gestures run on Ground Control or on the kiosk, Dismiss on face, Postpone on face and how many finger gestures are set up. Each is set on the kiosk, in Kiosk Satellite under **Settings > Ground Control > Vision Match**, **Settings > Screensaver > Face Detection** and **Settings > Gestures**. |
| Who is talking | The voice scores of the current command over the last 10 seconds, against the recognition and other voice thresholds, and the person recognized. |
| What it sees | Who Face Recognition sees, and each look of the current vision session over the last 10 seconds: a face looking at the kiosk or turned away, the hand and its finger count and Ground Control's time. |
| Looks | Each of the last 7 days: the kiosk's looks, the empty looks that found no face or hand and the vision sessions. |
| Events | What happened on the kiosk as it happens: a session starting, someone recognized or rejected, another voice, the end of speech, enrollments, vision sessions starting and ending, the screensaver dismissed and each finger gesture. |
| Recent sessions | The kiosk's latest sessions. Each opens on the [Sessions](sessions.md) page. |

### Chosen People Ground Control Doesn't Have

A kiosk set to answer chosen people sends the people it chose. When some of them aren't in this Ground Control, such as after moving to another Ground Control or forgetting a person, every command from them is rejected. The kiosk's page shows **Choose again** on the Recognition row, the Overview lists them as not on this Ground Control and the log names the kiosk. In Kiosk Satellite, the **People** setting names them in red.

To fix it, choose the people again in Kiosk Satellite, under **Settings > Ground Control > Voice Match > People**.
