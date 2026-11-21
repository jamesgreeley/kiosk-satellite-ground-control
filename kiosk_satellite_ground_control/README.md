# Kiosk Satellite Ground Control

Ground Control is a companion service for [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite). It runs on a computer or server in your home and handles the voice and vision processing that kiosks don't have the computing power for.

Kiosk Satellite Ground Control currently has two components: **Voice Match** which does speaker voice recognition, trimming and isolation. **Vision Match** which does facial recognition (coming soon).

**Voice Match** has three features:

- **Recognition:** who is speaking to a kiosk, from the voiceprints of the people enrolled.
- **Trimming:** a TV or another person before, between and after the user's words is cut out of the command.
- **Isolation:** other voices talking over the user's words are removed too. Best on an NVIDIA GPU.

Open **Ground Control** in the sidebar to enroll people and approve kiosks. The **Documentation** tab covers the options, and the [Ground Control docs](https://github.com/jxlarrea/kiosk-satellite-ground-control/blob/main/docs/getting-started.md) cover everything else.
