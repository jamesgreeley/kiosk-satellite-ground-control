# Changelog

All notable changes to Kiosk Satellite Ground Control are documented here.

## Unreleased

### Added
- **Spectrograms of a session's audio.** On a session's page, each audio row has a spectrogram under it: the wake word, what the kiosk heard and what went to the assistant. They share one brightness scale, so the voices left out or removed show up dark, and in follow mode what went to the assistant lines up under what was heard. A line follows playback. The browser draws them from the session's audio, so nothing more is kept on the server.
- **Update notice for Docker installs.** A standalone container checks GitHub for a new Ground Control version a minute after it starts and every 12 hours after that. When there is one, a card at the top of Overview shows it, with links to the changelog and the repository, and the log says so too. Set `UPDATE_CHECK=false` to turn it off. The Home Assistant app doesn't check, since Home Assistant offers its updates.

### Changed
- **README benchmarks in two tables.** One for the CPU and one for NVIDIA GPUs, with the speaker model as a row. The CPU table has the Intel N100 with the Light model too, its default. The GPU table has an estimate for the RTX 3050, and the RTX 4000's times are measured again on the CUDA 12 image: 0.11 s for Isolation and 0.12 s for every feature together. The installation docs had Isolation on an N100 at about 3 seconds instead of the measured 6.

## v2026.11.6 - 2026-11-21

### Added
- **Macs with Apple silicon.** Ground Control runs on a Mac with an M-series chip in Docker Desktop or OrbStack, with the CPU image. On a MacBook Pro M5, every feature together adds about 0.8 s to a 5 s command, faster than an i9-13900H. Ground Control picks the Accurate speaker model on an Apple chip by itself, and Light stays the default on other ARM CPUs. The installation docs have a Mac section and the README's benchmarks have an M5 column. Docker on a Mac can't use the GPU and its mDNS doesn't reach the network, so kiosks need the Mac's address entered.

## v2026.11.5 - 2026-11-21

### Changed
- **GPU memory in the docs.** The CUDA image needs 3 GB of free GPU memory, or 4 GB with Isolation, instead of a 4 GB GPU for everything. The installation page says how much Ground Control keeps on the GPU and how much more a command needs, for a GPU shared with other apps.
- **README CUDA badge.** It shows CUDA 12 and 13, since the amd64 CUDA image runs on CUDA 12 and the arm64 one on CUDA 13.

### Fixed
- **Isolation and the speaker model holding on to GPU memory (#11).** On a GPU, Isolation kept the memory its longest command needed, about 1 GB after a 3 s command and 3 GB after a 10 s one. The speaker model did the same for audio longer than 4 s, up to 1.6 GB more. A GPU shared with an LLM or speech-to-text had less room for them. Both now hand that memory back after each run. Ground Control keeps about 1.3 GB on the GPU, or 1.5 GB with Isolation, and a command takes 2 to 35 ms more.
- **Underscores missing in entity ids on the People page (#10).** With some fonts and zoom levels, Chrome hid the underscores in a person's to-do list, calendar and person fields, so `todo.tristyn_s_shopping_list` looked like `todo.tristyn s shopping list`. The value was always saved with its underscores. Text fields now show them, at the same size as before.

## v2026.11.4 - 2026-11-21

### Fixed
- **"Failed to load libnvrtc.so.12" in the CUDA image's log (#6).** On amd64, each GPU session printed "Failed to load libnvrtc.so.12", "Failed to load libcufft.so.11" and "Please follow https://onnxruntime.ai/docs/install/#cuda-and-cudnn to install CUDA." at startup. It was harmless: the image gets CUDA from its base image, Ground Control doesn't use those two libraries and the models ran on the GPU anyway. The check that printed it now runs once and only writes to the detailed log.

## v2026.11.3 - 2026-11-21

### Fixed
- **Home Assistant Cloud speech-to-text failing after a pause (#4).** With Trimming, nothing went to Home Assistant until the user's speech was accepted, so a pause after the wake word, or speech that took a moment to be accepted, ended the command with "400 Invalid HTTP request" after about 6 seconds with Home Assistant Cloud's speech-to-text. Ground Control now sends the kiosk a keepalive every half second until then, and the kiosk sends Home Assistant a little silence for each. It isn't speech, so the kiosk still cancels a command nobody talks in and still falls back to its microphone if Ground Control goes away. This needs the next Kiosk Satellite release. Older versions ignore it.

## v2026.11.2 - 2026-11-21

### Added
- **Documentation.** New pages in `docs/`: getting started, installation, kiosks, people, Voice Match, backup and moving to another machine, sessions, settings, and logs and issues. The README's Install section has its own parts for the Home Assistant app, with a button that adds the repository to Home Assistant, and for a standalone Docker container. The app's Info and Documentation tabs link to the docs. The README has badges for stars, downloads, the version and the build, and for the Home Assistant app, Docker and CUDA.

### Fixed
- **CUDA image on a GTX 10 series GPU (#3).** On a GTX 1080, the CUDA image stopped at startup with "CUBLAS failure 8: the function requires an architectural feature absent from the device", because CUDA 13 doesn't support GPUs older than Turing. The amd64 CUDA image now runs on CUDA 12, which supports every NVIDIA GPU from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell), and needs an NVIDIA driver from the 570 series or newer instead of 580. The arm64 image stays on CUDA 13 for the GB10.

### Changed
- **A CUDA image runs on the GPU or not at all.** When the GPU can't be used, the CUDA image stops with one line in the log that says why, such as no GPU visible to the container, a driver that is too old or an unsupported GPU, instead of a stack trace or quietly running on the CPU.
- **No waits for the GPU between checks.** On a GPU, each speaker model keeps a session for every window length up to 4 s, so a check never waits for the GPU to set up a new length, which took up to 0.2 s with CUDA 12. A speaker model now takes about 1.3 GB of GPU memory.

## v2026.11.1 - 2026-11-20

The first release of Kiosk Satellite Ground Control, a companion service for Kiosk Satellite that runs the voice and vision work too heavy for a kiosk on a computer or server in your home.

### Added
- **Voice Match: Recognition, Trimming and Isolation.** Who is speaking, and only their voice in the command.
- **Recognition.** Each voice command is checked against the people enrolled, and the kiosk knows who is speaking about a second in, while the person is still talking. A kiosk can answer anyone or only the people you choose. Each kiosk gets a Speaker sensor in Home Assistant, and an LLM conversation agent can tell who is speaking from the instructions on the Settings page, so "my calendar" or "my list" means the speaker's.
- **Trimming.** The kiosk sends everything its microphone hears and gets back only the speech of the person who said the wake word. A TV or another person before, between and after their words is cut out, long pauses are shortened and the command ends once the user stops talking, whatever the TV does. A follow-up in a conversation carries on with the same person, so a one word answer like "yes" goes through. Once the user is done, a voice has to sound clearly like them to start another turn, so a TV that briefly sounds a little like them can't interrupt a realtime answer.
- **Isolation.** Other voices talking over the user's words are removed too, using the user's voiceprint, once the user stops talking. It needs the Accurate speaker model and is off until turned on under **Settings > Voice Match**, and each kiosk can turn it off in Kiosk Satellite. While a turn is held for it, a little silence keeps the audio stream open, so speech-to-text that gives up on a quiet stream, like Home Assistant Cloud's, still gets the command. A 5 s command takes about 0.1 s on an NVIDIA RTX 4000 SFF Ada and about 1.2 s on four cores of an i9-13900H. On a CPU it runs a version of the model made for the CPU, twice as fast and nearly identical.
- **Two speaker models.** Accurate (ReDimNet2 B3, multilingual) is the default on an NVIDIA GPU and on x86 CPUs with more than 4 cores. Light (ReDimNet2 B1) is the default on ARM and on smaller x86 CPUs such as the Intel N100. Changing the model rebuilds everyone's voiceprint from the kept enrollment audio in the background, so nobody records again.
- **People.** Add, rename and forget people, and give each a ring color, a to-do list, calendar, playlist and Home Assistant person, which Kiosk Satellite uses when that person says "my". The fields suggest Home Assistant's entities and Music Assistant's playlists as they are typed in. A person linked to a Home Assistant person shows that person's picture.
- **Guided enrollment on a kiosk.** Start an enrollment from the app and the person reads aloud to a kiosk of the kind they talk to. The kiosk and the app show the same text, filling as it is spoken, so the person can read from a phone while walking around the room. Speech that doesn't sound like the person, like a TV, is left out of the voiceprint. Only kiosks with Allow enrollment here on are offered.
- **Kiosks.** Kiosks on the same network find Ground Control on their own, at the machine's address on its main network, and sign in with the Home Assistant token they already have. For kiosks that can't reach that address, like kiosks on another VLAN, **mdns_address** in the app's options or `MDNS_ADDRESS` in Docker sets the address they get. A new kiosk waits for approval on the Kiosks page. If a kiosk answers chosen people Ground Control doesn't have, as after moving to another Ground Control, its page and the log say so, and a rejected command says the voice isn't one of the people that kiosk answers. Each kiosk's page shows its connection, its Kiosk Satellite version, the Voice Match features it has on, who is talking right now with each check as it happens, and its latest sessions.
- **Saved sessions.** Each session keeps its events, every score, why it ended, its audio as heard and as sent and the wake word, for 7 days by default. The session page shows the waveform colored by who was talking and every step (Isolation included), and a session's voice can be added to a person. **Download** saves a session as one zip for an issue report: its audio as heard, as sent and the wake word, its record, and Ground Control's details with the log from while it ran. Keeping the audio can be turned off.
- **The web app.** A Ground Control panel in Home Assistant's sidebar as a Home Assistant app, or its own page with **Log in with Home Assistant** as a container. Pages for Overview, People, Kiosks, Sessions, Settings and Logs. **Logs** shows what Ground Control has been doing and downloads a report for an issue, with tokens left out.
- **Home Assistant app and Docker images.** One CPU image for the app and standalone Docker (amd64 and arm64), and a CUDA 13 image for NVIDIA GPUs, which needs an NVIDIA driver for CUDA 13 (580 or newer). `compose.yaml` and `compose.cuda.yaml` run them with Docker Compose. As an app, people, voiceprints and sessions are in the app's own folder, which the Samba share and File editor apps show, and a Home Assistant backup of the app keeps them. The app's Info page shows the full Kiosk Satellite Ground Control logo, its options have names and descriptions on its Configuration tab, and Home Assistant's watchdog can restart it if it stops answering. The models are in the repository, each with its license next to it, and the service checks each one against its pinned hash when it loads.
- **License.** Kiosk Satellite Ground Control is under CC BY-NC-ND 4.0, the same license as Kiosk Satellite: free for personal, non-commercial use. The ReDimNet2 and Silero VAD models keep their own MIT licenses.
