# Backup and Moving

Everything Ground Control keeps is in one data folder. Back it up, or copy it to another installation to move Ground Control to a new machine without enrolling anyone again.

## What Is in the Data Folder

| Item | What it holds |
| --- | --- |
| `people.json` | Everyone on the People page, with their color and Home Assistant links. |
| `voices/` | Each person's enrollment audio, one folder per person. |
| `profiles/` | The voiceprints, one file per speaker model. |
| `kiosks.json` | Every kiosk and whether it is approved. |
| `settings.json` | The settings changed in the app, such as the speaker model, the thresholds and Isolation. |
| `sessions/` | Saved sessions and their audio. Optional to keep: they expire after a few days anyway. |

Where the folder is:

| Installation | Data folder |
| --- | --- |
| Home Assistant app | `addon_configs/…_kiosk_satellite_ground_control` in the Samba share and File editor apps, `/addon_configs/…_kiosk_satellite_ground_control` from the SSH app. The start of the name depends on where the app was installed from. |
| Docker | The volume mounted at `/data`. With `docker run` as in the README it is `ground-control`. Docker Compose puts the project name in front, such as `kiosk-satellite-ground-control_ground-control`: `docker volume ls` lists it. Use that name in the commands below. |

## Backing Up

### Home Assistant App

Home Assistant's own backups include the app's data folder:

1. Go to **Settings > System > Backups**.
2. Back up now, with **Kiosk Satellite Ground Control** among the apps included. Automatic backups that include all apps cover it too.

To keep a copy outside Home Assistant, copy the app's folder from the Samba share instead.

### Docker

Stop the container, copy the volume to a file and start it again:

```bash
docker stop ground-control
docker run --rm -v ground-control:/data -v "$PWD":/backup alpine \
  tar czf /backup/ground-control.tgz -C /data .
docker start ground-control
```

With a folder mounted at `/data` instead of a volume, stop the container and copy that folder.

## Moving to Another Installation

The steps are the same whatever the old and new installations are:

1. **Install Ground Control on the new machine** and start it once, so its data folder exists. Use the same version as the old one, or newer.
2. **Stop both.** Ground Control writes to its data folder while it runs, so copy it only while stopped.
3. **Copy the old data folder's contents into the new one,** replacing what is there. Leave out `sessions/` if you don't need the old sessions.
4. **Start the new installation.** Keep the old one stopped, or uninstall it, so kiosks don't find two.
5. **Point the kiosks at the new machine.** Kiosks that find Ground Control on their own switch to it by themselves. A kiosk with the address entered under **Settings > Ground Control** in Kiosk Satellite needs the new address.

Everyone keeps their voice, and the kiosks stay approved and keep the people they answer. People keep the same IDs, so a kiosk set to answer chosen people doesn't need them chosen again.

If the new machine runs another speaker model, such as Accurate on a GPU after Light on a Raspberry Pi, the voiceprints for it are rebuilt from the enrollment audio when Ground Control starts. Nobody records again. A speaker model or setting changed in the app moves with `settings.json`, so check **Settings > Voice Match** on the new machine.

### From a Home Assistant App

Copy the app's folder from the Samba share, or from the SSH app, while the app is stopped:

- **To an app on another Home Assistant:** restore a backup of the app there under **Settings > System > Backups**, or copy the folder into the new app's folder.
- **To Docker:** copy the folder's contents into the new container's data folder. For the `ground-control` volume:

    ```bash
    docker stop ground-control
    docker run --rm -v ground-control:/data -v /path/to/the/copy:/from alpine \
      cp -a /from/. /data/
    docker start ground-control
    ```

### From Docker

Make a backup file as in [Backing Up](#docker), then:

- **To another Docker machine:** copy `ground-control.tgz` there and restore it into the volume while the container is stopped:

    ```bash
    docker stop ground-control
    docker run --rm -v ground-control:/data -v "$PWD":/backup alpine \
      tar xzf /backup/ground-control.tgz -C /data
    docker start ground-control
    ```

- **To a Home Assistant app:** stop the app, unpack `ground-control.tgz` and copy its contents into the app's folder with the Samba share, replacing what is there. Then start the app.

> **Info:** The copy replaces anyone enrolled on the new installation before the move. If a kiosk chose people on the new installation in the meantime, Kiosk Satellite names them in red under **People**. Choose them again there.
