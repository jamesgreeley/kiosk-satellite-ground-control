# Backup and Moving

Ground Control keeps all of its data in one folder. Back it up or copy it to another installation to move Ground Control to a new machine without enrolling anyone again.

## What Is in the Data Folder

| Item | What it holds |
| --- | --- |
| `people.json` | Everyone on the People page. It includes each person's color and Home Assistant links. |
| `voices/` | Each person's enrollment audio. Each person has their own folder. |
| `profiles/` | The voiceprints. There is one file per speaker model. |
| `kiosks.json` | Every kiosk and whether it is approved. |
| `settings.json` | The settings changed in the app, such as the speaker model, the thresholds and Isolation. |
| `sessions.db` | Saved sessions and each kiosk's look counts. You don't need to keep it because sessions expire after a few days anyway. |
| `sessions/` | The audio and frames of saved sessions. Like `sessions.db`, you don't need to keep it. |

The folder's location depends on the installation:

| Installation | Data folder |
| --- | --- |
| Home Assistant app | It is `addon_configs/…_kiosk_satellite_ground_control` in the Samba share and File editor apps and `/addon_configs/…_kiosk_satellite_ground_control` from the SSH app. The start of the name depends on where the app was installed from. |
| Docker | It is the volume mounted at `/data`. If you used `docker run` as in the README, it is named `ground-control`. Docker Compose puts the project name in front, such as `kiosk-satellite-ground-control_ground-control`. Run `docker volume ls` to find it. Use that name in the commands below. |

## Backing Up

### Home Assistant App

Home Assistant's own backups include the app's data folder:

1. Go to **Settings > System > Backups**.
2. Back up now and include **Kiosk Satellite Ground Control** among the apps. Automatic backups that include all apps cover it too.

To keep a copy outside Home Assistant, copy the app's folder from the Samba share instead.

### Docker

Stop the container, copy the volume to a file and start it again:

```bash
docker stop ground-control
docker run --rm -v ground-control:/data -v "$PWD":/backup alpine \
  tar czf /backup/ground-control.tgz -C /data .
docker start ground-control
```

If you mounted a folder at `/data` instead of a volume, stop the container and copy that folder.

## Moving to Another Installation

The steps are the same for any old and new installation:

1. **Install Ground Control on the new machine** and start it once to create its data folder. Use the same version as the old one or newer.
2. **Stop both.** Ground Control writes to its data folder while it runs, so copy it only while stopped.
3. **Copy the old data folder's contents into the new one.** Replace what is there. Leave out `sessions.db` and `sessions/` if you don't need the old sessions.
4. **Start the new installation.** Keep the old one stopped or uninstall it so only one installation is discovered on the network.
5. **Point the kiosks at the new machine.** Kiosks that connect through automatic discovery need no changes. If you entered the address in a kiosk under **Settings > Ground Control** in Kiosk Satellite, enter the new address there.

Everyone's voice stays enrolled. The kiosks stay approved and keep the people they answer. People keep the same IDs, so you don't need to choose them again on a kiosk set to answer chosen people.

The new machine might use another speaker model, such as Accurate on a GPU after Light on a Raspberry Pi. In that case the voiceprints for it are rebuilt from the enrollment audio when Ground Control starts. Nobody needs to record again. A speaker model or setting you changed in the app moves with `settings.json`, so check **Settings > Voice Match** on the new machine.

### From a Home Assistant App

While the app is stopped, copy the app's folder from the Samba share or the SSH app:

- **To an app on another Home Assistant:** restore a backup of the app there under **Settings > System > Backups** or copy the folder into the new app's folder.
- **To Docker:** copy the folder's contents into the new container's data folder. For the `ground-control` volume, run:

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

- **To a Home Assistant app:** stop the app, unpack `ground-control.tgz` and copy its contents into the app's folder with the Samba share. Replace what is there, then start the app.

> **Info:** The copy replaces anyone enrolled on the new installation before the move. If you chose people for a kiosk on the new installation in the meantime, they show in red under **People** in Kiosk Satellite. Choose them again there.
