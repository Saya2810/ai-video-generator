# AI Video Studio

<video src="examples/fox-wan-raw-832x480.mp4" controls muted width="100%"></video>

*Raw output of the motion step, 832×480, unedited.*

Turn a prompt or a script into a narrated, captioned video. Runs locally, with open models (Apple Silicon Mac).
No cloud, no API keys, no account.

- Narration: Kokoro (English) or Chatterbox (23 other languages)
- Script and storyboard: Qwen3-4B
- Images: FLUX.2 klein 4B
- Motion: Wan 2.2 (or a 2.5D camera for quick previews)
- Captions and sync check: Whisper, ffmpeg

## Install

```bash
git clone https://github.com/Saya2810/ai-video-studio.git
cd ai-video-studio
bash scripts/setup.sh
```

- The setup installs ffmpeg and uv, and downloads the core models into `models/`.
- If it stops, run it again. Finished downloads are kept.
- Check it with `./bin/aivs --help`.

## Use

Web studio:

```bash
./bin/aivs ui
```

Then open localhost in your browser.

Command line:

```bash
./bin/aivs make --prompt "Why octopuses are the most alien intelligence on Earth" --seconds 45
./bin/aivs make --script my_script.txt --lang de --aspect 9:16 --music wonder
./bin/aivs make --storyboard examples/night-migration.json
```

- Output: `projects/<name>/final.mp4`, plus `captions.srt` and `report.json`.
- Edit `projects/<name>/project.json`, then run `./bin/aivs render <name>`. Only changed scenes are recomputed.
- `./bin/aivs verify <name>` checks audio and caption sync.
- For a quick first test, add `--draft --motion parallax`.

### Main options

| Option | What it does |
|---|---|
| `--lang` | `en` (Kokoro) or one of 23 languages such as `de`, `fr`, `es` (Chatterbox) |
| `--aspect` | `16:9`, `9:16`, `1:1` |
| `--voice` | Kokoro voice, e.g. `af_heart`, `am_michael` |
| `--music` | `calm`, `wonder`, `uplifting`, `dark`, `none`, or your own file |
| `--captions` | `burn`, `srt`, `off` |
| `--motion` | `wan` (default), `parallax` (fast 2.5D camera), `camera` (flat zoom) |
| `--draft` | 720p preview, about twice as fast |
| `--style` | one style line for all scenes, e.g. `"watercolor, soft pastel"` |
| `--characters N` | dialogue between N speaking characters |
| `--person NAME=IMAGE` | a person who must look like the photo (repeatable) |
| `--object` / `--background` | same idea for objects and places |
| `--plan-only` | stop after the storyboard so you can edit it first |

Run `./bin/aivs make --help` for the full list.

## Optional extras

Each one is a separate script. Run only what you need.

| Feature | Command | Download |
|---|---|---|
| Animated scenes (`--motion wan`) | `bash scripts/setup_wan.sh` | ≈20 GB |
| Lip-sync for speaking characters | `bash scripts/setup_lipsync.sh` | ≈6–9 GB |
| Talking characters (InfiniteTalk) | `bash scripts/setup_talk.sh` | ≈30 GB, needs the two above |

Without them the studio still works and uses the 2.5D camera.

## Good to know

- Wan and lip-sync are slow: several minutes per second of video. Use `--draft` and `--motion parallax` to iterate.
- Faces are recreated by the image model. Likeness is good, not perfect.
- Music is a simple generated pad, not a score. Bring your own track for more.
- Projects go to `projects/` by default. Set `AIVS_PROJECTS=/path` to use another folder.

## Tests

```bash
envs/studio/bin/python tests/test_new_features.py
envs/studio/bin/python tests/test_dialogue.py
```

They use fakes for the models, so they need no downloads.

## Licence

Code: MIT, see [LICENSE](LICENSE), by https://github.com/Saya2810. Models and tools: see [THIRD_PARTY.md](THIRD_PARTY.md).
