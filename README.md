# AI Video Studio

<video src="examples/fox-wan-raw-832x480.mp4" controls muted width="100%"></video>

*Raw output of the Wan 2.2 motion step (832×480, one shot, not upscaled or edited). [Watch the file](examples/fox-wan-raw-832x480.mp4).*

Turn a prompt or a script into a finished, narrated, captioned 1080p video — entirely on an Apple Silicon Mac,
with free, commercially usable open models. No cloud, no API keys, no accounts.

```
idea or script ─► script + storyboard ─► narration ─► speech check + word timing ─► keyframes ─► motion ─► edit
  (+ your photos)  Qwen3-4B              Kokoro /       Whisper large-v3-turbo        FLUX.2       Wan 2.2   ffmpeg
                                         Chatterbox                                   klein 4B     TI2V-5B
                                                                                      (+ photos)   (or 2.5D camera)
```

Built and verified on a MacBook Pro M5 with 16 GB. Only one model is in memory at a time.

## Setup (once)

Requires macOS on Apple Silicon and [Homebrew](https://brew.sh).

```bash
bash scripts/setup.sh
```

This creates `envs/studio`, installs the package, and downloads about 12 GB of models into `models/`
(resumable — run it again if it is interrupted).

## Make a video

Web studio:

```bash
./bin/aivs ui
```

then open http://127.0.0.1:8760 — type an idea or paste a script, press **Create video**, edit the storyboard, re-render.

Command line:

```bash
./bin/aivs make --prompt "Why octopuses are the most alien intelligence on Earth" --seconds 45
```

```bash
./bin/aivs make --script my_script.txt --lang de --aspect 9:16 --music wonder
```

```bash
./bin/aivs make --storyboard examples/night-migration.json
```

The result is `projects/<name>/final.mp4`, with `captions.srt` and `report.json` (timings, loudness, per-line
speech accuracy) next to it.

Useful options (`./bin/aivs make --help` lists all):

| Option | Meaning |
|---|---|
| `--lang` | narration language: `en` uses Kokoro; `de fr es it pt nl pl tr sv da fi no ru ja ko zh hi ar el he ms sw` use Chatterbox Multilingual |
| `--aspect` | `16:9` (1920×1080), `9:16` (1080×1920), `1:1` (1080×1080) |
| `--voice` | Kokoro voice, e.g. `af_heart`, `am_michael`, `bf_emma`, `bm_george` |
| `--ref-audio` | 5–15 s sample of a voice you have the rights to; narration is spoken in that voice (Chatterbox) |
| `--music` | `calm`, `wonder`, `uplifting`, `dark`, `none`, or a path to your own track (ducked under the voice automatically) |
| `--captions` | `burn` (in picture + .srt), `srt`, `off` |
| `--motion` | `wan` (default: every scene is animated with Wan 2.2, ~10 min per scene — see below), `parallax` (depth-aware 2.5D camera over stills, fast), `camera` (flat zoom/pan) |
| `--wan-shots` | with `--motion wan`: only these shots are generated (e.g. `s02,s05`); the rest use parallax |
| `--person NAME=IMAGE` | photo of a person who must appear; use NAME in your script/idea (repeatable). Only photos of people who agreed |
| `--object NAME=IMAGE` | photo of an object/product that must look exactly like this (repeatable) |
| `--background NAME=IMAGE` | photo of a place; a scene set only there uses the photo itself as its picture (repeatable) |
| `--characters N` | the video becomes a dialogue between N speaking characters; those not given with `--person`/`--cast` are invented (name, look, voice) |
| `--narrator` | with `--characters`: an off-screen narrator may speak as well |
| `--cast FILE.json` | define characters: `[{"name":"Anna","gender":"female","age":"30s","look":"…","voice":"warm low voice","photo":"anna.jpg","voice_sample":"anna.wav"}]` |
| `--lipsync auto\|off` | mouths of characters speaking on screen follow their lines (needs `scripts/setup_lipsync.sh`; default auto) |
| `--references FILE.json` | same as above with descriptions: `[{"kind":"person","name":"Anna","description":"…","path":"anna.jpg"}]` |
| `--consistency` | `auto` (default: when the storyboard has a recurring subject, shot 1 becomes the look reference for all other keyframes), `on`, `off` |
| `--style` | one style line applied to every shot, e.g. `"watercolor illustration, soft pastel palette"` |
| `--draft` | 720p preview, about twice as fast |
| `--plan-only` | stop after the storyboard so you can edit `projects/<name>/project.json` first |
| `--replan` | write a fresh storyboard for an existing project name |

Re-render after editing a storyboard (only changed shots are recomputed):

```bash
./bin/aivs render night-migration
```

Check audio/caption sync of a finished video independently (re-transcribes the final file):

```bash
./bin/aivs verify night-migration
```

## Examples (made by this code on a MacBook Pro M5, 16 GB)

Sample storyboards with the measurement reports from the runs that made them (the rendered videos are not in the repo, except the demo clip at the top; render them yourself):

| Storyboard | How it was made | Render time |
|---|---|---|
| `examples/night-migration.json` (1920×1080, English) | `aivs make --prompt "The secret nightly migration of deep-sea creatures: ..." --seconds 40` | 6 min 41 s |
| `examples/leuchtturm-de.json` (1080×1920, German) | `aivs make --script examples/leuchtturm.de.txt --lang de --aspect 9:16 --music wonder` | 4 min 34 s |

Measurement reports: `examples/*.report.json`.

## Your own photos (people, objects, places)

```bash
./bin/aivs make --script story.txt --lang de --person "Anna=fotos/anna.jpg" --background "Küche=fotos/kueche.jpg"
```

In the web studio: **Add images** under the idea/script, give each image a name and a type, and use the name in the
text ("Anna betritt die Küche"). Every scene whose text mentions a name gets that photo; the planner also assigns
photos where they fit. In the project view you can tick photos per scene, add or remove photos and re-render.

How a scene is drawn: people/objects/places go to FLUX.2-klein as numbered reference images (people first, max 3 per
scene); a scene that only shows one place uses your photo itself as its picture. Then Wan animates that picture.
Uploads are converted to PNG (≤1536 px) and **all metadata (EXIF, GPS) is removed**; they stay in
`projects/<name>/refs/`. Person photos need the confirmation that the person agreed, and **with uploaded photos of
real people no sexual or nude scenes are generated** (checked at creation and before every render; invented
characters are not affected). Faces are recreated by the image model, so likeness is good but not perfect; it drifts
more the further the camera moves away from the original photo.

## Dialogues and characters

```bash
./bin/aivs make --prompt "Zwei Freundinnen streiten im Café über eine Erbschaft" --lang de --characters 2
./bin/aivs make --script dialog.txt --lang de --person "Anna=fotos/anna.jpg"     # dialog.txt: lines like "Anna: Hallo!"
```

- **How many speakers:** `--characters N` (web: *Speaking characters*). Characters you define (uploaded person
  photos, `--cast`) are used; the rest are invented by the planner with a name, a look and a voice description.
  A script whose lines start with `Name:` is a dialogue automatically; lines without a name, or `Erzähler:` /
  `Narrator:`, are spoken by the narrator.
- **Looks:** invented characters get a generated portrait; it (or your photo) is the reference image in every scene
  the character appears in, so the face stays recognisable from scene to scene.
- **Voices:** English uses a different Kokoro voice per character. Other languages: one voice sample per character is
  designed from its voice description (Qwen3-TTS VoiceDesign) and Chatterbox speaks all of the character's lines in
  it. Your own sample (`voice_sample`) replaces the designed voice. Captions show the speaker's name and colour.
- **Lip-sync (optional):** `bash scripts/setup_lipsync.sh` (once, ~9 GB). Scenes where a character speaks on screen
  are then lip-synced with LatentSync 1.6 after the motion step. Scenes are planned with the speaker facing the
  camera. When several people are in a scene, the speaker's face is recognised by comparing it with the character's
  photo/portrait, so only the speaker's mouth moves. If the speaker's face is not found (e.g. only the listener is
  visible, or the face is turned away), the scene keeps its clip (noted in `report.json → lipsync_notes`).
- In the web editor, **Motion for all scenes** switches an existing project to Wan (projects made before Wan
  became the default are saved with the 2.5D camera); each scene can still override it.
- In the web editor you can change, per scene, who speaks and whether they speak on screen; per character the look
  (→ new portrait) and the voice description (→ new voice), and listen to each voice.

## How quality is protected

- **Narration is never paraphrased.** With `--script` your text is split into shots deterministically; the LLM only adds visuals.
- **Every spoken line is checked.** Whisper transcribes each take; a line whose word error rate exceeds 25 % is re-synthesised (up to 3 takes, best one kept).
- **Sync is by construction.** Shot lengths are derived from the measured narration, lines are placed at exact sample positions, captions use word timings from the actual audio but always show your original text.
- **Broadcast-style audio.** Music is side-chain ducked under the voice; the mix is normalised to −14 LUFS with a two-pass EBU R128 filter.
- **Resumable.** Each artefact is cached by a hash of its inputs; an interrupted or edited project only recomputes what changed.

## Limits you should know about

- Default motion is Wan 2.2 image-to-video at real speed: **about 5 minutes per second of video** on a 16 GB M5
  (measured: one 3.4 s chunk ≈ 9½ min denoising + ~7 min decoding with other apps open; a 16 s video ≈ 1½ h).
  Softer than the 1080p stills (generated at 832×480 and upscaled). `--wan-length stretch` is the old, faster mode
  (one 2 s clip per scene slowed down: slow motion and ghosting). Use `--motion parallax` or `--draft` for quick
  previews. A scene where Wan fails, or all scenes if Wan is not installed, fall back to the 2.5D camera
  automatically (see `motion_fallbacks` in `report.json`).
- Projects created before Wan became the default keep their saved motion (`parallax`); re-render them with
  `--motion wan` to animate them.
- No sound effects. Music is a procedural ambient pad, not a composed score; bring your own track for more.
- Lip-sync only after `scripts/setup_lipsync.sh`; it reworks a 512 px face region (best for medium close-ups facing
  the camera). It is slow on a 16 GB M5: about **6 minutes per second of scene** (measured: a 6.4 s scene took
  38 min, ~3.7 min per 16 video frames), on top of Wan. A 4-line dialogue of ~17 s needs about 1¾ h for lip-sync.
  Faces in profile or turned away may not be found; such scenes keep their clip.
- Dialogue: one scene per spoken line, so dialogues have more scenes than narrated videos (and take longer with Wan).
- A recurring subject is kept consistent by using shot 1 as a reference image; this works well for places and
  objects, less reliably for faces.
- English narration (Kokoro) is the most natural; other languages use Chatterbox, which is slower.

## Generated motion (Wan 2.2 image-to-video, default)

```bash
bash scripts/setup_wan.sh          # once, ~20 GB
```

```bash
./bin/aivs render my-project --motion wan --wan-shots s02   # only some scenes
```

In the web editor each scene has its own motion choice (AI animation / 2.5D camera / flat camera).
This animates the keyframe of the chosen shots with Wan 2.2 TI2V-5B (Apache-2.0), so subjects really move.
Every scene is generated at real speed (24 fps) in chunks of up to 81 frames (3.4 s); each chunk continues from the
last frame of the previous one, and the chunks are joined without motion interpolation. Speaking characters get
"talking, expressive face, head movements, gestures" in the animation prompt. Measured on the M5 / 16 GB: one 81-frame
832×480 chunk ≈ 9½ min denoising + ~7 min VAE decoding (121-frame chunks peaked at 14.4 GB, hence 81). Expect it to look
softer than the 1080p stills (480p upscaled). It needs an extra ~20 GB download and a separate environment
(`envs/wan`); close other heavy apps while it runs.

## Talking characters (InfiniteTalk, optional)

```bash
bash scripts/setup_talk.sh         # once, ~30 GB download, ~20 GB after conversion
```

With this installed, a scene where a character speaks on screen is generated from its keyframe **and the spoken line**
by InfiniteTalk (Apache-2.0, built on Wan2.1-I2V-14B with the lightx2v 4-step distillation): head, face, expressions,
gestures and mouth all follow the audio, instead of Wan motion plus a repainted mouth. It runs on Apple MLX with the
14B model's blocks streamed from disk (8-bit), so it fits 16 GB. `--talk off` (settings `talk`) disables it; scenes
where it fails fall back to Wan + lip-sync.

## Moving to a GPU server later

Every model stage is an isolated worker process that reads a JSON job and writes a JSON result
(`aivs/workers/*.py`), and model ids are environment variables (`AIVS_MODEL_*`, see `aivs/config.py`).
Replacing a worker with a CUDA implementation, or running it on another machine, does not touch the pipeline,
timeline, or edit code.

## Tests

The offline tests use fakes for all models, so they need no downloads (about 10 s each):

```bash
envs/studio/bin/python tests/test_new_features.py
envs/studio/bin/python tests/test_dialogue.py
```

Generated test output goes to `test-results/`, which is not tracked.

## Repository layout

| Path | Content |
|---|---|
| `aivs/` | the pipeline (planner, TTS, images, motion, edit, captions) and the web studio (`aivs/web/`) |
| `aivs/workers/` | one worker per model stage, each a JSON-in / JSON-out process |
| `scripts/` | setup scripts for the optional backends and helper runners |
| `tests/` | offline tests |
| `examples/` | the demo clip shown above, two sample storyboards and their measurement reports |
| `evidence/licenses/` | licence texts of the models and tools used |
| `projects/`, `models/`, `envs/`, `vendor/`, `logs/` | created locally; not tracked (see `.gitignore`) |

## Licences

Code: MIT (`LICENSE`), by https://github.com/Saya2810. Models and tools: see `THIRD_PARTY.md` — all defaults are Apache-2.0 or MIT and
place no restrictions on commercial use of the output.
