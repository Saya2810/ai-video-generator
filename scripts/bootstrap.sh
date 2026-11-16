#!/bin/bash
# AI Video Studio — macOS bootstrap + smoke tests (Apple Silicon). bash 3.2 compatible.
# Everything stays inside this project folder (envs/, models/, vendor/, test-results/). Nothing needs sudo.
#
#   bash scripts/bootstrap.sh 0        # probe + uv + ffmpeg + assembly test        (no model downloads, ~2 min)
#   bash scripts/bootstrap.sh 1        # voice + transcription + planner LLM        (~12 GB download)
#   bash scripts/bootstrap.sh 2        # keyframe image model                       (~6 GB download)
#   bash scripts/bootstrap.sh 3        # video model LTX-2.3 q4 + Gemma encoder     (~30 GB download, slow)
#   bash scripts/bootstrap.sh 4        # music (ACE-Step 1.5) install check         (~1 GB, weights on first run)
#   bash scripts/bootstrap.sh 0 1 2    # several tiers in one go
#
# Every test is logged to test-results/mac-<timestamp>/ with PASS/FAIL, wall time and peak memory in results.tsv.
# A failing test does NOT stop the run: the logs are the deliverable for the next agent.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIERS="${*:-0}"
export HF_HOME="$ROOT/models/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export PATH="$ROOT/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"
RES="$ROOT/test-results/mac-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RES" "$ROOT/envs" "$ROOT/models" "$ROOT/vendor" "$ROOT/bin"
TSV="$RES/results.tsv"; printf "test\tstatus\twall_s\tpeak_rss_bytes\n" > "$TSV"
PY=3.12

run_test() {  # run_test <name> <command...>   (command must be a real executable, not a shell function)
  name="$1"; shift; log="$RES/$name.log"
  echo ">>> $name"; echo "\$ $*" > "$log"; t0=$(date +%s)
  if /usr/bin/time -l "$@" >> "$log" 2>&1; then st=PASS; else st=FAIL; fi
  t1=$(date +%s)
  peak=$(grep -E "maximum resident set size|peak memory footprint" "$log" | awk '{print $1}' | sort -n | tail -1)
  printf "%s\t%s\t%s\t%s\n" "$name" "$st" "$((t1-t0))" "${peak:-na}" >> "$TSV"
  echo "    $st in $((t1-t0))s (peak ${peak:-na} bytes)  log: $log"
}
has_tier() { case " $TIERS " in *" $1 "*) return 0;; *) return 1;; esac; }
free_gb() { df -g "$ROOT" | awk 'NR==2{print $4}'; }
need_gb() { f=$(free_gb); if [ "$f" -lt "$1" ]; then echo "!! only ${f} GB free, tier needs $1 GB — skipping"; return 1; fi; }

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (https://astral.sh/uv, MIT/Apache-2.0) into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh >> "$RES/uv_install.log" 2>&1
  fi
  command -v uv >/dev/null 2>&1 || { echo "!! uv not available — see $RES/uv_install.log"; exit 1; }
}
ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then return 0; fi
  if command -v brew >/dev/null 2>&1; then brew install ffmpeg >> "$RES/ffmpeg_install.log" 2>&1; fi
  if ! command -v ffmpeg >/dev/null 2>&1; then   # fallback: static build shipped by the imageio-ffmpeg wheel
    [ -d "$ROOT/envs/tools" ] || uv venv -q --python $PY "$ROOT/envs/tools"
    uv pip install -q -p "$ROOT/envs/tools" imageio-ffmpeg >> "$RES/ffmpeg_install.log" 2>&1
    exe=$("$ROOT/envs/tools/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
    [ -n "$exe" ] && ln -sf "$exe" "$ROOT/bin/ffmpeg"
  fi
}
mkenv() {  # mkenv <name> <pip specs...>
  n="$1"; shift
  [ -d "$ROOT/envs/$n" ] || uv venv -q --python $PY "$ROOT/envs/$n"
  run_test "install_$n" uv pip install -p "$ROOT/envs/$n" "$@"
}

ensure_uv

# ---------------------------------------------------------------- tier 0
if has_tier 0; then
  { echo "date: $(date)"; sw_vers; echo "chip: $(sysctl -n machdep.cpu.brand_string)"
    echo "mem_bytes: $(sysctl -n hw.memsize)"; echo "perf_cores: $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)"
    echo "eff_cores: $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null)"; echo "free_gb: $(free_gb)"
    echo "xcode_clt: $(xcode-select -p 2>&1)"; echo "uv: $(uv --version 2>&1)"
    echo "brew: $(command -v brew || echo none)"; echo "git: $(git --version 2>&1)"
    echo "python3: $(python3 --version 2>&1)"; echo "ollama: $(command -v ollama || echo none)"
    system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Cores|Metal" ; } > "$RES/probe.txt" 2>&1
  cat "$RES/probe.txt"
  ensure_ffmpeg
  { command -v ffmpeg; ffmpeg -version | head -1
    ffmpeg -hide_banner -filters 2>/dev/null | grep -E " (sidechaincompress|loudnorm|xfade|subtitles|amix|adelay) " | awk '{print $2}' | tr '\n' ' '
    echo; ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "libx264|h264_videotoolbox|hevc_videotoolbox|aac " ; } > "$RES/ffmpeg.txt" 2>&1
  run_test assemble bash "$ROOT/scripts/assemble_test.sh" "$RES/assemble"
  [ -d "$ROOT/envs/probe" ] || uv venv -q --python $PY "$ROOT/envs/probe"
  run_test install_mlx uv pip install -p "$ROOT/envs/probe" mlx
  run_test mlx_metal "$ROOT/envs/probe/bin/python" -c "import mlx.core as mx; print('mlx', mx.__version__ if hasattr(mx,'__version__') else ''); print('metal', mx.metal.is_available()); print(mx.device_info() if hasattr(mx,'device_info') else mx.metal.device_info()); a=mx.random.normal((4096,4096)); mx.eval(a@a); print('matmul ok')"
fi

# ---------------------------------------------------------------- tier 1  voice / ASR / planner
if has_tier 1 && need_gb 20; then
  A="$ROOT/envs/audio/bin/python"; O="$RES/audio"; mkdir -p "$O"
  mkenv audio "mlx-audio==0.6.5"
  run_test install_audio_kokoro_extras uv pip install -p "$ROOT/envs/audio" "misaki[en]"
  run_test tts_kokoro_en "$A" -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --lang_code a --voice af_heart \
     --text "The lighthouse keeper climbed the stairs as the storm rolled in over the bay." --output_path "$O" --file_prefix kokoro_en
  run_test tts_chatterbox_de "$A" -m mlx_audio.tts.generate --model mlx-community/chatterbox-multilingual-v3 --lang_code de \
     --text "Der Leuchtturmwärter stieg die Treppe hinauf, während der Sturm über die Bucht zog." --output_path "$O" --file_prefix chatterbox_de
  run_test tts_chatterbox_en "$A" -m mlx_audio.tts.generate --model mlx-community/chatterbox-multilingual-v3 --lang_code en \
     --text "The lighthouse keeper climbed the stairs as the storm rolled in over the bay." --output_path "$O" --file_prefix chatterbox_en
  wav=$(ls "$O"/chatterbox_de* 2>/dev/null | head -1)
  if [ -n "$wav" ]; then
    run_test stt_whisper_de "$A" -m mlx_audio.stt.generate --model mlx-community/whisper-large-v3-turbo-asr-fp16 --audio "$wav" \
       --language de --format json --output-path "$O/chatterbox_de_transcript"
  fi
  mkenv llm "mlx-lm==0.32.2"
  run_test llm_planner "$ROOT/envs/llm/bin/python" -m mlx_lm generate --model mlx-community/Qwen3-4B-Instruct-2507-4bit --max-tokens 400 \
     --prompt "Return only JSON: a 3-shot list for a 15 second video about a lighthouse in a storm. Each shot has: visual_prompt, duration_s, narration_de, sfx."
fi

# ---------------------------------------------------------------- tier 2  keyframe image
if has_tier 2 && need_gb 15; then
  O="$RES/image"; mkdir -p "$O"
  mkenv image "mflux==0.24.0"
  run_test img_zimage_turbo_q4 "$ROOT/envs/image/bin/mflux-generate-z-image-turbo" --model filipstrand/Z-Image-Turbo-mflux-4bit \
     --prompt "cinematic wide shot of a lighthouse on a rocky coast at dusk, storm clouds, crashing waves, volumetric light, 35mm film still" \
     --width 1024 --height 576 --steps 9 --seed 42 --output "$O/keyframe.png"
  [ -f "$O/keyframe.png" ] && cp "$O/keyframe.png" "$ROOT/test-results/latest_keyframe.png"
fi

# ---------------------------------------------------------------- tier 3  video (+ native audio)
if has_tier 3 && need_gb 45; then
  O="$RES/video"; mkdir -p "$O"; V="$ROOT/vendor/ltx-2-mlx"
  if [ ! -d "$V" ]; then git clone https://github.com/dgrauet/ltx-2-mlx.git "$V" >> "$RES/ltx_clone.log" 2>&1; (cd "$V" && git checkout -q 747b30e) >> "$RES/ltx_clone.log" 2>&1; fi
  run_test install_ltx uv sync --project "$V" --all-extras
  LTX="uv run --project $V ltx-2-mlx"
  M=dgrauet/ltx-2.3-mlx-q4
  # smallest meaningful run first: 2 s, 768x512, distilled 8-step, block streaming
  run_test ltx_t2v_768x512_49f $LTX generate --distilled --low-ram --model $M -f 49 --frame-rate 24 -H 512 -W 768 \
     -p "A lighthouse on a rocky coast at dusk, waves crash against the rocks, wind howls, a distant foghorn sounds" -o "$O/t2v_small.mp4"
  if grep -q "ltx_t2v_768x512_49f	PASS" "$TSV"; then
    K="$ROOT/test-results/latest_keyframe.png"
    if [ -f "$K" ]; then
      run_test ltx_i2v_1024x576_97f $LTX generate --distilled --low-ram --model $M -f 97 --frame-rate 24 -H 576 -W 1024 --image "$K" 0 1.0 \
         -p "Slow push-in on the lighthouse, waves crash, the lamp begins to rotate, thunder rumbles in the distance" -o "$O/i2v_576p.mp4"
    fi
    run_test ltx_t2v_1280x704_97f $LTX generate --distilled --low-ram --model $M -f 97 --frame-rate 24 -H 704 -W 1280 \
       -p "A lighthouse on a rocky coast at dusk, waves crash against the rocks, wind howls, a distant foghorn sounds" -o "$O/t2v_704p.mp4"
    W=$(ls "$ROOT"/test-results/mac-*/audio/chatterbox_en* 2>/dev/null | tail -1)
    if [ -n "$W" ]; then   # audio-to-video (lip/sound sync to our own TTS). Beta upstream — failure here is informative, not blocking.
      run_test ltx_a2v_768x512 $LTX a2v --low-ram --model $M --frame-rate 24 -H 512 -W 768 --audio "$W" \
         -p "Close-up of an old lighthouse keeper speaking to camera, warm lamp light, storm outside the window" -o "$O/a2v.mp4"
    fi
  fi
fi

# ---------------------------------------------------------------- tier 4  music
if has_tier 4 && need_gb 20; then
  V="$ROOT/vendor/ACE-Step-1.5"
  [ -d "$V" ] || git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "$V" >> "$RES/ace_clone.log" 2>&1
  run_test install_acestep uv sync --project "$V"
  run_test acestep_help uv run --project "$V" acestep --help
fi

rm -f "$ROOT/test-results/latest"; ln -s "$RES" "$ROOT/test-results/latest"
echo; echo "================ RESULTS ($RES) ================"; column -t -s "$(printf '\t')" "$TSV"
echo "Done. Tell Claude: 'bootstrap tiers $TIERS finished' — the logs are in test-results/latest/."
