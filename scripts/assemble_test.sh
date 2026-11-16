#!/usr/bin/env bash
# Assembly smoke test: proves the ffmpeg post-production chain the studio needs
# (clip crossfade, voice + ducked music + timed SFX mix, loudness normalisation, burned subtitles).
# Portable (macOS/Linux). Uses only synthetic sources, no model downloads.
# Usage: bash scripts/assemble_test.sh [out_dir] [ffmpeg_bin]
set -euo pipefail
OUT="${1:-$(cd "$(dirname "$0")/.." && pwd)/test-results/assemble}"
FF="${2:-ffmpeg}"; FP="${FF%ffmpeg}ffprobe"; command -v "$FP" >/dev/null 2>&1 || FP=""
mkdir -p "$OUT"; cd "$OUT"
W=1280; H=720; FPS=24
# two "shots"
"$FF" -y -v error -f lavfi -i "testsrc2=s=${W}x${H}:r=$FPS:d=4" -c:v libx264 -pix_fmt yuv420p shot1.mp4
"$FF" -y -v error -f lavfi -i "smptebars=s=${W}x${H}:r=$FPS:d=4" -c:v libx264 -pix_fmt yuv420p shot2.mp4
# stand-ins for voice / music / sfx
"$FF" -y -v error -f lavfi -i "sine=f=220:d=2.5" -af "tremolo=f=6:d=0.8" -ar 48000 voice1.wav
"$FF" -y -v error -f lavfi -i "sine=f=330:d=2.5" -af "tremolo=f=5:d=0.8" -ar 48000 voice2.wav
"$FF" -y -v error -f lavfi -i "sine=f=110:d=7.5" -ar 48000 music.wav
"$FF" -y -v error -f lavfi -i "anoisesrc=d=0.3:c=white:a=0.5" -ar 48000 sfx.wav
cat > subs.srt <<SRT
1
00:00:00,400 --> 00:00:02,900
Erste Zeile: Sprecherin (Deutsch)

2
00:00:04,000 --> 00:00:06,500
Second line: narrator (English)
SRT
# video: 0.5 s crossfade (4 + 4 - 0.5 = 7.5 s) + burned subtitles
# audio: voices placed by adelay, music ducked under voice via sidechain, SFX at 3.5 s, then EBU R128 normalise
"$FF" -y -v error -i shot1.mp4 -i shot2.mp4 -i voice1.wav -i voice2.wav -i music.wav -i sfx.wav -filter_complex "
[0:v][1:v]xfade=transition=fade:duration=0.5:offset=3.5,subtitles=subs.srt[v];
[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,adelay=400:all=1[v1];[3:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,adelay=4000:all=1[v2];
[v1][v2]amix=inputs=2:normalize=0,apad=whole_dur=7.5,asplit=2[vox][voxsc];
[4:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume=0.5[mus];[mus][voxsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck];
[5:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,adelay=3500:all=1[fx];
[vox][duck][fx]amix=inputs=3:normalize=0:duration=longest,loudnorm=I=-14:TP=-1.5:LRA=11[a]" \
 -map "[v]" -map "[a]" -t 7.5 -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -ar 48000 -b:a 192k -movflags +faststart final.mp4
if [ -n "$FP" ]; then "$FP" -v error -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels:format=duration -of compact final.mp4; fi
echo "ASSEMBLE_TEST PASS -> $OUT/final.mp4"
