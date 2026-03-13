#!/usr/bin/env python3
"""
Video Dubbing Pipeline: English -> Spanish
Uses: Faster Whisper (transcription) + Deep Translator (translation) + Edge TTS (speech synthesis) + FFmpeg (video processing)

Usage:
    source /home/xavi/dub_video/dubbing-env/bin/activate
    python dub_video.py input_video.mp4
    python dub_video.py input_video.mp4 --voice es-MX-DaliaNeural
    python dub_video.py input_video.mp4 --keep-original-audio --original-volume 0.1
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts
import srt
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from pydub import AudioSegment


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg."""
    print(f"[1/5] Extracting audio from video...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True, check=True
    )


def transcribe(audio_path: str, model_size: str = "large-v3", device: str = "cuda"):
    """Transcribe audio using Faster Whisper."""
    print(f"[2/5] Transcribing audio with Faster Whisper ({model_size})...")
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, info = model.transcribe(audio_path, beam_size=5, language="en", word_timestamps=True)

    result = []
    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
        print(f"  [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()}")

    print(f"  Detected language: {info.language} (probability: {info.language_probability:.2f})")
    return result


def translate_segments(segments: list, source: str = "en", target: str = "es"):
    """Translate transcribed segments to target language."""
    print(f"[3/5] Translating {len(segments)} segments to Spanish...")
    translator = GoogleTranslator(source=source, target=target)

    translated = []
    for seg in segments:
        try:
            translated_text = translator.translate(seg["text"])
            if not translated_text:
                translated_text = seg["text"]
        except Exception as e:
            print(f"  Translation warning: {e}")
            translated_text = seg["text"]
        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "original": seg["text"],
            "translated": translated_text,
        })
        print(f"  [{seg['start']:.1f}s] {seg['text']}")
        print(f"         -> {translated_text}")

    return translated


async def synthesize_segment(text: str, output_path: str, voice: str, rate: str = "+0%"):
    """Synthesize a single segment with edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
    except Exception as e:
        print(f"  TTS warning: failed for '{text[:40]}...' — {e}")


def _get_xtts_model(gpu: bool = True):
    """Load XTTS v2 model (cached after first call)."""
    if not hasattr(_get_xtts_model, "_model"):
        import torch
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load

        from TTS.api import TTS
        _get_xtts_model._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if gpu:
            _get_xtts_model._model.to("cuda")
    return _get_xtts_model._model


def synthesize_segment_xtts(text: str, output_path: str, speaker_wav: str,
                            language: str = "es", gpu: bool = True):
    """Synthesize a single segment with XTTS v2 voice cloning."""
    try:
        tts = _get_xtts_model(gpu=gpu)
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )
    except Exception as e:
        print(f"  XTTS warning: failed for '{text[:40]}...' — {e}")


async def synthesize_all(segments: list, output_dir: str, voice: str,
                         natural_pace: bool = True, max_speedup: float = 1.1,
                         tts_engine: str = "edge", speaker_wav: str = None,
                         gpu: bool = True):
    """Synthesize all translated segments.

    tts_engine: "edge" for Edge TTS, "xtts" for XTTS v2 voice cloning.
    speaker_wav: path to reference voice WAV (required for xtts).

    If natural_pace is True, segments play at natural TTS speed and are only
    sped up (by at most max_speedup) when they would overlap the next segment.
    """
    if tts_engine == "xtts":
        print(f"[4/5] Generating Spanish speech with XTTS v2 (voice clone)...")
        for i, seg in enumerate(segments):
            out_path = os.path.join(output_dir, f"seg_{i:04d}.wav")
            seg["audio_path"] = out_path
            synthesize_segment_xtts(seg["translated"], out_path, speaker_wav,
                                    gpu=gpu)
            print(f"  Segment {i}/{len(segments)-1}: {seg['translated'][:50]}...")
    else:
        print(f"[4/5] Generating Spanish speech with Edge TTS ({voice})...")
        tasks = []
        for i, seg in enumerate(segments):
            out_path = os.path.join(output_dir, f"seg_{i:04d}.mp3")
            seg["audio_path"] = out_path
            tasks.append(synthesize_segment(seg["translated"], out_path, voice))
        await asyncio.gather(*tasks)

    # Read durations for all segments
    for i, seg in enumerate(segments):
        if not os.path.exists(seg["audio_path"]):
            print(f"  Warning: Failed to generate audio for segment {i}")
            seg["tts_duration"] = 0
            continue
        tts_audio = AudioSegment.from_file(seg["audio_path"])
        seg["tts_duration"] = len(tts_audio) / 1000.0

    if natural_pace:
        # Natural pace: only speed up when TTS would overlap the next segment
        for i, seg in enumerate(segments):
            if seg["tts_duration"] <= 0:
                continue

            # Determine the deadline: start of the next segment, or end of video
            if i + 1 < len(segments):
                deadline = segments[i + 1]["start"]
            else:
                deadline = seg["end"] + 5.0  # last segment: allow 5s overflow

            available = deadline - seg["start"]
            if available <= 0:
                available = seg["end"] - seg["start"]

            if seg["tts_duration"] > available:
                speed_ratio = seg["tts_duration"] / available
                if speed_ratio <= max_speedup:
                    # Apply mild speedup to fit before next segment
                    atempo = speed_ratio
                else:
                    # Cap at max_speedup — segment will overflow slightly
                    atempo = max_speedup

                base_path = os.path.splitext(seg["audio_path"])[0]
                adjusted_path = base_path + "_adjusted.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", seg["audio_path"],
                     "-filter:a", f"atempo={atempo}",
                     adjusted_path],
                    capture_output=True, check=True
                )
                seg["audio_path"] = adjusted_path
                seg["tts_duration"] = seg["tts_duration"] / atempo
                print(f"  Segment {i}: {atempo:.2f}x speedup — {seg['translated'][:50]}...")
            else:
                print(f"  Segment {i}: natural pace — {seg['translated'][:50]}...")
    else:
        # Strict pace: old behavior — speed up to fit original segment timing
        for i, seg in enumerate(segments):
            if seg["tts_duration"] <= 0:
                continue
            target_duration = seg["end"] - seg["start"]
            if target_duration > 0:
                speed_ratio = seg["tts_duration"] / target_duration
                if speed_ratio > 1.3:
                    base_path = os.path.splitext(seg["audio_path"])[0]
                    adjusted_path = base_path + "_adjusted.wav"
                    atempo = min(speed_ratio, 2.0)
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", seg["audio_path"],
                         "-filter:a", f"atempo={atempo}",
                         adjusted_path],
                        capture_output=True, check=True
                    )
                    seg["audio_path"] = adjusted_path
            print(f"  Segment {i}: {seg['translated'][:50]}...")


def mix_audio(segments: list, total_duration: float, original_audio_path: str,
              keep_original: bool = False, original_volume: float = 0.1,
              natural_pace: bool = True):
    """Mix all TTS segments into a single audio track aligned to original timestamps."""
    print("[5/5] Mixing final audio track...")

    # Create a silent base track matching original video length
    base = AudioSegment.silent(duration=int(total_duration * 1000))

    for i, seg in enumerate(segments):
        if "audio_path" not in seg or not os.path.exists(seg["audio_path"]):
            continue
        tts_audio = AudioSegment.from_file(seg["audio_path"])
        position_ms = int(seg["start"] * 1000)

        if natural_pace:
            # Allow overflow into gap, but trim before the next segment starts
            if i + 1 < len(segments):
                max_end_ms = int(segments[i + 1]["start"] * 1000)
            else:
                max_end_ms = int(total_duration * 1000)
            max_duration_ms = max_end_ms - position_ms
        else:
            # Strict: trim to original segment boundary
            max_duration_ms = int((seg["end"] - seg["start"]) * 1000)

        if len(tts_audio) > max_duration_ms:
            tts_audio = tts_audio[:max_duration_ms]

        # Extend base if needed (last segment overflow)
        if position_ms + len(tts_audio) > len(base):
            extra = (position_ms + len(tts_audio)) - len(base)
            base = base + AudioSegment.silent(duration=extra)

        base = base.overlay(tts_audio, position=position_ms)

    if keep_original:
        original = AudioSegment.from_file(original_audio_path)
        # Lower the original audio volume
        original = original - (20 * (1 - original_volume))  # rough dB reduction
        # Make same length
        if len(original) > len(base):
            original = original[:len(base)]
        elif len(original) < len(base):
            original = original + AudioSegment.silent(duration=len(base) - len(original))
        base = base.overlay(original)

    return base


def merge_video_audio(video_path: str, audio_segment: AudioSegment, output_path: str):
    """Merge the dubbed audio back into the video."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_segment.export(tmp.name, format="wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", tmp.name,
             "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
             "-shortest", output_path],
            capture_output=True, check=True
        )
        os.unlink(tmp.name)
    print(f"\nDone! Output saved to: {output_path}")


def save_subtitles(segments: list, output_path: str):
    """Save translated segments as SRT subtitles."""
    import datetime
    subs = []
    for i, seg in enumerate(segments):
        subs.append(srt.Subtitle(
            index=i + 1,
            start=datetime.timedelta(seconds=seg["start"]),
            end=datetime.timedelta(seconds=seg["end"]),
            content=seg["translated"],
        ))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    print(f"Subtitles saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dub a video from English to Spanish")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output video path (default: input_es.mp4)")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for Whisper inference")
    parser.add_argument("--voice", default="es-ES-AlvaroNeural",
                        help="Edge TTS voice (e.g., es-ES-AlvaroNeural, es-MX-DaliaNeural, es-ES-ElviraNeural)")
    parser.add_argument("--keep-original-audio", action="store_true",
                        help="Keep original audio at low volume in background")
    parser.add_argument("--original-volume", type=float, default=0.1,
                        help="Volume of original audio when kept (0.0-1.0)")
    parser.add_argument("--subtitles", action="store_true",
                        help="Also generate Spanish SRT subtitle file")
    parser.add_argument("--strict-pace", action="store_true",
                        help="Force TTS to match original segment timing (old behavior). "
                             "Default is natural pace with overflow into gaps.")
    parser.add_argument("--max-speedup", type=float, default=1.1,
                        help="Max TTS speedup factor in natural pace mode (default: 1.1)")
    parser.add_argument("--tts-engine", default="edge", choices=["edge", "xtts"],
                        help="TTS engine: 'edge' for Edge TTS, 'xtts' for XTTS v2 voice cloning")
    parser.add_argument("--speaker-wav", default=None,
                        help="Path to reference voice WAV/MP3 for XTTS voice cloning")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_es.mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Extract audio
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)

        # Step 2: Transcribe
        segments = transcribe(audio_path, model_size=args.model, device=args.device)
        if not segments:
            print("No speech detected in the video.")
            sys.exit(1)

        # Save transcription
        with open(os.path.join(tmpdir, "transcription.json"), "w") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        # Step 3: Translate
        translated = translate_segments(segments)

        # Step 4: Synthesize TTS
        tts_dir = os.path.join(tmpdir, "tts")
        os.makedirs(tts_dir)
        natural_pace = not args.strict_pace
        use_gpu = args.device == "cuda"
        asyncio.run(synthesize_all(translated, tts_dir, args.voice,
                                   natural_pace=natural_pace,
                                   max_speedup=args.max_speedup,
                                   tts_engine=args.tts_engine,
                                   speaker_wav=args.speaker_wav,
                                   gpu=use_gpu))

        # Get total video duration
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True
        )
        total_duration = float(result.stdout.strip())

        # Step 5: Mix and merge
        final_audio = mix_audio(
            translated, total_duration, audio_path,
            keep_original=args.keep_original_audio,
            original_volume=args.original_volume,
            natural_pace=natural_pace
        )
        merge_video_audio(video_path, final_audio, output_path)

        # Optional: subtitles
        if args.subtitles:
            srt_path = output_path.replace(".mp4", ".srt")
            save_subtitles(translated, srt_path)


if __name__ == "__main__":
    main()
