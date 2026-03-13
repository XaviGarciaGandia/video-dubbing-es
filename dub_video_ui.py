#!/usr/bin/env python3
"""
Gradio Web UI for the Video Dubbing Pipeline.

Usage:
    source /home/xavi/dub_video/dubbing-env/bin/activate
    export LD_LIBRARY_PATH="/home/xavi/dub_video/dubbing-env/lib/python3.9/site-packages/nvidia/cublas/lib:/home/xavi/dub_video/dubbing-env/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
    python dub_video_ui.py
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from dub_video import (
    extract_audio,
    merge_video_audio,
    mix_audio,
    save_subtitles,
    synthesize_all,
    transcribe,
    translate_segments,
)

VOICES = {
    "Alvaro (Spain, male)": "es-ES-AlvaroNeural",
    "Elvira (Spain, female)": "es-ES-ElviraNeural",
    "Dalia (Mexico, female)": "es-MX-DaliaNeural",
    "Jorge (Mexico, male)": "es-MX-JorgeNeural",
    "Elena (Argentina, female)": "es-AR-ElenaNeural",
    "Tomas (Argentina, male)": "es-AR-TomasNeural",
    "Valentina (Colombia, female)": "es-CO-SalomeNeural",
    "Gonzalo (Colombia, male)": "es-CO-GonzaloNeural",
}

MODELS = ["large-v3", "medium", "small", "base", "tiny"]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


class LogCapture:
    """Captures print output and stores log lines."""

    def __init__(self):
        self.lines = []
        self._old_stdout = None
        self._old_stderr = None

    def start(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        if self._old_stdout:
            sys.stdout = self._old_stdout
        if self._old_stderr:
            sys.stderr = self._old_stderr

    def write(self, text):
        if text.strip():
            self.lines.append(text.rstrip())
        if self._old_stdout:
            self._old_stdout.write(text)

    def flush(self):
        if self._old_stdout:
            self._old_stdout.flush()

    def get_log(self):
        return "\n".join(self.lines)


def dub_single_video(video_path, engine, voice, ref_wav_path, model_size,
                     device, natural_pace, max_speedup, keep_original,
                     original_volume, gen_subtitles, output_dir=None,
                     watermark_path=None, watermark_opacity=0.3,
                     watermark_x="W-w-10", watermark_y="H-h-10",
                     intro_path=None, outro_path=None):
    """Core dubbing logic for a single video. Returns (output_video, output_srt, log_text, segments_text, error)."""
    log = LogCapture()
    log.start()

    try:
        use_gpu = device == "cuda"

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(video_path), "dubbed_output")
        os.makedirs(output_dir, exist_ok=True)

        stem = Path(video_path).stem
        suffix = "_xtts" if engine == "xtts" else "_es"
        output_video = os.path.join(output_dir, f"{stem}{suffix}.mp4")
        output_srt = os.path.join(output_dir, f"{stem}{suffix}.srt")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1
            print(f"[1/5] Extracting audio from {Path(video_path).name}...")
            audio_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(video_path, audio_path)

            # Step 2
            print(f"[2/5] Transcribing audio with Faster Whisper ({model_size})...")
            segments = transcribe(audio_path, model_size=model_size, device=device)
            if not segments:
                log.stop()
                return None, None, log.get_log(), "", "No speech detected in the video."
            print(f"  Found {len(segments)} segments")

            # Step 3
            print(f"[3/5] Translating {len(segments)} segments to Spanish...")
            translated = translate_segments(segments)

            # Step 4
            engine_label = "XTTS v2 (voice clone)" if engine == "xtts" else f"Edge TTS ({voice})"
            print(f"[4/5] Generating speech with {engine_label}...")
            tts_dir = os.path.join(tmpdir, "tts")
            os.makedirs(tts_dir)
            asyncio.run(synthesize_all(
                translated, tts_dir, voice,
                natural_pace=natural_pace,
                max_speedup=max_speedup,
                tts_engine=engine,
                speaker_wav=ref_wav_path,
                gpu=use_gpu,
            ))

            # Get total video duration
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, check=True,
            )
            total_duration = float(result.stdout.strip())

            # Step 5
            print("[5/5] Mixing final audio track...")
            final_audio = mix_audio(
                translated, total_duration, audio_path,
                keep_original=keep_original,
                original_volume=original_volume,
                natural_pace=natural_pace,
            )
            merge_video_audio(video_path, final_audio, output_video,
                              watermark_path=watermark_path,
                              watermark_opacity=watermark_opacity,
                              watermark_x=watermark_x,
                              watermark_y=watermark_y,
                              intro_path=intro_path,
                              outro_path=outro_path)
            print(f"Video saved: {output_video}")

            # Subtitles
            srt_file = None
            if gen_subtitles:
                save_subtitles(translated, output_srt)
                srt_file = output_srt
                print(f"Subtitles saved: {output_srt}")

        # Build segment text
        seg_lines = []
        for i, seg in enumerate(translated):
            seg_lines.append(
                f"[{seg['start']:.1f}s - {seg['end']:.1f}s]\n"
                f"  EN: {seg['original']}\n"
                f"  ES: {seg['translated']}"
            )
        segments_text = "\n\n".join(seg_lines)

        print("Done!")
        log.stop()
        return output_video, srt_file, log.get_log(), segments_text, None

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nERROR: {e}\n{tb}")
        log.stop()
        return None, None, log.get_log(), "", str(e)


# ── Single video handler ─────────────────────────────────────────────────────

def run_single(video_file, tts_engine, voice_name, speaker_wav, model_size,
               device, natural_pace, max_speedup, keep_original,
               original_volume, gen_subtitles,
               watermark_file, watermark_opacity, watermark_x, watermark_y,
               intro_file, outro_file,
               progress=gr.Progress(track_tqdm=False)):

    if video_file is None:
        raise gr.Error("Please upload a video file.")

    engine = "xtts" if tts_engine == "XTTS v2 (Voice Clone)" else "edge"
    if engine == "xtts" and speaker_wav is None:
        raise gr.Error("Please upload a voice reference audio for XTTS voice cloning.")

    voice = VOICES[voice_name]
    ref_wav_path = speaker_wav if engine == "xtts" else None

    progress(0.1, desc="Processing...")
    output_video, srt_file, log_text, segments_text, error = dub_single_video(
        video_file, engine, voice, ref_wav_path, model_size,
        device, natural_pace, max_speedup, keep_original,
        original_volume, gen_subtitles,
        watermark_path=watermark_file,
        watermark_opacity=watermark_opacity,
        watermark_x=watermark_x,
        watermark_y=watermark_y,
        intro_path=intro_file,
        outro_path=outro_file,
    )

    if error:
        log_text += f"\n\nFAILED: {error}"

    progress(1.0, desc="Done!" if not error else "Failed")

    download_files = []
    if output_video:
        download_files.append(output_video)
    if srt_file:
        download_files.append(srt_file)

    return output_video, download_files, log_text, segments_text


# ── Batch folder handler ─────────────────────────────────────────────────────

def run_batch(input_folder, output_folder, tts_engine, voice_name, speaker_wav,
              model_size, device, natural_pace, max_speedup, keep_original,
              original_volume, gen_subtitles,
              watermark_file, watermark_opacity, watermark_x, watermark_y,
              intro_file, outro_file,
              progress=gr.Progress(track_tqdm=False)):

    if not input_folder or not input_folder.strip():
        raise gr.Error("Please enter an input folder path.")

    input_folder = input_folder.strip()
    if not os.path.isdir(input_folder):
        raise gr.Error(f"Input folder not found: {input_folder}")

    engine = "xtts" if tts_engine == "XTTS v2 (Voice Clone)" else "edge"
    if engine == "xtts" and speaker_wav is None:
        raise gr.Error("Please upload a voice reference audio for XTTS voice cloning.")

    voice = VOICES[voice_name]
    ref_wav_path = speaker_wav if engine == "xtts" else None

    # Determine output folder
    if output_folder and output_folder.strip():
        out_dir = output_folder.strip()
    else:
        out_dir = os.path.join(input_folder, "dubbed_output")
    os.makedirs(out_dir, exist_ok=True)

    # Find all video files
    video_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        raise gr.Error(f"No video files found in {input_folder}")

    all_logs = []
    all_outputs = []
    summary_lines = []
    total = len(video_files)

    all_logs.append(f"Found {total} video(s) in {input_folder}\n")
    all_logs.append(f"Output folder: {out_dir}\n")
    all_logs.append("=" * 60)

    for idx, vpath in enumerate(video_files):
        fname = os.path.basename(vpath)
        all_logs.append(f"\n[{idx+1}/{total}] Processing: {fname}")
        all_logs.append("-" * 40)

        progress((idx / total), desc=f"[{idx+1}/{total}] {fname}")

        output_video, srt_file, log_text, segments_text, error = dub_single_video(
            vpath, engine, voice, ref_wav_path, model_size,
            device, natural_pace, max_speedup, keep_original,
            original_volume, gen_subtitles, output_dir=out_dir,
            watermark_path=watermark_file,
            watermark_opacity=watermark_opacity,
            watermark_x=watermark_x,
            watermark_y=watermark_y,
            intro_path=intro_file,
            outro_path=outro_file,
        )

        all_logs.append(log_text)

        if error:
            summary_lines.append(f"FAILED  {fname}: {error}")
            all_logs.append(f"\nFAILED: {error}")
        else:
            summary_lines.append(f"OK      {fname} -> {os.path.basename(output_video)}")
            if output_video:
                all_outputs.append(output_video)
            if srt_file:
                all_outputs.append(srt_file)

    all_logs.append("\n" + "=" * 60)
    all_logs.append("BATCH SUMMARY:")
    all_logs.extend(summary_lines)
    all_logs.append(f"\n{len(all_outputs)} file(s) generated in {out_dir}")

    progress(1.0, desc="Batch complete!")

    full_log = "\n".join(all_logs)
    summary = "\n".join(summary_lines)

    return all_outputs, full_log, summary


# ── UI toggle helpers ─────────────────────────────────────────────────────────

def toggle_tts_options(engine):
    if engine == "XTTS v2 (Voice Clone)":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Video Dubbing: EN -> ES", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Video Dubbing Pipeline: English -> Spanish")

    with gr.Row():
        # ── Left column: shared settings ──
        with gr.Column(scale=1):
            tts_engine_radio = gr.Radio(
                choices=["Edge TTS (preset voices)", "XTTS v2 (Voice Clone)"],
                value="Edge TTS (preset voices)",
                label="TTS Engine",
            )

            voice_dropdown = gr.Dropdown(
                choices=list(VOICES.keys()),
                value="Alvaro (Spain, male)",
                label="Spanish Voice (Edge TTS)",
                visible=True,
            )

            speaker_wav_input = gr.Audio(
                label="Voice Reference Audio (10-30s of clean speech)",
                type="filepath",
                visible=False,
            )

            model_dropdown = gr.Dropdown(
                choices=MODELS,
                value="large-v3",
                label="Whisper Model",
            )
            device_radio = gr.Radio(
                choices=["cuda", "cpu"],
                value="cuda",
                label="Device",
            )

            with gr.Accordion("Pacing", open=True):
                natural_pace_check = gr.Checkbox(
                    value=True,
                    label="Natural pace (overflow into gaps instead of speeding up)",
                )
                max_speedup_slider = gr.Slider(
                    minimum=1.0, maximum=1.5, step=0.05, value=1.1,
                    label="Max speedup factor",
                )

            with gr.Accordion("Audio mixing", open=False):
                keep_original_check = gr.Checkbox(
                    value=False,
                    label="Keep original audio in background",
                )
                original_vol_slider = gr.Slider(
                    minimum=0.0, maximum=0.5, step=0.05, value=0.1,
                    label="Original audio volume",
                )

            gen_subs_check = gr.Checkbox(
                value=True,
                label="Generate Spanish subtitles (.srt)",
            )

            with gr.Accordion("Watermark", open=False):
                watermark_input = gr.Image(
                    label="Watermark image (PNG with transparency recommended)",
                    type="filepath",
                )
                watermark_opacity_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                    label="Watermark opacity (0 = invisible, 1 = fully opaque)",
                )
                with gr.Row():
                    watermark_x_input = gr.Textbox(
                        value="W-w-10", label="X position",
                        info="FFmpeg expression. W-w-10 = right, 10 = left, (W-w)/2 = center",
                    )
                    watermark_y_input = gr.Textbox(
                        value="H-h-10", label="Y position",
                        info="FFmpeg expression. H-h-10 = bottom, 10 = top, (H-h)/2 = center",
                    )

            with gr.Accordion("Intro / Outro videos", open=False):
                intro_input = gr.Video(
                    label="Intro video (prepended before the dubbed video)",
                    sources=["upload"],
                )
                outro_input = gr.Video(
                    label="Outro video (appended after the dubbed video)",
                    sources=["upload"],
                )

        # ── Right column: tabs for single / batch ──
        with gr.Column(scale=1):
            with gr.Tabs():
                # ── Single video tab ──
                with gr.Tab("Single Video"):
                    video_input = gr.Video(label="Input Video", sources=["upload"])
                    single_run_btn = gr.Button("Dub Video", variant="primary", size="lg")
                    video_output = gr.Video(label="Dubbed Video")
                    single_files = gr.File(label="Download files", file_count="multiple")

                # ── Batch folder tab ──
                with gr.Tab("Batch (Folder)"):
                    gr.Markdown("Process all videos in a folder on this machine.")
                    input_folder = gr.Textbox(
                        label="Input Folder Path",
                        placeholder="/home/xavi/videos_to_dub/",
                    )
                    output_folder = gr.Textbox(
                        label="Output Folder Path (leave empty for input_folder/dubbed_output/)",
                        placeholder="",
                    )
                    batch_run_btn = gr.Button("Dub All Videos", variant="primary", size="lg")
                    batch_files = gr.File(label="Generated files", file_count="multiple")
                    batch_summary = gr.Textbox(
                        label="Batch Summary",
                        lines=8,
                        interactive=False,
                    )

            # ── Shared log area ──
            log_output = gr.Textbox(
                label="Processing Log (full detail + errors)",
                lines=15,
                interactive=False,
                show_copy_button=True,
            )

    with gr.Accordion("Transcription & Translation Details", open=False):
        segments_table = gr.Textbox(
            label="Segments (last processed video)",
            lines=15,
            interactive=False,
            show_copy_button=True,
        )

    # ── Events ────────────────────────────────────────────────────────────────

    tts_engine_radio.change(
        fn=toggle_tts_options,
        inputs=[tts_engine_radio],
        outputs=[voice_dropdown, speaker_wav_input],
    )

    single_run_btn.click(
        fn=run_single,
        inputs=[
            video_input, tts_engine_radio, voice_dropdown, speaker_wav_input,
            model_dropdown, device_radio, natural_pace_check, max_speedup_slider,
            keep_original_check, original_vol_slider, gen_subs_check,
            watermark_input, watermark_opacity_slider,
            watermark_x_input, watermark_y_input,
            intro_input, outro_input,
        ],
        outputs=[video_output, single_files, log_output, segments_table],
    )

    batch_run_btn.click(
        fn=run_batch,
        inputs=[
            input_folder, output_folder, tts_engine_radio, voice_dropdown,
            speaker_wav_input, model_dropdown, device_radio, natural_pace_check,
            max_speedup_slider, keep_original_check, original_vol_slider,
            gen_subs_check,
            watermark_input, watermark_opacity_slider,
            watermark_x_input, watermark_y_input,
            intro_input, outro_input,
        ],
        outputs=[batch_files, log_output, batch_summary],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
