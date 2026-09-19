from __future__ import annotations

import os
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from vidgoclip.ai import ollama_ready
from vidgoclip.config import OUTPUT_DIR, load_settings, save_settings
from vidgoclip.exporter import export_clip
from vidgoclip.media import executable_available, format_timestamp
from vidgoclip.models import AnalysisResult, Candidate
from vidgoclip.pipeline import analyze_video
from vidgoclip.preview import prepare_preview

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master: "VidGoClipApp") -> None:
        super().__init__(master)
        self.master_app = master
        self.settings = master.settings.copy()

        self.title("Vid-go-clip Settings")
        self.geometry("620x540")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text="ANALYSIS SETTINGS",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(22, 12))

        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 18))
        frame.grid_columnconfigure(1, weight=1)

        self.whisper_var = ctk.StringVar(
            value=str(self.settings.get("whisper_model", "small"))
        )
        self.ollama_var = ctk.StringVar(
            value=str(self.settings.get("ollama_model", "qwen3-vl:4b"))
        )
        self.min_var = ctk.StringVar(
            value=str(self.settings.get("min_clip_seconds", 18))
        )
        self.target_var = ctk.StringVar(
            value=str(self.settings.get("target_clip_seconds", 42))
        )
        self.max_var = ctk.StringVar(
            value=str(self.settings.get("max_clip_seconds", 75))
        )
        self.max_ai_var = ctk.StringVar(
            value=str(self.settings.get("max_ai_candidates", 36))
        )
        self.max_visual_var = ctk.StringVar(
            value=str(self.settings.get("max_visual_candidates", 12))
        )

        rows = [
            ("Whisper model", self.whisper_var),
            ("Ollama vision model", self.ollama_var),
            ("Minimum clip seconds", self.min_var),
            ("Target clip seconds", self.target_var),
            ("Maximum clip seconds", self.max_var),
            ("Semantic candidates", self.max_ai_var),
            ("Visual candidates", self.max_visual_var),
        ]

        for row, (label, variable) in enumerate(rows):
            ctk.CTkLabel(frame, text=label).grid(
                row=row, column=0, sticky="w", padx=14, pady=9
            )
            if label == "Whisper model":
                widget = ctk.CTkOptionMenu(
                    frame,
                    variable=variable,
                    values=["tiny", "base", "small", "medium", "large-v3"],
                )
            else:
                widget = ctk.CTkEntry(frame, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", padx=14, pady=9)

        ctk.CTkLabel(
            self,
            text=(
                "v0.2 uses word-level timestamps. Larger Whisper models can "
                "improve transcript/boundary quality but take more resources."
            ),
            wraplength=560,
            justify="left",
            text_color=("gray35", "gray70"),
        ).grid(row=2, column=0, sticky="w", padx=24, pady=(0, 16))

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 22))
        button_row.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            button_row,
            text="SAVE",
            height=42,
            command=self._save,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            button_row,
            text="Cancel",
            width=110,
            command=self.destroy,
        ).grid(row=0, column=1)

    def _save(self) -> None:
        try:
            minimum = float(self.min_var.get())
            target = float(self.target_var.get())
            maximum = float(self.max_var.get())
            max_ai = int(self.max_ai_var.get())
            max_visual = int(self.max_visual_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Numeric settings must be numbers.")
            return

        if not (5 <= minimum <= target <= maximum <= 180):
            messagebox.showerror(
                "Settings",
                "Use: 5 <= minimum <= target <= maximum <= 180 seconds.",
            )
            return

        self.master_app.settings.update(
            {
                "whisper_model": self.whisper_var.get(),
                "ollama_model": self.ollama_var.get().strip() or "qwen3-vl:4b",
                "min_clip_seconds": minimum,
                "target_clip_seconds": target,
                "max_clip_seconds": maximum,
                "max_ai_candidates": max(4, min(100, max_ai)),
                "max_visual_candidates": max(1, min(30, max_visual)),
            }
        )
        save_settings(self.master_app.settings)
        self.master_app._refresh_status()
        self.destroy()


class VidGoClipApp(ctk.CTk):
    PREVIEW_SIZE = (440, 248)

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.video_path: Path | None = None
        self.analysis: AnalysisResult | None = None
        self.candidate_by_id: dict[str, Candidate] = {}

        self.preview_capture = None
        self.preview_after_id = None
        self.preview_photo = None
        self.preview_video_path: Path | None = None
        self.preview_audio_path: Path | None = None
        self.preview_candidate_id: str | None = None
        self.preview_playing = False
        self.preview_fps = 30.0

        self.title("Vid-go-clip v0.2 — Smart Clipping")
        self.geometry("1380x900")
        self.minsize(1120, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._refresh_status()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="VID-GO-CLIP  •  SMART CLIPPING v0.2",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(16, 2))

        ctk.CTkLabel(
            header,
            text=(
                "Video + speech + audio energy → ranked moments → preview → "
                "smart vertical reframe + captions"
            ),
            text_color=("gray35", "gray70"),
        ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 16))

        self.stack_badge = ctk.CTkLabel(
            header,
            text="CHECKING...",
            corner_radius=12,
            padx=12,
            pady=6,
        )
        self.stack_badge.grid(row=0, column=1, rowspan=2, padx=(8, 8))

        ctk.CTkButton(
            header,
            text="Settings",
            width=100,
            command=lambda: SettingsDialog(self),
        ).grid(row=0, column=2, rowspan=2, padx=(0, 24))

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self)
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=18)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(3, weight=1)

        source = ctk.CTkFrame(body)
        source.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        source.grid_columnconfigure(0, weight=1)

        self.video_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            source,
            textvariable=self.video_var,
            height=42,
            placeholder_text="Choose a video to analyze...",
        ).grid(row=0, column=0, sticky="ew", padx=(12, 8), pady=12)
        ctk.CTkButton(
            source,
            text="Choose video",
            width=130,
            command=self._choose_video,
        ).grid(row=0, column=1, padx=(0, 12), pady=12)

        controls = ctk.CTkFrame(body)
        controls.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        controls.grid_columnconfigure(8, weight=1)

        ctk.CTkLabel(controls, text="Focus").grid(
            row=0, column=0, padx=(12, 6), pady=12
        )
        self.focus_var = ctk.StringVar(
            value=str(self.settings.get("analysis_focus", "Balanced"))
        )
        ctk.CTkOptionMenu(
            controls,
            variable=self.focus_var,
            values=[
                "Balanced",
                "Important",
                "Controversial",
                "Interesting",
                "Emotional",
            ],
            command=self._focus_changed,
            width=135,
        ).grid(row=0, column=1, padx=(0, 8), pady=12)

        self.vertical_var = ctk.BooleanVar(
            value=bool(self.settings.get("vertical_export", False))
        )
        ctk.CTkCheckBox(
            controls,
            text="9:16",
            variable=self.vertical_var,
            command=self._save_ui_settings,
        ).grid(row=0, column=2, padx=8, pady=12)

        self.smart_var = ctk.BooleanVar(
            value=bool(self.settings.get("smart_reframe", True))
        )
        ctk.CTkCheckBox(
            controls,
            text="Smart track",
            variable=self.smart_var,
            command=self._save_ui_settings,
        ).grid(row=0, column=3, padx=8, pady=12)

        self.caption_var = ctk.BooleanVar(
            value=bool(self.settings.get("burn_captions", True))
        )
        ctk.CTkCheckBox(
            controls,
            text="Captions",
            variable=self.caption_var,
            command=self._save_ui_settings,
        ).grid(row=0, column=4, padx=8, pady=12)

        self.force_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            controls,
            text="Reanalyze",
            variable=self.force_var,
        ).grid(row=0, column=5, padx=8, pady=12)

        self.analyze_button = ctk.CTkButton(
            controls,
            text="ANALYZE VIDEO",
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._start_analysis,
        )
        self.analyze_button.grid(row=0, column=6, padx=12, pady=12)

        self.progress = ctk.CTkProgressBar(body, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.progress.stop()
        self.progress.set(0)

        result_area = ctk.CTkFrame(body)
        result_area.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 8))
        result_area.grid_columnconfigure(0, weight=3)
        result_area.grid_columnconfigure(1, weight=2)
        result_area.grid_rowconfigure(0, weight=1)

        table_frame = ctk.CTkFrame(result_area)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        columns = (
            "rank", "time", "score", "important", "controversy",
            "interesting", "visual", "audio", "title",
        )
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        headings = {
            "rank": "#",
            "time": "Time",
            "score": "Score",
            "important": "Imp.",
            "controversy": "Contr.",
            "interesting": "Int.",
            "visual": "Visual",
            "audio": "Audio",
            "title": "Suggested moment",
        }
        widths = {
            "rank": 42,
            "time": 112,
            "score": 62,
            "important": 54,
            "controversy": 54,
            "interesting": 54,
            "visual": 54,
            "audio": 54,
            "title": 270,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(
                column,
                width=widths[column],
                minwidth=40,
                stretch=(column == "title"),
            )
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        detail = ctk.CTkFrame(result_area)
        detail.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        detail.grid_columnconfigure(0, weight=1)
        detail.grid_rowconfigure(5, weight=1)

        self.preview_label = ctk.CTkLabel(
            detail,
            text="Select a moment, then press PLAY PREVIEW",
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            fg_color=("gray80", "gray15"),
            corner_radius=8,
        )
        self.preview_label.grid(
            row=0, column=0, sticky="ew", padx=14, pady=(14, 6)
        )

        preview_buttons = ctk.CTkFrame(detail, fg_color="transparent")
        preview_buttons.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        preview_buttons.grid_columnconfigure(2, weight=1)
        self.preview_button = ctk.CTkButton(
            preview_buttons,
            text="PLAY PREVIEW",
            width=125,
            command=self._play_selected_preview,
            state="disabled",
        )
        self.preview_button.grid(row=0, column=0, padx=(0, 6))
        self.stop_preview_button = ctk.CTkButton(
            preview_buttons,
            text="STOP",
            width=80,
            command=self._stop_preview,
            state="disabled",
        )
        self.stop_preview_button.grid(row=0, column=1, padx=(0, 6))
        self.preview_time_label = ctk.CTkLabel(
            preview_buttons,
            text="",
            anchor="e",
        )
        self.preview_time_label.grid(row=0, column=2, sticky="e")

        self.detail_title = ctk.CTkLabel(
            detail,
            text="Select a moment",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
            justify="left",
            wraplength=440,
        )
        self.detail_title.grid(
            row=2, column=0, sticky="ew", padx=14, pady=(6, 6)
        )

        self.detail_reason = ctk.CTkLabel(
            detail,
            text="Ranked moments will appear here after analysis.",
            anchor="w",
            justify="left",
            wraplength=440,
            text_color=("gray30", "gray75"),
        )
        self.detail_reason.grid(
            row=3, column=0, sticky="ew", padx=14, pady=(0, 8)
        )

        ctk.CTkLabel(
            detail,
            text="Transcript",
            anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 4))

        self.transcript_box = ctk.CTkTextbox(detail, height=150)
        self.transcript_box.grid(
            row=5, column=0, sticky="nsew", padx=14, pady=(0, 12)
        )
        self.transcript_box.configure(state="disabled")

        self.status_label = ctk.CTkLabel(
            body,
            text="Choose a video to begin.",
            anchor="w",
            justify="left",
            wraplength=1200,
        )
        self.status_label.grid(
            row=4, column=0, sticky="ew", padx=12, pady=(0, 8)
        )

        actions = ctk.CTkFrame(body)
        actions.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 12))
        actions.grid_columnconfigure(3, weight=1)

        self.export_selected_button = ctk.CTkButton(
            actions,
            text="EXPORT SELECTED",
            command=self._export_selected,
            state="disabled",
        )
        self.export_selected_button.grid(row=0, column=0, padx=12, pady=12)

        self.export_top_button = ctk.CTkButton(
            actions,
            text="EXPORT TOP 5",
            command=self._export_top,
            state="disabled",
        )
        self.export_top_button.grid(row=0, column=1, padx=(0, 12), pady=12)

        ctk.CTkButton(
            actions,
            text="Open output folder",
            command=self._open_output,
        ).grid(row=0, column=2, padx=(0, 12), pady=12)

        ctk.CTkLabel(
            actions,
            text=(
                "Smart track follows faces/motion. Audio-event labels are "
                "heuristics, not definitive sound classification."
            ),
            anchor="e",
            text_color=("gray35", "gray70"),
        ).grid(row=0, column=3, sticky="e", padx=12, pady=12)

    def _refresh_status(self) -> None:
        ffmpeg_ok = executable_available("ffmpeg") and executable_available("ffprobe")
        ai_ok = ollama_ready(str(self.settings["ollama_base_url"]))
        if ffmpeg_ok and ai_ok:
            text = "LOCAL AI READY"
        elif not ffmpeg_ok and not ai_ok:
            text = "FFMPEG + OLLAMA NEEDED"
        elif not ffmpeg_ok:
            text = "FFMPEG NEEDED"
        else:
            text = "OLLAMA NEEDED"
        self.stack_badge.configure(text=text)

    def _choose_video(self) -> None:
        self._stop_preview()
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path = Path(path)
            self.video_var.set(path)
            self.status_label.configure(
                text=f"Ready to analyze: {self.video_path.name}"
            )

    def _save_ui_settings(self) -> None:
        self.settings["analysis_focus"] = self.focus_var.get()
        self.settings["vertical_export"] = bool(self.vertical_var.get())
        self.settings["smart_reframe"] = bool(self.smart_var.get())
        self.settings["burn_captions"] = bool(self.caption_var.get())
        save_settings(self.settings)

    def _focus_changed(self, _value: str) -> None:
        self._save_ui_settings()
        if self.video_path and self.analysis:
            self._start_analysis(force_override=False)

    def _set_busy(self, message: str) -> None:
        self.analyze_button.configure(state="disabled")
        self.export_selected_button.configure(state="disabled")
        self.export_top_button.configure(state="disabled")
        self.status_label.configure(text=message)
        self.progress.start()

    def _clear_busy(self) -> None:
        self.progress.stop()
        self.progress.set(0)
        self.analyze_button.configure(state="normal")
        if self.analysis and self.analysis.candidates:
            self.export_selected_button.configure(state="normal")
            self.export_top_button.configure(state="normal")
            self.preview_button.configure(state="normal")

    def _progress_message(self, message: str) -> None:
        self.after(0, lambda: self.status_label.configure(text=message))

    def _start_analysis(self, force_override: bool | None = None) -> None:
        self._stop_preview()
        if self.video_path is None:
            typed = self.video_var.get().strip()
            if typed:
                self.video_path = Path(typed)

        if self.video_path is None or not self.video_path.is_file():
            messagebox.showwarning("Video", "Choose a valid video first.")
            return

        if not executable_available("ffmpeg") or not executable_available("ffprobe"):
            messagebox.showerror(
                "FFmpeg required",
                "Run setup.bat first so FFmpeg is available.",
            )
            return

        self._save_ui_settings()
        force = self.force_var.get() if force_override is None else force_override
        self._set_busy("Starting smart video analysis...")

        threading.Thread(
            target=self._analysis_worker,
            args=(self.video_path, force),
            daemon=True,
        ).start()

    def _analysis_worker(self, video_path: Path, force: bool) -> None:
        try:
            result = analyze_video(
                video_path,
                settings=self.settings.copy(),
                progress=self._progress_message,
                force=force,
            )
        except Exception as exc:
            self.after(0, self._analysis_failed, str(exc))
            return
        self.after(0, self._analysis_finished, result)

    def _analysis_finished(self, result: AnalysisResult) -> None:
        self.analysis = result
        self.candidate_by_id = {c.id: c for c in result.candidates}

        for item in self.tree.get_children():
            self.tree.delete(item)

        for rank, candidate in enumerate(result.candidates, start=1):
            s = candidate.scores
            self.tree.insert(
                "",
                "end",
                iid=candidate.id,
                values=(
                    rank,
                    f"{format_timestamp(candidate.start)}–{format_timestamp(candidate.end)}",
                    f"{candidate.final_score:.0f}",
                    f"{s.get('importance', 0):.0f}",
                    f"{s.get('controversy', 0):.0f}",
                    f"{s.get('interest', 0):.0f}",
                    f"{s.get('visual', 0):.0f}",
                    f"{s.get('audio', 0):.0f}",
                    candidate.title or candidate.id,
                ),
            )

        self._clear_busy()
        self.force_var.set(False)
        notes = result.model_notes
        self.status_label.configure(
            text=(
                f"Found {len(result.candidates)} ranked moments • "
                f"Whisper {notes.get('whisper', '?')} • "
                f"word timing {notes.get('word_timing', '?')} • "
                f"audio {notes.get('audio_analysis', '?')} • "
                f"visual AI {notes.get('visual_ai', '?')}"
            )
        )
        if result.candidates:
            first = result.candidates[0]
            self.tree.selection_set(first.id)
            self.tree.focus(first.id)
            self._show_candidate(first)

    def _analysis_failed(self, error: str) -> None:
        self._clear_busy()
        self.status_label.configure(text=f"Operation failed: {error}")
        messagebox.showerror("Vid-go-clip", error)

    def _selection_changed(self, _event=None) -> None:
        self._stop_preview()
        selection = self.tree.selection()
        if not selection:
            return
        candidate = self.candidate_by_id.get(selection[0])
        if candidate:
            self._show_candidate(candidate)

    def _show_candidate(self, candidate: Candidate) -> None:
        s = candidate.scores
        self.detail_title.configure(
            text=(
                f"{candidate.title or candidate.id}\n"
                f"{format_timestamp(candidate.start)} – "
                f"{format_timestamp(candidate.end)} • "
                f"{candidate.final_score:.0f}/100"
            )
        )
        visual = (
            f"\nVisual: {candidate.visual_description}"
            if candidate.visual_description else ""
        )
        audio = (
            f"\nAudio: {candidate.audio_description}"
            if candidate.audio_description else ""
        )
        self.detail_reason.configure(
            text=(
                f"{candidate.reason}{visual}{audio}\n\n"
                f"Importance {s.get('importance', 0):.0f} • "
                f"Controversy {s.get('controversy', 0):.0f} • "
                f"Interest {s.get('interest', 0):.0f} • "
                f"Emotion {s.get('emotion', 0):.0f} • "
                f"Visual {s.get('visual', 0):.0f} • "
                f"Audio {s.get('audio', 0):.0f} • "
                f"Context {s.get('context', 0):.0f}"
            )
        )
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", candidate.text)
        self.transcript_box.configure(state="disabled")
        self.preview_label.configure(
            text="Press PLAY PREVIEW to watch this exact candidate",
            image=None,
        )
        self.preview_time_label.configure(
            text=f"{candidate.duration:.1f}s"
        )
        self.preview_button.configure(state="normal")

    def _selected_candidate(self) -> Candidate | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return self.candidate_by_id.get(selection[0])

    def _play_selected_preview(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None or self.video_path is None:
            return

        self._stop_preview()
        self.status_label.configure(text="Preparing in-app preview...")
        self.preview_button.configure(state="disabled")
        threading.Thread(
            target=self._preview_prepare_worker,
            args=(self.video_path, candidate),
            daemon=True,
        ).start()

    def _preview_prepare_worker(
        self,
        video_path: Path,
        candidate: Candidate,
    ) -> None:
        try:
            video, audio = prepare_preview(video_path, candidate)
        except Exception as exc:
            self.after(0, self._preview_failed, str(exc))
            return
        self.after(
            0,
            self._start_preview_files,
            candidate.id,
            video,
            audio,
        )

    def _preview_failed(self, error: str) -> None:
        self.preview_button.configure(state="normal")
        self.status_label.configure(text=f"Preview failed: {error}")

    def _start_preview_files(
        self,
        candidate_id: str,
        video_path: Path,
        audio_path: Path | None,
    ) -> None:
        self.preview_candidate_id = candidate_id
        self.preview_video_path = video_path
        self.preview_audio_path = audio_path
        self.preview_capture = cv2.VideoCapture(str(video_path))
        if not self.preview_capture.isOpened():
            self._preview_failed("OpenCV could not open the preview.")
            return

        self.preview_fps = float(
            self.preview_capture.get(cv2.CAP_PROP_FPS) or 30.0
        )
        self.preview_playing = True
        self.stop_preview_button.configure(state="normal")
        self.status_label.configure(text="Playing preview inside Vid-go-clip.")

        if audio_path is not None and os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(
                    str(audio_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
            except Exception:
                pass

        self._preview_tick()

    def _preview_tick(self) -> None:
        if not self.preview_playing or self.preview_capture is None:
            return

        ok, frame = self.preview_capture.read()
        if not ok:
            self._stop_preview()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", self.PREVIEW_SIZE, (15, 15, 15))
        x = (self.PREVIEW_SIZE[0] - image.width) // 2
        y = (self.PREVIEW_SIZE[1] - image.height) // 2
        canvas.paste(image, (x, y))
        self.preview_photo = ImageTk.PhotoImage(canvas)
        self.preview_label.configure(image=self.preview_photo, text="")

        frame_index = float(
            self.preview_capture.get(cv2.CAP_PROP_POS_FRAMES) or 0
        )
        seconds = frame_index / max(1.0, self.preview_fps)
        candidate = self._selected_candidate()
        total = candidate.duration if candidate else 0.0
        self.preview_time_label.configure(
            text=f"{seconds:.1f}s / {total:.1f}s"
        )

        delay = max(10, int(round(1000.0 / max(1.0, self.preview_fps))))
        self.preview_after_id = self.after(delay, self._preview_tick)

    def _stop_preview(self) -> None:
        self.preview_playing = False
        if self.preview_after_id is not None:
            try:
                self.after_cancel(self.preview_after_id)
            except Exception:
                pass
            self.preview_after_id = None

        if self.preview_capture is not None:
            try:
                self.preview_capture.release()
            except Exception:
                pass
            self.preview_capture = None

        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        if hasattr(self, "stop_preview_button"):
            self.stop_preview_button.configure(state="disabled")
        if hasattr(self, "preview_button") and self.analysis:
            self.preview_button.configure(state="normal")

    def _export_selected(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None or self.video_path is None:
            messagebox.showinfo("Export", "Select a ranked moment first.")
            return
        self._start_export([candidate])

    def _export_top(self) -> None:
        if self.analysis is None or self.video_path is None:
            return
        count = int(self.settings.get("default_export_count", 5))
        self._start_export(self.analysis.candidates[:count])

    def _start_export(self, candidates: list[Candidate]) -> None:
        self._stop_preview()
        self._save_ui_settings()
        self._set_busy(f"Exporting {len(candidates)} smart clip(s)...")
        options = {
            "vertical": bool(self.vertical_var.get()),
            "smart_reframe": bool(self.smart_var.get()),
            "burn_captions": bool(self.caption_var.get()),
            "caption_words_per_line": int(
                self.settings.get("caption_words_per_line", 7)
            ),
        }
        threading.Thread(
            target=self._export_worker,
            args=(candidates, options),
            daemon=True,
        ).start()

    def _export_worker(
        self,
        candidates: list[Candidate],
        options: dict,
    ) -> None:
        assert self.video_path is not None
        outputs = []
        analysis = self.analysis
        try:
            for index, candidate in enumerate(candidates, start=1):
                self._progress_message(
                    f"Exporting clip {index}/{len(candidates)}..."
                )
                outputs.append(
                    export_clip(
                        self.video_path,
                        candidate,
                        vertical=bool(options["vertical"]),
                        smart_reframe=bool(options["smart_reframe"]),
                        burn_captions=bool(options["burn_captions"]),
                        words=analysis.words if analysis else [],
                        transcript=analysis.transcript if analysis else [],
                        caption_words_per_line=int(
                            options["caption_words_per_line"]
                        ),
                        progress=self._progress_message,
                    )
                )
        except Exception as exc:
            self.after(0, self._analysis_failed, str(exc))
            return
        self.after(0, self._export_finished, outputs)

    def _export_finished(self, outputs: list[Path]) -> None:
        self._clear_busy()
        self.status_label.configure(
            text=f"Exported {len(outputs)} clip(s) to {OUTPUT_DIR.resolve()}"
        )
        messagebox.showinfo(
            "Export complete",
            f"Created {len(outputs)} clip(s).",
        )

    def _open_output(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(OUTPUT_DIR.resolve())
        else:
            messagebox.showinfo("Output folder", str(OUTPUT_DIR.resolve()))

    def _on_close(self) -> None:
        self._stop_preview()
        self.destroy()


if __name__ == "__main__":
    VidGoClipApp().mainloop()
