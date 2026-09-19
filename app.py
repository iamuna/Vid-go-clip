from __future__ import annotations

import os
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from vidgoclip.ai import ollama_ready
from vidgoclip.config import OUTPUT_DIR, load_settings, save_settings
from vidgoclip.exporter import export_clip
from vidgoclip.media import executable_available, format_timestamp
from vidgoclip.models import AnalysisResult, Candidate
from vidgoclip.pipeline import analyze_video

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master: "VidGoClipApp") -> None:
        super().__init__(master)
        self.master_app = master
        self.settings = master.settings.copy()

        self.title("Vid-go-clip Settings")
        self.geometry("620x520")
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
                "small is the default Whisper model for speed. Larger models "
                "can improve transcription but require more time/resources."
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
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.video_path: Path | None = None
        self.analysis: AnalysisResult | None = None
        self.candidate_by_id: dict[str, Candidate] = {}

        self.title("Vid-go-clip")
        self.geometry("1280x820")
        self.minsize(1020, 700)
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
            text="VID-GO-CLIP",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(16, 2))

        ctk.CTkLabel(
            header,
            text=(
                "Upload a long video → AI watches/listens → ranked moments → export clips"
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
        controls.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(controls, text="Focus").grid(
            row=0, column=0, padx=(12, 6), pady=12
        )
        self.focus_var = ctk.StringVar(
            value=str(self.settings.get("analysis_focus", "Balanced"))
        )
        self.focus_menu = ctk.CTkOptionMenu(
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
            width=140,
        )
        self.focus_menu.grid(row=0, column=1, padx=(0, 12), pady=12)

        self.vertical_var = ctk.BooleanVar(
            value=bool(self.settings.get("vertical_export", False))
        )
        ctk.CTkCheckBox(
            controls,
            text="Export 9:16 center crop",
            variable=self.vertical_var,
            command=self._save_ui_settings,
        ).grid(row=0, column=2, padx=12, pady=12)

        self.force_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            controls,
            text="Reanalyze from scratch",
            variable=self.force_var,
        ).grid(row=0, column=3, padx=12, pady=12)

        self.analyze_button = ctk.CTkButton(
            controls,
            text="ANALYZE VIDEO",
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._start_analysis,
        )
        self.analyze_button.grid(row=0, column=4, padx=12, pady=12)

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
            "rank", "time", "score", "important",
            "controversy", "interesting", "visual", "title",
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
            "title": "Suggested moment",
        }
        widths = {
            "rank": 42,
            "time": 110,
            "score": 65,
            "important": 58,
            "controversy": 58,
            "interesting": 58,
            "visual": 58,
            "title": 280,
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
        detail.grid_rowconfigure(2, weight=1)

        self.detail_title = ctk.CTkLabel(
            detail,
            text="Select a moment",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
            justify="left",
            wraplength=430,
        )
        self.detail_title.grid(
            row=0, column=0, sticky="ew", padx=14, pady=(14, 6)
        )

        self.detail_reason = ctk.CTkLabel(
            detail,
            text="Ranked moments will appear here after analysis.",
            anchor="w",
            justify="left",
            wraplength=430,
            text_color=("gray30", "gray75"),
        )
        self.detail_reason.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 8)
        )

        self.transcript_box = ctk.CTkTextbox(detail)
        self.transcript_box.grid(
            row=2, column=0, sticky="nsew", padx=14, pady=(0, 12)
        )
        self.transcript_box.configure(state="disabled")

        self.status_label = ctk.CTkLabel(
            body,
            text="Choose a video to begin.",
            anchor="w",
            justify="left",
            wraplength=1100,
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
                "Scores are editorial signals, not truth/fact ratings. "
                "Review context before publishing."
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
        self.settings["vertical_export"] = self.vertical_var.get()
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

    def _progress_message(self, message: str) -> None:
        self.after(0, lambda: self.status_label.configure(text=message))

    def _start_analysis(self, force_override: bool | None = None) -> None:
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
        force = (
            self.force_var.get()
            if force_override is None
            else force_override
        )
        self._set_busy("Starting video analysis...")

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
        self.candidate_by_id = {
            candidate.id: candidate
            for candidate in result.candidates
        }

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
                f"Visual AI {notes.get('visual_ai', '?')}"
            )
        )
        if result.candidates:
            first = result.candidates[0]
            self.tree.selection_set(first.id)
            self.tree.focus(first.id)
            self._show_candidate(first)

    def _analysis_failed(self, error: str) -> None:
        self._clear_busy()
        self.status_label.configure(text=f"Analysis failed: {error}")
        messagebox.showerror("Vid-go-clip", error)

    def _selection_changed(self, _event=None) -> None:
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
            f"\n\nVisual: {candidate.visual_description}"
            if candidate.visual_description
            else ""
        )
        self.detail_reason.configure(
            text=(
                f"{candidate.reason}{visual}\n\n"
                f"Importance {s.get('importance', 0):.0f} • "
                f"Controversy {s.get('controversy', 0):.0f} • "
                f"Interest {s.get('interest', 0):.0f} • "
                f"Emotion {s.get('emotion', 0):.0f} • "
                f"Visual {s.get('visual', 0):.0f} • "
                f"Context {s.get('context', 0):.0f}"
            )
        )
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", candidate.text)
        self.transcript_box.configure(state="disabled")

    def _selected_candidate(self) -> Candidate | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return self.candidate_by_id.get(selection[0])

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
        self._set_busy(
            f"Exporting {len(candidates)} clip(s)..."
        )
        vertical = bool(self.vertical_var.get())
        threading.Thread(
            target=self._export_worker,
            args=(candidates, vertical),
            daemon=True,
        ).start()

    def _export_worker(
        self,
        candidates: list[Candidate],
        vertical: bool,
    ) -> None:
        assert self.video_path is not None
        outputs = []
        try:
            for index, candidate in enumerate(candidates, start=1):
                self._progress_message(
                    f"Exporting clip {index}/{len(candidates)}..."
                )
                outputs.append(
                    export_clip(
                        self.video_path,
                        candidate,
                        vertical=vertical,
                    )
                )
        except Exception as exc:
            self.after(0, self._analysis_failed, str(exc))
            return
        self.after(0, self._export_finished, outputs)

    def _export_finished(self, outputs: list[Path]) -> None:
        self._clear_busy()
        self.status_label.configure(
            text=(
                f"Exported {len(outputs)} clip(s) to {OUTPUT_DIR.resolve()}"
            )
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


if __name__ == "__main__":
    VidGoClipApp().mainloop()
