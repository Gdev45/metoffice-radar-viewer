import os
import re
import tarfile
import tempfile
import shutil
import gzip
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import OrderedDict

import iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tkinterdnd2 import DND_FILES, TkinterDnD


APP_TITLE = "Met Office Radar Viewer (High Detail)"

# ------------------------------------------------------------
# Standard Met Office Radar Levels & Palette
# ------------------------------------------------------------

DEFAULT_LEVELS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 128.0]

DEFAULT_COLORS = [
    "#b8e6ff", "#4da6ff", "#0066ff", "#00cc66",
    "#00ff00", "#ccff00", "#ffff00", "#ffcc00", "#ff9900",
    "#ff3300", "#cc0000", "#990099", "#660066"
]

# Precise UK View Bounding Box [min_lon, max_lon, min_lat, max_lat]
UK_EXTENT = [-8.5, 2.5, 49.5, 59.0]

# ------------------------------------------------------------
# Met Office radar projection
# ------------------------------------------------------------

radar_crs = ccrs.TransverseMercator(
    central_longitude=-2,
    central_latitude=49,
    false_easting=400000,
    false_northing=-100000,
    scale_factor=0.9996012717,
    globe=ccrs.Globe(
        ellipse="airy",
        semimajor_axis=6377563.396,
        semiminor_axis=6356256.91
    )
)

# ------------------------------------------------------------
# Dynamic PAL Palette File Parser
# ------------------------------------------------------------

def parse_pal_file(filepath):
    """Parses .pal files and dynamically calculates exact boundaries for N colors."""
    rgb_list = []
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if any(line.startswith(h) for h in ["JASC-PAL", "RIFF", "PAL", "0100"]):
            continue

        if line.startswith("#") and len(line) in (7, 9):
            try:
                r = int(line[1:3], 16) / 255.0
                g = int(line[3:5], 16) / 255.0
                b = int(line[5:7], 16) / 255.0
                rgb_list.append((r, g, b))
                continue
            except ValueError:
                pass

        clean_line = re.sub(r"[,;:\t]+", " ", line)
        parts = clean_line.split()

        numbers = []
        for part in parts:
            try:
                numbers.append(int(part))
            except ValueError:
                pass

        if len(numbers) == 1:
            continue

        if len(numbers) >= 4:
            numbers = numbers[-3:]

        if len(numbers) == 3:
            r, g, b = numbers
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                rgb_list.append((r / 255.0, g / 255.0, b / 255.0))

    if not rgb_list:
        raise ValueError("Could not extract any valid RGB color data from this .pal file.")

    custom_cmap = ListedColormap(rgb_list)
    num_colors = len(rgb_list)

    if num_colors == len(DEFAULT_COLORS):
        custom_levels = list(DEFAULT_LEVELS)
    else:
        custom_levels = list(np.geomspace(0.1, 128.0, num_colors + 1))

    custom_norm = BoundaryNorm(custom_levels, custom_cmap.N)

    return custom_cmap, custom_norm, custom_levels

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_timestamp(filename):
    name = os.path.basename(filename)
    match = re.search(r"_(\d{12})_1km-composite\.dat\.gz$", name)
    return match.group(1) if match else None


def format_timestamp(filename):
    timestamp = get_timestamp(filename)
    if timestamp is None:
        return os.path.basename(filename)

    return (
        f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}  "
        f"{timestamp[8:10]}:{timestamp[10:12]} UTC"
    )

# ------------------------------------------------------------
# Main Application
# ------------------------------------------------------------

class RadarViewer:

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1250x950")
        self.root.minsize(900, 700)
        self.root.configure(bg="#c0c0c0")

        # Palette state
        self.palettes = {
            "Default Palette": (
                ListedColormap(DEFAULT_COLORS),
                BoundaryNorm(DEFAULT_LEVELS, len(DEFAULT_COLORS)),
                DEFAULT_LEVELS
            )
        }
        self.current_cmap, self.current_norm, self.current_levels = self.palettes["Default Palette"]

        # Files & Cache
        self.files = []
        self.index = 0
        self.data_cache = OrderedDict()
        self.max_cache = 50
        self.x_edges = None
        self.y_edges = None

        # Animation state
        self.playing = False
        self.after_id = None
        self.temp_directory = None

        # Plot components
        self.mesh = None
        self.colorbar = None
        self.gridlines = None

        self.build_interface()
        self.build_map()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ========================================================
    # USER INTERFACE
    # ========================================================

    def build_interface(self):
        style = ttk.Style()
        
        available_themes = style.theme_names()
        if "winnative" in available_themes:
            style.theme_use("winnative")
        elif "classic" in available_themes:
            style.theme_use("classic")
        else:
            style.theme_use("clam")

        win_gray = "#c0c0c0"
        font_plain = ("MS Sans Serif", 9)
        font_bold = ("MS Sans Serif", 9, "bold")

        style.configure(".", background=win_gray, foreground="black", font=font_plain)
        style.configure("TFrame", background=win_gray)
        style.configure("TLabel", background=win_gray, foreground="black", font=font_plain)
        style.configure("TButton", background=win_gray, foreground="black", font=font_plain, borderwidth=2, relief="raised")
        style.configure("TCombobox", fieldbackground="white", background=win_gray, foreground="black", font=font_plain)
        style.configure("Horizontal.TScale", background=win_gray)

        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")

        self.drop_frame = tk.Frame(top, bg="#ffffff", bd=2, relief="sunken")
        self.drop_frame.pack(fill="x", pady=(0, 6))

        self.drop_label = tk.Label(
            self.drop_frame,
            text="Drop Met Office radar file here",
            font=font_bold,
            bg="#ffffff",
            fg="#000000",
            anchor="center",
            pady=6
        )
        self.drop_label.pack(fill="x")

        controls = ttk.Frame(top)
        controls.pack(fill="x")

        ttk.Button(controls, text="Open File...", command=self.open_file).pack(side="left", padx=2)
        ttk.Button(controls, text="|<", width=3, command=self.first).pack(side="left", padx=2)
        ttk.Button(controls, text="<", width=3, command=self.previous).pack(side="left", padx=2)

        self.play_button = ttk.Button(controls, text="Play", width=6, command=self.toggle_play)
        self.play_button.pack(side="left", padx=2)

        ttk.Button(controls, text=">", width=3, command=self.next).pack(side="left", padx=2)
        ttk.Button(controls, text=">|", width=3, command=self.last).pack(side="left", padx=2)

        ttk.Label(controls, text="Speed:").pack(side="left", padx=(8, 2))
        self.speed = ttk.Combobox(
            controls,
            values=["100 ms", "250 ms", "500 ms", "1 second", "2 seconds"],
            state="readonly",
            width=8
        )
        self.speed.current(2)
        self.speed.pack(side="left")

        ttk.Label(controls, text="Palette:").pack(side="left", padx=(10, 2))
        self.palette_combo = ttk.Combobox(
            controls,
            values=list(self.palettes.keys()),
            state="readonly",
            width=14
        )
        self.palette_combo.current(0)
        self.palette_combo.bind("<<ComboboxSelected>>", self.change_palette)
        self.palette_combo.pack(side="left")

        ttk.Button(controls, text="Load PAL...", command=self.open_pal_file).pack(side="left", padx=2)

        self.info = ttk.Label(controls, text="No radar loaded")
        self.info.pack(side="right", padx=5)

        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=1,
            orient="horizontal",
            command=self.slider_changed
        )
        self.slider.pack(fill="x", padx=8, pady=(2, 0))

        self.scan_label = ttk.Label(
            self.root,
            text="Drop a Met Office .tar file to begin",
            anchor="center"
        )
        self.scan_label.pack(fill="x", padx=8, pady=4)

        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.drop_event)
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.drop_event)

    # ========================================================
    # HIGH-DETAIL OPTIMIZED MAP SETUP
    # ========================================================

    def build_map(self):
        self.figure = plt.Figure(figsize=(10, 8), dpi=100, facecolor="#000000")
        self.ax = self.figure.add_subplot(111, projection=radar_crs)
        self.ax.set_facecolor("#000000")
        self.ax.set_extent(UK_EXTENT, crs=ccrs.PlateCarree())

        # High resolution 10m feature set for clear detail
        RES = "10m"
        YELLOW_MAIN = "#f5e000"
        COUNTY_GRAY = "#3a3a3a"
        ROAD_DARK = "#282828"
        RIVER_BLUE = "#1a385c"

        # 1. County & Regional Lines
        counties = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale=RES,
            edgecolor=COUNTY_GRAY,
            facecolor='none'
        )
        self.ax.add_feature(counties, linewidth=0.4, linestyle=":", zorder=1)

        # 2. Major Rivers
        rivers = cfeature.NaturalEarthFeature(
            category='physical',
            name='rivers_lake_centerlines',
            scale=RES,
            edgecolor=RIVER_BLUE,
            facecolor='none'
        )
        self.ax.add_feature(rivers, linewidth=0.5, zorder=1)

        # 3. Primary Road Infrastructure
        roads = cfeature.NaturalEarthFeature(
            category='cultural',
            name='roads',
            scale=RES,
            edgecolor=ROAD_DARK,
            facecolor='none'
        )
        self.ax.add_feature(roads, linewidth=0.3, zorder=1)

        # 4. National Borders
        borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale=RES,
            edgecolor=YELLOW_MAIN,
            facecolor='none'
        )
        self.ax.add_feature(borders, linewidth=0.6, zorder=3)

        # 5. Crisp Coastlines
        coastline = cfeature.NaturalEarthFeature(
            category='physical',
            name='coastline',
            scale=RES,
            edgecolor=YELLOW_MAIN,
            facecolor='none'
        )
        self.ax.add_feature(coastline, linewidth=0.8, zorder=3)

        # Latitude / Longitude Overlay Grid
        self.gridlines = self.ax.gridlines(
            draw_labels=False,
            linewidth=0.3,
            color="#222222",
            alpha=0.5,
            linestyle="--"
        )

        self.ax.set_title(
            "Met Office UK Radar",
            color="#d0d0d0",
            fontsize=10,
            fontfamily="sans-serif"
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().configure(bg="#000000")
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(0, 6))

    # ========================================================
    # PALETTE MANAGEMENT
    # ========================================================

    def open_pal_file(self):
        filename = filedialog.askopenfilename(
            title="Open Palette File",
            filetypes=[("Palette Files", "*.pal"), ("All Files", "*")]
        )
        if filename:
            try:
                name = os.path.basename(filename)
                cmap, norm, levels = parse_pal_file(filename)
                self.palettes[name] = (cmap, norm, levels)
                self.palette_combo["values"] = list(self.palettes.keys())
                self.palette_combo.set(name)
                self.apply_palette(name)
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"Failed to load .pal file:\n\n{e}")

    def change_palette(self, event=None):
        selected = self.palette_combo.get()
        self.apply_palette(selected)

    def apply_palette(self, palette_name):
        if palette_name in self.palettes:
            self.current_cmap, self.current_norm, self.current_levels = self.palettes[palette_name]
            if self.mesh is not None:
                self.mesh.set_cmap(self.current_cmap)
                self.mesh.set_norm(self.current_norm)
                if self.colorbar is not None:
                    self.colorbar.remove()
                
                ticks = [round(lvl, 2) for lvl in self.current_levels] if len(self.current_levels) <= 15 else None
                
                self.colorbar = self.figure.colorbar(
                    self.mesh, ax=self.ax,
                    orientation="horizontal",
                    pad=0.02,
                    shrink=0.5,
                    fraction=0.02,
                    aspect=35,
                    ticks=ticks,
                    extend="max"
                )
                self.colorbar.set_label("mm/h", color="#cccccc", fontsize=8, fontfamily="sans-serif")
                self.colorbar.ax.xaxis.set_tick_params(color="#cccccc", labelcolor="#cccccc", labelsize=7)
                self.colorbar.outline.set_edgecolor("#444444")
                self.canvas.draw_idle()

    # ========================================================
    # FILE / ARCHIVE LOADING
    # ========================================================

    def open_file(self):
        filename = filedialog.askopenfilename(
            title="Open Met Office radar",
            filetypes=[
                ("Met Office radar", "*.tar *.gz"),
                ("TAR files", "*.tar"),
                ("Radar files", "*.gz"),
                ("All files", "*")
            ]
        )
        if filename:
            self.load_file(filename)

    def drop_event(self, event):
        try:
            files = self.root.tk.splitlist(event.data)
            if not files:
                return
            
            filepath = files[0]
            if filepath.lower().endswith(".pal"):
                name = os.path.basename(filepath)
                cmap, norm, levels = parse_pal_file(filepath)
                self.palettes[name] = (cmap, norm, levels)
                self.palette_combo["values"] = list(self.palettes.keys())
                self.palette_combo.set(name)
                self.apply_palette(name)
            else:
                self.load_file(filepath)
        except Exception as error:
            messagebox.showerror(APP_TITLE, str(error))

    def load_file(self, filename):
        self.stop()
        filename = os.path.abspath(filename)

        def process_archive():
            try:
                self.data_cache.clear()
                self.x_edges = None
                self.y_edges = None

                if tarfile.is_tarfile(filename):
                    if self.temp_directory:
                        self.temp_directory.cleanup()

                    self.temp_directory = tempfile.TemporaryDirectory(prefix="metoffice_radar_")

                    extracted = []
                    with tarfile.open(filename, "r:*") as archive:
                        for member in archive.getmembers():
                            name = os.path.basename(member.name)
                            if not member.isfile() or not name.endswith(".dat.gz") or "1km-composite" not in name:
                                continue

                            destination = os.path.join(self.temp_directory.name, name)
                            source = archive.extractfile(member)
                            if source:
                                with open(destination, "wb") as output:
                                    shutil.copyfileobj(source, output)
                                extracted.append(destination)
                    files = extracted

                elif filename.endswith(".dat.gz"):
                    files = [filename]
                else:
                    raise ValueError("Please drop a Met Office .tar, .dat.gz, or .pal file.")

                files.sort(key=lambda f: get_timestamp(f) or "")
                if not files:
                    raise ValueError("No Met Office 1 km radar scans were found.")

                self.root.after(0, lambda: self._finish_loading(files, filename))

            except Exception as error:
                self.root.after(0, lambda: messagebox.showerror(APP_TITLE, f"Could not open radar file:\n\n{error}"))

        self.info.config(text="Extracting scans...")
        threading.Thread(target=process_archive, daemon=True).start()

    def _finish_loading(self, files, filename):
        self.files = files
        self.index = 0
        self.slider.configure(to=len(self.files) - 1)
        self.info.config(text=f"{len(self.files)} scans loaded")
        self.drop_label.config(text=os.path.basename(filename))
        self.load_scan(0)

    # ========================================================
    # SCAN PROCESSING & DATA NORMALIZATION
    # ========================================================

    def read_scan(self, index):
        if index in self.data_cache:
            self.data_cache.move_to_end(index)
            return self.data_cache[index]

        filename = self.files[index]

        try:
            cube = iris.load_cube(filename)
        except Exception:
            if filename.endswith(".gz"):
                temp_dir = self.temp_directory.name if self.temp_directory else tempfile.gettempdir()
                temp_dat = os.path.join(temp_dir, os.path.basename(filename)[:-3])

                if not os.path.exists(temp_dat):
                    with gzip.open(filename, "rb") as source, open(temp_dat, "wb") as destination:
                        shutil.copyfileobj(source, destination)

                cube = iris.load_cube(temp_dat)
            else:
                raise

        data = cube.data
        if np.ma.isMaskedArray(data):
            data = data.filled(np.nan)

        data = np.asarray(data, dtype=np.float32)

        valid_mask = ~np.isnan(data)
        max_val = np.nanmax(data) if np.any(valid_mask) else 0

        if max_val > 128.0:
            if max_val > 1000.0:
                data = data / 100.0
            else:
                data = data / 32.0

        data[data < 0.1] = np.nan

        if self.x_edges is None:
            x_pts = cube.coord("projection_x_coordinate").points
            y_pts = cube.coord("projection_y_coordinate").points

            dx = abs(x_pts[1] - x_pts[0]) if len(x_pts) > 1 else 1000.0
            dy = abs(y_pts[1] - y_pts[0]) if len(y_pts) > 1 else 1000.0

            self.x_edges = np.append(x_pts - dx / 2.0, x_pts[-1] + dx / 2.0)
            self.y_edges = np.append(y_pts - dy / 2.0, y_pts[-1] + dy / 2.0)

        self.data_cache[index] = data
        if len(self.data_cache) > self.max_cache:
            self.data_cache.popitem(last=False)

        return data

    # ========================================================
    # SCAN DISPLAY
    # ========================================================

    def load_scan(self, index):
        if not self.files:
            return

        index = max(0, min(index, len(self.files) - 1))
        self.index = index

        try:
            data = self.read_scan(index)

            if self.mesh is None:
                self.mesh = self.ax.pcolormesh(
                    self.x_edges, self.y_edges, data,
                    transform=radar_crs,
                    cmap=self.current_cmap,
                    norm=self.current_norm,
                    zorder=2,
                    rasterized=True
                )
                
                ticks = [round(lvl, 2) for lvl in self.current_levels] if len(self.current_levels) <= 15 else None

                self.colorbar = self.figure.colorbar(
                    self.mesh, ax=self.ax,
                    orientation="horizontal",
                    pad=0.02,
                    shrink=0.5,
                    fraction=0.02,
                    aspect=35,
                    ticks=ticks,
                    extend="max"
                )
                self.colorbar.set_label("mm/h", color="#cccccc", fontsize=8, fontfamily="sans-serif")
                self.colorbar.ax.xaxis.set_tick_params(color="#cccccc", labelcolor="#cccccc", labelsize=7)
                self.colorbar.outline.set_edgecolor("#444444")
            else:
                self.mesh.set_array(data.ravel())

            timestamp = format_timestamp(self.files[index])
            self.ax.set_title(f"Met Office UK Radar — {timestamp}", color="#d0d0d0", fontsize=10, fontfamily="sans-serif")
            self.slider.set(index)
            self.scan_label.config(
                text=f"Scan {index + 1} of {len(self.files)}    •    {timestamp}"
            )

            self.canvas.draw_idle()

        except Exception as error:
            messagebox.showerror(APP_TITLE, f"Could not load radar scan:\n\n{error}")

    # ========================================================
    # CONTROLS & NAVIGATION
    # ========================================================

    def slider_changed(self, value):
        if self.playing or not self.files:
            return
        index = int(float(value))
        if index != self.index:
            self.load_scan(index)

    def first(self):
        self.stop()
        self.load_scan(0)

    def last(self):
        self.stop()
        self.load_scan(len(self.files) - 1)

    def previous(self):
        self.stop()
        self.load_scan(max(0, self.index - 1))

    def next(self):
        self.stop()
        self.load_scan(min(len(self.files) - 1, self.index + 1))

    def toggle_play(self):
        if self.playing:
            self.stop()
        else:
            self.playing = True
            self.play_button.config(text="Pause")
            self.play_animation()

    def stop(self):
        self.playing = False
        self.play_button.config(text="Play")
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def play_animation(self):
        if not self.playing:
            return

        next_index = self.index + 1
        if next_index >= len(self.files):
            next_index = 0

        self.load_scan(next_index)

        delays = {
            "100 ms": 100,
            "250 ms": 250,
            "500 ms": 500,
            "1 second": 1000,
            "2 seconds": 2000
        }
        delay = delays.get(self.speed.get(), 500)
        self.after_id = self.root.after(delay, self.play_animation)

    def close(self):
        self.stop()
        if self.temp_directory:
            self.temp_directory.cleanup()
        plt.close(self.figure)
        self.root.destroy()


# ============================================================
# EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = RadarViewer(root)
    root.mainloop()
