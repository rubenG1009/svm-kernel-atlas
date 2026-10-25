"""Render a silent, captioned 4:5 video from the experiment's actual models."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiment import SEED, fit_model, run_experiment


ROOT = Path(__file__).parent
OUT = ROOT / "output"
SIZE = (720, 900)
FPS = 20
DURATION = 26
BG = "#091320"
CARD = "#111f30"
WHITE = "#f2f6fb"
MUTED = "#aac0d0"
BLUE = "#61b6f8"
ORANGE = "#ffaf71"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def chart(run, model):
    """Draw the fitted boundary and the same fixed train/test observations."""
    X, y = run["X"], run["y"]
    lo, hi = X.min(axis=0) - .35, X.max(axis=0) + .35
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 240),
                         np.linspace(lo[1], hi[1], 240))
    score = model.decision_function(np.column_stack((xx.ravel(), yy.ravel())))
    score = score.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.4, 5.5), dpi=100)
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.contourf(xx, yy, score, levels=[-1e9, 0, 1e9],
                colors=["#17314a", "#493025"], alpha=.95)
    ax.contour(xx, yy, score, levels=[0], colors=[WHITE], linewidths=2.4)
    for label, color in enumerate((BLUE, ORANGE)):
        train = run["train_idx"][y[run["train_idx"]] == label]
        test = run["test_idx"][y[run["test_idx"]] == label]
        ax.scatter(X[train, 0], X[train, 1], c=color, s=18,
                   alpha=.85, edgecolors="none", rasterized=True)
        ax.scatter(X[test, 0], X[test, 1], facecolors="none",
                   edgecolors=color, s=48, linewidths=1.4, rasterized=True)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.035, right=.965, bottom=.035, top=.965)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(rgba, "RGBA").convert("RGB")


def panel_for(time):
    if time < 3:
        return "circles_linear", "One dataset.", "Different boundaries.", "INTRO"
    if time < 8:
        return "circles_linear", "LINEAR KERNEL", "58.0% test accuracy", "CIRCLES"
    if time < 13:
        return "circles_rbf", "RBF KERNEL", "96.7% test accuracy", "CIRCLES"
    if time < 16:
        return "moons_01", "RBF · gamma 0.1", "87.3% test accuracy", "MOONS"
    if time < 19:
        return "moons_1", "RBF · gamma 1", "96.7% test accuracy", "MOONS"
    if time < 22:
        return "moons_10", "RBF · gamma 10", "95.3% test accuracy", "MOONS"
    return "moons_1", "Higher gamma ≠ better here", "Check held-out data.", "TAKEAWAY"


def make_frame(time, panels):
    key, heading, result, _ = panel_for(time)
    img = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((30, 20, 690, 63), radius=15, fill="#1a3347")
    draw.text((51, 30), "SVM  /  KERNEL LAB", font=font(20, True), fill=BLUE)
    draw.text((30, 86), heading, font=font(47, True), fill=WHITE)
    draw.text((30, 150), result, font=font(30, True), fill=ORANGE if key == "circles_linear" else BLUE)

    # The wipe switches between real fitted states. It does not invent an
    # intermediate classifier or move any observation.
    plot = panels[key]
    for switch, previous in ((8, "circles_linear"), (16, "moons_01"),
                             (19, "moons_1"), (22, "moons_10")):
        if switch <= time < switch + .55:
            width = int(plot.width * (time - switch) / .55)
            reveal = panels[previous].copy()
            reveal.paste(plot.crop((0, 0, width, plot.height)), (0, 0))
            plot = reveal
            break
    img.paste(plot, (40, 205))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((30, 774, 690, 840), radius=16, fill="#152b3e")
    if time < 13:
        note = "Same 150 test points · +58 correct with RBF" if time >= 8 else "Filled = train    ○ = test    line = boundary"
    elif time < 22:
        note = "Only gamma changes · same train/test split"
    else:
        note = "Synthetic 2D · one split · seed 42"
    draw.text((49, 793), note, font=font(22), fill=WHITE)
    draw.text((30, 854), "C = 1  ·  500 samples/dataset  ·  70/30 split", font=font(18), fill=MUTED)
    draw.rounded_rectangle((30, 887, 690, 892), radius=2, fill="#244058")
    draw.rounded_rectangle((30, 887, 30 + int(660 * time / DURATION), 892), radius=2, fill=BLUE)
    return img


def render(output):
    runs = run_experiment()
    circles, moons = runs["circles"], runs["moons"]
    panels = {
        "circles_linear": chart(circles, circles["models"]["linear"]),
        "circles_rbf": chart(circles, circles["models"]["rbf"]),
    }
    for gamma, key in ((.1, "moons_01"), (1, "moons_1"), (10, "moons_10")):
        model, _ = fit_model(moons["X"], moons["y"], moons["train_idx"],
                             moons["test_idx"], "rbf", gamma=gamma)
        panels[key] = chart(moons, model)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), SIZE, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
        quality=7, macro_block_size=2,
    )
    writer.send(None)
    snapshots = {}
    try:
        for number in range(DURATION * FPS):
            time = number / FPS
            frame = make_frame(time, panels)
            writer.send(frame.tobytes())
            if number in (20, 100, 200, 280, 340, 400, 480):
                snapshots[number] = frame.resize((288, 360))
    finally:
        writer.close()

    cover = make_frame(9, panels)
    cover.save(output.with_name("kernel_story_cover.png"), optimize=True)
    sheet = Image.new("RGB", (288 * 4, 360 * 2), BG)
    for i, frame in enumerate(snapshots.values()):
        sheet.paste(frame, ((i % 4) * 288, (i // 4) * 360))
    sheet.save(output.with_name("kernel_story_preview.png"), optimize=True)
    print(f"Video: {output} ({DURATION}s, {SIZE[0]}×{SIZE[1]}, {FPS} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "kernel_story.mp4")
    render(parser.parse_args().output)
