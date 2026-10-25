"""Render a 3D SVM decision-surface explainer from the fitted experiment."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from PIL import Image, ImageDraw, ImageFont

from experiment import fit_model, run_experiment


ROOT = Path(__file__).parent
SIZE = (720, 900)
FPS = 20
DURATION = 27
INK = "#eaf4fc"
BLUE = "#58b8ff"
ORANGE = "#ffaf74"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def states():
    runs = run_experiment()
    circles, moons = runs["circles"], runs["moons"]
    result = [
        ("circles_linear", circles, circles["models"]["linear"],
         circles["metrics"]["linear"], "LINEAR · CIRCLES", "58.0% test accuracy"),
        ("circles_rbf", circles, circles["models"]["rbf"],
         circles["metrics"]["rbf"], "RBF · CIRCLES", "96.7% test accuracy"),
    ]
    for gamma, label in ((.1, "0.1"), (1, "1"), (10, "10")):
        model, metrics = fit_model(moons["X"], moons["y"],
                                   moons["train_idx"], moons["test_idx"],
                                   "rbf", gamma=gamma)
        result.append((f"moons_{label}", moons, model, metrics,
                       f"RBF · MOONS · γ {label}",
                       f"{metrics['accuracy_test']:.1%} test accuracy"))
    return result


def score_surface(run, model):
    X = run["X"]
    lo, hi = X.min(axis=0) - .35, X.max(axis=0) + .35
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 46),
                         np.linspace(lo[1], hi[1], 46))
    score = model.decision_function(np.column_stack((xx.ravel(), yy.ravel())))
    return xx, yy, score.reshape(xx.shape), lo, hi


def chart(run, surface, azimuth):
    xx, yy, score, lo, hi = surface
    fig = plt.figure(figsize=(6.4, 5.6), dpi=100, facecolor="#102235")
    ax = fig.add_subplot(111, projection="3d", facecolor="#102235")
    ax.view_init(elev=27, azim=azimuth)
    ax.set_proj_type("persp", focal_length=1.15)
    ax.set_box_aspect((1, 1, .72), zoom=.88)

    blue, orange = np.array(to_rgba(BLUE)), np.array(to_rgba(ORANGE))
    mix = 1 / (1 + np.exp(-2.3 * score))
    colors = blue[None, None, :] * (1 - mix[..., None]) + orange[None, None, :] * mix[..., None]
    colors[..., 3] = .82
    ax.plot_surface(xx, yy, np.clip(score, -2.2, 2.2), facecolors=colors,
                    rstride=1, cstride=1, shade=False, linewidth=0,
                    antialiased=False, zorder=1)
    ax.contour(xx, yy, score, zdir="z", offset=-2.65, levels=[0],
               colors=[INK], linewidths=2.5)

    X, y = run["X"], run["y"]
    for label, color in ((0, BLUE), (1, ORANGE)):
        train = run["train_idx"][y[run["train_idx"]] == label]
        test = run["test_idx"][y[run["test_idx"]] == label]
        ax.scatter(X[train, 0], X[train, 1], np.full(len(train), -2.65),
                   c=color, s=8, depthshade=False, edgecolors="none")
        ax.scatter(X[test, 0], X[test, 1], np.full(len(test), -2.65),
                   facecolors="none", edgecolors=color, s=24,
                   linewidths=.8, depthshade=False)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(-2.65, 2.65))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.07, .14, .22, .75))
        axis.pane.set_edgecolor((.25, .44, .56, .5))
        axis.line.set_color((.4, .65, .8, .45))
        axis._axinfo["grid"]["color"] = (.35, .55, .65, .15)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), "RGBA").convert("RGB")
    plt.close(fig)
    return img


def current_state(t):
    if t < 3:
        return 0, "What does an SVM see?", "A boundary has a 3D shape."
    if t < 8:
        return 0, "A straight surface", "87 / 150 test points correct"
    if t < 13:
        return 1, "A curved surface", "145 / 150 test points correct"
    if t < 16:
        return 2, "Change one parameter", "γ = 0.1  ·  131 / 150 correct"
    if t < 19:
        return 3, "Change one parameter", "γ = 1  ·  145 / 150 correct"
    if t < 22:
        return 4, "Change one parameter", "γ = 10  ·  143 / 150 correct"
    return 3, "Score shapes the surface", "Boundary = where score is zero"


def compose(t, state, plot):
    index, title, sub = current_state(t)
    img = Image.new("RGB", SIZE, "#081521")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((28, 18, 692, 67), radius=16, fill="#183247")
    draw.text((49, 29), "SVM  /  DECISION SURFACE", font=font(21, True), fill=BLUE)
    draw.text((29, 87), title, font=font(40, True), fill=INK)
    draw.text((30, 146), sub, font=font(28, True), fill=ORANGE)
    draw.rounded_rectangle((28, 194, 692, 762), radius=22, fill="#102235", outline="#35546a", width=2)
    img.paste(plot, (40, 200))
    draw = ImageDraw.Draw(img)
    # A compact instrument frame: bolts, rulers and a model readout.
    for x in (45, 675):
        for y in (211, 745):
            draw.ellipse((x-3, y-3, x+3, y+3), fill="#81a5bd")
    for n in range(25):
        x = 72 + n * 24
        draw.line((x, 754, x, 749 if n % 5 else 743), fill="#42647a", width=2)
    draw.rounded_rectangle((29, 780, 691, 842), radius=15, fill="#173246")
    model_label = state[4] if t < 22 else "SYNTHETIC 2D · ONE SPLIT"
    draw.text((48, 792), model_label, font=font(23, True), fill=BLUE)
    draw.text((29, 855), "Floor: data + boundary  ·  height: score (clipped ±2.2)", font=font(18), fill="#b0c7d4")
    draw.rounded_rectangle((29, 887, 691, 892), radius=2, fill="#25445a")
    draw.rounded_rectangle((29, 887, 29 + int(662 * t / DURATION), 892), radius=2, fill=BLUE)
    return img


def render(output):
    state_list = states()
    surfaces = [score_surface(run, model) for _, run, model, *_ in state_list]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), SIZE, fps=FPS,
                                         codec="libx264", pix_fmt_out="yuv420p",
                                         quality=7, macro_block_size=2)
    writer.send(None)
    previews = []
    last_plot = None
    last_index = None
    last_angle = None
    try:
        for frame in range(DURATION * FPS):
            t = frame / FPS
            index, _, _ = current_state(t)
            # A slow orbit reveals the decision surface without moving data.
            angle = round(-58 + 17 * np.sin(2 * np.pi * t / 13), 1)
            if index != last_index or angle != last_angle:
                last_plot = chart(state_list[index][1], surfaces[index], angle)
                last_index, last_angle = index, angle
            image = compose(t, state_list[index], last_plot)
            writer.send(image.tobytes())
            if frame in (20, 90, 180, 280, 340, 400, 480):
                previews.append(image.resize((288, 360)))
    finally:
        writer.close()
    cover = compose(9.0, state_list[1], chart(state_list[1][1], surfaces[1], -52))
    cover.save(output.with_name("kernel_story_3d_cover.png"), optimize=True)
    sheet = Image.new("RGB", (1152, 720), "#081521")
    for n, image in enumerate(previews):
        sheet.paste(image, ((n % 4) * 288, (n // 4) * 360))
    sheet.save(output.with_name("kernel_story_3d_preview.png"), optimize=True)
    print(f"Video: {output} ({DURATION}s, {SIZE[0]}×{SIZE[1]}, {FPS}fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "kernel_story_3d.mp4")
    render(parser.parse_args().output)
