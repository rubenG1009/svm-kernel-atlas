(() => {
  "use strict";

  const data = window.SVM_DATA;
  const loading = document.getElementById("loading");
  if (!data || !window.Plotly) {
    loading.textContent = "Dashboard assets could not load. Rebuild data.js and refresh.";
    return;
  }

  const $ = (id) => document.getElementById(id);
  const plotA = $("plot-a");
  const plotB = $("plot-b");
  const blue = "#5dbdff";
  const orange = "#ffad70";
  const floor = data.floorZ;
  const clip = data.scoreClip;
  const defaultCamera = { eye: { x: 1.35, y: 1.3, z: 1.08 }, up: { x: 0, y: 0, z: 1 } };
  const cameraRange = { min: 1.25, max: 2.75 };
  const state = {
    dataset: "circles", kernel: "rbf", gamma: "1", view: "3d",
    compare: false, compareKey: "linear", camera: structuredClone(defaultCamera),
    showSurface: true, showTrain: true, showTest: true,
    showErrors: false, showSupport: false,
  };
  const params = new URLSearchParams(window.location.search);
  if (["circles", "moons"].includes(params.get("dataset"))) state.dataset = params.get("dataset");
  if (["linear", "rbf", "poly", "laplacian", "rational_quadratic", "cosine"].includes(params.get("kernel"))) state.kernel = params.get("kernel");
  if (["0.1", "1", "10"].includes(params.get("gamma"))) state.gamma = params.get("gamma");
  if (["2d", "3d"].includes(params.get("view"))) state.view = params.get("view");
  if (params.has("compare") && data.datasets[state.dataset].models[params.get("compare")]) {
    state.compare = true;
    state.compareKey = params.get("compare");
  }
  let plotAReady = false;
  let plotBReady = false;
  let syncingCamera = false;
  let renderQueue = Promise.resolve();

  function key() { return ["linear", "cosine"].includes(state.kernel) ? state.kernel : `${state.kernel}:${state.gamma}`; }
  function scenario(modelKey) { return data.datasets[state.dataset].models[modelKey]; }
  function label(modelKey) {
    if (modelKey === "linear") return "Linear";
    if (modelKey === "cosine") return "Cosine";
    const [kernel, gamma] = modelKey.split(":");
    const names = { rbf: "RBF", poly: "Polynomial", laplacian: "Laplacian", rational_quadratic: "Rational quadratic" };
    return `${names[kernel]}, γ ${gamma}`;
  }
  function percent(value) { return `${(value * 100).toFixed(1)}%`; }
  function signed(value) { return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`; }

  function boundedCamera(raw) {
    const camera = structuredClone(raw);
    const eye = camera.eye;
    if (!eye) return { camera, changed: false };
    const distance = Math.hypot(eye.x, eye.y, eye.z);
    if (!Number.isFinite(distance) || distance === 0) return { camera: structuredClone(defaultCamera), changed: true };
    const bounded = Math.max(cameraRange.min, Math.min(cameraRange.max, distance));
    if (Math.abs(distance - bounded) < .001) return { camera, changed: false };
    const factor = bounded / distance;
    camera.eye = { x: eye.x * factor, y: eye.y * factor, z: eye.z * factor };
    return { camera, changed: true };
  }

  function drawHero() {
    const svg = $("hero-plot");
    if (!svg) return;
    const dataset = data.datasets.circles;
    const model = dataset.models["rbf:1"];
    const [[minX, minY], [maxX, maxY]] = dataset.bounds;
    const scale = Math.min(370 / (maxX - minX), 245 / (maxY - minY));
    const x = value => 220 + (value - (minX + maxX) / 2) * scale;
    const y = value => 160 - (value - (minY + maxY) / 2) * scale;
    const add = (name, attributes) => {
      const element = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [attribute, value] of Object.entries(attributes)) element.setAttribute(attribute, value);
      svg.append(element);
      return element;
    };
    for (let grid = -4; grid <= 4; grid++) {
      const position = 220 + grid * 40;
      add("line", { x1: position, x2: position, y1: 16, y2: 304, class: "art-grid" });
    }
    for (let grid = -3; grid <= 3; grid++) {
      const position = 160 + grid * 40;
      add("line", { x1: 16, x2: 424, y1: position, y2: position, class: "art-grid" });
    }
    model.points.filter((_, index) => index % 5 === 0).forEach(point => {
      add("circle", { cx: x(point.x), cy: y(point.y), r: point.split === "test" ? 3.2 : 2.4, class: `art-point-${point.true}` });
    });
    for (const line of model.boundary) {
      if (line.length < 2) continue;
      add("path", { d: line.map(([px, py], index) => `${index ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`).join(" "), class: "art-boundary" });
    }
    const labelX = add("text", { x: 411, y: 302, class: "art-label" });
    labelX.textContent = "X₁";
    const labelY = add("text", { x: 23, y: 28, class: "art-label" });
    labelY.textContent = "X₂";
  }

  function updateControls() {
    document.querySelectorAll("[data-dataset]").forEach(button => {
      const active = button.dataset.dataset === state.dataset;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-kernel]").forEach(button => {
      const active = button.dataset.kernel === state.kernel;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === state.view;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    $("gamma").disabled = ["linear", "cosine"].includes(state.kernel);
    $("gamma").value = state.gamma;
    $("kernel-explainer").textContent = {
      linear: "A straight boundary in the standardized feature plane.",
      rbf: "Squared Euclidean distance creates local curved regions.",
      poly: "A degree-three polynomial combines feature interactions.",
      laplacian: "Manhattan distance changes the geometry of local similarity.",
      rational_quadratic: "A multi-scale similarity with shape parameter α = 1.",
      cosine: "Compares direction, largely discarding distance from the center.",
    }[state.kernel];
    $("compare").checked = state.compare;
    $("compare-options").hidden = !state.compare;
    $("compare-kernel").value = state.compareKey;
    $("show-surface").checked = state.showSurface;
    $("show-train").checked = state.showTrain;
    $("show-test").checked = state.showTest;
    $("show-errors").checked = state.showErrors;
    $("show-support").checked = state.showSupport;
    $("plot-grid").classList.toggle("comparing", state.compare);
    $("secondary-unit").hidden = !state.compare;
    $("reset-camera").disabled = state.view !== "3d";
    $("view-instructions").innerHTML = state.view === "3d"
      ? "Drag to orbit <span>↗</span> Scroll to zoom <span>↗</span> Hover to inspect"
      : "Drag to pan <span>↗</span> Scroll to zoom <span>↗</span> Hover to inspect";
    $("chart-note").textContent = state.view === "3d"
      ? "Height = signed decision score, clipped at ±2.2"
      : "Color = signed decision score, clipped at ±2.2";
  }

  function updateReadout() {
    const model = scenario(key());
    const metrics = model.metrics;
    $("chart-title").textContent = `${state.dataset === "circles" ? "Circles" : "Moons"} · ${label(key())}`;
    $("test-accuracy").textContent = percent(metrics.accuracy_test);
    $("correct-count").textContent = `${metrics.correct_test} / ${metrics.test_total} correct`;
    $("train-accuracy").textContent = percent(metrics.accuracy_train);
    $("support-count").textContent = String(metrics.support_vectors);
    const c = metrics.confusion;
    $("confusion-matrix").innerHTML = `<span class="axis"></span><span class="axis">PRED 0</span><span class="axis">PRED 1</span><span class="axis">TRUE 0</span><span class="cell diag">${c[0][0]}</span><span class="cell">${c[0][1]}</span><span class="axis">TRUE 1</span><span class="cell">${c[1][0]}</span><span class="cell diag">${c[1][1]}</span>`;
    $("comparison-readout").hidden = !state.compare;
    if (state.compare) {
      const other = scenario(state.compareKey).metrics;
      const points = metrics.correct_test - other.correct_test;
      const delta = 100 * (metrics.accuracy_test - other.accuracy_test);
      $("comparison-text").textContent = `${label(key())}: ${metrics.correct_test}/${metrics.test_total} correct. ${label(state.compareKey)}: ${other.correct_test}/${other.test_total}. Selected model: ${signed(delta)} percentage points and ${points >= 0 ? "+" : ""}${points} correct predictions.`;
    }
  }

  function boundaryTraces(model, threeD) {
    const x = [], y = [], z = [];
    for (const line of model.boundary) {
      for (const point of line) {
        x.push(point[0]); y.push(point[1]); if (threeD) z.push(floor + .02);
      }
      x.push(null); y.push(null); if (threeD) z.push(null);
    }
    return [{
      type: threeD ? "scatter3d" : "scatter", mode: "lines", x, y,
      ...(threeD ? { z } : {}), line: { color: "#f1f8fc", width: threeD ? 7 : 3 },
      hoverinfo: "skip", showlegend: false, name: "Score = 0",
    }];
  }

  function pointTraces(model, threeD) {
    const traces = [];
    for (const split of ["train", "test"]) {
      if ((split === "train" && !state.showTrain) || (split === "test" && !state.showTest)) continue;
      for (const cls of [0, 1]) {
        const points = model.points.filter(point => point.split === split && point.true === cls);
        traces.push({
          type: threeD ? "scatter3d" : "scatter", mode: "markers",
          x: points.map(point => point.x), y: points.map(point => point.y),
          ...(threeD ? { z: points.map(() => floor + .05) } : {}),
          customdata: points.map(point => [point.id, point.true, point.pred, point.score, point.split, point.support]),
          marker: {
            color: cls === 0 ? blue : orange,
            size: split === "test" ? (threeD ? 5.5 : 9) : (threeD ? 3.1 : 5),
            opacity: split === "test" ? 1 : .8,
            symbol: split === "test" ? "circle-open" : "circle",
            line: { color: cls === 0 ? blue : orange, width: split === "test" ? 2 : 0 },
          },
          hovertemplate: "Point %{customdata[0]}<br>True class %{customdata[1]} · predicted %{customdata[2]}<br>Decision score %{customdata[3]:.3f}<br>%{customdata[4]}<extra></extra>",
          name: `Class ${cls} · ${split}`, showlegend: false,
        });
      }
    }
    if (state.showErrors && state.showTest) {
      const wrong = model.points.filter(point => point.split === "test" && point.true !== point.pred);
      traces.push({ type: threeD ? "scatter3d" : "scatter", mode: "markers",
        x: wrong.map(point => point.x), y: wrong.map(point => point.y),
        ...(threeD ? { z: wrong.map(() => floor + .12) } : {}),
        marker: { symbol: "x", size: threeD ? 7 : 12, color: "#ff586c", line: { width: 2, color: "#ff586c" } },
        hoverinfo: "skip", showlegend: false, name: "Test error" });
    }
    if (state.showSupport && state.showTrain) {
      const support = model.points.filter(point => point.support);
      traces.push({ type: threeD ? "scatter3d" : "scatter", mode: "markers",
        x: support.map(point => point.x), y: support.map(point => point.y),
        ...(threeD ? { z: support.map(() => floor + .1) } : {}),
        marker: { symbol: "diamond-open", size: threeD ? 6 : 10, color: "#e9dd88", line: { width: 1, color: "#e9dd88" } },
        hoverinfo: "skip", showlegend: false, name: "Support vector" });
    }
    return traces;
  }

  function traces(modelKey) {
    const dataset = data.datasets[state.dataset];
    const model = dataset.models[modelKey];
    const threeD = state.view === "3d";
    const result = [];
    if (state.showSurface) {
      const z = model.scoreGrid.map(row => row.map(score => Math.max(-clip, Math.min(clip, score))));
      const base = {
        x: dataset.xGrid, y: dataset.yGrid, z,
        colorscale: [[0, "#1b6597"], [.45, "#4384a5"], [.5, "#173143"], [.55, "#a7785e"], [1, "#f4a96c"]],
        zmin: -clip, zmax: clip, showscale: false, hoverinfo: "skip", showlegend: false,
      };
      if (threeD) result.push({ ...base, type: "surface", surfacecolor: z, opacity: .69 });
      else result.push({ ...base, type: "heatmap", opacity: .65 });
    }
    result.push(...boundaryTraces(model, threeD), ...pointTraces(model, threeD));
    return result;
  }

  function axis(title, range) {
    return { title: { text: title, font: { size: 11, color: "#b9d4e1" } }, range,
      color: "#9ab6c4", gridcolor: "#25475c", zerolinecolor: "#36536a",
      backgroundcolor: "#102638", showbackground: true, tickfont: { size: 10 } };
  }

  function layout() {
    const bounds = data.datasets[state.dataset].bounds;
    const shared = { paper_bgcolor: "#102638", plot_bgcolor: "#102638", font: { color: "#c0d5e1", family: "Inter, sans-serif" }, margin: { l: 7, r: 7, t: 10, b: 8 }, showlegend: false, uirevision: state.view };
    if (state.view === "3d") return { ...shared,
      scene: { xaxis: axis("x₁", [bounds[0][0], bounds[1][0]]),
        yaxis: axis("x₂", [bounds[0][1], bounds[1][1]]),
        zaxis: axis("signed score", [floor, 2.65]),
        camera: structuredClone(state.camera), dragmode: "orbit", uirevision: "shared-camera",
        aspectmode: "manual", aspectratio: { x: 1, y: 1, z: .82 }, bgcolor: "#102638" },
    };
    return { ...shared, margin: { l: 45, r: 20, t: 14, b: 45 },
      xaxis: { title: "x₁", range: [bounds[0][0], bounds[1][0]], color: "#9ab6c4", gridcolor: "#25475c", zeroline: false, scaleanchor: "y" },
      yaxis: { title: "x₂", range: [bounds[0][1], bounds[1][1]], color: "#9ab6c4", gridcolor: "#25475c", zeroline: false },
    };
  }

  function inspectPoint(event) {
    const point = event.points?.[0]?.customdata;
    if (!point) return;
    const [id, truth, prediction, score, split, support] = point;
    $("point-detail").textContent = `Point ${id} · ${split}. True class ${truth}; predicted class ${prediction}. Decision score ${Number(score).toFixed(3)}. ${truth === prediction ? "Correct prediction." : "Incorrect prediction."}${support ? " Support vector." : ""}`;
  }

  function attachPlotEvents(div, other) {
    div.on("plotly_hover", inspectPoint);
    div.on("plotly_relayout", async update => {
      if (state.view !== "3d" || syncingCamera || !update) return;
      if (!Object.keys(update).some(field => field.startsWith("scene.camera"))) return;
      const camera = div.layout?.scene?.camera;
      if (!camera) return;
      const bounded = boundedCamera(camera);
      state.camera = bounded.camera;
      if (!bounded.changed && (!state.compare || !other.layout?.scene)) return;
      syncingCamera = true;
      try {
        if (bounded.changed) await Plotly.relayout(div, { "scene.camera": state.camera });
        if (state.compare && other.layout?.scene) await Plotly.relayout(other, { "scene.camera": state.camera });
      }
      finally { syncingCamera = false; }
    });
  }

  async function render() {
    updateControls();
    updateReadout();
    loading.hidden = false;
    try {
      await Plotly.react(plotA, traces(key()), layout(), { responsive: true, displaylogo: false, scrollZoom: true });
      if (!plotAReady) { attachPlotEvents(plotA, plotB); plotAReady = true; }
      if (state.compare) {
        await Plotly.react(plotB, traces(state.compareKey), layout(), { responsive: true, displaylogo: false, scrollZoom: true });
        if (!plotBReady) { attachPlotEvents(plotB, plotA); plotBReady = true; }
      }
      await new Promise(resolve => requestAnimationFrame(resolve));
      Plotly.Plots.resize(plotA);
      if (state.compare) Plotly.Plots.resize(plotB);
      loading.hidden = true;
    } catch (error) {
      loading.textContent = `Could not draw the chart: ${error.message}`;
      console.error(error);
    }
  }

  function scheduleRender() { renderQueue = renderQueue.then(render).catch(error => { loading.textContent = error.message; console.error(error); }); }
  document.querySelectorAll("[data-dataset]").forEach(button => button.addEventListener("click", () => { state.dataset = button.dataset.dataset; scheduleRender(); }));
  document.querySelectorAll("[data-kernel]").forEach(button => button.addEventListener("click", () => { state.kernel = button.dataset.kernel; scheduleRender(); }));
  document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => { state.view = button.dataset.view; scheduleRender(); }));
  $("gamma").addEventListener("change", event => { state.gamma = event.target.value; scheduleRender(); });
  $("compare").addEventListener("change", event => { state.compare = event.target.checked; scheduleRender(); });
  $("compare-kernel").addEventListener("change", event => { state.compareKey = event.target.value; scheduleRender(); });
  for (const [id, property] of [["show-surface", "showSurface"], ["show-train", "showTrain"], ["show-test", "showTest"], ["show-errors", "showErrors"], ["show-support", "showSupport"]]) {
    $(id).addEventListener("change", event => { state[property] = event.target.checked; scheduleRender(); });
  }
  $("reset-camera").addEventListener("click", async () => {
    state.camera = structuredClone(defaultCamera);
    if (state.view !== "3d") return;
    syncingCamera = true;
    try {
      await Plotly.relayout(plotA, { "scene.camera": state.camera });
      if (state.compare && plotBReady) await Plotly.relayout(plotB, { "scene.camera": state.camera });
    } finally { syncingCamera = false; }
  });
  $("export-image").addEventListener("click", () => Plotly.downloadImage(plotA, {
    format: "png", width: 1200, height: 900, filename: `svm-${state.dataset}-${key().replace(":", "-")}`,
  }));
  drawHero();
  scheduleRender();
})();
