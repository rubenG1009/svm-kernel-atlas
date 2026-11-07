(() => {
  "use strict";
  const data = window.SVM_DATA, plot = document.getElementById("guide-plot"), otherPlot = document.getElementById("guide-plot-b");
  if (!data || !window.Plotly || !plot) return;
  const $ = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const kernels = ["linear", "rbf", "poly", "laplacian", "rational_quadratic", "cosine"];
  const names = { linear: "Linear", rbf: "RBF", poly: "Polynomial", laplacian: "Laplacian", rational_quadratic: "Rational quadratic", cosine: "Cosine" };
  const state = { dataset: ["circles", "moons"].includes(params.get("dataset")) ? params.get("dataset") : "circles",
    kernel: kernels.includes(params.get("kernel")) ? params.get("kernel") : "rbf",
    gamma: ["0.1", "1", "10"].includes(params.get("gamma")) ? params.get("gamma") : "1",
    view: params.get("view") === "2d" ? "2d" : "3d", compare: "", step: ["0", "1", "2", "3"].includes(params.get("step")) ? Number(params.get("step")) : 2, pointId: null, pointSource: "primary", timer: null,
    showTrain: params.get("train") !== "0", showTest: params.get("test") !== "0", showErrors: params.get("errors") === "1" };
  if (data.datasets[state.dataset].models[params.get("compare")]) state.compare = params.get("compare");
  const dataset = () => data.datasets[state.dataset];
  const key = () => ["linear", "cosine"].includes(state.kernel) ? state.kernel : `${state.kernel}:${state.gamma}`;
  const model = () => dataset().models[key()];
  const label = modelKey => { const [kernel, gamma] = modelKey.split(":"); return names[kernel] + (gamma ? ` · γ ${gamma}` : ""); };
  const steps = [
    ["Start with the points.", "Two classes live on the same flat input plane. Each dot is one observation."],
    ["Find the boundary.", "The white line marks where the decision score is zero. The sign on either side determines the predicted class."],
    ["Reveal the surface.", "In 3D, height is the model’s signed decision score. In 2D, color shows that score. It is not a third input feature."],
    ["Find the support vectors.", "Gold diamonds highlight training points selected as support vectors by the fitted model."],
  ];
  const kernelNotes = {
    linear: "A straight boundary in the input plane. Gamma does not apply.",
    rbf: "Similarity depends on squared Euclidean distance; gamma controls how local it is.",
    poly: "A degree-three polynomial captures feature interactions; gamma scales the input term.",
    laplacian: "Similarity depends on Manhattan distance; gamma controls its decay.",
    rational_quadratic: "A distance-based similarity with shape parameter α = 1; gamma scales distance.",
    cosine: "Compares direction after scaling. Gamma does not apply.",
  };
  const axis = (title, range) => ({ title: { text: title, font: { color: "#bed0dd", size: 12 } }, range,
    color: "#9fb5c5", gridcolor: "#345268", zerolinecolor: "#49647a", backgroundcolor: "#102638", showbackground: true, tickfont: { size: 10 } });
  let camera = { eye: { x: 1.45, y: 1.45, z: 1.05 } }, syncing = false;
  function layout() {
    const bounds = dataset().bounds;
    const shared = { paper_bgcolor: "#102638", plot_bgcolor: "#102638", margin: { l: 0, r: 0, t: 0, b: 0 },
      showlegend: false, font: { family: "Inter, sans-serif", color: "#dce9f5" }, uirevision: `${state.dataset}-${state.view}` };
    if (state.view === "3d") return { ...shared, scene: { xaxis: axis("x₁", [bounds[0][0], bounds[1][0]]),
      yaxis: axis("x₂", [bounds[0][1], bounds[1][1]]), zaxis: axis("decision score", [-2.3, 2.3]),
      camera: structuredClone(camera), dragmode: "orbit", aspectmode: "manual", aspectratio: { x: 1, y: 1, z: .72 }, bgcolor: "#102638", uirevision: `${state.dataset}-camera` } };
    return { ...shared, margin: { l: 42, r: 16, t: 18, b: 42 },
      xaxis: { title: "x₁", range: [bounds[0][0], bounds[1][0]], color: "#9fb5c5", gridcolor: "#345268", scaleanchor: "y" },
      yaxis: { title: "x₂", range: [bounds[0][1], bounds[1][1]], color: "#9fb5c5", gridcolor: "#345268" } };
  }
  function traces(modelKey) {
    const current = dataset().models[modelKey], out = [], threeD = state.view === "3d";
    if (state.step >= 2) {
      const z = current.scoreGrid.map(row => row.map(value => Math.max(-data.scoreClip, Math.min(data.scoreClip, value))));
      out.push({ type: threeD ? "surface" : "heatmap", x: dataset().xGrid, y: dataset().yGrid, z,
        ...(threeD ? { surfacecolor: z, cmin: -data.scoreClip, cmax: data.scoreClip } : { zmin: -data.scoreClip, zmax: data.scoreClip }),
        colorscale: [[0, "#286ca0"], [.5, "#35556b"], [1, "#f4a96c"]], opacity: .67, showscale: false, hoverinfo: "skip", name: "Decision score" });
    }
    if (state.step >= 1) {
      const x = [], y = [], z = [];
      current.boundary.forEach(line => { line.forEach(([px, py]) => { x.push(px); y.push(py); z.push(0); }); x.push(null); y.push(null); z.push(null); });
      out.push({ type: threeD ? "scatter3d" : "scatter", mode: "lines", x, y, ...(threeD ? { z } : {}), line: { color: "#ffffff", width: threeD ? 8 : 3 }, hoverinfo: "skip", name: "Score = 0" });
    }
    for (const split of ["train", "test"]) for (const cls of [0, 1]) {
      if ((split === "train" && !state.showTrain) || (split === "test" && !state.showTest)) continue;
      const points = current.points.filter(p => p.true === cls && p.split === split), color = cls === 0 ? "#5dbdff" : "#ffad70";
      out.push({ type: threeD ? "scatter3d" : "scatter", mode: "markers", x: points.map(p => p.x), y: points.map(p => p.y), ...(threeD ? { z: points.map(() => 0) } : {}),
        customdata: points.map(p => p.id), marker: { color, size: split === "test" ? (threeD ? 5 : 8) : (threeD ? 3.2 : 5), opacity: .9,
          symbol: split === "test" ? "circle-open" : "circle", line: { color, width: split === "test" ? 2 : 0 } },
        hovertemplate: "Point %{customdata}<extra></extra>", name: `Class ${cls} · ${split}` });
    }
    if (state.showErrors && state.showTest) {
      const points = current.points.filter(p => p.split === "test" && p.true !== p.pred);
      out.push({ type: threeD ? "scatter3d" : "scatter", mode: "markers", x: points.map(p => p.x), y: points.map(p => p.y), ...(threeD ? { z: points.map(() => .03) } : {}),
        customdata: points.map(p => p.id), marker: { symbol: "x", size: threeD ? 7 : 10, color: "#ff647b", line: { width: 2, color: "#ff647b" } }, name: "Test errors" });
    }
    if (state.step >= 3 && state.showTrain) {
      const points = current.points.filter(p => p.support && p.split === "train");
      out.push({ type: threeD ? "scatter3d" : "scatter", mode: "markers", x: points.map(p => p.x), y: points.map(p => p.y), ...(threeD ? { z: points.map(() => .025) } : {}),
        customdata: points.map(p => p.id), marker: { symbol: "diamond-open", size: threeD ? 7 : 10, color: "#ffdc85", line: { width: 2, color: "#ffdc85" } },
        hovertemplate: "Support vector %{customdata}<extra></extra>", name: "Support vectors" });
    }
    return out;
  }
  function describePoint() {
    const modelKey = state.pointSource === "comparison" && state.compare ? state.compare : key();
    const point = dataset().models[modelKey].points.find(p => String(p.id) === String(state.pointId));
    $("guide-point-detail").textContent = point ? `${label(modelKey)} · point ${point.id} · ${point.split}. True class ${point.true}; predicted ${point.pred}. Score ${Number(point.score).toFixed(3)}. ${point.true === point.pred ? "Correct." : "Incorrect."}${point.support ? " Support vector." : ""}` : "Select a test point or click any dot.";
  }
  function updateUI() {
    document.querySelectorAll("[data-guide-dataset]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.guideDataset === state.dataset)));
    document.querySelectorAll("[data-guide-step]").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.guideStep) === state.step)));
    $("guide-kernel").value = state.kernel; $("guide-gamma").value = state.gamma; $("guide-gamma").disabled = ["linear", "cosine"].includes(state.kernel);
    $("guide-view").value = state.view; $("guide-compare").value = state.compare;
    $("guide-kernel-note").textContent = kernelNotes[state.kernel];
    $("guide-train").checked = state.showTrain; $("guide-test").checked = state.showTest; $("guide-errors").checked = state.showErrors;
    $("guide-counter").textContent = `${String(state.step + 1).padStart(2, "0")} / 04`;
    $("guide-title").textContent = steps[state.step][0];
    $("guide-description").textContent = state.step === 3 && !state.showTrain
      ? "Turn on training points to see the support vectors." : steps[state.step][1];
    const { correct_test, test_total } = model().metrics; $("guide-score").textContent = `${correct_test} / ${test_total}`;
    const metrics = model().metrics;
    $("guide-extra-metrics").textContent = `Training accuracy: ${(metrics.accuracy_train * 100).toFixed(1)}% · Support vectors: ${metrics.support_vectors}`;
    for (let row = 0; row < 2; row++) for (let col = 0; col < 2; col++) $(`guide-m${row}${col}`).textContent = String(metrics.confusion[row][col]);
    $("guide-primary-label").textContent = label(key()); $("guide-secondary-label").textContent = state.compare ? label(state.compare) : "Comparison";
    $("guide-secondary").hidden = !state.compare; $("guide-plots").classList.toggle("comparing", !!state.compare);
    $("guide-comparison-result").hidden = !state.compare;
    if (state.compare) { const c = dataset().models[state.compare].metrics; const difference = correct_test - c.correct_test;
      $("guide-comparison-result").textContent = `${label(state.compare)}: ${c.correct_test} / ${c.test_total} correct. Selected model: ${difference > 0 ? "+" : ""}${difference} correct predictions on the same test points.`; }
    $("guide-gesture").textContent = state.view === "3d" ? "Drag to rotate · scroll to zoom" : "Drag to pan · scroll to zoom";
    $("guide-caption").textContent = state.view === "3d" ? "Height is the decision score, not a third input feature. Surface clipped at ±2.2." : "Color is the decision score, clipped at ±2.2; the boundary is score zero.";
    describePoint();
  }
  function populatePoints() {
    const select = $("guide-point"); select.textContent = "";
    const placeholder = document.createElement("option"); placeholder.value = ""; placeholder.textContent = "Choose a point"; select.appendChild(placeholder);
    dataset().models.linear.points.filter(p => p.split === "test").filter((_, i) => i % 8 === 0).forEach(p => { const o = document.createElement("option"); o.value = p.id; o.textContent = `Point ${p.id}`; select.appendChild(o); });
    state.pointId = null; state.pointSource = "primary";
  }
  function attach(div, other, comparison) {
    div.on("plotly_click", event => {
      const id = event.points?.[0]?.customdata; if (id === undefined || id === null) return;
      state.pointId = id; state.pointSource = comparison ? "comparison" : "primary"; const select = $("guide-point");
      if (![...select.options].some(o => o.value === String(id))) { const o = document.createElement("option"); o.value = id; o.textContent = `Point ${id} (selected)`; select.appendChild(o); }
      select.value = String(id); describePoint();
    });
    div.on("plotly_relayout", async update => {
      if (state.view !== "3d" || syncing || !Object.keys(update || {}).some(k => k.startsWith("scene.camera"))) return;
      if (!div.layout?.scene?.camera) return; camera = structuredClone(div.layout.scene.camera);
      if (state.compare && other.layout?.scene) { syncing = true; try { await Plotly.relayout(other, { "scene.camera": camera }); } finally { syncing = false; } }
    });
  }
  let ready = false, otherReady = false, queue = Promise.resolve();
  function render() {
    updateUI();
    queue = queue.then(async () => {
      await Plotly.react(plot, traces(key()), layout(), { responsive: true, displaylogo: false, scrollZoom: true });
      if (!ready) { attach(plot, otherPlot, false); ready = true; }
      if (state.compare) { await Plotly.react(otherPlot, traces(state.compare), layout(), { responsive: true, displaylogo: false, scrollZoom: true }); if (!otherReady) { attach(otherPlot, plot, true); otherReady = true; } }
      await new Promise(resolve => requestAnimationFrame(resolve)); Plotly.Plots.resize(plot); if (state.compare) Plotly.Plots.resize(otherPlot);
      $("guide-error").hidden = true;
    }).catch(error => { $("guide-error").textContent = "The chart could not be drawn. Try refreshing or switching views."; $("guide-error").hidden = false; console.error(error); });
  }
  function stopTour() { clearInterval(state.timer); state.timer = null; $("guide-play").textContent = "Play the tour ▶"; }
  populatePoints();
  $("guide-point").addEventListener("change", e => { state.pointId = e.target.value || null; state.pointSource = "primary"; describePoint(); });
  document.querySelectorAll("[data-guide-dataset]").forEach(b => b.addEventListener("click", () => { stopTour(); state.dataset = b.dataset.guideDataset; populatePoints(); render(); }));
  document.querySelectorAll("[data-guide-step]").forEach(b => b.addEventListener("click", () => { stopTour(); state.step = Number(b.dataset.guideStep); render(); }));
  for (const [id, property] of [["guide-kernel", "kernel"], ["guide-gamma", "gamma"], ["guide-view", "view"], ["guide-compare", "compare"]]) $(id).addEventListener("change", e => { stopTour(); state[property] = e.target.value; render(); });
  for (const [id, property] of [["guide-train", "showTrain"], ["guide-test", "showTest"], ["guide-errors", "showErrors"]]) $(id).addEventListener("change", e => { state[property] = e.target.checked; render(); });
  $("guide-reset").addEventListener("click", () => { camera = { eye: { x: 1.45, y: 1.45, z: 1.05 } }; if (ready) { if (state.view === "3d") Plotly.relayout(plot, { "scene.camera": camera }); else Plotly.relayout(plot, { "xaxis.autorange": true, "yaxis.autorange": true }); if (state.compare && otherReady) { if (state.view === "3d") Plotly.relayout(otherPlot, { "scene.camera": camera }); else Plotly.relayout(otherPlot, { "xaxis.autorange": true, "yaxis.autorange": true }); } } });
  $("guide-export").addEventListener("click", () => { if (ready) Plotly.downloadImage(plot, { format: "png", width: 1200, height: 900, filename: `svm-${state.dataset}-${key().replace(":", "-")}` }); });
  $("guide-share").addEventListener("click", async () => {
    const url = new URL(location.href);
    for (const [name, value] of Object.entries({ dataset: state.dataset, kernel: state.kernel, gamma: state.gamma, view: state.view })) url.searchParams.set(name, value);
    url.searchParams.set("step", String(state.step)); url.searchParams.set("train", state.showTrain ? "1" : "0");
    url.searchParams.set("test", state.showTest ? "1" : "0"); url.searchParams.set("errors", state.showErrors ? "1" : "0");
    if (state.compare) url.searchParams.set("compare", state.compare); else url.searchParams.delete("compare");
    url.hash = "explore";
    try {
      if (!navigator.clipboard) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(url.href);
      const local = url.protocol === "file:" || ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
      $("guide-share-status").textContent = local ? "Local link copied · works on this device" : "View link copied";
    } catch (_) { $("guide-share-status").textContent = "Could not copy the view link. Please try again."; }
  });
  window.addEventListener("svm:comparison", () => { stopTour(); Object.assign(state, { dataset: "circles", kernel: "rbf", gamma: "1", view: "2d", compare: "linear", step: 2, showTrain: true, showTest: true, showErrors: false }); populatePoints(); render(); });
  $("guide-play").addEventListener("click", () => { if (state.timer) { stopTour(); return; } state.step = 0; render(); $("guide-play").textContent = "Pause tour";
    state.timer = setInterval(() => { if (state.step >= 3) { stopTour(); return; } state.step++; render(); }, 4500); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopTour(); });
  updateUI();
  if ("IntersectionObserver" in window) { const observer = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) { observer.disconnect(); render(); } }, { rootMargin: "300px" }); observer.observe($("explore")); } else render();
})();
