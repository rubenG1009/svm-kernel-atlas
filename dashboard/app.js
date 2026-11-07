(() => {
  "use strict";
  const data = window.SVM_DATA;
  const loading = document.getElementById("loading");
  if (!data || !window.Plotly) { loading.textContent = "Dashboard assets could not load. Rebuild data.js and refresh."; return; }
  loading.hidden = true;
  const $ = id => document.getElementById(id);
  function drawBoundary(svgId, modelKey) {
    const svg = $(svgId);
    if (!svg) return;
    const dataset = data.datasets.circles;
    const model = dataset.models[modelKey];
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

  drawBoundary("hero-plot", "rbf:1");
  for (const [id, modelKey] of [["linear", "linear"], ["rbf", "rbf:1"]]) {
    drawBoundary(`story-${id}`, modelKey);
    const { correct_test, test_total } = data.datasets.circles.models[modelKey].metrics;
    $(`story-${id}-score`).textContent = `${correct_test} / ${test_total}`;
  }
  $("try-comparison").addEventListener("click", () => {
    window.dispatchEvent(new CustomEvent("svm:comparison"));
    $("explore").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
