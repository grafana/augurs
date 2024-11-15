import uPlot from "./dist/uPlot/uPlot.esm.js";

import { getSize } from "./helpers.js";
import { legendAsTooltipPlugin } from "./plugins.js";

const worker = new Worker("./outlier.worker.js", {
  type: "module",
});

function setUpPlot(data, outlyingSeries) {
  const opts = {
    ...getSize(),
    series: [
      {},
      ...data.slice(1).map((_, i) => {
        const isOutlier = outlyingSeries.has(i);
        return {
          label: `${i + 1} (${isOutlier ? "outlier" : "normal"})`,
          stroke: isOutlier ? "red" : "blue",
          width: 1,
        };
      }),
    ],
    plugins: [legendAsTooltipPlugin()],
  };
  const u = uPlot(opts, data, document.getElementById("outlier-plot"));
  window.addEventListener("resize", () => {
    u.setSize(getSize());
  });
  return u;
}

let data;
let start;

worker.onmessage = async (e) => {
  if (e.data === "ready") {
    start = performance.now();
    const opts = { sensitivity: 0.8 };
    data = await fetch("./outlier.data.json").then((res) => res.json());
    const series = data.data.slice(1).map((arr) => new Float64Array(arr));
    worker.postMessage({ opts, data: series });
  } else {
    const elapsed = performance.now() - start;
    const outliers = e.data;
    const outlyingSeries = new Set(outliers.outlyingSeries);
    setUpPlot(data.data, outlyingSeries);
    document.getElementById("outlier-title").innerText =
      `Outlier detection with DBSCAN - done in ${elapsed}ms`;
  }
};
export default undefined;
