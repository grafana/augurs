import uPlot from "./dist/uPlot/uPlot.esm.js";

import { getSize } from "./helpers.js";
import { legendAsTooltipPlugin } from "./plugins.js";

const worker = new Worker("./clustering.worker.js", {
  type: "module",
});

function setUpPlot(data, labels) {
  const opts = {
    ...getSize(),
    series: [
      {},
      ...data.slice(1).map((_, i) => {
        const cluster = labels[i];
        return {
          label: `${i + 1} (cluster ${labels[i]})`,
          stroke: cluster === -1 ? "black" : cluster === 0 ? "blue" : "red",
          width: 1,
        };
      }),
    ],
    plugins: [legendAsTooltipPlugin()],
  };
  const u = uPlot(opts, data, document.getElementById("clustering-plot"));
  window.addEventListener("resize", () => {
    u.setSize(getSize());
  });
  return u;
}

let start;
let data;

worker.onmessage = async (e) => {
  if (e.data === "ready") {
    start = performance.now();
    const dtwOpts = { window: 2 };
    const dbscanOpts = { epsilon: 5000, minClusterSize: 2 };
    data = await fetch("./outlier.data.json").then((res) => res.json());
    const series = data.data.slice(1).map((arr) => new Float64Array(arr));
    worker.postMessage({ dtwOpts, dbscanOpts, data: series });
  } else {
    const elapsed = performance.now() - start;
    const clusterLabels = e.data;
    setUpPlot(data.data, clusterLabels);
    document.getElementById("clustering-title").innerText =
      `Clustering with DBSCAN - done in ${elapsed}ms`;
  }
};
export default undefined;
