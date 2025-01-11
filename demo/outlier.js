import uPlot from "./dist/uPlot/uPlot.esm.js";

import { getSize } from "./helpers.js";
import { legendAsTooltipPlugin } from "./plugins.js";

class OutlierWorker {
  constructor() {
    this.worker = new Worker("./outlier.worker.js", { type: "module" });
    this.dataPromise = fetch("./outlier.data.json").then((res) => res.json());
    this.dataPromise.then((data) => {
      this.data = data.data;
    });
  }

  static create = () => {
    return new Promise((resolve, reject) => {
      const worker = new OutlierWorker();
      worker.worker.onmessage = (e) => {
        if (e.data === "ready") {
          worker.dataPromise.then(() => resolve(worker));
        } else {
          reject();
        }
      }
    })
  }

  detect = async (opts) => {
    return new Promise((resolve) => {
      const start = performance.now();
      this.worker.postMessage({
        opts,
        data: this.data.slice(1).map(arr => new Float64Array(arr)),
      });
      this.worker.onmessage = (e) => {
        const elapsed = (performance.now() - start).toFixed(0);
        resolve({ outliers: e.data, elapsed });
      };
    });
  }
}

function setUpPlot(data) {
  const opts = {
    ...getSize(),
    series: [
      {},
      ...data.slice(1).map((_, i) => {
        return {
          label: `${i + 1}`,
          stroke: "black",
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

async function main() {
  const worker = await OutlierWorker.create();

  const u = setUpPlot(worker.data);
  async function runOutlierDetection(opts) {
    const { outliers, elapsed } = await worker.detect(opts);
    outliers.seriesResults.forEach((res, i) => {
      const seriesIdx = i + 1;
      u.delSeries(seriesIdx);
      u.addSeries({
        label: `${i} (${res.isOutlier ? "outlier" : "normal"})`,
        stroke: res.isOutlier ? "red" : "black",
        width: 1,
      }, seriesIdx);
    });
    u.redraw();
    document.getElementById("outlier-title").innerText = `Outlier detection with DBSCAN - done in ${elapsed}ms`;
  }
  runOutlierDetection({ sensitivity: 0.8 });

  document.getElementById("outlier-sensitivity").addEventListener("input", function() {
    runOutlierDetection({ sensitivity: parseFloat(this.value) });
  })
}

export default main;
