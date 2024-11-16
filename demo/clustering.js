import uPlot from "./dist/uPlot/uPlot.esm.js";

import { getSize } from "./helpers.js";
import { legendAsTooltipPlugin } from "./plugins.js";

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
  const u = uPlot(opts, data, document.getElementById("clustering-plot"));
  window.addEventListener("resize", () => {
    u.setSize(getSize());
  });
  return u;
}

class ClusteringWorker {
  constructor() {
    this.worker = new Worker("./clustering.worker.js", { type: "module" });
    this.dataPromise = fetch("./outlier.data.json").then((res) => res.json());
    this.dataPromise.then((data) => {
      this.data = data.data;
    });
  }

  static create = () => {
    return new Promise((resolve, reject) => {
      const worker = new ClusteringWorker();
      worker.worker.onmessage = (e) => {
        if (e.data === "ready") {
          worker.dataPromise.then(() => resolve(worker));
        } else {
          reject();
        }
      }
    })
  }

  cluster = async (dtwOpts, dbscanOpts) => {
    return new Promise((resolve, reject) => {
      const start = performance.now();
      this.worker.postMessage({
        dtwOpts,
        dbscanOpts,
        data: this.data.slice(1).map(arr => new Float64Array(arr)),
      });
      this.worker.onmessage = (e) => {
        const elapsed = (performance.now() - start).toFixed(0);
        resolve({ clusterLabels: e.data, elapsed });
      };
    });
  }
}

async function main() {
  const worker = await ClusteringWorker.create();

  const u = setUpPlot(worker.data);
  async function runClustering(dtwOpts, dbscanOpts) {
    const { clusterLabels, elapsed } = await worker.cluster(dtwOpts, dbscanOpts);
    clusterLabels.forEach((cluster, i) => {
      const seriesIdx = i + 1;
      u.delSeries(seriesIdx);
      u.addSeries({
        label: `${i} (cluster ${cluster})`,
        stroke: cluster === -1 ? "black" : cluster === 0 ? "blue" : cluster === 1 ? "red" : "yellow",
        width: 1,
      }, seriesIdx);
    });
    u.redraw()
    document.getElementById("clustering-title").innerText = `Clustering with DBSCAN - done in ${elapsed}ms`;
  }
  const dtwOpts = { window: 2 };
  const dbscanOpts = { epsilon: 5000, minClusterSize: 2 };
  runClustering(dtwOpts, dbscanOpts);

  document.getElementById("clustering-dtw-window").addEventListener("change", function() {
    dtwOpts.window = parseFloat(this.value);
    runClustering(dtwOpts, dbscanOpts);
  });
  document.getElementById("clustering-dbscan-epsilon").addEventListener("change", function() {
    dbscanOpts.epsilon = parseFloat(this.value);
    runClustering(dtwOpts, dbscanOpts);
  });
  document.getElementById("clustering-dbscan-min-cluster-size").addEventListener("change", function() {
    dbscanOpts.minClusterSize = parseInt(this.value);
    runClustering(dtwOpts, dbscanOpts);
  });
}

export default main;
