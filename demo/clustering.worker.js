import initDtw, { Dtw } from "./dist/@bsull/augurs/dtw.js";
import initClustering, {
  DbscanClusterer,
} from "./dist/@bsull/augurs/clustering.js";

await Promise.all([initDtw(), initClustering()]);

self.onmessage = (e) => {
  const { dtwOpts, dbscanOpts, data } = e.data;
  const dtw = Dtw.euclidean(dtwOpts);
  const distanceMatrix = dtw.distanceMatrix(data);
  const clusterer = new DbscanClusterer(dbscanOpts);
  const labels = clusterer.fit(distanceMatrix);
  self.postMessage(labels);
};
self.postMessage("ready");
