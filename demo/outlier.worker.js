import initOutlier, { OutlierDetector } from "./dist/@bsull/augurs/outlier.js";

await initOutlier();

self.onmessage = (e) => {
  const { opts, data } = e.data;
  const detector = OutlierDetector.dbscan(opts);
  const outliers = detector.preprocess(data).detect();
  self.postMessage(outliers);
};
self.postMessage("ready");
