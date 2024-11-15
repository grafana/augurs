import initChangepoint, {
  ChangepointDetector,
} from "./dist/@bsull/augurs/changepoint.js";

await initChangepoint();

self.onmessage = (e) => {
  const { y } = e.data;
  const cpd = new ChangepointDetector("normal-gamma");
  const cps = cpd.detectChangepoints(y);
  self.postMessage(cps);
};
self.postMessage("ready");
