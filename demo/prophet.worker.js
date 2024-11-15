import initProphet, {
  initLogging,
  Prophet,
} from "./dist/@bsull/augurs/prophet.js";
import { optimizer } from "./dist/@bsull/augurs-prophet-wasmstan/prophet-wasmstan.js";

await initProphet();

self.onmessage = (e) => {
  const df = {
    ds: e.data.ds,
    y: e.data.y,
  };
  const prophet = new Prophet({
    optimizer,
    uncertaintySamples: 500,
    intervalWidth: 0.8,
  });
  prophet.fit(df);
  const predictions = prophet.predict();
  self.postMessage(predictions);
};
self.postMessage("ready");
