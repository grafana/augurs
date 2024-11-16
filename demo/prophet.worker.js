import initProphet, {
  initLogging,
  Prophet,
} from "./dist/@bsull/augurs/prophet.js";
import { optimizer } from "./dist/@bsull/augurs-prophet-wasmstan/prophet-wasmstan.js";

await initProphet();

self.onmessage = (e) => {
  const { data, opts } = e.data;
  const df = {
    ds: data.ds,
    y: data.y,
  };
  const prophet = new Prophet({
    optimizer,
    uncertaintySamples: 500,
    intervalWidth: 0.8,
    ...(opts ?? {})
  });
  prophet.fit(df);
  const predictions = prophet.predict();
  self.postMessage(predictions);
};
self.postMessage("ready");
