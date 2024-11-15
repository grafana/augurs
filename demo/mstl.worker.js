import initMstl, { MSTL } from "./dist/@bsull/augurs/mstl.js";
import initSeasonality, {
  seasonalities,
} from "./dist/@bsull/augurs/seasons.js";

await Promise.all([initMstl(), initSeasonality()]);

self.onmessage = (e) => {
  const { ds, y } = e.data;
  const seasons = seasonalities(ds);
  const mstl = MSTL.ets(seasons);
  mstl.fit(e.data.y);
  const predictions = mstl.predictInSample(0.8);
  self.postMessage(predictions);
};
self.postMessage("ready");
