import uPlot from "./dist/uPlot/uPlot.esm.js";

// converts the legend into a simple tooltip
export function legendAsTooltipPlugin({
  className,
  style = { backgroundColor: "#f5f6fa", color: "black" },
} = {}) {
  let legendEl;

  function init(u, opts) {
    legendEl = u.root.querySelector(".u-legend");

    legendEl.classList.remove("u-inline");
    className && legendEl.classList.add(className);

    uPlot.assign(legendEl.style, {
      textAlign: "left",
      pointerEvents: "none",
      display: "none",
      position: "absolute",
      left: 0,
      top: 0,
      zIndex: 100,
      boxShadow: "2px 2px 4px rgba(0,0,0,0.5)",
      ...style,
    });

    // hide series color markers
    const idents = legendEl.querySelectorAll(".u-marker");

    for (let i = 0; i < idents.length; i++) idents[i].style.display = "none";

    const overEl = u.over;
    overEl.style.overflow = "visible";

    // move legend into plot bounds
    overEl.appendChild(legendEl);

    // show/hide tooltip on enter/exit
    overEl.addEventListener("mouseenter", () => {
      legendEl.style.display = null;
    });
    overEl.addEventListener("mouseleave", () => {
      legendEl.style.display = "none";
    });

    // let tooltip exit plot
    // overEl.style.overflow = "visible";
  }

  function update(u) {
    const { left, top } = u.cursor;
    legendEl.style.transform = "translate(" + left + "px, " + top + "px)";
  }

  return {
    hooks: {
      init: init,
      setCursor: update,
    },
  };
}
