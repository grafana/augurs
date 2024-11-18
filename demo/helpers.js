export function getSize() {
  const margin = window.innerWidth <= 768 ? 30 : 100;
  const denominator =
    window.innerWidth <= 480 ? 2.5 : window.innerWidth <= 768 ? 3 : 4;
  return {
    width: Math.min(window.innerWidth - margin, 1100),
    height: window.innerHeight / denominator,
  };
}
