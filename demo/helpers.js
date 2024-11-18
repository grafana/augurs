export function getSize() {
  const margin = window.innerWidth <= 768 ? 30 : 100;
  const BREAKPOINTS = {
    MOBILE: 480,
    TABLET: 768
  };
  
  const DENOMINATORS = {
    MOBILE: 2.5,    // Larger height ratio for mobile
    TABLET: 3,      // Medium height ratio for tablet
    DESKTOP: 4      // Smaller height ratio for desktop
  };
  
  const denominator = 
    window.innerWidth <= BREAKPOINTS.MOBILE ? DENOMINATORS.MOBILE :
    window.innerWidth <= BREAKPOINTS.TABLET ? DENOMINATORS.TABLET :
    DENOMINATORS.DESKTOP;
  return {
    width: Math.min(window.innerWidth - margin, 1100),
    height: window.innerHeight / denominator,
  };
}
