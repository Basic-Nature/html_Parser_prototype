/**
 * Decorative F2 cosmic-observatory backdrop.
 *
 * T1 restores visual identity only. Runtime/parser/socket authority remains
 * outside this component.
 */
export function CosmicBackdrop() {
  return (
    <div className="blf2-cosmic-backdrop" aria-hidden="true">
      <div className="blf2-cosmic-fractal" />
      <div className="blf2-cosmic-stars blf2-cosmic-stars--far" />
      <div className="blf2-cosmic-stars blf2-cosmic-stars--near" />
      <div className="blf2-cosmic-horizon" />

      <div className="blf2-cosmic-system">
        <div className="blf2-cosmic-sun" />

        <div className="blf2-cosmic-earth-system">
          <div className="blf2-cosmic-earth">
            <span className="blf2-cosmic-earth-atmosphere" />
          </div>

          <div className="blf2-cosmic-moon-orbit">
            <div className="blf2-cosmic-moon-anchor">
              <span className="blf2-cosmic-moon" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
