/* Minimal initializer: add `left-sidebar-centered` to <body> on desktop load
   without overriding user toggles. Safe to include as a small standalone script. */
(function(){
  function applyLeftSidebarCentered(){
    try{
      if (!window.matchMedia('(min-width: 768px)').matches) return; // only desktop-ish
      var b = document.body;
      // Respect existing user toggles or explicit state
      if (b.classList.contains('right-sidebar-collapsed')) return;
      if (document.querySelector('#sidebar.sidebar-open')) return;
      if (b.classList.contains('left-sidebar-centered')) return;
      b.classList.add('left-sidebar-centered');
    }catch(e){ /* fail quietly */ }
  }
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', applyLeftSidebarCentered);
  } else {
    applyLeftSidebarCentered();
  }
})();
