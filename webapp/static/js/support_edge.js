/* Candidate; no tracking, no network or map listeners. */
'use strict';
(function(){
  function start(){
    const root=document.getElementById('epSupportEdge');
    const trigger=document.getElementById('epSupportEdgeTrigger');
    const card=document.getElementById('epSupportEdgeCard');
    const close=document.getElementById('epSupportEdgeClose');
    const shell=document.getElementById('observatoryShell');
    if(!root||!trigger||!card||!close||!shell)return;
    const coarse=window.matchMedia('(hover: none), (pointer: coarse)');
    let closeTimer=null, wasFocusRestore=false;
    const isOpen=()=>!card.hidden;
    function setOpen(v){if(v&&!allowed())return;card.hidden=!v;trigger.setAttribute('aria-expanded',String(v));}
    function allowed(){
      if(coarse.matches)return false;
      const r=shell.getBoundingClientRect(), w=document.documentElement.clientWidth;
      // The cue and entire card must fit completely left of the real shell.
      const fit=r.left>=324 && r.left<=w && r.width>0 && w>=1250;
      if(!fit)return false;
      const hit=document.elementFromPoint(30,Math.round(window.innerHeight*.65));
      if(hit && shell.contains(hit))return false;
      return true;
    }
    function update(){const visible=allowed();root.hidden=!visible;if(!visible)setOpen(false);}
    function cancel(){if(closeTimer!==null){clearTimeout(closeTimer);closeTimer=null;}}
    function deferClose(){cancel();closeTimer=setTimeout(()=>{if(!root.matches(':hover')&&!card.contains(document.activeElement)&&document.activeElement!==trigger)setOpen(false);},85);}
    function returnFocus(){wasFocusRestore=true;trigger.focus({preventScroll:true});wasFocusRestore=false;}
    trigger.addEventListener('mouseenter',()=>{cancel();setOpen(true)});
    trigger.addEventListener('focus',()=>{if(!wasFocusRestore){cancel();setOpen(true)}});
    trigger.addEventListener('click',()=>{cancel();setOpen(true)}); // Idempotent under hover/focus.
    trigger.addEventListener('mouseleave',deferClose);
    trigger.addEventListener('blur',deferClose);
    card.addEventListener('mouseenter',()=>{cancel();setOpen(true)});
    card.addEventListener('mouseleave',deferClose);
    card.addEventListener('focusin',cancel);
    card.addEventListener('focusout',deferClose);
    close.addEventListener('click',()=>{cancel();setOpen(false);returnFocus()});
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&isOpen()){e.stopPropagation();cancel();setOpen(false);if(root.contains(document.activeElement))returnFocus();}});
    document.addEventListener('pointerdown',e=>{if(isOpen()&&!root.contains(e.target)){cancel();setOpen(false)}});
    window.addEventListener('resize',update);
    window.addEventListener('scroll',update,{passive:true});
    if(window.ResizeObserver){const ro=new ResizeObserver(update);ro.observe(shell);}
    update();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
})();
