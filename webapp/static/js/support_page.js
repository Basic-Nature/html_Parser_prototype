'use strict';
(function(){
  const snapshot=window.EP_SUPPORT_SNAPSHOT;
  const $=(id)=>document.getElementById(id);
  const currency=(amount)=>amount===null||amount===undefined?'—':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(amount);
  const clear=(node)=>{while(node.firstChild)node.removeChild(node.firstChild)};
  const label=(p)=>{const [y,m]=p.split('-');return new Date(Date.UTC(Number(y),Number(m)-1,1)).toLocaleDateString('en-US',{month:'short',year:'numeric',timeZone:'UTC'})};
  function render(){
    if(!snapshot || snapshot.publish_approved!==false || !Array.isArray(snapshot.monthly) || snapshot.monthly.length===0){
      document.querySelector('.hero-figure').textContent='Unavailable';
      const s=document.getElementById('support-filter-status');
      if(s)s.textContent='Financial snapshot awaits publication review.';
      return;
    }
    const totals=snapshot.founder;
    $('hero-provider-total').textContent=currency(totals.provider_record_payment_subtotal_usd);
    $('provider-total').textContent=currency(totals.provider_record_payment_subtotal_usd);
    $('azure-total').textContent=currency(totals.invoice_paid_status_azure_usd);
    $('github-total').textContent=currency(totals.applied_payment_github_usd);
    $('godaddy-total').textContent=currency(totals.received_payment_godaddy_usd);
    for(const el of document.querySelectorAll('[data-fund]')){
      const amount=snapshot.contributions[el.getAttribute('data-fund')];
      el.textContent=amount===null||amount===undefined?'—':currency(amount);
      el.setAttribute('aria-label',amount===null||amount===undefined?'Value unavailable; no verified contribution ledger':currency(amount));
    }
    function draw(){
      const year=$('year-filter').value;
      const rows=snapshot.monthly.filter(x=>year==='all'||x.period.startsWith(year));
      const chart=$('monthly-chart'),tbody=$('expense-body'),foot=$('expense-foot');
      const status=$('support-filter-status');
      clear(chart);clear(tbody);clear(foot);
      const max=Math.max(...rows.map(r=>r.provider_payment_sum_usd),1);
      for(const r of rows){
        const col=document.createElement('div');col.className='bar-col';
        const stack=document.createElement('div');stack.className='bar-stack';
        stack.setAttribute('role','img');stack.tabIndex=0;
        stack.setAttribute('aria-label',label(r.period)+': Azure '+currency(r.azure_paid_invoice_usd)+', GitHub '+currency(r.github_applied_payments_usd)+', GoDaddy '+currency(r.godaddy_paid_receipt_usd));
        const parts=[['azure',r.azure_paid_invoice_usd],['github',r.github_applied_payments_usd],['godaddy',r.godaddy_paid_receipt_usd]];
        for(const [type,val] of parts){
          if(val===null||val===undefined||val<=0)continue;
          const bar=document.createElement('div');bar.className='bar-segment bar-'+type;
          bar.style.height=((val/max)*155)+'px';bar.title=type+': '+currency(val);stack.appendChild(bar);
        }
        col.appendChild(stack);
        const short=document.createElement('span');short.className='bar-label';short.textContent=r.period.substring(2,4)+'/'+r.period.substring(5,7);col.appendChild(short);chart.appendChild(col);
        const tr=document.createElement('tr');
        const cells=[label(r.period),currency(r.azure_paid_invoice_usd),currency(r.github_applied_payments_usd),currency(r.godaddy_paid_receipt_usd),currency(r.provider_payment_sum_usd),r.incomplete_period?'Partial — Azure pending':'Historical record'];
        cells.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i>=1&&i<=4)td.className='align-right';if(i===5)td.className=r.incomplete_period?'partial-note':'coverage-note';tr.appendChild(td)});
        tbody.appendChild(tr);
      }
      if(status) status.textContent=`Showing ${rows.length} billing months for ${year==='all'?'all periods':year}.`;
      const tr=document.createElement('tr');
      const v=[year==='all'?'Selected periods total':year+' provider payments',...['azure_paid_invoice_usd','github_applied_payments_usd','godaddy_paid_receipt_usd','provider_payment_sum_usd'].map(key=>currency(rows.reduce((s,r)=>s+(r[key]??0),0))),'Mixed evidence'];
      v.forEach((s,i)=>{const td=document.createElement('td');td.textContent=s;if(i>=1&&i<=4)td.className='align-right';tr.appendChild(td)});foot.appendChild(tr);
    }
    $('year-filter').addEventListener('change',draw);draw();
  }
  document.addEventListener('DOMContentLoaded',render);
}());
