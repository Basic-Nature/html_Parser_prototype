(() => {
  "use strict";
  const root = document.getElementById('projectWorkspace');
  if (!root) return;
  const byId = (id) => document.getElementById(id);
  const notice = byId('projectNotice');
  const list = byId('projectList');
  const form = byId('projectCreateForm');
  const detail = byId('projectDetail');
  const csrf = root.dataset.csrf;
  let active = null;
  let available = [];
  // Retain the idempotency token across uncertain network failures. A changed
  // draft is a new operation and must receive its own token.
  let pendingCreate = null;
  let preflightEpoch = 0;
  function resetRunPreflight() {
    preflightEpoch += 1;
    byId('projectRunPreflightResult').textContent = 'No eligibility review requested. No run was started.';
  }
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const states = new Set('AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR VI AS GU MP'.split(' '));
  function msg(s) { notice.textContent = s; }
  async function api(path, method='GET', data=null) {
    const headers = {'Accept':'application/json'};
    if (data !== null) {headers['Content-Type']='application/json';headers['X-CSRFToken']=csrf;}
    const res = await fetch(path, {method, headers, credentials:'same-origin',
      cache:'no-store', body: data === null ? undefined : JSON.stringify(data)});
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) { const error = new Error(payload.error || 'request_failed');error.status=res.status;throw error; }
    return payload;
  }
  function node(tag, text, attrs={}) {
    const el = document.createElement(tag);
    if (text !== null) el.textContent=String(text);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
    return el;
  }
  function rowLabel(label, input) {
    const wrap = node('label',label);wrap.appendChild(input);return wrap;
  }
  function scopeRow(value={}) {
    const row = node('div',null,{class:'scope-row'});
    for (const [field,label,placeholder] of [
      ['election_year','Year','2024'],['election_date','Election date','YYYY-MM-DD'],
      ['state_code','State/territory code','AZ'],['jurisdiction','County/locality','Pima'],
      ['contest','Contest','President']]) {
      const input=node('input',null,{name:field,placeholder:placeholder,autocomplete:'off'});
      if (value[field] !== undefined && value[field] !== null) input.value=String(value[field]);
      if (field==='state_code') input.maxLength=2;
      if (field==='election_year') input.inputMode='numeric';
      row.appendChild(rowLabel(label,input));
    }
    const sel=node('select',null,{name:'granularity'});
    for (const option of ['','statewide','county','municipality','precinct','unknown']) {
      const op=node('option',option||'Unspecified',{value:option});sel.appendChild(op);
    }
    sel.value=value.granularity || '';
    row.appendChild(rowLabel('Reporting granularity',sel));
    const remove = node('button','Remove this scope',{type:'button'});
    remove.addEventListener('click',()=>row.remove());row.appendChild(remove);
    return row;
  }
  function draftState() {
    const state = new URLSearchParams(location.search).get('state');
    return state && states.has(state.toUpperCase()) ? state.toUpperCase() : null;
  }
  function renderDetail(data) {
    active=data;detail.hidden=false;
    resetRunPreflight();
    byId('projectTitle').textContent=data.title;
    byId('projectMeta').textContent=`${data.lifecycle} · version ${data.row_version} · Project ${data.id}`;
    byId('projectBallotLens').href=`/ballot_lens?project_id=${encodeURIComponent(data.id)}`;
    const scopes=byId('projectScopes');scopes.replaceChildren();
    for (const item of data.scopes) scopes.appendChild(scopeRow(item));
    if (!data.scopes.length) scopes.appendChild(scopeRow({state_code:draftState()}));
    const sourceList=byId('projectSources');sourceList.replaceChildren();
    const runRefSelect=byId('projectRunSourceRefSelect');
    runRefSelect.replaceChildren(node('option','Select a saved source reference',{value:''}));
    for (const ref of data.source_refs) {
      runRefSelect.appendChild(node('option',`Binding ${ref.registry_binding_id} · revision ${ref.registry_revision_id}`,{value:ref.id}));
    }
    for (const ref of data.source_refs) {
      const li=node('li',null);
      li.appendChild(node('span',`Registry binding ${ref.registry_binding_id} · revision ${ref.registry_revision_id} · reference only`));
      const remove=node('button','Remove',{type:'button'});
      remove.addEventListener('click',async()=>{
        try{await openAfter(()=>api(`/api/projects/v1/${data.id}/source-refs/${ref.id}`,'DELETE',
          {expected_version:active.row_version}));}catch(e){msg(`Could not remove reference: ${e.message}`);}
      });li.appendChild(remove);sourceList.appendChild(li);
    }
    const select=byId('projectSourceSelect');select.replaceChildren(node('option','Select approved source metadata',{value:''}));
    for (const src of available) select.appendChild(node('option',`${src.year} · ${src.state} · ${src.contest} · ${src.scope} · ${src.format}`,{value:src.registry_binding_id}));
    if (location.pathname !== `/projects/${data.id}`) history.replaceState(null,'',`/projects/${data.id}`);
  }
  async function loadDetail(id) {
    if (!UUID.test(id)) {msg('Invalid project selector.');return;}
    try {renderDetail(await api(`/api/projects/v1/${encodeURIComponent(id)}`));msg('Project saved and available for resumption.');}
    catch(e){msg(`Project unavailable (${e.message}).`);detail.hidden=true;}
  }
  async function openAfter(action) {
    const payload=await action();
    renderDetail(payload);msg('Project changes saved.');await reloadList();
  }
  async function reloadList() {
    const payload=await api('/api/projects/v1');list.replaceChildren();
    for (const project of payload.projects) {
      const li=node('li',null);
      const link=node('a',project.title,{href:`/projects/${encodeURIComponent(project.id)}`});
      link.addEventListener('click',(event)=>{event.preventDefault();loadDetail(project.id);});
      li.appendChild(link);list.appendChild(li);
    }
    if (!payload.projects.length) list.appendChild(node('li','No projects yet. Create your first investigation.'));
  }
  form.addEventListener('input', () => {pendingCreate = null;});
  form.addEventListener('submit',async(event)=>{
    event.preventDefault();
    const title = form.elements.namedItem('title').value;
    const description = form.elements.namedItem('description').value;
    if (!pendingCreate || pendingCreate.title !== title || pendingCreate.description !== description) {
      pendingCreate = {title, description, idempotency_key:crypto.randomUUID()};
    }
    try {
      await openAfter(()=>api('/api/projects/v1','POST',pendingCreate));
      pendingCreate = null;
      form.reset();
    } catch(e){msg(`Could not create project: ${e.message}`);}
  });
  byId('projectAddScope').addEventListener('click',()=>{
    const container=byId('projectScopes');
    if (container.children.length >= 20){msg('Maximum of 20 scope groups.');return;}
    container.appendChild(scopeRow());
  });
  byId('projectScopeForm').addEventListener('submit',async(event)=>{
    event.preventDefault();if (!active) return;
    const scopes=[];
    for (const row of byId('projectScopes').children) {
      const vals={};for(const field of ['election_year','election_date','state_code','jurisdiction','contest','granularity']){
        let value=row.querySelector(`[name="${field}"]`).value.trim();
        if (field==='state_code') value=value.toUpperCase();
        vals[field]=value ? (field==='election_year' ? Number(value) : value) : null;
      }scopes.push(vals);
    }
    try{await openAfter(()=>api(`/api/projects/v1/${active.id}/scopes`,'PUT',
      {scopes,expected_version:active.row_version}));}catch(e){msg(`Could not save scopes: ${e.message}`);}
  });
  byId('projectSourceForm').addEventListener('submit',async(event)=>{
    event.preventDefault();if (!active) return;
    const registry_binding_id=byId('projectSourceSelect').value;
    try{await openAfter(()=>api(`/api/projects/v1/${active.id}/source-refs`,'POST',
      {registry_binding_id,expected_version:active.row_version}));}
    catch(e){msg(`Could not associate source: ${e.message}`);}
  });
  byId('projectRunPreflightForm').addEventListener('input', resetRunPreflight);
  byId('projectRunPreflightForm').addEventListener('submit', async(event) => {
    event.preventDefault();
    if (!active || active.lifecycle !== 'active') return;
    const sourceRefId=byId('projectRunSourceRefSelect').value;
    const workflowItemId=byId('projectRunWorkflowItem').value.trim().toLowerCase();
    const output=byId('projectRunPreflightResult');
    const epoch=++preflightEpoch;
    const projectId=active.id;
    const projectVersion=active.row_version;
    if (!UUID.test(sourceRefId) || !UUID.test(workflowItemId) ||
        !active.source_refs.some(ref => ref.id === sourceRefId)) {
      output.textContent='Select a saved source reference and enter a valid Workflow item UUID. No run was started.';
      return;
    }
    output.textContent='Reviewing current Project, Workflow, and Registry eligibility. No run was started.';
    const params=new URLSearchParams({source_ref_id:sourceRefId,
      workflow_item_id:workflowItemId,expected_project_version:String(projectVersion)});
    try {
      const preview=await api(`/api/projects/v1/${encodeURIComponent(projectId)}/run-preflight?${params}`);
      if (epoch!==preflightEpoch || !active || active.id!==projectId || active.row_version!==projectVersion) return;
      if (preview.contract!=='project_run_preflight_v1' || preview.project_id!==projectId ||
          preview.source_ref_id!==sourceRefId || preview.workflow_item_id!==workflowItemId ||
          preview.eligible_for_confirmation_review!==true || preview.confirmation_enabled!==false ||
          preview.execution_authorized!==false || preview.run_dispatched!==false ||
          preview.source_url_disclosed!==false) throw new Error('Unexpected preflight response');
      const label=preview.source_label;
      if (!label || !['year','state','contest','scope','format'].every(key => typeof label[key]==='string')) {
        throw new Error('Invalid source label');
      }
      output.textContent=`Eligible to review: ${label.year} · ${label.state} · ${label.contest} · ${label.scope} · ${label.format}. Confirmation and execution are not enabled in J2A. No run was started.`;
    } catch(e) {
      if (epoch!==preflightEpoch) return;
      output.textContent=`Eligibility not available (${e.message}). No run was started.`;
    }
  });
  (async()=>{
    try{
      await reloadList();
      const result=await api('/api/projects/v1/source-options');available=result.sources;
      form.hidden=false;
      const pathMatch=location.pathname.match(/^\/projects\/([a-zA-Z0-9-]+)$/);
      if(pathMatch)await loadDetail(pathMatch[1]);
      else msg('Create a project or reopen an existing investigation.');
    }catch(e){form.hidden=true;byId('projectAccessHelp').hidden=false;
      msg(e.status===503?'Projects are not yet enabled in this environment.':
        'You need authorized contributor access to create or resume projects.');}
  })();
})();
