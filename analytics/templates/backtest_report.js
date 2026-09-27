/* Percent fields are ratios. All untrusted labels enter the DOM through textContent.
 * A serialized update queue holds the relayout lock until every Plotly promise settles.
 * Selection uses fill_id, independent of table pages, chart modes and rounded prices. */
"use strict";
const payload = JSON.parse(document.getElementById('payload').textContent);
const D = payload.report, figures = payload.figures, $ = id => document.getElementById(id);
const days = D.account_daily.map(r => r.date), full = [days[0], days.at(-1)];
const state = {globalRange:[...full], lastInteractedRange:null, localRange:{}, selectedFillId:null,
  modes:{}, filters:{symbol:'', action:'', start:'', end:''}, page:0, group:null, linked:true};
const graphs = new Map(), pending = new Map(), byFill = new Map(D.fills.map(f => [f.fill_id,f]));
const stockIndex = new Map(D.stocks.map((s,i) => [s.symbol,i]));
let locked = false, queue = Promise.resolve();
const fmt = (x,n=2) => x == null ? '—' : Number(x).toLocaleString('zh-CN',{minimumFractionDigits:n,maximumFractionDigits:n});
const pct = x => x == null ? '—' : fmt(x*100)+'%';
function text(tag, value, parent) {const e=document.createElement(tag);e.textContent=value;if(parent)parent.append(e);return e;}
function safe(task) {return Promise.resolve().then(task).catch(e => {$('error').textContent=e.message;console.error(e);});}
function serial(task) {queue=queue.catch(()=>{}).then(async()=>{locked=true;try{await task();}finally{locked=false;}});return queue;}
function axisRange(r) {
  if(r[0]===r[1]) {
    // Center the midnight daily point with half a day on each side; UTC arithmetic
    // is used only for calendar strings, never for converting actual fill times.
    const t=Date.parse(r[0]+'T00:00:00Z');
    return [t-43200000,t+43200000].map(v=>new Date(v).toISOString().slice(0,19).replace('T',' '));
  }
  return [r[0]+' 00:00:00',r[1]+' 23:59:59'];
}
function inputs(r) {$('start').value=r[0];$('end').value=r[1];}
function normalized(r) {
  const a=String(r[0]).slice(0,10), b=String(r[1]).slice(0,10);
  if(!a || !b || a>b) throw Error('开始日期不能晚于结束日期');
  return [a<full[0]?full[0]:a>full[1]?full[1]:a,b>full[1]?full[1]:b<full[0]?full[0]:b];
}
async function rangeTo(r, group=null, global=false) {
  r=normalized(r);
  if(global || state.linked) {state.globalRange=[...r];inputs(r);group=null;}
  const groups=group ? [group] : ['account','drawdown','comparison',...D.stocks.map((_,i)=>'s'+i)];
  groups.forEach(g=>state.localRange[g]=[...r]);
  await serial(async()=>{await Promise.all([...graphs.values()].filter(g=>groups.includes(g.group)).map(async g=>{
    const target=axisRange(r), old=g.el.layout.xaxis.range;
    if(JSON.stringify(old)!==JSON.stringify(target)) await Plotly.relayout(g.el,{'xaxis.range':target,'xaxis.autorange':false});
  }));});
}
async function create(id,group) {
  const el=$(id), f=figures[id];
  f.layout.xaxis.range=axisRange(state.localRange[group]||state.globalRange);
  await Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false,showSendToCloud:false,
    modeBarButtonsToRemove:['toImage','sendDataToCloud'],scrollZoom:true});
  graphs.set(id,{el,group});
  // Range may have changed while an offscreen chart was being initialized.
  await serial(()=>Plotly.relayout(el,{'xaxis.range':axisRange(state.localRange[group]||state.globalRange),'xaxis.autorange':false}));
  el.on('plotly_relayout', event=>{
    if(locked) return;
    let r=event['xaxis.range'] || (event['xaxis.range[0]']!=null ? [event['xaxis.range[0]'],event['xaxis.range[1]']] : null);
    if(event['xaxis.autorange'])r=full;
    if(!r)return;
    safe(async()=>{r=normalized(r);state.lastInteractedRange=[...r];await rangeTo(r,group);});
  });
  if(id.endsWith('-price')) el.on('plotly_click',event=>{
    const point=event.points[0], c=point.customdata;
    if(c) {safe(()=>showDay(c[0],c[1],c[2]));return;}
    // Plotly may pick the overlapping close/MA trace instead of a visible fill glyph.
    // Resolve that day's nearest group without moving actual dates or prices.
    const stock=D.stocks[Number(group.slice(1))], date=String(point.x).slice(0,10);
    const matches=stock.markers.filter(m=>m.date===date).sort((a,b)=>Math.abs(a.weighted_price-point.y)-Math.abs(b.weighted_price-point.y));
    if(matches.length)safe(()=>showDay(stock.symbol,date,matches[0].action_type));
  });
}
async function ensureStock(i) {
  if(pending.has(i)) return pending.get(i);
  const task=(async()=>{for(const part of ['price','volume','position'])await create('s'+i+'-'+part,'s'+i);})();
  pending.set(i,task);
  try{await task;}catch(e){pending.delete(i);throw e;}
}
function filtered() {const f=state.filters;return D.fills.filter(r=>(!f.symbol||r.symbol===f.symbol)&&(!f.action||r.action_type===f.action)&&(!f.start||r.date>=f.start)&&(!f.end||r.date<=f.end));}
async function clearSelection() {
  if(state.selectedFillId){const f=byFill.get(state.selectedFillId), id='s'+stockIndex.get(f.symbol)+'-price';
    if(graphs.has(id))await Plotly.restyle($(id),{x:[[]],y:[[]],text:[[]]},[10]);}
  state.selectedFillId=null;$('details').textContent='请选择成交查看详情';
}
function renderTable() {
  const all=filtered(), max=Math.max(0,Math.ceil(all.length/50)-1);state.page=Math.min(max,state.page);
  $('rows').replaceChildren();
  for(const f of all.slice(state.page*50,state.page*50+50)) {
    const tr=document.createElement('tr');tr.dataset.fill=f.fill_id;tr.tabIndex=0;
    if(f.fill_id===state.selectedFillId)tr.classList.add('selected');
    if(state.group&&f.symbol===state.group[0]&&f.date===state.group[1]&&f.action_type===state.group[2])tr.classList.add('group');
    for(const v of [f.symbol,f.ts,f.action_type,fmt(f.price,4),fmt(f.shares,0),fmt(f.amount),fmt(f.fees),fmt(f.realized_pnl),fmt(f.shares_after,0)])text('td',v,tr);
    tr.title='实际价格原始精度：'+f.price;tr.onclick=()=>safe(()=>selectFill(f.fill_id));
    tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();safe(()=>selectFill(f.fill_id));}};
    $('rows').append(tr);
  }
  $('page').textContent=` ${state.page+1} / ${max+1} 页（${all.length} 笔） `;
  $('prev').disabled=state.page===0;$('next').disabled=state.page===max;
  const f=state.filters;$('filter-status').textContent=`筛选：${f.symbol||'全部股票'} · ${f.action||'全部动作'} · ${f.start||'开始'} 至 ${f.end||'结束'}`+(D.fills.length?'':' · 本次回测无成交');
}
async function changeFilters() {
  for(const k of ['symbol','action','start','end'])state.filters[k]=$('filter-'+k).value;
  state.page=0;
  if(state.selectedFillId&&!filtered().some(f=>f.fill_id===state.selectedFillId))await clearSelection();
  renderTable();
}
async function showDay(symbol,date,action) {
  $('filter-symbol').value=symbol;$('filter-action').value='';$('filter-start').value=date;$('filter-end').value=date;
  state.group=[symbol,date,action];await changeFilters();$('trades').scrollIntoView({block:'start'});
}
async function selectFill(id) {
  const f=byFill.get(id), i=stockIndex.get(f.symbol), group='s'+i;
  await ensureStock(i);await clearSelection();state.selectedFillId=id;
  const r=state.localRange[group]||state.globalRange;
  if(f.date<r[0]||f.date>r[1]) {const d=days.indexOf(f.date), target=[days[Math.max(0,d-10)],days[Math.min(days.length-1,d+10)]];state.lastInteractedRange=[...target];await rangeTo(target,group);}
  // A legend click (or isolating another trace) can hide this trace. An explicit
  // table selection must restore the marker without changing the other traces.
  await Plotly.restyle($(group+'-price'),{visible:true,x:[[f.date]],y:[[f.price]],text:[[f.ts+' '+f.action_type+' '+f.price+' 元']]},[10]);
  $('details').textContent=`${f.symbol} ${f.ts} ${f.action_type}\n实际成交价 ${f.price} 元；${f.shares} 股；费用 ${fmt(f.fees)} 元；卖出已实现盈亏 ${fmt(f.realized_pnl)} 元；成交后成本 ${fmt(f.cost_basis_after,6)} 元`;
  renderTable();$(group).scrollIntoView({block:'start'});
}
async function boot() {
  inputs(full);$('range').textContent=`${D.meta.start_ts} — ${D.meta.end_ts}（上海本地时间）`;
  const s=D.summary;
  for(const [label,value] of [['初始资金',fmt(s.initial_cash)],['期末权益',fmt(s.final_equity)],['累计收益',pct(s.total_return)],['成交 / 卖出笔数',s.fill_count+' / '+s.sell_count],['已实现盈亏',fmt(s.realized_pnl)],['最大回撤',pct(s.max_drawdown)]]){const e=text('div',label,$('summary'));e.className='metric';text('strong',value,e);}
  $('comparison-note').textContent='共同价格基准时刻：'+D.comparison.base_ts+'；缺少共同起点：'+(D.comparison.excluded_symbols.join('、')||'无');
  $('notes').textContent=D.meta.price_basis+'。'+D.meta.aggregation_note;
  D.warnings.forEach(w=>text('p',(w.symbol?w.symbol+'：':'')+w.message,$('warnings')));
  const observer=new IntersectionObserver(entries=>{entries.filter(e=>e.isIntersecting).forEach(e=>{observer.unobserve(e.target);safe(()=>ensureStock(Number(e.target.dataset.index)));});},{rootMargin:'400px'});
  for(const [i,stock] of D.stocks.entries()) {
    const group='s'+i;state.modes[group]='kline';
    const link=text('button',stock.symbol,$('directory'));link.onclick=()=>safe(async()=>{await ensureStock(i);$(group).scrollIntoView();});
    const option=text('option',stock.symbol,$('filter-symbol'));option.value=stock.symbol;
    const section=document.createElement('section');section.id=group;section.className='stock';section.dataset.index=i;
    text('h2',stock.symbol,section);
    text('p',`期末持仓 ${fmt(stock.daily.at(-1).shares,0)} 股 · `+(stock.markers.length?'实际成交见图与明细':'本次回测无成交')+(stock.daily.every(r=>r.close==null)?' · 无可用行情':''),section);
    const button=text('button','切换收盘折线',section);button.dataset.mode=group;
    button.onclick=()=>safe(async()=>{await ensureStock(i);const line=state.modes[group]==='kline';state.modes[group]=line?'line':'kline';await Plotly.restyle($(group+'-price'),{visible:[!line,line]},[0,1]);button.textContent=line?'切换日 K 线':'切换收盘折线';});
    for(const part of ['price','volume','position']){const div=document.createElement('div');div.id=group+'-'+part;div.className='plot '+part;section.append(div);}
    $('stocks').append(section);observer.observe(section);
  }
  for(const id of ['account','drawdown','comparison'])await create(id,id);
  for(const k of ['symbol','action','start','end'])$('filter-'+k).onchange=()=>safe(changeFilters);
  $('clear').onclick=()=>safe(async()=>{for(const k of ['symbol','action','start','end'])$('filter-'+k).value='';state.group=null;await changeFilters();});
  $('prev').onclick=()=>{state.page--;renderTable();};$('next').onclick=()=>{state.page++;renderTable();};
  $('apply').onclick=()=>safe(async()=>{const r=normalized([$('start').value,$('end').value]);await rangeTo(r,null,true);state.lastInteractedRange=r;$('error').textContent='';});
  $('reset').onclick=()=>safe(async()=>{await rangeTo(full,null,true);state.lastInteractedRange=[...full];$('error').textContent='';});
  $('link').onchange=()=>safe(async()=>{state.linked=$('link').checked;$('link-state').textContent=state.linked?'已联动':'未联动';if(state.linked)await rangeTo(state.lastInteractedRange||state.globalRange,null,true);});
  renderTable();document.body.dataset.ready='true';
}
// Exposed read-only by convention for reproducible offline browser acceptance.
window.backtestReport={state,ensureStock,selectFill,showDay,rangeTo,graphs};
safe(boot);
