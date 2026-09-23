"""Build two self-contained, interactive SVG explainers. No model data used."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

CSS = r'''
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;color:#182d42;background:#f8fafc}
*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1100px;margin-inline:auto}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#52718c;font-weight:750}
h1{font-size:clamp(23px,3.5vw,34px);margin:9px 0 8px;letter-spacing:-.04em}p{font-size:14px;line-height:1.65;margin:0;color:#526679}
.stages{display:flex;gap:8px;margin:20px 0 12px}.stage{flex:1;padding:9px;border-radius:8px;font-size:12px;background:#eaf0f5;color:#627385;text-align:center}.stage.active{background:#193d58;color:white}
svg{display:block;width:100%;height:auto;background:white;border:1px solid #dde6ee;border-radius:14px}svg text{font-family:inherit;fill:#233c51}svg .small{font-size:12px;fill:#60778a}svg .label{font-size:15px;font-weight:650}svg .large{font-size:28px;font-weight:750;letter-spacing:-.04em}
.insight{min-height:65px;margin-top:12px;padding:13px 16px;border-left:3px solid #0f9488;background:#edf8f6;border-radius:0 8px 8px 0;color:#274d50;font-size:14px;line-height:1.55}
.controls{display:flex;align-items:center;gap:12px;margin-top:16px}button{border:1px solid #cad9e4;border-radius:8px;background:white;color:#1a3d59;padding:9px 16px;font:600 13px inherit;cursor:pointer;min-width:78px}button:hover{background:#e7f1f8}button:focus-visible,input:focus-visible{outline:3px solid #3b82f6;outline-offset:3px}input{min-width:30px;flex:1;accent-color:#187c86}output{font-variant-numeric:tabular-nums;font-size:12px;color:#52718c;min-width:35px}
.caption{margin-top:12px;font-size:12px;color:#728495}.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:#526679}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
@media(max-width:540px){body{padding:12px}.stages{gap:4px}.stage{font-size:10px;padding:8px 3px}.insight{font-size:12px;min-height:89px}button{padding:8px;min-width:60px}.controls{gap:8px}svg .small{font-size:14px}}
'''

COMMON = r'''
const svg=document.querySelector('svg'), ns='http://www.w3.org/2000/svg';
function el(tag,attrs={},text=''){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;svg.append(n);return n;}
function rect(x,y,w,h,fill,more={}){return el('rect',{x,y,width:w,height:h,fill,...more});}
function txt(x,y,text,cls='small',more={}){return el('text',{x,y,class:cls,...more},text);}
const clamp=x=>Math.max(0,Math.min(1,x));
const ease=x=>{x=clamp(x);return x*x*(3-2*x);};
const lerp=(a,b,t)=>a+(b-a)*t;
const stages=[...document.querySelectorAll('.stage')],insight=document.querySelector('.insight');
function stage(index,message){stages.forEach((n,i)=>{n.classList.toggle('active',i===index);n.setAttribute('aria-current',i===index?'step':'false');});insight.textContent=message;}
'''

PLAYER = r'''
const slider=document.querySelector('#scrub'),play=document.querySelector('#play'),time=document.querySelector('output');
let progress=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,visible=true,last=0;
function paint(){render(progress);slider.value=progress*100;time.textContent=Math.round(progress*100)+'%';play.textContent=playing?'Pause':'Play';play.setAttribute('aria-label',playing?'Pause animation':'Play animation');}
play.onclick=()=>{if(progress>=1&&!playing)progress=0;playing=!playing;last=0;paint();};
document.querySelector('#replay').onclick=()=>{progress=0;playing=true;last=0;paint();};
slider.oninput=()=>{progress=Number(slider.value)/100;playing=false;paint();};
new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;last=0;}).observe(svg);
document.addEventListener('visibilitychange',()=>{last=0;});
function tick(now){if(playing&&visible&&!document.hidden){if(last)progress=Math.min(1,progress+(now-last)/14000);if(progress>=1)playing=false;paint();}last=now;requestAnimationFrame(tick);}
paint();requestAnimationFrame(tick);
if(window.parent!==window)new ResizeObserver(()=>window.parent.postMessage({type:'spark-animation-height',height:Math.ceil(document.body.getBoundingClientRect().height)+24},location.origin)).observe(document.body);
'''

REBLOCK = r'''
const colors=['#147d92','#efaa42','#8565c4','#dd728b'];
const q=[0,1,2,3,1,2,3,0,2,3,0,1,3,0,1,2],k=[2,0,3,1,0,3,1,2,3,1,2,0,1,2,0,3];
function target(types){const order=types.map((v,i)=>i).sort((a,b)=>types[a]-types[b]||a-b);return types.map((_,i)=>order.indexOf(i));}
const qp=target(q),kp=target(k),mx=488,my=114,s=20;
// Stable, balanced toy partitions: first coarse preference, then within each parent.
function parentOrder(types){return target(types.map(c=>Math.floor(c/2)));}
const qm=parentOrder(q),km=parentOrder(k);
svg.setAttribute('viewBox','0 0 850 825');
txt(28,34,'GROUP BY ATTENTION PREFERENCE','label');
txt(28,57,'16 query tokens · 4 tokens per toy block');
txt(mx,34,'THE SAME ATTENTION, REORDERED','label');
txt(mx,57,'Rows: Q · Columns: K (V follows K)');
for(let b=0;b<4;b++){
 rect(25,89+b*67,330,58,'#f5f8fb',{rx:9,stroke:'#dde6ee'});
 txt(37,113+b*67,'BLOCK '+(b+1));
}
const tokens=q.map((c,i)=>{const n=el('g');n.append(el('circle',{r:17,fill:colors[c]}));n.append(txt(0,5,''+(i+1),'small',{'text-anchor':'middle',style:'fill:white;font-weight:700'}));return n;});
const rowDots=q.map((c,i)=>el('circle',{r:6,fill:colors[c]}));
const colDots=k.map((c,i)=>el('circle',{r:6,fill:colors[c]}));
const cells=[];
for(let i=0;i<16;i++)for(let j=0;j<16;j++){
 const match=q[i]===k[j];cells.push({i,j,n:rect(0,0,s-1,s-1,match?'#218a9a':'#eaf2f5',{rx:2}),mass:Math.exp(match?2:-2)});
}
for(let i=0;i<=4;i++){
 el('line',{x1:mx,y1:my+4*s*i,x2:mx+16*s,y2:my+4*s*i,stroke:'#9bb1c1','stroke-width':1});
 el('line',{x1:mx+4*s*i,y1:my,x2:mx+4*s*i,y2:my+16*s,stroke:'#9bb1c1','stroke-width':1});
}
const selected=Array.from({length:4},()=>rect(0,0,4*s,4*s,'none',{stroke:'#df9b26','stroke-width':3,rx:3}));
txt(28,395,'EXACT-BLOCK BUDGET','small');
txt(28,426,'1 of 4 per query block','label');
txt(28,456,'Tokens and attention values stay unchanged.');
txt(488,470,'Attention mass inside selected blocks');
const massText=txt(488,505,'25.0%','large');
txt(615,503,'Illustrative toy attention','small');
const position=(idx)=>[130+(idx%4)*58,118+Math.floor(idx/4)*67];
// The tree records the same token membership as the animated permutation.
txt(28,554,'REPEAT THE SPLIT INSIDE EACH CHILD','label');
txt(28,577,'Stop when a group reaches the block size: 4 tokens in this toy.');
const treeNodes=[];
function treeNode(x,y,w,label,ids){
 const box=rect(x,y,w,48,'#f5f8fb',{rx:8,stroke:'#c5d4df','stroke-width':2});
 txt(x+w/2,y+19,label,'label',{'text-anchor':'middle'});
 txt(x+w/2,y+37,ids,'small',{'text-anchor':'middle'});
 treeNodes.push(box);
}
const edges=[];
for(const [x1,y1,x2,y2] of [[425,639,225,667],[425,639,625,667],[225,715,125,746],[225,715,325,746],[625,715,525,746],[625,715,725,746]]){
 edges.push(el('line',{x1,y1,x2,y2,stroke:'#c5d4df','stroke-width':2}));
}
treeNode(325,591,200,'Parent · 16 tokens','split into equal children');
treeNode(125,667,200,'Child A · 8 tokens','repeat split within A');
treeNode(525,667,200,'Child B · 8 tokens','repeat split within B');
for(let c=0;c<4;c++)treeNode(35+c*200,746,180,'Leaf '+(c+1)+' · 4 tokens',q.flatMap((v,i)=>v===c?[i+1]:[]).join(' · '));
function render(t){
 const first=ease((t-.12)/.2),second=ease((t-.44)/.22);
 const route=t<.12?0:t<.44?1:t<.72?2:3;
 const moving=(first>0&&first<1)||(second>0&&second<1);
 const qpos=q.map((_,i)=>lerp(lerp(i,qm[i],first),qp[i],second));
 const kpos=k.map((_,i)=>lerp(lerp(i,km[i],first),kp[i],second));
 stage(route,[
 'Start with 16 tokens. Fixed blocks mix attention preferences; the root contains every token.',
 'Split by coarse preference to fill each child’s assigned capacity.',
 'Recurse: apply the split separately inside A and B. Tokens stay within their parent, producing four leaves of 4 tokens. Stop at the leaf block size.',
 'Reorder Q and K independently so that tokens within each leaf are contiguous; V follows K. Then select attention blocks and return the output to the original Q order.'
 ][route]);
 treeNodes.forEach((n,i)=>{
   const reached=i===0||i<3&&first===1||i>=3&&second===1;
   const active=route===0?i===0:route===1?i===0:route===2?i===1||i===2:i>=3;
   n.setAttribute('fill',reached?'#edf8f6':'#f5f8fb');
   n.setAttribute('stroke',active?'#148d95':'#c5d4df');
   n.setAttribute('stroke-width',active?3:1);
 });
 edges.forEach((n,i)=>n.setAttribute('stroke',i<2&&first===1||i>=2&&second===1?'#148d95':'#c5d4df'));
 tokens.forEach((n,i)=>{const p=position(i),m=position(qm[i]),r=position(qp[i]);n.setAttribute('transform',`translate(${lerp(lerp(p[0],m[0],first),r[0],second)},${lerp(lerp(p[1],m[1],first),r[1],second)})`);});
 rowDots.forEach((n,i)=>{n.setAttribute('cx',mx-16);n.setAttribute('cy',my+(qpos[i]+.5)*s);});
 colDots.forEach((n,i)=>{n.setAttribute('cy',my-16);n.setAttribute('cx',mx+(kpos[i]+.5)*s);});
 cells.forEach(({i,j,n})=>{n.setAttribute('x',mx+kpos[j]*s);n.setAttribute('y',my+qpos[i]*s);});
 const blockMass=Array.from({length:4},()=>[0,0,0,0]);
 cells.forEach(({i,j,mass})=>{blockMass[Math.floor(qpos[i]/4)][Math.floor(kpos[j]/4)]+=mass;});
 const chosen=blockMass.map(row=>row.indexOf(Math.max(...row)));
 selected.forEach((n,b)=>{n.setAttribute('x',mx+chosen[b]*4*s);n.setAttribute('y',my+b*4*s);n.setAttribute('opacity',moving?0:1);});
 const retained=blockMass.reduce((v,row,b)=>v+row[chosen[b]],0)/cells.reduce((v,c)=>v+c.mass,0);
 massText.textContent=moving?'Moving…':(100*retained).toFixed(1)+'%';
 massText.setAttribute('font-size',moving?'21':'28');
 svg.dataset.retainedMass=retained;svg.dataset.progress=t;svg.dataset.phase=route;svg.dataset.queryOrder=JSON.stringify(qpos);svg.dataset.keyOrder=JSON.stringify(kpos);
}
'''

REWEIGHT = r'''
const logits=[-2,-1,1,2],values=[-1,-.5,.5,1],weights=logits.map(Math.exp);
const mass=weights.reduce((a,b)=>a+b,0),value=weights.reduce((a,w,i)=>a+w*values[i],0)/mass;
const exactMass=8,exactValue=.1,dense=(exactMass*exactValue+mass*value)/(exactMass+mass);
const orange='#e6a03b',teal='#148d95';
txt(28,34,'ONE COMPRESSED K/V BLOCK','label');
txt(28,57,'At representative query a · four toy tokens');
const bars=[],barLabels=[];
for(let i=0;i<4;i++){
 const x=44+i*78;
 rect(x,265-weights[i]*21,44,weights[i]*21,'none',{stroke:'#9caebb','stroke-dasharray':'4 3',rx:3});
 bars.push(rect(x,244,44,21,teal,{rx:3}));
 barLabels.push(txt(x+22,234,'1.00','small',{'text-anchor':'middle'}));
 txt(x+22,289,'z = '+logits[i],'small',{'text-anchor':'middle'});
 txt(x+22,312,'v = '+values[i],'small',{'text-anchor':'middle'});
}
txt(28,86,'Contribution to attention mass: exp(z)');
el('line',{x1:28,y1:265,x2:365,y2:265,stroke:'#c1cfd9'});
txt(28,353,'COMPRESSED MASS','small');
const massLabel=txt(28,386,'4.00','large');
txt(205,353,'COMPRESSED VALUE','small');
const valueLabel=txt(205,386,'0.000','large');
txt(28,424,'Dashed outlines: exact token contributions.');
txt(28,448,'Mean pooling underestimates the attention mass.');
txt(455,34,'EXACT + COMPRESSED BRANCHES','label');
txt(455,57,'Normalize both branches together');
txt(455,94,'Dense reference at a','label');
const denseShare=mass/(exactMass+mass),width=350;
rect(455,109,width*(1-denseShare),34,orange,{rx:4});
rect(455+width*(1-denseShare),109,width*denseShare,34,teal);
txt(455,171,'Current approximation','label');
const exactBar=rect(455,186,width*2/3,34,orange,{rx:4});
const compressedBar=rect(455+width*2/3,186,width/3,34,teal);
const shareLabel=txt(455,245,'Compressed share: 33.3%','label');
txt(455,292,'ATTENTION OUTPUT','small');
const outputLabel=txt(455,328,'0.067','large');
txt(605,328,'Dense: '+dense.toFixed(3),'label');
const errorLabel=txt(455,359,'Absolute error: '+Math.abs(.8/12-dense).toFixed(3));
rect(455,386,12,12,orange,{rx:2});txt(475,397,'Exact branch');
rect(635,386,12,12,teal,{rx:2});txt(655,397,'Compressed branch');
txt(455,432,'Exact branch stays fixed: mass 8, value 0.1.');
const formula=txt(28,498,'Block mean: mass = 4 exp(mean z), value = mean v','label');
function render(t){
 const m=ease((t-.2)/.3),v=ease((t-.6)/.25),currentMass=lerp(4,mass,m),currentValue=value*v;
 const share=currentMass/(exactMass+currentMass),output=(exactMass*exactValue+currentMass*currentValue)/(exactMass+currentMass);
 const phase=t<.2?0:t<.6?1:2;
 stage(phase,[
 'Mean pooling underestimates attention mass and treats all values equally.',
 'A log-mass bias restores the block’s attention mass.',
 'The same scores form the block’s weighted value summary.'
 ][phase]);
 bars.forEach((n,i)=>{const w=lerp(1,weights[i],m);n.setAttribute('y',265-w*21);n.setAttribute('height',w*21);barLabels[i].setAttribute('y',255-w*21);barLabels[i].textContent=w.toFixed(2);});
 massLabel.textContent=currentMass.toFixed(2);valueLabel.textContent=currentValue.toFixed(3);
 exactBar.setAttribute('width',width*(1-share));compressedBar.setAttribute('x',455+width*(1-share));compressedBar.setAttribute('width',width*share);
 shareLabel.textContent='Compressed share: '+(share*100).toFixed(1)+'%';
 outputLabel.textContent=output.toFixed(3);errorLabel.textContent='Absolute error: '+Math.abs(output-dense).toFixed(3);
 formula.textContent=phase===0?'Block mean: mass = 4 exp(mean z), value = mean v':phase===1?'Log-mass bias restores log-sum-exp(z)':'Weighted value: Σ softmax(z) · v';
 svg.dataset.mass=currentMass;svg.dataset.value=currentValue;svg.dataset.output=output;svg.dataset.dense=dense;svg.dataset.progress=t;
}
'''


def page(kind, title, description, stages, code, caption):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Spark-Attn</title><style>{CSS}</style></head><body>
<div class="eyebrow">Spark-Attn / {kind} / interactive illustration</div>
<h1>{title}</h1><p>{description}</p>
<div class="stages" aria-label="Animation stages">{''.join('<div class="stage">'+s+'</div>' for s in stages)}</div>
<svg viewBox="0 0 850 530" role="img" aria-label="{description}"><title>{title}</title><desc>{caption}</desc></svg>
<div class="insight"></div>
<div class="controls"><button id="play" type="button">Pause</button><button id="replay" type="button">Replay</button><label for="scrub" class="eyebrow">Progress</label><input id="scrub" aria-label="Animation progress" type="range" min="0" max="100" step="0.1" value="0"><output>0%</output></div>
<p class="caption">{caption}</p>
<script>{COMMON}\n{code}\n{PLAYER}</script></body></html>'''


def main():
    reblock = page('01', 'Better blocks, same attention',
        'Spark-Reblock groups tokens with similar attention preferences before block selection.',
        ['01 · Root', '02 · Split parent', '03 · Recurse in children', '04 · Leaf blocks'], REBLOCK,
        'Illustrative 16-token example, not measured model results. Colors denote four attention-preference groups. Toy blocks contain 4 tokens; production blocks contain 64. The binary splits illustrate recursion with predetermined toy preferences; production uses learned similarity, configured fanout (8-way in the blog ablation), and 64-token leaves.')
    reweight = page('02', 'Reweight the summary, restore the mass',
        'Spark-Reweight combines weighted summaries with a log-mass bias.',
        ['01 · Mean pooling', '02 · Restore attention mass', '03 · Weighted value'], REWEIGHT,
        'Illustrative scalar example, not measured model results. The exact branch has mass 8 and value 0.1. The four compressed tokens have logits [-2, -1, 1, 2] and values [-1, -0.5, 0.5, 1]. Corrections are shown in stages for explanation; the implementation computes them together.')
    for name, content in [('spark-reblock.html', reblock), ('spark-reweight.html', reweight)]:
        (HERE/name).write_text(content)
        print(name, len(content.encode()), 'bytes')


if __name__ == '__main__':
    main()
