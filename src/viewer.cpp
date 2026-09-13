#include "surface.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace minimal {
namespace {
void write_vertices(std::ostream &out, const std::vector<Vec3> &vertices) {
    out << '[';
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (i)
            out << ',';
        out << '[' << vertices[i].x << ',' << vertices[i].y << ',' << vertices[i].z << ']';
    }
    out << ']';
}

void write_faces(std::ostream &out, const std::vector<std::array<int, 3>> &faces) {
    out << '[';
    for (size_t i = 0; i < faces.size(); ++i) {
        if (i)
            out << ',';
        out << '[' << faces[i][0] << ',' << faces[i][1] << ',' << faces[i][2] << ']';
    }
    out << ']';
}
} // namespace

void write_html(const Mesh &m, const std::string &path,
                const std::vector<AnimationTopology> &topologies,
                const std::vector<AnimationFrame> &frames) {
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Не удалось записать HTML.");
    out << std::setprecision(17);
    out << R"HTML(<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Эволюция минимальной поверхности</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#111d2eeb;--line:#28435d;--text:#eef5ff;--muted:#9eb2c8;--accent:#5ed3cf;--rim:#ff776e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;touch-action:none}.panel{position:absolute;top:20px;left:20px;width:min(410px,calc(100vw - 40px));padding:18px;background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 14px 40px #0007;backdrop-filter:blur(10px)}
h1{font-size:21px;margin:0 0 8px}.stats{color:var(--muted);line-height:1.5;min-height:44px}.area{font-size:26px;color:var(--accent);font-variant-numeric:tabular-nums;margin:10px 0 2px}.reduction{color:var(--muted);font-size:13px}.controls{position:absolute;left:20px;right:20px;bottom:20px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 14px 40px #0007;backdrop-filter:blur(10px)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.timeline{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin-top:12px}button,select,label{border:1px solid var(--line);background:#17283d;color:var(--text);border-radius:8px;padding:8px 11px;font:inherit}button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:not-allowed}label{display:flex;gap:7px;align-items:center;padding:7px 10px}input[type=range]{width:100%;accent-color:var(--accent)}.hint{position:absolute;right:24px;top:24px;color:var(--muted);font-size:13px;text-align:right;pointer-events:none}.badge{display:inline-block;color:var(--rim);margin-top:6px;font-size:13px}
@media(max-width:700px){.panel{top:10px;left:10px;width:calc(100vw - 20px);padding:13px}.controls{left:10px;right:10px;bottom:10px}.hint{display:none}.timeline{grid-template-columns:1fr}.timeline span{display:none}}
</style>
</head>
<body>
<canvas id="view" aria-label="Анимация минимизации треугольной поверхности"></canvas>
<section class="panel">
  <h1>Эволюция минимальной поверхности</h1>
  <div class="area" id="area"></div>
  <div class="reduction" id="reduction"></div>
  <div class="stats" id="stats"></div>
  <div class="badge">Красный контур закреплён</div>
</section>
<div class="hint">Вращение — перетаскивание<br>Масштаб — колесо мыши</div>
<section class="controls">
  <div class="row">
    <button id="play">▶ Запустить</button>
    <button id="reset">↺ В начало</button>
    <label>Скорость <select id="speed"><option value="0.5">0,5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
    <label><input id="wire" type="checkbox" checked> Сетка</label>
    <label><input id="ghost" type="checkbox"> Исходная поверхность</label>
    <button id="record">Скачать WebM</button>
  </div>
  <div class="timeline"><span>Исходная</span><input id="timeline" type="range" min="0" step="0.01"><span>Минимальная</span></div>
</section>
<script>
const scaleFactor=)HTML"
        << m.scale << ", topologies=";
    out << '[';
    for (size_t i = 0; i < topologies.size(); ++i) {
        if (i)
            out << ',';
        write_faces(out, topologies[i].faces);
    }
    out << "], frames=[";
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i)
            out << ',';
        out << "{v:";
        write_vertices(out, frames[i].vertices);
        out << ",t:" << frames[i].topology << ",a:" << frames[i].area << ",r:" << frames[i].residual
            << ",l:" << frames[i].level << ",i:" << frames[i].iteration << '}';
    }
    out << R"HTML(];
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const playButton=document.getElementById('play'),resetButton=document.getElementById('reset');
const timeline=document.getElementById('timeline'),speed=document.getElementById('speed');
const wire=document.getElementById('wire'),ghost=document.getElementById('ghost'),recordButton=document.getElementById('record');
timeline.max=Math.max(0,frames.length-1);recordButton.disabled=frames.length<2||!window.MediaRecorder||!canvas.captureStream;
let playhead=0,playing=false,lastTime=0,playGeneration=0,yaw=.42,pitch=-.9,zoom=1,drag=null,recorder=null,chunks=[];
const bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
for(const frame of frames)for(const p of frame.v)for(let k=0;k<3;k++){bounds.min[k]=Math.min(bounds.min[k],p[k]);bounds.max[k]=Math.max(bounds.max[k],p[k]);}
const center=bounds.min.map((v,k)=>(v+bounds.max[k])/2),radius=Math.max(...bounds.max.map((v,k)=>v-bounds.min[k]))||1;
const rims=topologies.map(faces=>{const edges=new Map();for(const f of faces)for(let i=0;i<3;i++){const a=f[i],b=f[(i+1)%3],key=[Math.min(a,b),Math.max(a,b)].join(',');edges.set(key,(edges.get(key)||0)+1);}return [...edges].filter(([,n])=>n===1).map(([key])=>key.split(',').map(Number));});
function current(){
 const left=Math.floor(playhead),right=Math.min(frames.length-1,left+1),mix=playhead-left,a=frames[left],b=frames[right];
 if(a.t!==b.t||a.v.length!==b.v.length)return mix<.5?{...a,mix:0}:{...b,mix:0};
 return {v:a.v.map((p,j)=>p.map((x,k)=>x+(b.v[j][k]-x)*mix)),t:a.t,a:a.a+(b.a-a.a)*mix,r:a.r+(b.r-a.r)*mix,l:mix<.5?a.l:b.l,i:mix<.5?a.i:b.i,mix};
}
function project(points,w,h){
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),size=Math.min(w,h)*.72*zoom/radius;
 return points.map(p=>{const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=cy*x-sy*y,b=sy*x+cy*y;return {p:[w/2+a*size,h*.54-(cp*b-sp*z)*size],z:sp*b+cp*z,r:[a,cp*b-sp*z,sp*b+cp*z]};});
}
function surface(points,faces,alpha,wireOnly,w,h){
 const q=project(points,w,h),sorted=faces.map(f=>({f,z:f.reduce((s,i)=>s+q[i].z,0)/3})).sort((a,b)=>a.z-b.z);
 ctx.globalAlpha=alpha;
 for(const {f} of sorted){const [a,b,c]=f.map(i=>q[i].r),u=b.map((x,i)=>x-a[i]),v=c.map((x,i)=>x-a[i]);const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],light=.28+.72*Math.abs((n[0]*.25+n[1]*.4+n[2]*.88)/(Math.hypot(...n)||1));
  ctx.beginPath();f.forEach((i,j)=>j?ctx.lineTo(...q[i].p):ctx.moveTo(...q[i].p));ctx.closePath();
  if(!wireOnly){ctx.fillStyle=`hsl(181 52% ${24+light*38}%)`;ctx.fill();}
  if(wire.checked||wireOnly){ctx.strokeStyle=wireOnly?'#dceaff':'#153d4a';ctx.lineWidth=wireOnly?.65:.5;ctx.stroke();}}
 ctx.globalAlpha=1;return q;
}
function draw(){
 const dpr=devicePixelRatio||1,w=innerWidth,h=innerHeight,cw=Math.round(w*dpr),ch=Math.round(h*dpr);if(canvas.width!==cw||canvas.height!==ch){canvas.width=cw;canvas.height=ch;}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#0b1220';ctx.fillRect(0,0,w,h);
 if(!frames.length)return;const frame=current(),faces=topologies[frame.t];
 const q=surface(frame.v,faces,1,false,w,h);if(ghost.checked&&playhead>.01)surface(frames[0].v,topologies[frames[0].t],.45,true,w,h);ctx.strokeStyle='#ff776e';ctx.lineWidth=2.2;ctx.beginPath();for(const [a,b] of rims[frame.t]){ctx.moveTo(...q[a].p);ctx.lineTo(...q[b].p);}ctx.stroke();
 const initial=frames[0].a,reduction=initial?100*(initial-frame.a)/initial:0;
 document.getElementById('area').textContent=`Площадь: ${frame.a.toPrecision(9)}`;
 document.getElementById('reduction').textContent=`Изменение от начала: ${reduction.toFixed(3)}%`;
 document.getElementById('stats').textContent=`Кадр ${Math.min(frames.length,Math.floor(playhead)+1)} из ${frames.length} · уровень сетки ${frame.l} · итерация ${frame.i} · ${frame.v.length} вершин · ${faces.length} треугольников · невязка ${frame.r.toExponential(2)}`;
 timeline.value=playhead;
}
function setPlaying(value){playing=value;const generation=++playGeneration;playButton.textContent=playing?'Ⅱ Пауза':'▶ Запустить';if(playing){if(playhead>=frames.length-1)playhead=0;lastTime=performance.now();requestAnimationFrame(now=>tick(now,generation));}}
function finishRecording(){if(recorder&&recorder.state==='recording')setTimeout(()=>recorder.stop(),250);}
function tick(now,generation){if(!playing||generation!==playGeneration)return;const dt=Math.min(100,now-lastTime);lastTime=now;playhead+=dt*Number(speed.value)/650;if(playhead>=frames.length-1){playhead=frames.length-1;setPlaying(false);draw();finishRecording();return;}draw();requestAnimationFrame(next=>tick(next,generation));}
playButton.onclick=()=>{if(playing){setPlaying(false);return;}const requested=Number(timeline.value);if(Number.isFinite(requested))playhead=Math.max(0,Math.min(frames.length-1,requested));setPlaying(true);};resetButton.onclick=()=>{setPlaying(false);playhead=0;draw();};timeline.onpointerdown=()=>setPlaying(false);timeline.oninput=()=>{setPlaying(false);playhead=Number(timeline.value);draw();};
wire.onchange=ghost.onchange=draw;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw();};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(4,Math.max(.2,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
recordButton.onclick=()=>{setPlaying(false);chunks=[];playhead=0;draw();const stream=canvas.captureStream(30),options={};for(const type of ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'])if(MediaRecorder.isTypeSupported(type)){options.mimeType=type;break;}recorder=new MediaRecorder(stream,options);recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};recorder.onstop=()=>{const blob=new Blob(chunks,{type:recorder.mimeType||'video/webm'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='minimal-surface.webm';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);for(const control of [playButton,resetButton,timeline,speed,recordButton])control.disabled=false;};for(const control of [playButton,resetButton,timeline,speed,recordButton])control.disabled=true;recorder.start();setPlaying(true);};
onresize=draw;draw();
</script>
</body>
</html>)HTML";
    if (!out)
        throw std::runtime_error("Ошибка записи HTML.");
}
} // namespace minimal
