#include "surface.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace minimal {
void write_html(const Mesh &m, const std::string &path) {
    std::ofstream o(path);
    if (!o)
        throw std::runtime_error("Не удалось записать HTML.");
    o << R"HTML(<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Минимальная поверхность</title><style>
*{box-sizing:border-box}body{margin:0;background:#101823;color:#e9eef6;font:16px system-ui}header{position:absolute;top:24px;left:28px;right:24px;pointer-events:none}h1{font-size:24px;margin:0 0 10px}p{color:#a9bacf;margin:6px 0;font-size:14px}label{pointer-events:auto;display:inline-block;margin-top:12px;padding:8px;background:#233449;border-radius:6px}canvas{display:block;width:100vw;height:100vh;touch-action:none}footer{position:absolute;bottom:20px;left:28px;color:#a9bacf;font-size:13px}</style>
<canvas id="view" aria-label="Трёхмерная треугольная поверхность"></canvas><header><h1>Минимальная поверхность</h1><p id="stats"></p><p>Красный контур — закреплённая граница. Поверхность — плоские треугольники.</p><label><input id="wire" type="checkbox" checked> Показать сетку</label></header><footer>Вращение: перетащите мышью или пальцем · Масштаб: колесо мыши</footer><script>
const vertices=)HTML"
      << std::setprecision(17) << '[';
    for (size_t i = 0; i < m.vertices.size(); ++i) {
        auto p = m.vertices[i];
        if (i)
            o << ',';
        o << '[' << p.x << ',' << p.y << ',' << p.z << ']';
    }
    o << "], faces=[";
    for (size_t i = 0; i < m.faces.size(); ++i) {
        auto f = m.faces[i];
        if (i)
            o << ',';
        o << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']';
    }
    o << "], boundary=[";
    for (size_t i = 0; i < m.boundary.size(); ++i) {
        if (i)
            o << ',';
        o << (m.boundary[i] ? "true" : "false");
    }
    o << "], area=" << evaluate(m, false).area * m.scale * m.scale << ";\n";
    o << R"HTML(
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),wire=document.getElementById('wire');
document.getElementById('stats').textContent=`${vertices.length} вершин · ${faces.length} треугольников · площадь ${area.toPrecision(9)}`;
const center=[0,0,0];vertices.forEach(p=>p.forEach((v,i)=>center[i]+=v/vertices.length));
const points=vertices.map(p=>p.map((v,i)=>v-center[i]));let yaw=.4,pitch=-.9,zoom=1,drag=null;
const edges=new Map();for(const f of faces)for(let i=0;i<3;i++){const a=f[i],b=f[(i+1)%3],key=[Math.min(a,b),Math.max(a,b)].join(',');edges.set(key,(edges.get(key)||0)+1);}
const rim=[...edges].filter(([k,n])=>n===1).map(([k])=>k.split(',').map(Number));
function draw(){const dpr=devicePixelRatio||1,w=innerWidth,h=innerHeight;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),scale=Math.min(w,h)*.68*zoom;
 const rotated=points.map(([x,y,z])=>{const a=cy*x-sy*y,b=sy*x+cy*y;return [a,cp*b-sp*z,sp*b+cp*z];});
 const screen=rotated.map(p=>[w/2+p[0]*scale,h*.57-p[1]*scale]);
 const sorted=faces.map(f=>({f,z:f.reduce((s,i)=>s+rotated[i][2],0)/3})).sort((a,b)=>a.z-b.z);
 for(const {f} of sorted){const [a,b,c]=f.map(i=>rotated[i]),u=b.map((v,i)=>v-a[i]),v=c.map((v,i)=>v-a[i]);const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];const light=.3+.7*Math.abs((n[0]*.25+n[1]*.4+n[2]*.88)/(Math.hypot(...n)||1));
 ctx.beginPath();f.forEach((i,j)=>j?ctx.lineTo(...screen[i]):ctx.moveTo(...screen[i]));ctx.closePath();ctx.fillStyle=`hsl(187 47% ${25+light*35}%)`;ctx.fill();if(wire.checked){ctx.strokeStyle='#173b4990';ctx.lineWidth=.55;ctx.stroke();}}
 ctx.strokeStyle='#ff776e';ctx.lineWidth=2;ctx.beginPath();for(const [a,b] of rim){ctx.moveTo(...screen[a]);ctx.lineTo(...screen[b]);}ctx.stroke();}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw();};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(4,Math.max(.2,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});wire.onchange=draw;onresize=draw;draw();
</script></html>)HTML";
    if (!o)
        throw std::runtime_error("Ошибка записи HTML.");
}
} // namespace minimal
