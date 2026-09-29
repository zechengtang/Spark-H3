"""Build a local, static visual gallery for the 72 ComfyUI VDN8 videos.

The output keeps the measured data beside the videos. It creates a symlink to
the original media, so the 7.7 GiB of MP4 files are not duplicated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_vdn8_four_model_72_20260929")
VIDEOS = Path("/autodl-fs/data/h3_outputs/comfyui_vdn8_four_model_72_20260929")
MODELS = ("minimax_h3", "lightx2v", "larryvrh", "comfyui")
DURATIONS = ("5s", "10s", "14p4s")
METHODS = ("dense", "official_sol", "spark_10pct", "spark_20pct", "spark_114blocks", "spark_228blocks")


def build_data() -> dict:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    scores = {
        (row["model"], row["duration"], row["method"]): row
        for row in csv.DictReader((ROOT / "results.csv").open(newline=""))
    }
    cases = {case["label"]: case for case in protocol["cases"]}
    data = {"seed": protocol["seed"], "cases": cases, "results": {}}
    for model in MODELS:
        data["results"][model] = {}
        for duration in DURATIONS:
            data["results"][model][duration] = {}
            dense_record = json.loads((ROOT / model / "records" / f"{duration}_dense.json").read_text())
            dense_seconds = dense_record["sampler_seconds"]
            for method in METHODS:
                file = VIDEOS / model / f"{duration}_{method}.mp4"
                if not file.is_file():
                    raise FileNotFoundError(file)
                row = scores.get((model, duration, method))
                data["results"][model][duration][method] = {
                    "src": f"videos/{model}/{duration}_{method}.mp4",
                    "poster": f"posters/{model}/{duration}_{method}.jpg",
                    "seconds": dense_seconds if row is None else float(row["sparse_sampler_seconds"]),
                    "speedup": None if row is None else float(row["speedup"]),
                    "psnr": None if row is None else float(row["psnr_db"]),
                    "ssim": None if row is None else float(row["ssim"]),
                    "lpips": None if row is None else float(row["lpips"]),
                    "timing_note": (
                        "首个稀疏步含明显初始化开销"
                        if model in ("lightx2v", "comfyui") and duration == "14p4s" and method == "spark_10pct"
                        else None
                    ),
                }
    return data


def create_posters() -> None:
    """First-frame posters let the page remain useful before video downloads."""
    for model in MODELS:
        (ROOT / "posters" / model).mkdir(parents=True, exist_ok=True)
        for duration in DURATIONS:
            for method in METHODS:
                source = VIDEOS / model / f"{duration}_{method}.mp4"
                target = ROOT / "posters" / model / f"{duration}_{method}.jpg"
                if target.is_file():
                    continue
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(source),
                     "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "4", "-y", str(target)],
                    check=True,
                )


def main() -> None:
    data = build_data()
    media_link = ROOT / "videos"
    if media_link.is_symlink():
        if media_link.resolve() != VIDEOS.resolve():
            raise RuntimeError(f"unexpected video link: {media_link}")
    elif media_link.exists():
        raise RuntimeError(f"video link path already exists: {media_link}")
    else:
        media_link.symlink_to(VIDEOS, target_is_directory=True)
    create_posters()
    script_data = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = TEMPLATE.replace("__DATA__", script_data)
    (ROOT / "index.html").write_text(page)
    print(ROOT / "index.html")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Prompt 08 · ComfyUI H3 视频对照</title>
  <style>
    :root{--ink:#163026;--dark:#14281f;--paper:#f3f4ed;--card:#fff;--line:#d8dfd4;--muted:#637369;--accent:#b8ed81;--green:#507f3a}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
    button{font:inherit}button:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #7fb945;outline-offset:3px}
    .hero{background:var(--dark);color:#eef4e9;padding:clamp(28px,5vw,72px) max(24px,calc((100vw - 1512px)/2));border-bottom:5px solid var(--accent)}
    .eyebrow{color:var(--accent);font:700 11px/1.4 ui-monospace,monospace;letter-spacing:.18em;text-transform:uppercase}
    h1{font:500 clamp(38px,5vw,76px)/1.09 Georgia,"Noto Serif CJK SC",serif;letter-spacing:-.035em;margin:16px 0 20px;max-width:960px}
    .hero p{max-width:880px;margin:0;color:#c3d1c5;font-size:17px;line-height:1.75}
    .hero-meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:28px}.hero-meta span{border:1px solid #49604f;color:#d9e6d7;padding:8px 11px;font:12px ui-monospace,monospace}
    main{max-width:1560px;margin:auto;padding:34px 24px 72px}
    .controls{display:grid;grid-template-columns:1fr auto;gap:20px 28px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:28px}
    .control-title{font:700 10px ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 11px}
    .switches{display:flex;flex-wrap:wrap;gap:8px}.switches button,.actions button{border:1px solid #bdcbb9;background:#fff;color:var(--ink);padding:10px 15px;cursor:pointer;transition:background .15s,border-color .15s,transform .15s}
    .switches button:hover,.actions button:hover{border-color:var(--green);transform:translateY(-1px)}.switches button[aria-pressed="true"]{background:var(--dark);color:#ecf3e8;border-color:var(--dark)}
    .actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.actions button{white-space:nowrap}.actions button.primary{background:#ddefc9;border-color:#b2d194}
    .case-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin:35px 0 17px}.case-head h2{font:500 clamp(25px,3vw,40px)/1.2 Georgia,"Noto Serif CJK SC",serif;margin:0}.case-head p{color:var(--muted);margin:6px 0 0}.legend{font:12px/1.6 ui-monospace,monospace;color:var(--muted);text-align:right}
    .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{min-width:0;border:1px solid var(--line);background:var(--card);box-shadow:0 8px 24px #1630260a;overflow:hidden}.card.dense{border-color:#91ab81}
    .card-head{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:16px 16px 13px}.card-title{margin:0;font-size:17px;font-weight:650}.card-index{font:700 11px ui-monospace,monospace;color:var(--green);margin-right:10px}.card-kind{font:700 10px ui-monospace,monospace;color:var(--green);text-transform:uppercase;letter-spacing:.1em;white-space:nowrap}
    video{display:block;width:100%;aspect-ratio:7/4;object-fit:cover;background:#14201b}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--line)}.metric{padding:12px 13px;border-right:1px solid var(--line);min-width:0}.metric:last-child{border-right:0}.metric label{display:block;text-transform:uppercase;color:var(--muted);font:700 9px/1.4 ui-monospace,monospace;letter-spacing:.05em}.metric strong{display:block;margin-top:4px;font:600 clamp(12px,1vw,18px)/1.2 ui-monospace,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}.metric small{color:var(--muted);font-size:10px}.speed{padding:9px 14px;background:#edf4e7;color:#456f31;font:700 11px ui-monospace,monospace}.speed.base{background:#e9eee8;color:#5f7063}
    details.prompt{margin-top:28px;background:var(--dark);color:#edf4e9;border:1px solid #2b4635}details.prompt summary{list-style:none;cursor:pointer;padding:18px 22px;display:flex;justify-content:space-between;gap:20px;align-items:center}details.prompt summary::-webkit-details-marker{display:none}.prompt-label{font:700 11px ui-monospace,monospace;color:var(--accent);letter-spacing:.13em;text-transform:uppercase}.prompt-sub{color:#adc2b1;font-size:13px}.prompt-body{white-space:pre-wrap;overflow-wrap:anywhere;max-height:520px;overflow:auto;border-top:1px solid #34503c;padding:20px 22px;font:15px/1.8 Georgia,"Noto Serif CJK SC",serif}
    .notes{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:35px;padding-top:25px;border-top:1px solid var(--line)}.notes h3{font-size:16px;margin:0 0 8px}.notes p{margin:0;color:#4d5e53}.notes a{color:#315c32;text-underline-offset:3px}.warn{color:#8c542b!important}
    @media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{grid-template-columns:1fr}.actions{justify-content:flex-start}}
    @media(max-width:650px){main{padding:24px 14px 50px}.hero{padding:38px 19px}.hero p{font-size:15px}.grid{grid-template-columns:1fr}.case-head{display:block}.legend{text-align:left;margin-top:12px}.metric strong{font-size:15px}.notes{grid-template-columns:1fr}.switches button{padding:9px 12px;font-size:13px}}
  </style>
</head>
<body>
  <header class="hero">
    <div class="eyebrow">MiniMax-H3 / ComfyUI / Prompt 08</div>
    <h1>同一 Prompt，六种注意力方案</h1>
    <p>选定模型和时长后，直接并排观看 Dense、官方 Sol 与四种 Spark 输出。每组保持相同的 prompt、seed、模型和采样步数；画质指标以该组 Dense 视频为参照。</p>
    <div class="hero-meta"><span>72 条带音频视频</span><span>4 个模型 × 3 个时长 × 6 种方案</span><span>1344 × 768 · 24 fps · seed 42</span></div>
  </header>
  <main>
    <section class="controls" aria-label="结果筛选">
      <div><p class="control-title">模型 / sampler steps</p><div class="switches" id="models"></div></div>
      <div><p class="control-title">时长 / frames</p><div class="switches" id="durations"></div></div>
      <div class="actions"><button type="button" class="primary" id="play-all">同时播放（静音）</button><button type="button" id="pause-all">全部暂停</button><button type="button" id="reset-all">回到开头</button></div>
    </section>
    <div class="case-head"><div><h2 id="case-title"></h2><p id="case-subtitle"></p></div><div class="legend">PSNR ↑ · SSIM ↑ · LPIPS ↓<br>去噪时间仅含 Sampler，单位秒</div></div>
    <section class="grid" id="grid" aria-live="polite" aria-label="视频对照"></section>
    <details class="prompt"><summary><span><span class="prompt-label">完整 Prompt</span><br><span class="prompt-sub">当前时长使用的原文，点击展开</span></span><span aria-hidden="true">＋</span></summary><div class="prompt-body" id="prompt-text"></div></details>
    <section class="notes"><div><h3>如何阅读</h3><p>每张卡片都能单独播放并打开声音。“同时播放”会将六段视频静音，以便对齐比较。PSNR、SSIM、LPIPS 比较视频帧与同组 Dense 的接近程度，不评价声音或绝对观感。</p></div><div><h3>计时范围</h3><p>页面显示原实验的单次去噪计时。每模型只做过 5 秒 Dense 两步预热，稀疏模式与各时长没有逐一预热；14.4 秒的 LightX2V、ComfyUI 官方 LoRA 的 Spark 10% 首个稀疏步含明显初始化开销。<a href="results.csv">下载全部评分与计时</a> · <a href="report.md">实验报告</a></p></div></section>
  </main>
  <script id="experiment-data" type="application/json">__DATA__</script>
  <script>
  (() => {
    const data = JSON.parse(document.getElementById('experiment-data').textContent);
    const models = [['minimax_h3','MiniMax-H3 · 20 步'],['lightx2v','LightX2V · 8 步'],['larryvrh','Larryvrh · 8 步'],['comfyui','ComfyUI 官方 LoRA · 8 步']];
    const durations = [['5s','5s · 124 帧'],['10s','10s · 243 帧'],['14p4s','14.4s · 345 帧']];
    const methods = [['dense','Dense'],['official_sol','官方 Sol'],['spark_10pct','Spark 10%'],['spark_20pct','Spark 20%'],['spark_114blocks','Spark 114 blocks'],['spark_228blocks','Spark 228 blocks']];
    const state = {model:'minimax_h3', duration:'5s'};
    const $ = id => document.getElementById(id);
    const format = (v,d) => v === null ? '—' : Number(v).toFixed(d);
    function makeSwitches(id, items, key) {
      const root=$(id); root.replaceChildren();
      for(const [value,label] of items){
        const b=document.createElement('button'); b.type='button'; b.textContent=label;
        b.setAttribute('aria-pressed',String(state[key]===value));
        b.addEventListener('click',()=>{state[key]=value; render()}); root.append(b);
      }
    }
    function metric(label,value,unit='') {
      const box=document.createElement('div'); box.className='metric';
      const l=document.createElement('label');l.textContent=label;
      const strong=document.createElement('strong');strong.textContent=value;
      if(unit){const small=document.createElement('small');small.textContent=' '+unit;strong.append(small)}
      box.append(l,strong);return box;
    }
    function makeCard(method,index,row) {
      const article=document.createElement('article');article.className='card'+(method==='dense'?' dense':'');
      const head=document.createElement('div');head.className='card-head';
      const title=document.createElement('h3');title.className='card-title';
      const no=document.createElement('span');no.className='card-index';no.textContent=String(index+1).padStart(2,'0');
      title.append(no,document.createTextNode(methods[index][1]));
      const kind=document.createElement('span');kind.className='card-kind';kind.textContent=method==='dense'?'Reference':'Variant';head.append(title,kind);
      const video=document.createElement('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.poster=row.poster;video.src=row.src;video.setAttribute('aria-label',methods[index][1]+' 视频');
      const speed=document.createElement('div');speed.className='speed'+(method==='dense'?' base':'');speed.textContent=method==='dense'?'Dense 基准':`${format(row.speedup,2)}× 相对于 Dense${row.timing_note?' · 首步含初始化':''}`;
      if(row.timing_note)speed.title=row.timing_note;
      const metrics=document.createElement('div');metrics.className='metrics';
      metrics.append(metric('PSNR',format(row.psnr,2),'dB'),metric('SSIM',format(row.ssim,3)),metric('LPIPS',format(row.lpips,3)),metric('去噪',format(row.seconds,1),'s'));
      article.append(head,video,speed,metrics);return article;
    }
    function render() {
      for(const video of $('grid').querySelectorAll('video')){video.pause();video.removeAttribute('src');video.load()}
      makeSwitches('models',models,'model');makeSwitches('durations',durations,'duration');
      const model=models.find(x=>x[0]===state.model)[1].split(' · ')[0];
      const duration=durations.find(x=>x[0]===state.duration)[1];
      const c=data.cases[state.duration];
      $('case-title').textContent=model+' / '+duration;
      $('case-subtitle').textContent=`VDN prompt 8 · ${c.width} × ${c.height} · 24 fps · seed ${data.seed}`;
      $('prompt-text').textContent=c.prompt;
      const grid=$('grid');grid.replaceChildren();
      methods.forEach(([method],index)=>grid.append(makeCard(method,index,data.results[state.model][state.duration][method])));
      const hash=`${state.model}/${state.duration}`;history.replaceState(null,'','#'+hash);
    }
    $('play-all').addEventListener('click',()=>{$('grid').querySelectorAll('video').forEach(v=>{v.muted=true;v.currentTime=0;v.play().catch(()=>{})})});
    $('pause-all').addEventListener('click',()=>{$('grid').querySelectorAll('video').forEach(v=>v.pause())});
    $('reset-all').addEventListener('click',()=>{$('grid').querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0})});
    const match=location.hash.slice(1).split('/');if(models.some(x=>x[0]===match[0]))state.model=match[0];if(durations.some(x=>x[0]===match[1]))state.duration=match[1];
    render();
  })();
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
