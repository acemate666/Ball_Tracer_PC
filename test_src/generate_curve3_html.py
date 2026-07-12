# -*- coding: utf-8 -*-
"""
Generate interactive HTML directly from JSON data.

Supported inputs:
1. tracker_*.json        Raw tracker output
2. *_replay.json         Replay output from test_curve3_replay.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _merge_racket_json(base_data: dict, racket_data: dict, racket_json_path: str | None) -> dict:
    merged = copy.deepcopy(base_data)
    merged_cfg = merged.setdefault("config", {})
    merged_summary = merged.setdefault("summary", {})
    merged_frames = merged.setdefault("frames", [])
    racket_cfg = racket_data.get("config", {})
    racket_summary = racket_data.get("summary", {})

    if racket_json_path:
        merged_cfg["racket_json_path"] = str(racket_json_path)

    for key in (
        "distance_unit",
        "first_frame_exposure_pc",
        "video_frame_mapping_exact",
        "racket_model_path",
        "racket_conf_threshold",
    ):
        if key in racket_cfg:
            merged_cfg[key] = racket_cfg[key]

    for key in (
        "video_frame_mapping_exact",
        "video_frames_mapped",
        "racket_observations_3d",
        "racket_frames_processed",
    ):
        if key in racket_summary:
            merged_summary[key] = racket_summary[key]

    if "racket_observations" in racket_data:
        merged["racket_observations"] = racket_data["racket_observations"]

    frame_by_idx = {
        frame.get("idx"): frame
        for frame in merged_frames
        if isinstance(frame, dict) and isinstance(frame.get("idx"), int)
    }

    for racket_frame in racket_data.get("frames", []):
        if not isinstance(racket_frame, dict):
            continue
        frame_idx = racket_frame.get("idx")
        target = frame_by_idx.get(frame_idx) if isinstance(frame_idx, int) else None
        if target is None:
            target = {}
            if isinstance(frame_idx, int):
                target["idx"] = frame_idx
                frame_by_idx[frame_idx] = target
            merged_frames.append(target)
        for key, value in racket_frame.items():
            if key == "idx":
                continue
            target[key] = value

    merged["frames"] = sorted(
        merged_frames,
        key=lambda frame: (
            0,
            frame.get("idx"),
        )
        if isinstance(frame, dict) and isinstance(frame.get("idx"), int)
        else (1, 0),
    )
    return merged


def generate_html(
    input_path: str,
    output_path: str,
    racket_json_path: str | None = None,
    arm_json_path: str | None = None,
    rk_tracking_json_path: str | None = None,
) -> None:
    data = _load_json(input_path)
    if racket_json_path:
        data = _merge_racket_json(data, _load_json(racket_json_path), racket_json_path)
    if arm_json_path:
        data["arm"] = _load_json(arm_json_path)
    if rk_tracking_json_path:
        data["rk_tracking"] = _load_json(rk_tracking_json_path)
    data_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("%%DATA_JSON%%", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Interactive HTML saved: {output_path}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Tracker / Curve3 Interactive</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0}
.hdr{padding:16px 24px;background:#16213e;border-bottom:1px solid #0f3460}
.hdr h1{font-size:20px;color:#e94560;margin-bottom:6px}
.hdr .st{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#a0a0c0}
.hdr .st span{background:#0f3460;padding:3px 10px;border-radius:4px}
.hdr .st .v{color:#e94560;font-weight:600}
.tabs{display:flex;gap:3px;padding:10px 24px 0}
.tab{padding:7px 18px;cursor:pointer;background:#16213e;border:1px solid #0f3460;border-bottom:none;
     border-radius:6px 6px 0 0;font-size:12px;color:#a0a0c0;user-select:none}
.tab:hover{background:#1a1a3e;color:#fff}
.tab.on{background:#1a1a2e;color:#e94560;border-color:#e94560;border-bottom:1px solid #1a1a2e}
.pnl{display:none}.pnl.on{display:block}
.cc{width:100%;padding:8px 20px}
.zt{display:flex;justify-content:flex-end;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 10px}
.ztl{font-size:12px;color:#a0a0c0}
.zb{appearance:none;border:1px solid #0f3460;background:#16213e;color:#d7d7eb;border-radius:999px;padding:4px 10px;
    font:inherit;font-size:12px;cursor:pointer;transition:background .18s ease,border-color .18s ease,transform .18s ease}
.zb:hover{background:#1a1a3e;border-color:#e94560;transform:translateY(-1px)}
.zb.on{background:#0f3460;border-color:#5cd0ff;color:#fff}
.zr{font-size:12px;color:#a0a0c0;min-width:44px;text-align:right}
.lc{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 10px}
.lb{appearance:none;display:inline-flex;align-items:center;gap:8px;border:1px solid #0f3460;background:#16213e;color:#d7d7eb;user-select:none;
    border-radius:999px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer;transition:background .18s ease,border-color .18s ease,opacity .18s ease,transform .18s ease}
.lb:hover{background:#1a1a3e;border-color:#e94560;transform:translateY(-1px)}
.lb.off{opacity:.45}
.ls{width:10px;height:10px;border-radius:999px;flex:0 0 10px;box-shadow:0 0 0 1px rgba(255,255,255,.15)}
.zx{overflow:hidden;padding-bottom:6px;border-radius:16px;transition:box-shadow .18s ease}
.cc.zoom-active .zx{box-shadow:0 0 0 1px rgba(92,208,255,.55),0 0 0 4px rgba(92,208,255,.10)}
.cb{width:100%;min-width:100%;height:780px;min-height:780px}
.cbt{width:100%;min-width:100%;height:2000px;min-height:2000px}
.armEv{padding:0 24px 4px;font-size:12px;color:#a0a0c0;line-height:1.7}
.armEv b{color:#e94560;font-weight:600}
.rkCtl{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 10px;font-size:12px;color:#a0a0c0}
.rkCtl input{width:92px;border:1px solid #0f3460;background:#16213e;color:#fff;border-radius:4px;padding:4px 6px;font:inherit}
</style>
</head>
<body>
<div class="hdr">
  <h1>Tracker / Curve3 Interactive</h1>
  <div class="st" id="st"></div>
</div>
<div class="tabs">
  <div class="tab on" onclick="sw(0)">All-in-One</div>
  <div class="tab" onclick="sw(1)">X / Y / Z Subplots</div>
  <div class="tab" onclick="sw(2)">3D Trajectory</div>
  <div class="tab" onclick="sw(3)">Car Location</div>
  <div class="tab" id="tabArm" onclick="sw(4)">Arm</div>
  <div class="tab" id="tabRk" onclick="sw(5)">RK Move</div>
  <div class="tab" id="tabRkSignals" onclick="sw(6)">RK Signals</div>
</div>
<div id="p0" class="pnl on"><div class="cc"><div class="lc" id="l0"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c0" data-action="out">X-</button><button type="button" class="zb on" data-plot="c0" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c0" data-action="in">X+</button><span id="c0r" class="zr">1.00x</span></div><div class="zx"><div id="c0" class="cb"></div></div></div></div>
<div id="p1" class="pnl"><div class="cc"><div class="lc" id="l1"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c1" data-action="out">X-</button><button type="button" class="zb on" data-plot="c1" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c1" data-action="in">X+</button><span id="c1r" class="zr">1.00x</span></div><div class="zx"><div id="c1" class="cb"></div></div></div></div>
<div id="p2" class="pnl"><div class="cc"><div class="lc" id="l2"></div><div class="zt"><span class="ztl">X zoom</span><button type="button" class="zb" data-plot="c2" data-action="out">X-</button><button type="button" class="zb on" data-plot="c2" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c2" data-action="in">X+</button><span id="c2r" class="zr">n/a</span></div><div class="zx"><div id="c2" class="cb"></div></div></div></div>
<div id="p3" class="pnl"><div class="cc"><div class="lc" id="l3"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c3" data-action="out">X-</button><button type="button" class="zb on" data-plot="c3" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c3" data-action="in">X+</button><span id="c3r" class="zr">1.00x</span></div><div class="zx"><div id="c3" class="cb"></div></div></div></div>
<div id="p4" class="pnl">
  <div class="armEv" id="armEv"></div>
  <div class="cc"><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c4" data-action="out">X-</button><button type="button" class="zb on" data-plot="c4" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c4" data-action="in">X+</button><span id="c4r" class="zr">1.00x</span></div><div class="zx"><div id="c4" class="cbt"></div></div><div class="lc" id="l4" style="margin:10px 0 0"></div></div>
</div>
<div id="p5" class="pnl">
  <div class="cc">
    <div class="rkCtl"><span>RK offset(s)</span><input id="rkOff" type="number" step="0.001" value="0"><button type="button" class="zb" id="rkApply">Apply</button><button type="button" class="zb" id="rkAuto">Auto align</button><span id="rkInfo"></span></div>
    <div class="lc" id="l5"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c5" data-action="out">X-</button><button type="button" class="zb on" data-plot="c5" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c5" data-action="in">X+</button><span id="c5r" class="zr">1.00x</span></div><div class="zx"><div id="c5" class="cb"></div></div>
  </div>
</div>
<div id="p6" class="pnl">
  <div class="cc">
    <div class="rkCtl"><span>RK offset(s)</span><input id="rkSigOff" type="number" step="0.001" value="0"><button type="button" class="zb" id="rkSigApply">Apply</button><button type="button" class="zb" id="rkSigAuto">Auto align</button><span id="rkSigInfo"></span></div>
    <div class="lc" id="l6"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c6" data-action="out">X-</button><button type="button" class="zb on" data-plot="c6" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c6" data-action="in">X+</button><span id="c6r" class="zr">1.00x</span></div><div class="zx"><div id="c6" class="cb"></div></div>
  </div>
</div>

<script>
const D = %%DATA_JSON%%;
(function(){
const cfg = D.config || {};
const summary = D.summary || {};
const PLOT_CONFIG = {
  responsive:true,
  displayModeBar:true,
  scrollZoom:false,
  plotGlPixelRatio:1,
};
const preds = D.predictions || [];
const frames = Array.isArray(D.frames) ? D.frames : [];
const writtenVideoFrameIds = Array.isArray(D.video_frame_indices) ? D.video_frame_indices : [];
const distanceScale = cfg.distance_unit === 'm' ? 1.0 : 0.001;
const scaleVec3 = p => ({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale});
const obsRaw = (D.observations || []).map(o=>scaleVec3(o));
const racketObsRaw = (D.racket_observations || []).map(o=>scaleVec3(o));
const carFull = (D.car_locs || []).map(c=>({...c, x:c.x*distanceScale, y:c.y*distanceScale, z:c.z*distanceScale}));
const s0Full = preds.filter(p=>p.stage===0).map(p=>({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale}));
const s1Full = preds.filter(p=>p.stage===1).map(p=>({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale}));
const resets = D.reset_times || summary.reset_times || [];
const throws = D.throws || [];
const ARM = (D.arm && Array.isArray(D.arm.states) && D.arm.states.length) ? D.arm : null;
const RK = (D.rk_tracking && D.rk_tracking.world && Array.isArray(D.rk_tracking.world.t)) ? D.rk_tracking : null;
if(!ARM){
  const tabArm=document.getElementById('tabArm');
  if(tabArm) tabArm.style.display='none';
}
if(!RK){
  const tabRk=document.getElementById('tabRk');
  if(tabRk) tabRk.style.display='none';
  const tabRkSignals=document.getElementById('tabRkSignals');
  if(tabRkSignals) tabRkSignals.style.display='none';
}
const sourceType = cfg.replay_source ? 'Replay JSON' : 'Tracker JSON';
const fps = cfg.fps || summary.actual_fps;
const durationS = cfg.duration_s || summary.duration_s;
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const firstNumeric = (items, key) => {
  for (const item of items) {
    if (item && isNum(item[key])) return item[key];
  }
  return null;
};
const firstFrameT0 =
  isNum(cfg.first_frame_exposure_pc) ? cfg.first_frame_exposure_pc :
  (frames.length > 0 && isNum(frames[0].exposure_pc) ? frames[0].exposure_pc : null);
const fallbackT0 = [firstNumeric(obsRaw, 't'), firstNumeric(racketObsRaw, 't'), firstNumeric(carFull, 't'), firstNumeric(preds, 'ct')]
  .find(v => v !== null);
const t0 = firstFrameT0 !== null ? firstFrameT0 : (fallbackT0 !== null ? fallbackT0 : 0);
const relTime = v => isNum(v) ? (v - t0) : 0;
const frameSeries = frames
  .filter(f => f && isNum(f.exposure_pc))
  .map(f => ({
    ...f,
    rel_s: isNum(f.elapsed_s) ? f.elapsed_s : relTime(f.exposure_pc),
    video_frame_idx: Number.isInteger(f.video_frame_idx) ? f.video_frame_idx : null,
  }));
const frameByIdx = new Map(frameSeries.filter(f => Number.isInteger(f.idx)).map(f => [f.idx, f]));
const mappedFramesFromVideo = writtenVideoFrameIds
  .map((frameId, videoFrameIdx) => {
    const f = frameByIdx.get(frameId);
    return f ? ({...f, video_frame_idx: videoFrameIdx}) : null;
  })
  .filter(Boolean);
const explicitVideoLinkedFrames = frameSeries.filter(f => f.video_frame_idx !== null);
const videoLinkedFrames = mappedFramesFromVideo.length > 0
  ? mappedFramesFromVideo
  : explicitVideoLinkedFrames;
const preferredFrames = videoLinkedFrames.length > 0 ? videoLinkedFrames : frameSeries;
const frameBallObs = preferredFrames
  .filter(f => f.ball3d)
  .map(f => ({
    ...scaleVec3(f.ball3d),
    t: f.exposure_pc,
    rel_s: f.rel_s,
    idx: f.idx,
    video_frame_idx: f.video_frame_idx,
  }));
const frameRacketObs = preferredFrames
  .filter(f => f.racket3d)
  .map(f => ({
    ...scaleVec3(f.racket3d),
    t: f.exposure_pc,
    rel_s: f.rel_s,
    idx: f.idx,
    video_frame_idx: f.video_frame_idx,
  }));
const obsFull = frameBallObs.length > 0
  ? frameBallObs
  : obsRaw.map(o => ({...o, rel_s: relTime(o.t), idx: null, video_frame_idx: null}));
const racketFull = frameRacketObs.length > 0
  ? frameRacketObs
  : racketObsRaw.map(o => ({
      ...o,
      rel_s: isNum(o.elapsed_s) ? o.elapsed_s : relTime(o.t),
      idx: Number.isInteger(o.frame_idx) ? o.frame_idx : null,
      video_frame_idx: Number.isInteger(o.video_frame_idx) ? o.video_frame_idx : null,
    }));
const obs = obsFull;
const racket = racketFull;
const car = carFull;
const s0 = s0Full;
const s1 = s1Full;
const pairedFrames = preferredFrames.filter(f => f.ball3d && f.racket3d);
const frameStartLabel =
  frames.length > 0 && isNum(frames[0].exposure_pc)
    ? `${Number(frames[0].exposure_pc).toFixed(6)}s`
    : null;
const ballSourceLabel = frameBallObs.length > 0 ? 'video-linked frames' : 'tracker observations';
const racketSourceLabel = frameRacketObs.length > 0 ? 'video-linked frames' : 'racket observations';
const g2 = trace => ({type:'scattergl', ...trace});
const buildPlots = [];
const builtPlots = new Set();
function ensurePlot(idx){
  if(builtPlots.has(idx)) return;
  const builder = buildPlots[idx];
  if(typeof builder !== 'function') return;
  builder();
  builtPlots.add(idx);
}
window.ensurePlot = ensurePlot;

const stat=(k,v)=>`<span>${k}: <span class="v">${v!=null?v:'-'}</span></span>`;
document.getElementById('st').innerHTML=[
  stat('Source', sourceType),
  cfg.replay_source ? stat('Replay source', cfg.replay_source) : '',
  isNum(cfg.first_frame_exposure_pc) ? stat('t0 perf', Number(cfg.first_frame_exposure_pc).toFixed(6)+'s') : '',
  frameStartLabel ? stat('Frame0 perf', frameStartLabel) : '',
  cfg.video_frame_mapping_exact != null ? stat('Frame map', cfg.video_frame_mapping_exact ? 'exact' : 'fallback') : '',
  summary.video_frames_mapped ? stat('Mapped video frames', summary.video_frames_mapped) : '',
  stat('Ball 3D', obsFull.length),
  stat('Ball src', ballSourceLabel),
  racketFull.length ? stat('Racket 3D', racketFull.length) : '',
  racket.length ? stat('Racket src', racketSourceLabel) : '',
  videoLinkedFrames.length ? stat('Video-linked frames', videoLinkedFrames.length) : '',
  pairedFrames.length ? stat('Ball+Racket same-frame', pairedFrames.length) : '',
  stat('S0 preds', s0Full.length),
  stat('S1 preds', s1Full.length),
  carFull.length ? stat('Car locs', carFull.length) : '',
  (cfg.car_localizer && Number.isInteger(cfg.car_localizer.sample_every_frames))
    ? stat('Car sample', `1/${cfg.car_localizer.sample_every_frames}`)
    : '',
  stat('2D render', 'full scattergl'),
  stat('Resets', resets.length),
  ARM ? stat('Arm states', ARM.states.length) : '',
  ARM ? stat('Arm cmds', ARM.commands.length) : '',
  ARM && ARM.duration_sec ? stat('Arm bag', ARM.duration_sec.toFixed(1)+'s') : '',
  RK ? stat('RK topics', Object.keys(RK.counts || {}).length) : '',
  RK && RK.world ? stat('RK world ball', RK.world.t.length) : '',
  throws.length ? stat('Throws', throws.length) : '',
  fps ? stat('FPS', fps.toFixed ? fps.toFixed(1) : fps) : '',
  cfg.noise_mm!=null ? stat('Noise', cfg.noise_mm+'mm') : '',
  cfg.cor!=null ? stat('COR', cfg.cor) : '',
  cfg.ideal_hit_z!=null ? stat('ideal_hit_z', (cfg.ideal_hit_z*distanceScale).toFixed(2)+'m') : '',
  cfg.min_stage1_points ? stat('min_s1', cfg.min_stage1_points) : '',
  durationS ? stat('Duration', durationS.toFixed ? durationS.toFixed(1)+'s' : durationS+'s') : '',
].filter(Boolean).join('');

const DL={paper_bgcolor:'#1a1a2e',plot_bgcolor:'#16213e',font:{color:'#e0e0e0',size:11},
  legend:{bgcolor:'rgba(22,33,62,0.9)',bordercolor:'#0f3460',borderwidth:1,font:{size:10},itemsizing:'constant'},
  hovermode:'closest',margin:{l:60,r:30,t:40,b:50}};
const GS={gridcolor:'#0f3460',zerolinecolor:'#0f3460'};
const predRemainingMs = p => (p && isNum(p.ht) && isNum(p.ct)) ? (p.ht - p.ct) * 1000 : null;

buildPlots[0] = () => {
  const oT=obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t));
  const rT=racket.map(r=>isNum(r.rel_s) ? r.rel_s : relTime(r.t));
  const tr=[
    g2({x:oT, y:obs.map(o=>o.x), name:'Ball X', mode:'markers',
     marker:{color:'#7f8c8d',symbol:'circle',size:2,opacity:0.5},
     hovertemplate:'t=%{x:.3f}s<br>x=%{y:.3f} m<extra>Ball X</extra>',
     visible:'legendonly'}),
    g2({x:oT, y:obs.map(o=>o.y), name:'Ball Y', mode:'markers',
     marker:{color:'#95a5a6',symbol:'circle',size:2,opacity:0.5},
     hovertemplate:'t=%{x:.3f}s<br>y=%{y:.3f} m<extra>Ball Y</extra>',
     visible:'legendonly'}),
    g2({x:oT, y:obs.map(o=>o.z), name:'Ball Z', mode:'markers',
     marker:{color:'#bdc3c7',symbol:'circle',size:2.5,opacity:0.6},
     hovertemplate:'t=%{x:.3f}s<br>z=%{y:.3f} m<extra>Ball Z</extra>'}),

    ...(racket.length ? [
    g2({x:rT, y:racket.map(r=>r.x), name:'Racket X', mode:'markers',
     marker:{color:'#ff66cc',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket x=%{y:.3f} m<extra>Racket X</extra>',
     visible:'legendonly'}),
    g2({x:rT, y:racket.map(r=>r.y), name:'Racket Y', mode:'markers',
     marker:{color:'#ff33aa',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket y=%{y:.3f} m<extra>Racket Y</extra>',
     visible:'legendonly'}),
    g2({x:rT, y:racket.map(r=>r.z), name:'Racket Z', mode:'markers',
     marker:{color:'#cc00ff',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket z=%{y:.3f} m<extra>Racket Z</extra>'}),
    ] : []),

    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.x), name:'S0 X', mode:'markers',
     marker:{color:'#3498db',symbol:'triangle-up',size:5},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0 X</extra>'}),
    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.y), name:'S0 Y', mode:'markers',
     marker:{color:'#2980b9',symbol:'triangle-up',size:5},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0 Y</extra>'}),
    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.z), name:'S0 Z', mode:'markers',
     marker:{color:'#1abc9c',symbol:'triangle-up',size:5},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0 Z</extra>'}),

    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.x), name:'S1 X', mode:'markers',
     marker:{color:'#e74c3c',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 X</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.y), name:'S1 Y', mode:'markers',
     marker:{color:'#c0392b',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 Y</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.z), name:'S1 Z', mode:'markers',
     marker:{color:'#e67e22',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 Z</extra>'}),

    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(predRemainingMs), name:'S0 remaining(ms)', mode:'markers',
     marker:{color:'#9b59b6',symbol:'triangle-up',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S0 remaining</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(predRemainingMs), name:'S1 remaining(ms)', mode:'markers',
     marker:{color:'#8e44ad',symbol:'square',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S1 remaining</extra>'}),

    // compute_latency = compute_t - ct（算完时刻 − 曝光时刻；tracker 内部耗时）
    // 旧 JSON 没有 compute_t 时过滤掉，避免 null 画成 0
    g2({x:s0.filter(p=>p.compute_t!=null).map(p=>relTime(p.ct)),
        y:s0.filter(p=>p.compute_t!=null).map(p=>(p.compute_t-p.ct)*1000),
        name:'S0 compute(ms)', mode:'markers',
        marker:{color:'#f39c12',symbol:'triangle-up',size:4}, yaxis:'y2',
        hovertemplate:'t=%{x:.3f}s<br>compute=%{y:.1f} ms<extra>S0 compute</extra>'}),
    g2({x:s1.filter(p=>p.compute_t!=null).map(p=>relTime(p.ct)),
        y:s1.filter(p=>p.compute_t!=null).map(p=>(p.compute_t-p.ct)*1000),
        name:'S1 compute(ms)', mode:'markers',
        marker:{color:'#d35400',symbol:'square',size:4}, yaxis:'y2',
        hovertemplate:'t=%{x:.3f}s<br>compute=%{y:.1f} ms<extra>S1 compute</extra>'}),

    ...(car.length ? [
    g2({x:car.map(c=>relTime(c.t)), y:car.map(c=>c.x), name:'Car X', mode:'markers',
     marker:{color:'#2ecc71',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car x=%{y:.3f} m<extra>Car X</extra>',
     visible:'legendonly'}),
    g2({x:car.map(c=>relTime(c.t)), y:car.map(c=>c.y), name:'Car Y', mode:'markers',
     marker:{color:'#27ae60',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car y=%{y:.3f} m<extra>Car Y</extra>',
     visible:'legendonly'}),
    g2({x:car.map(c=>relTime(c.t)), y:car.map(c=>c.z), name:'Car Z', mode:'markers',
     marker:{color:'#f1c40f',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car z=%{y:.3f} m<extra>Car Z</extra>',
     visible:'legendonly'}),
    ] : []),
  ];

  Plotly.newPlot('c0',tr,{
    ...DL,
    title:{text:'All Curves - click legend to toggle, scroll to zoom',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS}, yaxis:{title:'Value (m)',...GS},
    yaxis2:{title:'Remaining / compute (ms)',...GS,overlaying:'y',side:'right'},
  },PLOT_CONFIG).then(()=>{wl('c0','l0');wz('c0');});
};

buildPlots[1] = () => {
  const oT=obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t));
  const rT=racket.map(r=>isNum(r.rel_s) ? r.rel_s : relTime(r.t));
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push(g2({x:oT,y:obs.map(o=>o[k]),name:`Ball ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#7f8c8d',symbol:'circle',size:2,opacity:0.4},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Ball ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'}));
    if (racket.length) {
      tr.push(g2({x:rT,y:racket.map(r=>r[k]),name:`Racket ${k.toUpperCase()}`,mode:'markers',
        marker:{color:['#ff66cc','#ff33aa','#cc00ff'][i],symbol:'x',size:4},
        hovertemplate:`t=%{x:.3f}s<br>racket ${k}=%{y:.3f} m<extra>Racket</extra>`,
        yaxis:ya,xaxis:'x'}));
    }
    tr.push(g2({x:s0.map(p=>relTime(p.ct)),y:s0.map(p=>p[k]),name:`S0 ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#3498db',symbol:'triangle-up',size:4},
      customdata:s0.map(predRemainingMs),
      hovertemplate:`t=%{x:.3f}s<br>pred ${k}=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0</extra>`,
      yaxis:ya,xaxis:'x'}));
    tr.push(g2({x:s1.map(p=>relTime(p.ct)),y:s1.map(p=>p[k]),name:`S1 ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#e74c3c',symbol:'square',size:4,line:{width:0.5,color:'#fff'}},
      customdata:s1.map(predRemainingMs),
      hovertemplate:`t=%{x:.3f}s<br>pred ${k}=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1</extra>`,
      yaxis:ya,xaxis:'x'}));
  });
  tr.push(g2({x:s0.map(p=>relTime(p.ct)),y:s0.map(predRemainingMs),name:'S0 remaining',mode:'markers',
    marker:{color:'#9b59b6',symbol:'triangle-up',size:3},
    hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S0</extra>',
    yaxis:'y4',xaxis:'x'}));
  tr.push(g2({x:s1.map(p=>relTime(p.ct)),y:s1.map(predRemainingMs),name:'S1 remaining',mode:'markers',
    marker:{color:'#8e44ad',symbol:'square',size:3},
    hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S1</extra>',
    yaxis:'y4',xaxis:'x'}));
  tr.push(g2({x:s0.filter(p=>p.compute_t!=null).map(p=>relTime(p.ct)),
    y:s0.filter(p=>p.compute_t!=null).map(p=>(p.compute_t-p.ct)*1000),
    name:'S0 compute',mode:'markers',
    marker:{color:'#f39c12',symbol:'triangle-up',size:3},
    hovertemplate:'t=%{x:.3f}s<br>compute=%{y:.1f} ms<extra>S0</extra>',
    yaxis:'y4',xaxis:'x'}));
  tr.push(g2({x:s1.filter(p=>p.compute_t!=null).map(p=>relTime(p.ct)),
    y:s1.filter(p=>p.compute_t!=null).map(p=>(p.compute_t-p.ct)*1000),
    name:'S1 compute',mode:'markers',
    marker:{color:'#d35400',symbol:'square',size:3},
    hovertemplate:'t=%{x:.3f}s<br>compute=%{y:.1f} ms<extra>S1</extra>',
    yaxis:'y4',xaxis:'x'}));

  Plotly.newPlot('c1',tr,{
    ...DL,
    title:{text:'X / Y / Z / Remaining (shared time axis)',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS,domain:[0,1],anchor:'y4'},
    yaxis:{title:'X (m)',...GS,domain:[0.78,1]},
    yaxis2:{title:'Y (m)',...GS,domain:[0.53,0.75]},
    yaxis3:{title:'Z (m)',...GS,domain:[0.28,0.50]},
    yaxis4:{title:'Remaining / compute (ms)',...GS,domain:[0.0,0.25]},
  },PLOT_CONFIG).then(()=>{wl('c1','l1');wz('c1');});
};

buildPlots[2] = () => {
  Plotly.newPlot('c2',[
    {x:obs.map(o=>o.x),y:obs.map(o=>o.y),z:obs.map(o=>o.z),
     mode:'markers',type:'scatter3d',name:'Ball',
      marker:{color:obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t)),colorscale:'Viridis',size:2,opacity:0.5,
       colorbar:{title:'t(s)',len:0.5,tickfont:{color:'#e0e0e0'},titlefont:{color:'#e0e0e0'}}},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Ball</extra>',
      text:obs.map(o=>(isNum(o.rel_s) ? o.rel_s : relTime(o.t)).toFixed(3))},
    ...(racket.length ? [{
     x:racket.map(r=>r.x),y:racket.map(r=>r.y),z:racket.map(r=>r.z),
     mode:'markers',type:'scatter3d',name:'Racket',
     marker:{color:'#ff33aa',size:4,symbol:'diamond'},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Racket</extra>',
     text:racket.map(r=>(isNum(r.rel_s) ? r.rel_s : relTime(r.t)).toFixed(3))
    }] : []),
    {x:s0.map(p=>p.x),y:s0.map(p=>p.y),z:s0.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S0 pred',
     marker:{color:'#3498db',size:4,symbol:'diamond'},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>remaining=%{customdata:.1f} ms<extra>S0</extra>',
     text:s0.map(p=>relTime(p.ct).toFixed(3))},
    {x:s1.map(p=>p.x),y:s1.map(p=>p.y),z:s1.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S1 pred',
     marker:{color:'#e74c3c',size:4,symbol:'diamond',line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>remaining=%{customdata:.1f} ms<extra>S1</extra>',
      text:s1.map(p=>relTime(p.ct).toFixed(3))},
    ...(car.length ? [{x:car.map(c=>c.x),y:car.map(c=>c.y),z:car.map(c=>c.z),
     mode:'markers',type:'scatter3d',name:'Car',
     marker:{color:'#2ecc71',size:4,symbol:'square'},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Car</extra>',
      text:car.map(c=>relTime(c.t).toFixed(3))}] : []),
  ],{
    ...DL,
    title:{text:'3D Trajectory',font:{size:13,color:'#a0a0c0'}},
    scene:{xaxis:{title:'X(m)',...GS,backgroundcolor:'#16213e'},
           yaxis:{title:'Y(m)',...GS,backgroundcolor:'#16213e'},
           zaxis:{title:'Z(m)',...GS,backgroundcolor:'#16213e'},bgcolor:'#16213e'},
  },PLOT_CONFIG).then(()=>{wl('c2','l2');wz('c2');});
};

buildPlots[3] = () => {
  if(car.length <= 0) return;
  const cT=car.map(c=>relTime(c.t));
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push(g2({x:cT,y:car.map(c=>c[k]),name:`Car ${k.toUpperCase()}`,mode:'markers',
      marker:{color:['#2ecc71','#27ae60','#f1c40f'][i],size:2},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Car ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'}));
  });
  tr.push(g2({x:cT,y:car.map(c=>c.yaw),name:'Car Yaw',mode:'markers',
    marker:{color:'#e94560',size:2},
    hovertemplate:'t=%{x:.3f}s<br>yaw=%{y:.4f}rad<extra>Car Yaw</extra>',
    yaxis:'y4',xaxis:'x'}));
  tr.push(g2({x:cT,y:car.map(c=>c.reprojection_error),name:'Reproj Err',mode:'markers',
    marker:{color:'#e67e22',size:2},
    hovertemplate:'t=%{x:.3f}s<br>err=%{y:.2f} px<extra>Reproj</extra>',
    yaxis:'y5',xaxis:'x'}));

  Plotly.newPlot('c3',tr,{
    ...DL,
    title:{text:'Car Location (X / Y / Z / Yaw / Reproj)',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS,domain:[0,1],anchor:'y5'},
    yaxis:{title:'X (m)',...GS,domain:[0.82,1]},
    yaxis2:{title:'Y (m)',...GS,domain:[0.62,0.79]},
    yaxis3:{title:'Z (m)',...GS,domain:[0.42,0.59]},
    yaxis4:{title:'Yaw (rad)',...GS,domain:[0.22,0.39]},
    yaxis5:{title:'Reproj (px)',...GS,domain:[0.0,0.19]},
  },PLOT_CONFIG).then(()=>{wl('c3','l3');wz('c3');});
};

buildPlots[4] = () => {
  if(!ARM) return;
  const escA = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const J = ARM.joint_names || ['joint1','joint2','joint3','joint4','joint5','joint6'];
  const JC = ['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2'];
  const AXC = ['#2ecc71','#f1c40f','#5cd0ff'];
  const events = ARM.events || [];
  // 渲染点数控制（session viewer 的窗口化思路）：hit/predict 事件前后保留全
  // 分辨率，窗口外抽稀到 2Hz。软渲染 WebGL 画不动 15 万点全量 32Hz 数据。
  const hitTs = [];
  events.forEach(e=>{
    if(e.topic.indexOf('hit')>=0 || e.topic.indexOf('predict')>=0){
      if(!hitTs.length || e.t - hitTs[hitTs.length-1] > 1.0) hitTs.push(e.t);
    }
  });
  const windows = hitTs.map(t=>[t-2.0, t+4.0]);
  const inWin = t => windows.some(w=>t>=w[0]&&t<=w[1]);
  const thin = (rows, outStep) => {
    const out=[]; let lastT=-1e9;
    rows.forEach(r=>{
      if(inWin(r.t) || r.t-lastT>=outStep){ out.push(r); lastT=r.t; }
    });
    return out;
  };
  const states = thin(ARM.states, 0.5), cmds = thin(ARM.commands || [], 0.5);
  const sT = states.map(s=>s.t);
  // 命令流在两次动作之间有长间隔，插入 null 断开连线，避免斜拉直线
  const gapX = rows => {const X=[];let prev=null;
    rows.forEach(r=>{if(prev!==null&&r.t-prev>0.5){X.push(null);}X.push(r.t);prev=r.t;});
    return X;};
  const gapY = (rows,get) => {const Y=[];let prev=null;
    rows.forEach(r=>{if(prev!==null&&r.t-prev>0.5){Y.push(null);}Y.push(get(r));prev=r.t;});
    return Y;};
  const cX = gapX(cmds);
  // 用 SVG scatter（非 scattergl）：lines 模式下每条 trace 只是一条 path，
  // 软渲染 WebGL 的机器上 scattergl 数万点会把渲染器卡死，SVG 反而流畅。
  const sv = t => ({type:'scatter', ...t});
  const fieldSeries = field => {
    const tr=[];
    J.forEach((name,i)=>{
      tr.push(sv({x:cX, y:gapY(cmds,r=>(r[field]&&r[field][i]!=null)?r[field][i]:null),
        name:`${name} target`, mode:'lines', line:{color:JC[i%JC.length],width:2},
        hovertemplate:`t=%{x:.3f}s<br>%{y:.4f}<extra>${name} target</extra>`}));
      tr.push(sv({x:sT, y:states.map(r=>(r[field]&&r[field][i]!=null)?r[field][i]:null),
        name:`${name} actual`, mode:'lines', line:{color:JC[i%JC.length],width:1,dash:'dot'},
        hovertemplate:`t=%{x:.3f}s<br>%{y:.4f}<extra>${name} actual</extra>`}));
    });
    return tr;
  };
  const tcpSeries = () => {
    const tr=[];
    ['x','y','z'].forEach((ax,i)=>{
      tr.push(sv({x:cX, y:gapY(cmds,r=>(r.tcp&&r.tcp[i]!=null)?r.tcp[i]:null),
        name:`TCP ${ax} target`, mode:'lines', line:{color:AXC[i],width:2},
        hovertemplate:`t=%{x:.3f}s<br>${ax}=%{y:.4f} m<extra>TCP ${ax} target</extra>`}));
      tr.push(sv({x:sT, y:states.map(r=>(r.tcp&&r.tcp[i]!=null)?r.tcp[i]:null),
        name:`TCP ${ax} actual`, mode:'lines', line:{color:AXC[i],width:1,dash:'dot'},
        hovertemplate:`t=%{x:.3f}s<br>${ax}=%{y:.4f} m<extra>TCP ${ax} actual</extra>`}));
    });
    return tr;
  };
  const evShapes = events.map(e=>({type:'line',xref:'x',yref:'paper',x0:e.t,x1:e.t,y0:0,y1:1,
    line:{color:(e.topic.indexOf('hit')>=0||e.topic.indexOf('predict')>=0)?'#e94560':'#f1c40f',width:1,dash:'dot'},
    opacity:0.55}));
  document.getElementById('armEv').innerHTML =
    (ARM.fk_source ? `FK: ${escA(ARM.fk_source)} &nbsp; ` : '') +
    (events.length
      ? 'Events: ' + events.map(e=>`<b>${e.t.toFixed(2)}s</b> ${escA(e.topic.replace('/tennis/',''))} ${escA(e.text).slice(0,60)}`).join(' &nbsp;|&nbsp; ')
      : 'Events: none');
  // 单 plot 四层 subplot（同 Car Location 模式）：只占一个 WebGL context，
  // 拆成 4 个独立 scattergl plot 会同时创建 4 个 context，软渲染机器直接卡死。
  const bindAxis=(traces,ya)=>traces.map(t=>({...t,xaxis:'x',yaxis:ya}));
  const tr=[
    ...bindAxis(fieldSeries('position'),'y'),
    ...bindAxis(fieldSeries('velocity'),'y2'),
    ...bindAxis(fieldSeries('effort'),'y3'),
    ...bindAxis(tcpSeries(),'y4'),
  ];
  Plotly.newPlot('c4',tr,{
    ...DL,
    showlegend:false,
    title:{text:'Arm — target(solid) vs actual(dot): Position / Velocity / Effort / TCP(FK)',
      font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Bag time (s)',...GS,domain:[0,1],anchor:'y4'},
    yaxis:{title:'Position (rad)',...GS,domain:[0.79,1]},
    yaxis2:{title:'Velocity (rad/s)',...GS,domain:[0.53,0.76]},
    yaxis3:{title:'Effort (Nm)',...GS,domain:[0.27,0.50]},
    yaxis4:{title:'TCP (m)',...GS,domain:[0.0,0.24]},
    shapes:evShapes,
  },PLOT_CONFIG).then(()=>{wl('c4','l4');wz('c4');});
};

buildPlots[6] = () => {
  ensurePlot(5);
  if(typeof window.__buildRkSignals === 'function') window.__buildRkSignals();
};

buildPlots[5] = () => {
  if(!RK) return;
  const ts = series => (series && Array.isArray(series.t)) ? series.t : [];
  const ys = (series, key) => (series && series.y && Array.isArray(series.y[key])) ? series.y[key] : [];
  const pairs = (series, key) => ts(series).map((t,i)=>({t:Number(t), v:Number(ys(series,key)[i])}))
    .filter(p=>isNum(p.t)&&isNum(p.v))
    .sort((a,b)=>a.t-b.t);
  const pcRows = obs.map(o=>({t:isNum(o.rel_s)?o.rel_s:relTime(o.t), x:o.x, y:o.y, z:o.z}))
    .filter(p=>isNum(p.t))
    .sort((a,b)=>a.t-b.t);
  const pcCarRows = car.map(c=>({t:isNum(c.elapsed_s)?c.elapsed_s:relTime(c.t), x:c.x, y:c.y, z:c.z, yaw:c.yaw}))
    .filter(p=>isNum(p.t))
    .sort((a,b)=>a.t-b.t);
  const nearest = (rows, t) => {
    let lo=0, hi=rows.length;
    while(lo<hi){
      const mid=(lo+hi)>>1;
      if(rows[mid].t<t) lo=mid+1; else hi=mid;
    }
    const cand=[];
    if(lo<rows.length) cand.push(rows[lo]);
    if(lo>0) cand.push(rows[lo-1]);
    if(!cand.length) return null;
    return cand.reduce((best,row)=>Math.abs(row.t-t)<Math.abs(best.t-t)?row:best,cand[0]);
  };
  const lerp = (a,b,f) => a + (b-a)*f;
  // 有界线性插值：t 落在相邻两行之间且间隔 ≤ maxGap 才给结果，绝不外推
  const interpRow = (rows, t, maxGap) => {
    if(!rows.length) return null;
    let lo=0, hi=rows.length;
    while(lo<hi){
      const mid=(lo+hi)>>1;
      if(rows[mid].t<t) lo=mid+1; else hi=mid;
    }
    if(lo<=0||lo>=rows.length) return null;
    const a=rows[lo-1], b=rows[lo];
    if(!(t>=a.t&&t<=b.t)||b.t-a.t>maxGap) return null;
    return {a, b, f:(t-a.t)/Math.max(1e-9,b.t-a.t)};
  };
  const interpPcVal = (rows,t,key,maxGap) => {
    const s=interpRow(rows,t,maxGap);
    return (s && isNum(s.a[key]) && isNum(s.b[key])) ? lerp(s.a[key],s.b[key],s.f) : null;
  };
  // 时间对齐：只用 RK 球在运动的段（|dy/dt|>1.5 m/s），对 PC 观测插值比 y、取中位差。
  // 旧的 Z 最近邻 MAE 会被 RK 锁死球/躺地球段拖进假极小（0712 场实测差 21s），已替换。
  const rkMovY = (()=>{
    const rows=pairs(RK.world,'y');
    const out=[];
    for(let i=1;i<rows.length;i++){
      const dt=rows[i].t-rows[i-1].t;
      if(dt>0&&dt<0.2&&Math.abs(rows[i].v-rows[i-1].v)/dt>1.5) out.push(rows[i]);
    }
    return out;
  })();
  const scoreOffset = off => {
    if(rkMovY.length<30 || pcRows.length<10) return null;
    const step=Math.max(1,Math.floor(rkMovY.length/400));
    const ds=[];
    for(let i=0;i<rkMovY.length;i+=step){
      const v=interpPcVal(pcRows, rkMovY[i].t+off, 'y', 0.08);
      if(v==null) continue;
      ds.push(Math.abs(v-rkMovY[i].v));
    }
    if(ds.length<15) return null;
    ds.sort((a,b)=>a-b);
    return {err:ds[ds.length>>1], n:ds.length};
  };
  const estimateOffset = () => {
    let best=null;
    for(let off=-30.0; off<=30.0001; off+=0.02){
      const s=scoreOffset(off);
      if(!s) continue;
      if(!best || s.err<best.err) best={off, ...s};
    }
    if(best){
      for(let off=best.off-0.05; off<=best.off+0.0501; off+=0.002){
        const s=scoreOffset(off);
        if(s && s.err<best.err) best={off, ...s};
      }
    }
    return best || {off:0, err:null, n:0};
  };
  const auto = estimateOffset();
  let rkOffset = Math.round(auto.off*1000)/1000;
  const input = document.getElementById('rkOff');
  const info = document.getElementById('rkInfo');
  const shifted = xs => xs.map(x=>isNum(Number(x)) ? Number(x)+rkOffset : null);
  const setInfo = () => {
    const errText = auto.err==null ? 'n/a' : `${auto.err.toFixed(3)}m / ${auto.n} pts`;
    info.textContent = `display t = RK t + ${rkOffset.toFixed(3)}s; auto traj |dy| ${errText}`;
  };
  const tr = (series,key,name,axis,color,mode='markers',extra={}) => g2({
    x:shifted(ts(series)), y:ys(series,key), name, mode,
    marker:{color,size:3}, line:{color,width:1.4},
    yaxis:axis, xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}<extra>${name}</extra>`,
    ...extra,
  });
  const rkPredTr = (key,name,color,extra={}) => tr(RK.pred,key,name,'y',color,'markers',{
    customdata:ys(RK.pred,'duration').map(v=>isNum(v)?v*1000:null),
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>remaining=%{customdata:.1f} ms<extra>${name}</extra>`,
    ...extra,
  });
  const rkRemainingTr = () => g2({
    x:shifted(ts(RK.pred)), y:ys(RK.pred,'duration').map(v=>isNum(v)?v*1000:null),
    name:'RK Predict remaining(ms)', mode:'markers',
    marker:{color:'#fde047',size:4,symbol:'triangle-up'}, yaxis:'y2', xaxis:'x',
    hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>RK Predict remaining</extra>',
  });
  // PC hit 预测 S0+S1 合成一条（按 ct 排序）；stage 用点形区分：S0=三角、S1=方块
  const sAll = [...s0, ...s1].sort((a,b)=>((a.ct||0)-(b.ct||0)));
  const stageSym = s => s===0 ? 'triangle-up' : 'square';
  const pcHitTr = (key,name,color,extra={}) => g2({
    x:sAll.map(p=>relTime(p.ct)), y:sAll.map(p=>p[key]), name, mode:'markers',
    customdata:sAll.map(p=>[predRemainingMs(p), p.stage]),
    marker:{color,size:5,symbol:sAll.map(p=>stageSym(p.stage))}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>S%{customdata[1]} remaining=%{customdata[0]:.1f} ms<extra>${name}</extra>`,
    ...extra,
  });
  const pcHitRemainingTr = () => g2({
    x:sAll.map(p=>relTime(p.ct)), y:sAll.map(predRemainingMs),
    name:'PC Hit remaining(ms)', mode:'markers',
    customdata:sAll.map(p=>p.stage),
    marker:{color:'#8e44ad',size:4,symbol:sAll.map(p=>stageSym(p.stage))}, yaxis:'y2', xaxis:'x',
    hovertemplate:'t=%{x:.3f}s<br>S%{customdata} remaining=%{y:.1f} ms<extra>PC Hit remaining</extra>',
  });
  // 分抛：按 ht_rel 聚类 RK 预测消息，取每抛最终 ht（击球时刻，RK 轴）+ 最后一条 ref 值
  const rkPredStage = ys(RK.pred,'stage');
  const rkPredDurMs = ys(RK.pred,'duration').map(v=>isNum(v)?v*1000:null);
  const rkThrows = (()=>{
    const t=ts(RK.pred), ht=ys(RK.pred,'ht_rel');
    const relx=ys(RK.pred,'rel_x'), rely=ys(RK.pred,'rel_y'), relz=ys(RK.pred,'rel_z');
    const out=[];
    for(let i=0;i<t.length;i++){
      const ti=Number(t[i]);
      if(!isNum(ti) || !isNum(ht[i])) continue;
      const cur=out[out.length-1];
      const upd = th => {
        th.ht=ht[i]; th.lastT=ti;
        if(isNum(relx[i])&&isNum(rely[i])&&isNum(relz[i])){
          th.stage=rkPredStage[i]; th.rel_x=relx[i]; th.rel_y=rely[i]; th.rel_z=relz[i];
          th.lastRelIdx=i;
        }
      };
      if(cur && Math.abs(ht[i]-cur.ht)<0.8 && ti-cur.lastT<2.0){
        upd(cur);
      } else {
        const th={ht:ht[i], lastT:ti, stage:null, rel_x:null, rel_y:null, rel_z:null, lastRelIdx:null};
        upd(th);
        out.push(th);
      }
    }
    return out;
  })();
  // RK ref：/predict_hit_pos 的 rel_x/rel_y/rel_z（击球点相对「击球时刻预测车位姿」车体系，臂端消费的量）。
  // 每抛最后一条 ref 只画一次：作为 star 挪到最终 ht 处（击球时刻臂在执行的参考，
  // 与 PC 真值 star 同横坐标可垂直对比），不再在其原消息时刻重复画常规点。
  const rkRefRows = key => {
    const t=ts(RK.pred), val=ys(RK.pred,key);
    const finalIdx=new Set(rkThrows.map(th=>th.lastRelIdx).filter(i=>i!=null));
    const rows=[];
    for(let i=0;i<t.length;i++){
      if(finalIdx.has(i)) continue;
      const ti=Number(t[i]);
      if(!isNum(ti) || !isNum(val[i])) continue;
      rows.push({t:ti+rkOffset, v:val[i], stage:rkPredStage[i], sym:stageSym(rkPredStage[i]), size:5,
                 note:isNum(rkPredDurMs[i])?`remaining=${rkPredDurMs[i].toFixed(1)} ms`:''});
    }
    rkThrows.forEach(th=>{
      if(!isNum(th[key])) return;
      rows.push({t:th.ht+rkOffset, v:th[key], stage:th.stage, sym:'star', size:11, note:'@ht final ref'});
    });
    return rows.sort((a,b)=>a.t-b.t);
  };
  const rkRefTr = (key,name,color,extra={}) => {
    const rows=rkRefRows(key);
    return g2({
      x:rows.map(r=>r.t), y:rows.map(r=>r.v), name, mode:'markers',
      customdata:rows.map(r=>[r.note, r.stage]),
      marker:{color, size:rows.map(r=>r.size), symbol:rows.map(r=>r.sym)}, yaxis:'y', xaxis:'x',
      hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>S%{customdata[1]} %{customdata[0]}<extra>${name}</extra>`,
      ...extra,
    });
  };
  const ballAt = t => {
    const s=interpRow(pcRows,t,0.12);
    if(!s) return null;
    return {x:lerp(s.a.x,s.b.x,s.f), y:lerp(s.a.y,s.b.y,s.f), z:lerp(s.a.z,s.b.z,s.f)};
  };
  const carAt = t => {
    const s=interpRow(pcCarRows,t,0.5);
    if(s && isNum(s.a.yaw) && isNum(s.b.yaw)){
      const dyaw=Math.atan2(Math.sin(s.b.yaw-s.a.yaw),Math.cos(s.b.yaw-s.a.yaw));
      return {x:lerp(s.a.x,s.b.x,s.f), y:lerp(s.a.y,s.b.y,s.f), yaw:s.a.yaw+dyaw*s.f};
    }
    const n=nearest(pcCarRows,t);
    return (n && Math.abs(n.t-t)<=0.3 && isNum(n.yaw)) ? {x:n.x, y:n.y, yaw:n.yaw} : null;
  };
  const relToCar = (b,c) => {
    const dx=b.x-c.x, dy=b.y-c.y, cy=Math.cos(c.yaw), sy=Math.sin(c.yaw);
    return {x:cy*dx+sy*dy, y:-sy*dx+cy*dy, z:b.z};
  };
  // PC 真值：球相对车体系位置，只画每抛最终 ht 处的插值 star（被相邻观测夹住才给，绝不外推）。
  // PC 在击球前丢球的抛没有 star；ht 前的过程真值与 PC Ball 曲线重复，不再展示。
  const truthRows = () => {
    const rows=[];
    rkThrows.forEach(th=>{
      const tHit=th.ht+rkOffset;
      const b=ballAt(tHit), c=carAt(tHit);
      if(b && c){
        const r=relToCar(b,c);
        rows.push({t:tHit, x:r.x, y:r.y, z:r.z});
      }
    });
    return rows.sort((a,b)=>a.t-b.t);
  };
  const truthTr = (rows,key,name,color,extra={}) => g2({
    x:rows.map(r=>r.t), y:rows.map(r=>r[key]), name, mode:'markers',
    marker:{color, size:11, symbol:'star'},
    yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m @ht<extra>${name}</extra>`,
    ...extra,
  });
  const pcTr = (key,name,color,extra={}) => g2({
    x:pcRows.map(p=>p.t), y:pcRows.map(p=>p[key]), name, mode:'markers',
    marker:{color,size:2.5}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const pcCarTr = (key,name,color,extra={}) => g2({
    x:pcCarRows.map(c=>c.t), y:pcCarRows.map(c=>c[key]), name, mode:'markers',
    marker:{color,size:2.5,symbol:'diamond'}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const pcCarYawTr = () => g2({
    x:pcCarRows.map(c=>c.t), y:pcCarRows.map(c=>isNum(c.yaw)?c.yaw*10:null),
    name:'PC Car Yaw x10', mode:'markers',
    customdata:pcCarRows.map(c=>c.yaw),
    marker:{color:'#f472b6',size:2.5,symbol:'diamond'}, yaxis:'y3', xaxis:'x',
    hovertemplate:'t=%{x:.3f}s<br>PC Car Yaw=%{customdata:.4f}rad<br>display=%{y:.3f}<extra>PC Car Yaw x10</extra>',
    visible:'legendonly',
  });
  const makeRelSeries = (rows, carRows) => {
    const out={t:[], y:{dx:[], dy:[], dist:[]}};
    rows.forEach(row=>{
      const c=nearest(carRows, row.t);
      if(!c || Math.abs(c.t-row.t)>0.08 || !isNum(row.x) || !isNum(row.y) || !isNum(c.x) || !isNum(c.y)) return;
      const dx=row.x-c.x, dy=row.y-c.y;
      out.t.push(row.t);
      out.y.dx.push(dx);
      out.y.dy.push(dy);
      out.y.dist.push(Math.hypot(dx, dy));
    });
    return out;
  };
  const rkRel = {t:[], y:{dx:[], dy:[], dist:[]}};
  ts(RK.world).forEach((t,i)=>{
    const x=ys(RK.world,'x')[i], y=ys(RK.world,'y')[i];
    const cx=ys(RK.world,'bot_x')[i], cy=ys(RK.world,'bot_y')[i];
    if(!isNum(t) || !isNum(x) || !isNum(y) || !isNum(cx) || !isNum(cy)) return;
    const dx=x-cx, dy=y-cy;
    rkRel.t.push(t);
    rkRel.y.dx.push(dx);
    rkRel.y.dy.push(dy);
    rkRel.y.dist.push(Math.hypot(dx, dy));
  });
  const pcRel = makeRelSeries(pcRows, pcCarRows);
  const pcRelTr = (series,key,name,color,extra={}) => g2({
    x:series.t, y:series.y[key], name, mode:'markers',
    marker:{color,size:3,symbol:'x'}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const traceData = () => [
    pcTr('x','PC Ball X','#7f8c8d',{visible:'legendonly'}),
    pcTr('y','PC Ball Y','#95a5a6',{visible:'legendonly'}),
    pcTr('z','PC Ball Z','#bdc3c7'),
    tr(RK.world,'x','RK World X','y','#3498db','markers',{visible:'legendonly'}),
    tr(RK.world,'y','RK World Y','y','#2980b9','markers',{visible:'legendonly'}),
    tr(RK.world,'z','RK World Z','y','#5cd0ff'),
    rkPredTr('x','RK Predict X','#f97316',{visible:'legendonly',marker:{color:'#f97316',size:6,symbol:'triangle-up'}}),
    rkPredTr('y','RK Predict Y','#fb923c',{visible:'legendonly',marker:{color:'#fb923c',size:6,symbol:'triangle-up'}}),
    rkPredTr('z','RK Predict Z','#e94560',{marker:{color:'#e94560',size:6,symbol:'triangle-up'}}),
    pcHitTr('x','PC Hit X','#fb7185',{visible:'legendonly'}),
    pcHitTr('y','PC Hit Y','#f43f5e',{visible:'legendonly'}),
    pcHitTr('z','PC Hit Z','#e11d48'),
    rkRefTr('rel_x','RK Ref X','#a3e635'),
    rkRefTr('rel_y','RK Ref Y','#4d7c0f',{visible:'legendonly'}),
    rkRefTr('rel_z','RK Ref Z','#84cc16'),
    ...(rows=>[
      truthTr(rows,'x','PC Truth X','#e2e8f0'),
      truthTr(rows,'y','PC Truth Y','#94a3b8',{visible:'legendonly'}),
      truthTr(rows,'z','PC Truth Z','#ffffff'),
    ])(truthRows()),
    tr(RK.estimate,'x','RK Estimate X','y','#facc15','markers',{visible:'legendonly',marker:{color:'#facc15',size:4}}),
    tr(RK.estimate,'y','RK Estimate Y','y','#fde047','markers',{visible:'legendonly',marker:{color:'#fde047',size:4}}),
    tr(RK.estimate,'z','RK Estimate Z','y','#f1c40f','markers',{visible:'legendonly',marker:{color:'#f1c40f',size:4}}),
    pcCarTr('x','PC Car X','#d946ef'),
    pcCarTr('y','PC Car Y','#c084fc'),
    pcCarTr('z','PC Car Z','#a78bfa',{visible:'legendonly'}),
    pcCarYawTr(),
    tr(RK.bot,'x','Bot X','y','#67e8c3'),
    tr(RK.bot,'y','Bot Y','y','#9fffce'),
    g2({x:shifted(ts(RK.bot)), y:ys(RK.bot,'yaw').map(v=>isNum(v)?v*10:null), name:'Bot Yaw x10', mode:'markers',
      customdata:ys(RK.bot,'yaw'),
      marker:{color:'#5eead4',size:2.5,symbol:'diamond'}, yaxis:'y3', xaxis:'x',
      hovertemplate:'t=%{x:.3f}s<br>Bot Yaw=%{customdata:.4f}rad<br>display=%{y:.3f}<extra>Bot Yaw x10</extra>',
      visible:'legendonly'}),
    tr(RK.bot,'target_x','Bot Target X','y','#ffd27f','markers',{visible:'legendonly'}),
    tr(RK.bot,'target_y','Bot Target Y','y','#ff9f7f','markers',{visible:'legendonly'}),
    tr(rkRel,'dx','RK Ball-Car dX','y','#ef4444','markers',{visible:'legendonly',marker:{color:'#ef4444',size:3,symbol:'cross'}}),
    tr(rkRel,'dy','RK Ball-Car dY','y','#f97316','markers',{visible:'legendonly',marker:{color:'#f97316',size:3,symbol:'cross'}}),
    tr(rkRel,'dist','RK Ball-Car XY Dist','y','#facc15','markers',{marker:{color:'#facc15',size:3,symbol:'cross'}}),
    pcRelTr(pcRel,'dx','PC Ball-Car dX','#ec4899',{visible:'legendonly'}),
    pcRelTr(pcRel,'dy','PC Ball-Car dY','#d946ef',{visible:'legendonly'}),
    pcRelTr(pcRel,'dist','PC Ball-Car XY Dist','#a855f7'),
    pcHitRemainingTr(),
    rkRemainingTr(),
  ];
  const layout = () => ({
    ...DL,
    title:{text:'RK Move ball and car positions aligned to PC timeline',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'PC report time (s)',...GS,domain:[0,1],anchor:'y'},
    yaxis:{title:'Ball + Car XYZ/XY (m)',...GS,domain:[0,1]},
    yaxis2:{title:'Remaining (ms)',...GS,overlaying:'y',side:'right'},
    yaxis3:{title:'Yaw rad x10',...GS,overlaying:'y',side:'right',position:0.94},
  });
  const redraw = () => {
    setInfo();
    syncSignalControls();
    const jobs=[Plotly.react('c5', traceData(), layout(), PLOT_CONFIG).then(()=>tl('c5','l5'))];
    if(document.getElementById('c6') && builtPlots.has(6)){
      jobs.push(Plotly.react('c6', signalTraceData(), signalLayout(), PLOT_CONFIG).then(()=>tl('c6','l6')));
    }
    return Promise.all(jobs);
  };
  if(input) input.value = rkOffset.toFixed(3);
  setInfo();
  Plotly.newPlot('c5', traceData(), layout(), PLOT_CONFIG).then(()=>{wl('c5','l5');wz('c5');});
  const apply = document.getElementById('rkApply');
  if(apply) apply.addEventListener('click',()=>{
    const v=Number(input.value);
    rkOffset=isNum(v) ? v : 0;
    redraw();
  });
  const autoBtn = document.getElementById('rkAuto');
  if(autoBtn) autoBtn.addEventListener('click',()=>{
    rkOffset=Math.round(auto.off*1000)/1000;
    if(input) input.value=rkOffset.toFixed(3);
    redraw();
  });

  const signalTraceData = () => [
    tr(RK.camera_cmd,'position','Camera Cmd Pos','y','#b197fc'),
    tr(RK.camera_motor,'position','Camera Motor Pos','y','#7fd1ff'),
    tr(RK.steer_cmd,'position','Steer Cmd Pos','y2','#f59e0b'),
    tr(RK.steer_motor,'position','Steer Motor Pos','y2','#facc15'),
    tr(RK.wheels_cmd,'current_avg','Wheel Current Avg','y3','#ff7f7f'),
    tr(RK.wheels_cmd,'speed_avg','Wheel Speed Avg','y3','#67e8c3','markers',{visible:'legendonly'}),
    tr(RK.wheels_pos_diff,'value_avg','Wheel PosDiff Avg','y3','#c084fc','markers',{visible:'legendonly'}),
    tr(RK.imu,'yaw_speed','IMU Yaw Speed','y4','#94a3b8','markers',{visible:'legendonly'}),
  ];
  const signalLayout = () => ({
    ...DL,
    title:{text:'RK move signals aligned to PC timeline',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'PC report time (s)',...GS,domain:[0,1],anchor:'y4'},
    yaxis:{title:'Camera pos',...GS,domain:[0.78,1]},
    yaxis2:{title:'Steer pos',...GS,domain:[0.52,0.74]},
    yaxis3:{title:'Wheels avg',...GS,domain:[0.26,0.48]},
    yaxis4:{title:'IMU',...GS,domain:[0.0,0.22]},
  });
  const sigInput = document.getElementById('rkSigOff');
  const sigInfo = document.getElementById('rkSigInfo');
  const syncSignalControls = () => {
    if(sigInput) sigInput.value = rkOffset.toFixed(3);
    if(sigInfo) sigInfo.textContent = info.textContent;
  };
  syncSignalControls();
  const sigApply = document.getElementById('rkSigApply');
  if(sigApply) sigApply.addEventListener('click',()=>{
    const v=Number(sigInput.value);
    rkOffset=isNum(v) ? v : 0;
    if(input) input.value=rkOffset.toFixed(3);
    redraw().then(syncSignalControls);
  });
  const sigAuto = document.getElementById('rkSigAuto');
  if(sigAuto) sigAuto.addEventListener('click',()=>{
    rkOffset=Math.round(auto.off*1000)/1000;
    if(input) input.value=rkOffset.toFixed(3);
    redraw().then(syncSignalControls);
  });

  window.__buildRkSignals = () => {
    syncSignalControls();
    Plotly.newPlot('c6', signalTraceData(), signalLayout(), PLOT_CONFIG).then(()=>{wl('c6','l6');wz('c6');});
  };
};
})();

function sw(i){
  ensurePlot(i);
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('on',j===i));
  document.querySelectorAll('.pnl').forEach((p,j)=>p.classList.toggle('on',j===i));
  window.dispatchEvent(new Event('resize'));
}

function tv(trace){return trace&&trace.visible!=='legendonly'&&trace.visible!==false}
function tc(trace){
  if(!trace) return '#d7d7eb';
  if(trace.line&&typeof trace.line.color==='string') return trace.line.color;
  if(trace.marker&&typeof trace.marker.color==='string') return trace.marker.color;
  return '#d7d7eb';
}
function tl(plotId,ctrlId){
  const plot=document.getElementById(plotId);
  const ctrl=document.getElementById(ctrlId);
  if(!plot||!ctrl||!plot.data) return;
  ctrl.innerHTML=plot.data.map((trace,idx)=>{
    const on=tv(trace);
    const name=trace&&trace.name?trace.name:`trace ${idx+1}`;
    return `<button type="button" class="lb${on?'':' off'}" data-plot="${plotId}" data-index="${idx}" aria-pressed="${on?'true':'false'}"><span class="ls" style="background:${tc(trace)}"></span><span>${name}</span></button>`;
  }).join('');
}
function wl(plotId,ctrlId){
  const plot=document.getElementById(plotId);
  const ctrl=document.getElementById(ctrlId);
  if(!plot||!ctrl) return;
  tl(plotId,ctrlId);
  // 事件委托挂在容器上：单击后按钮条会整体重建，直接绑在按钮上 dblclick 收不到
  ctrl.addEventListener('click',ev=>{
    const btn=ev.target.closest('.lb');
    if(!btn) return;
    const idx=Number(btn.dataset.index);
    const next=!tv(plot.data[idx]);
    Plotly.restyle(plotId,{visible:next?true:'legendonly'},[idx]).then(()=>tl(plotId,ctrlId));
  });
  // 双击一个系列：只显示它；已是 solo 时再双击恢复全部
  ctrl.addEventListener('dblclick',ev=>{
    const btn=ev.target.closest('.lb');
    if(!btn) return;
    const idx=Number(btn.dataset.index);
    const alreadySolo=plot.data.every((t,j)=>tv(t)===(j===idx));
    const vis=plot.data.map((t,j)=>(alreadySolo||j===idx)?true:'legendonly');
    Plotly.restyle(plotId,{visible:vis}).then(()=>tl(plotId,ctrlId));
  });
  plot.on('plotly_restyle',()=>tl(plotId,ctrlId));
}

const ZSTEP=1.35, ZMAX=200.0;
let AP=null;
const ZSTATE={};
function gp(id){return document.getElementById(id)}
function sap(id){
  AP=id;
  document.querySelectorAll('.cc').forEach(cc=>{
    const plot=cc.querySelector('.cb');
    cc.classList.toggle('zoom-active',!!plot&&plot.id===id);
  });
}
function nx(plot){
  const xs=[];
  (plot?.data||[]).forEach(trace=>{
    (trace?.x||[]).forEach(v=>{
      if(typeof v==='number'&&Number.isFinite(v)) xs.push(v);
    });
  });
  return xs;
}
function fx(id){
  const cached=ZSTATE[id]?.fullRange;
  if(cached) return [...cached];
  const plot=gp(id);
  if(!plot) return null;
  let range=null;
  const axis=plot._fullLayout&&plot._fullLayout.xaxis;
  if(!axis) return null;
  if(axis&&Array.isArray(axis.range)&&axis.range.length===2){
    const a=Number(axis.range[0]), b=Number(axis.range[1]);
    if(Number.isFinite(a)&&Number.isFinite(b)&&b>a) range=[a,b];
  }
  if(!range){
    const xs=nx(plot);
    if(!xs.length) return null;
    range=[Math.min(...xs),Math.max(...xs)];
  }
  ZSTATE[id]={...(ZSTATE[id]||{}),fullRange:range};
  return [...range];
}
function cx(id){
  const plot=gp(id);
  if(!plot) return null;
  const axis=plot._fullLayout&&plot._fullLayout.xaxis;
  if(axis&&Array.isArray(axis.range)&&axis.range.length===2){
    const a=Number(axis.range[0]), b=Number(axis.range[1]);
    if(Number.isFinite(a)&&Number.isFinite(b)&&b>a) return [a,b];
  }
  return fx(id);
}
function qx(range,fullRange){
  const f0=fullRange[0], f1=fullRange[1], fs=f1-f0;
  if(!(fs>0)) return [f0,f1];
  let a=Number(range[0]), b=Number(range[1]);
  let span=b-a;
  const minSpan=Math.max(fs/ZMAX,1e-6);
  if(!(span>0)) span=minSpan;
  span=Math.max(minSpan,Math.min(fs,span));
  const center=(a+b)/2;
  a=center-span/2;
  b=center+span/2;
  if(a<f0){b+=f0-a;a=f0;}
  if(b>f1){a-=b-f1;b=f1;}
  if(a<f0)a=f0;
  if(b>f1)b=f1;
  return [a,b];
}
function ux(id){
  const full=fx(id), cur=cx(id), readout=document.getElementById(`${id}r`);
  if(!full||!cur){
    if(readout) readout.textContent='n/a';
    return;
  }
  const factor=Math.max(1,(full[1]-full[0])/Math.max(1e-9,cur[1]-cur[0]));
  if(readout) readout.textContent=`${factor.toFixed(2)}x`;
  document.querySelectorAll(`.zb[data-plot="${id}"]`).forEach(btn=>{
    btn.classList.toggle('on',btn.dataset.action==='reset'&&Math.abs(factor-1)<1e-3);
  });
}
function rx(id,range){
  return Plotly.relayout(id,{'xaxis.range':range,'xaxis.autorange':false}).then(()=>ux(id));
}
function mx(id,event){
  const plot=gp(id);
  const axis=plot?._fullLayout?.xaxis;
  const cur=cx(id);
  if(!plot||!axis||!cur) return null;
  const rect=plot.getBoundingClientRect();
  const axisOffset=Number(axis._offset), axisLength=Number(axis._length);
  if(!Number.isFinite(axisOffset)||!Number.isFinite(axisLength)||!(axisLength>0)) return null;
  const rawPixel=Number(event.clientX)-rect.left-axisOffset;
  const pixel=Math.max(0,Math.min(axisLength,rawPixel));
  if(typeof axis.p2l==='function'){
    const converted=Number(axis.p2l(pixel));
    if(Number.isFinite(converted)) return converted;
  }
  const ratio=pixel/axisLength;
  return cur[0]+ratio*(cur[1]-cur[0]);
}
function zx(id,spanFactor,centerX=null){
  const full=fx(id), cur=cx(id);
  if(!full||!cur) return Promise.resolve();
  const center=(typeof centerX==='number'&&Number.isFinite(centerX))?centerX:((cur[0]+cur[1])/2);
  const next=qx([center-((cur[1]-cur[0])*spanFactor)/2,center+((cur[1]-cur[0])*spanFactor)/2],full);
  return rx(id,next);
}
function zxReset(id){
  const full=fx(id);
  if(!full) return Promise.resolve();
  return rx(id,full);
}
function wz(id){
  const plot=gp(id);
  if(!plot) return;
  fx(id);
  ux(id);
  plot.addEventListener('pointerdown',()=>sap(id));
  plot.addEventListener('wheel',event=>{
    if(AP!==id) return;
    if(event.ctrlKey||event.metaKey) return;
    if(!fx(id)) return;
    event.preventDefault();
    zx(id,event.deltaY<0?(1/ZSTEP):ZSTEP,mx(id,event));
  },{passive:false});
  plot.on('plotly_relayout',()=>ux(id));
}
document.querySelectorAll('.zb[data-plot]').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const id=btn.dataset.plot;
    sap(id);
    if(btn.dataset.action==='in') zx(id,1/ZSTEP);
    else if(btn.dataset.action==='out') zx(id,ZSTEP);
    else zxReset(id);
  });
});
ensurePlot(0);
sap('c0');
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="curve3_output/curve3_result.json")
    parser.add_argument("--racket-json", default=None)
    parser.add_argument(
        "--arm-json", default=None,
        help="extract_arm_bag.py 输出的机械臂 JSON；缺省时自动探测 <input>_arm.json",
    )
    parser.add_argument("--rk-tracking-json", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = os.path.splitext(args.input)[0]
    out = args.output or (base + ".html")
    arm_json = args.arm_json
    if arm_json is None:
        candidate = base + "_arm.json"
        if os.path.exists(candidate):
            arm_json = candidate
    rk_tracking_json = args.rk_tracking_json
    if rk_tracking_json is None:
        candidate = base + "_rk_tracking.json"
        if os.path.exists(candidate):
            rk_tracking_json = candidate
    generate_html(args.input, out, args.racket_json, arm_json, rk_tracking_json)


if __name__ == "__main__":
    main()
