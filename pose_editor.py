"""
Interactive drag-and-drop keypoint editor — Gradio 6 compatible.

Data is embedded in the DOM via html_template's ${value} substitution.
js_on_load clones the canvas (to strip stale listeners) and sets up
fresh event handlers every time the component value changes.
"""

import json
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from scipy.interpolate import RBFInterpolator

from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton, joints_dict

_CANVAS_MAX_PX = 1200
_ANGLE_OUTPUT_DIR = Path(__file__).parent / "temp" / "açılar"
_ANGLE_CONFIDENCE_THRESHOLD = 0.01
_STANDARD_ANGLE_DEFINITIONS = [
    (9, 7, 13, "R.Omuz"),
    (8, 6, 12, "L.Omuz"),
    (7, 9, 11, "R.Dirsek"),
    (6, 8, 10, "L.Dirsek"),
    (7, 13, 16, "R.Kalca"),
    (6, 12, 15, "L.Kalca"),
    (13, 16, 18, "R.Diz"),
    (12, 15, 17, "L.Diz"),
    (16, 18, 22, "R.AyakBilegi"),
    (15, 17, 19, "L.AyakBilegi"),
    (24, 18, 22, "R.AyakYonu"),
    (21, 17, 19, "L.AyakYonu"),
]

# ── HTML template ────────────────────────────────────────────────────────────
# ${value} is replaced by Gradio with the component value (base64-encoded JSON).
# It is placed inside a hidden <div> so the canvas JS can read it from the DOM.
EDITOR_HTML_TEMPLATE = """
<div class="pe-wrap">
  <div class="pe-data" style="display:none">${value}</div>
  <canvas class="pe-canvas"></canvas>
  <div class="pe-bar">
    <button class="pe-btn pe-reset"  data-action="reset">&#8617; Reset</button>
    <button class="pe-btn pe-names"  data-action="names">&#128065; Names</button>
    <button class="pe-btn pe-save"   data-action="save">&#128190; Save PNG</button>
    <button class="pe-btn pe-angles" data-action="angles">&#128208; A&#231;&#305;lar</button>
    <button class="pe-btn pe-custom-angle" data-action="custom-angle">&#8736; &#214;zel A&#231;&#305;</button>
    <button class="pe-btn pe-fullscreen" data-action="fullscreen">&#x2922; Tam Ekran</button>
    <button class="pe-btn pe-keypoints" data-action="keypoints">&#x25CF; Noktalar</button>
    <span class="pe-info">Goruntu yukleniyor...</span>
    <div class="pe-nav-group">
      <button class="pe-btn pe-prev" data-action="prev">&#9664; Prev</button>
      <button class="pe-btn pe-next" data-action="next">Next &#9654;</button>
    </div>
  </div>
</div>
"""

# ── Scoped CSS ───────────────────────────────────────────────────────────────
EDITOR_CSS_TEMPLATE = """
.pe-wrap {
  background: #1e1e2e; border-radius: 8px; padding: 10px;
  display: flex; flex-direction: column; gap: 8px; user-select: none;
}
.pe-canvas {
  display: block; max-width: 100%; border: 2px solid #555;
  border-radius: 4px; cursor: crosshair;
}
.pe-bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.pe-btn {
  padding: 6px 14px; border: none; border-radius: 5px;
  cursor: pointer; font-size: 13px; font-weight: 600; color: #fff;
}
.pe-reset  { background: #c0392b; }
.pe-names  { background: #2980b9; }
.pe-save   { background: #8e44ad; }
.pe-angles { background: #27ae60; }
.pe-custom-angle { background: #16a085; }
.pe-custom-angle.is-active { outline: 2px solid #fff; box-shadow: 0 0 0 2px rgba(22,160,133,.45); }
.pe-fullscreen { background: #e67e22; }
.pe-keypoints { background: #d35400; }
.pe-info   { font-size: 12px; color: #aaa; font-family: monospace; }
.pe-nav-group { margin-left: auto; display: flex; gap: 6px; }
.pe-prev   { background: #555e6e; }
.pe-next   { background: #555e6e; }

.pe-wrap:fullscreen { 
  padding: 15px; 
  background: #1e1e2e; 
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}
.pe-wrap:fullscreen .pe-canvas { 
  flex-grow: 1;
  width: 100%; 
  height: 0;
  object-fit: contain; 
  margin: 0 auto; 
  border: none;
}
.pe-wrap:fullscreen .pe-bar {
  flex-shrink: 0;
  margin-top: 15px;
}
/* Safari support */
.pe-wrap:-webkit-full-screen { 
  padding: 15px; 
  background: #1e1e2e; 
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}
.pe-wrap:-webkit-full-screen .pe-canvas { 
  flex-grow: 1;
  width: 100%; 
  height: 0;
  object-fit: contain; 
  margin: 0 auto; 
  border: none;
}
.pe-wrap:-webkit-full-screen .pe-bar {
  flex-shrink: 0;
  margin-top: 15px;
}
"""

# ── js_on_load ───────────────────────────────────────────────────────────────
# js_on_load fires ONCE on component mount (value is empty at that point).
# A MutationObserver catches Gradio's subsequent template re-renders (when
# the Python side sets a new value).  The dedup guard (_peKey) prevents
# re-initialisation on DOM mutations caused by info.textContent updates
# during drag — so drag never interrupts itself.  When data genuinely
# changes, the old canvas is replaced with a clone to strip stale listeners.
EDITOR_JS_ON_LOAD = r"""
function bootEditor() {
  var dataEl = element.querySelector('.pe-data');
  if (!dataEl) return;
  var raw = (dataEl.textContent || '').trim();
  if (!raw || raw.length < 20) return;

  /* same payload as last init → nothing to do (e.g. info.textContent mutation) */
  if (element._peKey === raw) return;
  element._peKey = raw;

  var DATA;
  try { DATA = JSON.parse(atob(raw)); } catch(e) { return; }

  /* replace canvas with a clone to remove ALL stale event listeners */
  var oldCanvas = element.querySelector('.pe-canvas');
  if (!oldCanvas) return;
  var canvas = oldCanvas.cloneNode(false);
  oldCanvas.parentNode.replaceChild(canvas, oldCanvas);

  var ctx  = canvas.getContext('2d');
  var info = element.querySelector('.pe-info');

  var R = 7;
  var origKps   = JSON.parse(JSON.stringify(DATA.kps));
  var keypoints = JSON.parse(JSON.stringify(DATA.kps));
  var skeleton  = DATA.sk;
  var names     = DATA.nm;
  var csScale   = DATA.cs;
  var dragging  = null;
  var hoveredAngle = null;
  var angleHitboxes = [];
  var customAngleMode = false;
  var customAnglePick = [];
  var customAngles = (DATA.manual_angles || []).map(function(a) {
    var pts = Array.isArray(a) ? a : (a.keypoint_indices || a.points || []);
    return [Number(pts[0]), Number(pts[1]), Number(pts[2])];
  }).filter(function(a) {
    return a.length === 3 && a.every(function(v) { return Number.isInteger(v); });
  });
  var showNames  = true;
  var showAngles = false;
  var showKeypoints = true;

  /* ── zoom / pan state ── */
  var zoom      = 1.0, panX = 0, panY = 0;
  var isPanning = false;
  var panStart  = {x:0, y:0}, panOrigin = {x:0, y:0};
  var MIN_ZOOM  = 0.5,  MAX_ZOOM = 10.0;

  function emitKeypointsToHiddenOutput() {
    var payload = JSON.stringify(editorStatePayload());
    var container = document.querySelector('#kp_editor_output');
    var el = container
      ? (container.querySelector('textarea') || container.querySelector('input[type="text"]') || container.querySelector('input'))
      : null;
    if (!el) return;
    var proto = el.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    var nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (nativeSetter && nativeSetter.set) {
      nativeSetter.set.call(el, payload);
    } else {
      el.value = payload;
    }
    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function kpC(i) { return 'hsl(' + Math.round(i / keypoints.length * 300) + ',80%,55%)'; }
  function skC(i) { return 'hsl(' + Math.round(i / skeleton.length  * 300) + ',70%,50%)'; }
  var ANGLE_CONF_THR = 0.01;

  function validAngleKeypoint(kp) {
    return kp &&
      Number.isFinite(kp.x) && Number.isFinite(kp.y) &&
      kp.c >= ANGLE_CONF_THR &&
      !(Math.abs(kp.x) < 1e-6 && Math.abs(kp.y) < 1e-6);
  }

  function editorStatePayload() {
    return {
      keypoints: keypoints,
      orig_keypoints: origKps,
      canvas_scale: csScale,
      show_standard_angles: showAngles,
      manual_angles: customAngles.map(function(a) {
        return { keypoint_indices: [a[0], a[1], a[2]] };
      })
    };
  }

  /* load image then size + draw canvas */
  var img = new window.Image();
  img.onload = function() {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    render();
    emitKeypointsToHiddenOutput();
    if (info) info.textContent =
      'Noktalari surukleleyin  (' + keypoints.length + ' keypoint)';
  };
  img.onerror = function() {
    if (info) info.textContent = 'Goruntu yuklenemedi!';
  };
  img.src = DATA.img;

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(img, 0, 0);

    /* constant screen-space sizes regardless of zoom level */
    var lw  = 2.5 / zoom;   /* skeleton line width  */
    var rr  = R   / zoom;   /* keypoint dot radius  */
    var fs  = 11  / zoom;   /* font size (px)       */

    /* skeleton lines */
    if (showKeypoints) {
      for (var si = 0; si < skeleton.length; si++) {
        var a = keypoints[skeleton[si][0]];
        var b = keypoints[skeleton[si][1]];
        if (!a || !b || a.c < 0.1 || b.c < 0.1) continue;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = skC(si);
        ctx.lineWidth = lw;
        ctx.stroke();
      }
    }

    /* keypoint dots */
    if (showKeypoints) {
      for (var i = 0; i < keypoints.length; i++) {
        var kp = keypoints[i];
        if (kp.c < 0.1) continue;
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, rr, 0, 2 * Math.PI);
        ctx.fillStyle = kpC(i);
        ctx.fill();
        ctx.strokeStyle = (dragging === i) ? '#fff' : 'rgba(255,255,255,.7)';
        ctx.lineWidth   = (dragging === i) ? 2 / zoom : 1.5 / zoom;
        ctx.stroke();
        if (showNames) {
          ctx.fillStyle = '#fff';
          ctx.font = fs + 'px sans-serif';
          ctx.fillText(names[i], kp.x + rr + 2 / zoom, kp.y + 4 / zoom);
        }
      }
    }
    angleHitboxes = [];
    drawAngles();
    drawCustomAngles();
    drawCustomAnglePick();
    drawAngleTooltip();
    ctx.restore();
  }

  /* ── 2-D joint-angle helpers ─────────────────────────────────────────── */
  function calcAngle(a, b, c) {
    /* Returns the angle at vertex B (in degrees) formed by rays B→A and B→C */
    var bax = a.x - b.x, bay = a.y - b.y;
    var bcx = c.x - b.x, bcy = c.y - b.y;
    var dot = bax * bcx + bay * bcy;
    var ma  = Math.sqrt(bax * bax + bay * bay);
    var mc  = Math.sqrt(bcx * bcx + bcy * bcy);
    if (ma < 1 || mc < 1) return null;
    return Math.round(Math.acos(Math.max(-1, Math.min(1, dot / (ma * mc)))) * 180 / Math.PI);
  }

  function drawAngles() {
    if (!showAngles) return;
    /* [idxA, idxB(vertex), idxC, label]  — standard COCO-25 indices */
    var ANG = [
      [9,  7,  13, 'R.Omuz'],
      [8,  6,  12, 'L.Omuz'],
      [7,  9,  11, 'R.Dirsek'],
      [6,  8,  10, 'L.Dirsek'],
      [7,  13, 16, 'R.Kalca'],
      [6,  12, 15, 'L.Kalca'],
      [13, 16, 18, 'R.Diz'],
      [12, 15, 17, 'L.Diz'],
      [16, 18, 22, 'R.AyakBilegi'],
      [15, 17, 19, 'L.AyakBilegi'],
      [24, 18, 22, 'R.AyakYonu'],
      [21, 17, 19, 'L.AyakYonu']
    ];
    ctx.save();
    for (var ai = 0; ai < ANG.length; ai++) {
      var ia = ANG[ai][0], ib = ANG[ai][1], ic = ANG[ai][2];
      if (ia >= keypoints.length || ib >= keypoints.length || ic >= keypoints.length) continue;
      var ka = keypoints[ia], kb = keypoints[ib], kc = keypoints[ic];
      if (!validAngleKeypoint(ka) || !validAngleKeypoint(kb) || !validAngleKeypoint(kc)) continue;
      var ang = calcAngle(ka, kb, kc);
      if (ang === null) continue;

      var dA   = Math.sqrt((ka.x-kb.x)*(ka.x-kb.x) + (ka.y-kb.y)*(ka.y-kb.y));
      var dC   = Math.sqrt((kc.x-kb.x)*(kc.x-kb.x) + (kc.y-kb.y)*(kc.y-kb.y));
      var arcR = Math.max(12 / zoom, Math.min(30 / zoom, Math.min(dA, dC) * 0.35));

      var angA = Math.atan2(ka.y - kb.y, ka.x - kb.x);
      var angC = Math.atan2(kc.y - kb.y, kc.x - kb.x);

      /* Always draw the minor arc */
      var diff = angC - angA;
      while (diff >  Math.PI) diff -= 2 * Math.PI;
      while (diff < -Math.PI) diff += 2 * Math.PI;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(kb.x, kb.y, arcR, angA, angC, diff < 0);
      ctx.strokeStyle = 'rgba(255,50,50,0.95)';
      ctx.lineWidth = 2 / zoom;
      ctx.stroke();

      /* Dashed radii from vertex */
      ctx.setLineDash([4 / zoom, 3 / zoom]);
      ctx.beginPath();
      ctx.moveTo(kb.x, kb.y);
      ctx.lineTo(kb.x + arcR * Math.cos(angA), kb.y + arcR * Math.sin(angA));
      ctx.moveTo(kb.x, kb.y);
      ctx.lineTo(kb.x + arcR * Math.cos(angC), kb.y + arcR * Math.sin(angC));
      ctx.strokeStyle = 'rgba(255,50,50,0.6)';
      ctx.lineWidth = 1.5 / zoom;
      ctx.stroke();
      ctx.setLineDash([]);

      /* Label — placed along the bisector of the two limb vectors */
      var mvx = (ka.x - kb.x) / (dA || 1) + (kc.x - kb.x) / (dC || 1);
      var mvy = (ka.y - kb.y) / (dA || 1) + (kc.y - kb.y) / (dC || 1);
      var mv  = Math.sqrt(mvx * mvx + mvy * mvy) || 1;
      var tx  = kb.x + (arcR + 18 / zoom) * mvx / mv;
      var ty  = kb.y + (arcR + 18 / zoom) * mvy / mv;
      ctx.font      = 'bold ' + (13 / zoom) + 'px sans-serif';
      ctx.lineWidth = 3 / zoom;
      ctx.strokeStyle = 'rgba(0,0,0,0.85)';
      ctx.strokeText(ang + '\u00b0', tx, ty);
      ctx.fillStyle = '#FF3333';
      ctx.fillText(ang + '\u00b0', tx, ty);

      angleHitboxes.push({
        x: tx,
        y: ty,
        r: Math.max(24 / zoom, arcR + 14 / zoom),
        label: ANG[ai][3],
        angle: ang,
        points: [
          nameForKeypoint(ia) + ' [' + ia + ']',
          nameForKeypoint(ib) + ' [' + ib + ']',
          nameForKeypoint(ic) + ' [' + ic + ']'
        ],
        vertex: nameForKeypoint(ib) + ' [' + ib + ']'
      });
    }
    ctx.restore();
  }

  function drawCustomAngles() {
    if (!customAngles.length) return;
    ctx.save();
    for (var ai = 0; ai < customAngles.length; ai++) {
      var def = customAngles[ai];
      drawCustomAngle(def[0], def[1], def[2], 'Ozel Aci ' + (ai + 1));
    }
    ctx.restore();
  }

  function drawCustomAngle(ia, ib, ic, label) {
    if (ia >= keypoints.length || ib >= keypoints.length || ic >= keypoints.length) return;
    var ka = keypoints[ia], kb = keypoints[ib], kc = keypoints[ic];
    if (!validAngleKeypoint(ka) || !validAngleKeypoint(kb) || !validAngleKeypoint(kc)) return;
    var ang = calcAngle(ka, kb, kc);
    if (ang === null) return;

    var dA   = Math.sqrt((ka.x-kb.x)*(ka.x-kb.x) + (ka.y-kb.y)*(ka.y-kb.y));
    var dC   = Math.sqrt((kc.x-kb.x)*(kc.x-kb.x) + (kc.y-kb.y)*(kc.y-kb.y));
    var arcR = Math.max(12 / zoom, Math.min(30 / zoom, Math.min(dA, dC) * 0.35));

    var angA = Math.atan2(ka.y - kb.y, ka.x - kb.x);
    var angC = Math.atan2(kc.y - kb.y, kc.x - kb.x);
    var diff = angC - angA;
    while (diff >  Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(kb.x, kb.y, arcR, angA, angC, diff < 0);
    ctx.strokeStyle = 'rgba(0,209,255,0.95)';
    ctx.lineWidth = 2 / zoom;
    ctx.stroke();

    ctx.setLineDash([4 / zoom, 3 / zoom]);
    ctx.beginPath();
    ctx.moveTo(kb.x, kb.y);
    ctx.lineTo(kb.x + arcR * Math.cos(angA), kb.y + arcR * Math.sin(angA));
    ctx.moveTo(kb.x, kb.y);
    ctx.lineTo(kb.x + arcR * Math.cos(angC), kb.y + arcR * Math.sin(angC));
    ctx.strokeStyle = 'rgba(0,209,255,0.55)';
    ctx.lineWidth = 1.5 / zoom;
    ctx.stroke();
    ctx.setLineDash([]);

    var mvx = (ka.x - kb.x) / (dA || 1) + (kc.x - kb.x) / (dC || 1);
    var mvy = (ka.y - kb.y) / (dA || 1) + (kc.y - kb.y) / (dC || 1);
    var mv  = Math.sqrt(mvx * mvx + mvy * mvy) || 1;
    var tx  = kb.x + (arcR + 18 / zoom) * mvx / mv;
    var ty  = kb.y + (arcR + 18 / zoom) * mvy / mv;
    ctx.font      = 'bold ' + (13 / zoom) + 'px sans-serif';
    ctx.lineWidth = 3 / zoom;
    ctx.strokeStyle = 'rgba(0,0,0,0.85)';
    ctx.strokeText(ang + '\u00b0', tx, ty);
    ctx.fillStyle = '#00d1ff';
    ctx.fillText(ang + '\u00b0', tx, ty);

    angleHitboxes.push({
      x: tx,
      y: ty,
      r: Math.max(24 / zoom, arcR + 14 / zoom),
      label: label,
      angle: ang,
      points: [
        nameForKeypoint(ia) + ' [' + ia + ']',
        nameForKeypoint(ib) + ' [' + ib + ']',
        nameForKeypoint(ic) + ' [' + ic + ']'
      ],
      vertex: nameForKeypoint(ib) + ' [' + ib + ']'
    });
  }

  function drawCustomAnglePick() {
    if (!customAngleMode || !customAnglePick.length) return;
    ctx.save();
    ctx.setLineDash([]);
    for (var i = 0; i < customAnglePick.length; i++) {
      var idx = customAnglePick[i];
      var kp = keypoints[idx];
      if (!kp || kp.c < 0.1) continue;
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, (R + 5) / zoom, 0, 2 * Math.PI);
      ctx.strokeStyle = i === 1 ? '#ffd166' : '#00d1ff';
      ctx.lineWidth = 3 / zoom;
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold ' + (13 / zoom) + 'px sans-serif';
      ctx.strokeStyle = 'rgba(0,0,0,0.9)';
      ctx.lineWidth = 3 / zoom;
      ctx.strokeText(String(i + 1), kp.x + 10 / zoom, kp.y - 8 / zoom);
      ctx.fillText(String(i + 1), kp.x + 10 / zoom, kp.y - 8 / zoom);
    }
    if (customAnglePick.length >= 2) {
      var a = keypoints[customAnglePick[0]];
      var b = keypoints[customAnglePick[1]];
      if (a && b) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = 'rgba(0,209,255,0.65)';
        ctx.lineWidth = 2 / zoom;
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  function nameForKeypoint(i) {
    return (names && names[i]) ? names[i] : String(i);
  }

  function findHoveredAngle(p) {
    if (!showAngles && !customAngles.length) return null;
    for (var i = angleHitboxes.length - 1; i >= 0; i--) {
      var h = angleHitboxes[i];
      var dx = p.x - h.x, dy = p.y - h.y;
      if (Math.sqrt(dx * dx + dy * dy) <= h.r) return h;
    }
    return null;
  }

  function drawAngleTooltip() {
    if (!hoveredAngle) return;

    var lines = [
      hoveredAngle.label + ': ' + hoveredAngle.angle + '\u00b0',
      'Keypointler: ' + hoveredAngle.points.join(' - '),
      'Merkez: ' + hoveredAngle.vertex
    ];
    var fontSize = 12 / zoom;
    var pad = 8 / zoom;
    var lineH = 17 / zoom;
    ctx.font = 'bold ' + fontSize + 'px sans-serif';

    var w = 0;
    for (var i = 0; i < lines.length; i++) {
      w = Math.max(w, ctx.measureText(lines[i]).width);
    }
    var boxW = w + pad * 2;
    var boxH = lineH * lines.length + pad * 2;
    var x = hoveredAngle.x + 16 / zoom;
    var y = hoveredAngle.y - boxH - 12 / zoom;

    if (x + boxW > canvas.width) x = hoveredAngle.x - boxW - 16 / zoom;
    if (y < 0) y = hoveredAngle.y + 18 / zoom;
    x = clamp(x, 2 / zoom, canvas.width - boxW - 2 / zoom);
    y = clamp(y, 2 / zoom, canvas.height - boxH - 2 / zoom);

    ctx.save();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(20,20,28,0.94)';
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1.5 / zoom;
    roundedRectPath(x, y, boxW, boxH, 6 / zoom);
    ctx.fill();
    ctx.stroke();

    for (var li = 0; li < lines.length; li++) {
      ctx.fillStyle = li === 0 ? '#ff6b6b' : '#ffffff';
      ctx.font = (li === 0 ? 'bold ' : '') + fontSize + 'px sans-serif';
      ctx.fillText(lines[li], x + pad, y + pad + lineH * (li + 0.72));
    }
    ctx.restore();
  }

  function roundedRectPath(x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  /* coordinate helpers */
  function getRawPos(e) {
    var r  = canvas.getBoundingClientRect();
    var canvasAspect = canvas.width / (canvas.height || 1);
    var rectAspect = r.width / (r.height || 1);
    var renderW = r.width, renderH = r.height;
    var offsetX = 0, offsetY = 0;
    
    if (canvasAspect > rectAspect) {
        renderH = r.width / canvasAspect;
        offsetY = (r.height - renderH) / 2;
    } else {
        renderW = r.height * canvasAspect;
        offsetX = (r.width - renderW) / 2;
    }
    
    var sx = canvas.width / renderW;
    var sy = canvas.height / renderH;
    var cx = e.touches ? e.touches[0].clientX : e.clientX;
    var cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - r.left - offsetX) * sx, y: (cy - r.top - offsetY) * sy };
  }

  function getPos(e) {
    /* World (keypoint) coords — accounts for current zoom & pan */
    var raw = getRawPos(e);
    return { x: (raw.x - panX) / zoom, y: (raw.y - panY) / zoom };
  }

  function nearest(p) {
    if (!showKeypoints) return null;
    var best = null, bestD = R + 15;  /* 22-px hit radius in canvas space */
    for (var i = 0; i < keypoints.length; i++) {
      var k = keypoints[i];
      if (k.c < 0.1) continue;
      var d = Math.sqrt((k.x - p.x) * (k.x - p.x) + (k.y - p.y) * (k.y - p.y));
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function setCustomAngleMode(active) {
    customAngleMode = active;
    customAnglePick = [];
    if (customAngleMode) showKeypoints = true;
    var btn = element.querySelector('[data-action="custom-angle"]');
    if (btn) btn.classList.toggle('is-active', customAngleMode);
    canvas.style.cursor = customAngleMode ? 'copy' : 'crosshair';
    render();
    if (info) {
      info.textContent = customAngleMode
        ? 'Ozel aci: 1. nokta, merkez nokta, 3. nokta seklinde secin.'
        : 'Ozel aci secimi kapatildi.';
    }
  }

  function pickCustomAngleKeypoint(idx) {
    if (idx === null || idx === undefined) return;
    if (customAnglePick.indexOf(idx) !== -1) {
      if (info) info.textContent = 'Bu keypoint zaten secildi; farkli bir nokta secin.';
      return;
    }
    customAnglePick.push(idx);
    if (customAnglePick.length < 3) {
      render();
      if (info) {
        var step = customAnglePick.length === 1 ? 'Merkez noktayi secin.' : '3. noktayi secin.';
        info.textContent = 'Ozel aci: ' + customAnglePick.length + '/3 secildi. ' + step;
      }
      return;
    }

    customAngles.push([customAnglePick[0], customAnglePick[1], customAnglePick[2]]);
    var a = keypoints[customAnglePick[0]];
    var b = keypoints[customAnglePick[1]];
    var c = keypoints[customAnglePick[2]];
    var ang = (a && b && c) ? calcAngle(a, b, c) : null;
    customAnglePick = [];
    render();
    emitKeypointsToHiddenOutput();
    if (info) {
      info.textContent = ang === null
        ? 'Ozel aci eklenemedi; secilen noktalari kontrol edin.'
        : 'Ozel aci eklendi: ' + ang + '\u00b0. Yeni aci icin 1. noktayi secin.';
    }
  }

  function finishDrag() {
    if (dragging !== null && info)
      info.textContent = names[dragging] + '  (' +
        Math.round(keypoints[dragging].x / csScale) + ', ' +
        Math.round(keypoints[dragging].y / csScale) + ')';
    if (dragging !== null) emitKeypointsToHiddenOutput();
    dragging = null;
    canvas.style.cursor = 'crosshair';
  }

  /* mouse
     IMPORTANT: do NOT set info.textContent inside mousemove.
     Any DOM text mutation triggers the MutationObserver, which schedules
     bootEditor() 80ms after every drag-pause.  Keep the DOM silent during
     drag; update info only on mousedown (which joint) and mouseup (final pos). */
  /* ── Scroll-to-zoom (cursor-centred) ── */
  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    var factor  = e.deltaY < 0 ? 1.15 : (1 / 1.15);
    var newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));
    var raw     = getRawPos(e);
    /* keep the hovered world-point fixed on screen */
    var wx = (raw.x - panX) / zoom;
    var wy = (raw.y - panY) / zoom;
    panX  = raw.x - wx * newZoom;
    panY  = raw.y - wy * newZoom;
    zoom  = newZoom;
    if (info) info.textContent = 'Zoom: ' + Math.round(zoom * 100) + '%  (Cift tikla sifirla)';
    render();
  }, { passive: false });

  /* ── Double-click → reset zoom ── */
  canvas.addEventListener('dblclick', function(e) {
    if (nearest(getPos(e)) === null) {
      zoom = 1.0; panX = 0; panY = 0;
      render();
      if (info) info.textContent = 'Zoom sifirlandi.';
    }
  });

  canvas.addEventListener('mousedown', function(e) {
    e.preventDefault();
    if (e.button === 1) {        /* middle button → always pan */
      isPanning = true;
      panStart  = getRawPos(e);
      panOrigin = {x: panX, y: panY};
      canvas.style.cursor = 'move';
      return;
    }
    var kpIdx = nearest(getPos(e));
    if (customAngleMode && e.button === 0) {
      if (kpIdx !== null) {
        pickCustomAngleKeypoint(kpIdx);
        return;
      }
      isPanning = true;
      panStart  = getRawPos(e);
      panOrigin = {x: panX, y: panY};
      canvas.style.cursor = 'move';
      return;
    }
    if (kpIdx !== null) {        /* left click on keypoint → drag */
      dragging = kpIdx;
      canvas.style.cursor = 'grabbing';
      if (info) info.textContent = names[dragging] + ' surukleniyor...';
    } else {                     /* left click on empty area → pan */
      isPanning = true;
      panStart  = getRawPos(e);
      panOrigin = {x: panX, y: panY};
      canvas.style.cursor = 'move';
    }
  });

  canvas.addEventListener('mousemove', function(e) {
    e.preventDefault();
    if (isPanning) {
      var raw = getRawPos(e);
      panX = panOrigin.x + (raw.x - panStart.x);
      panY = panOrigin.y + (raw.y - panStart.y);
      render();
      return;
    }
    var p = getPos(e);
    if (dragging === null) {
      var nextHoveredAngle = findHoveredAngle(p);
      if (nextHoveredAngle !== hoveredAngle) {
        hoveredAngle = nextHoveredAngle;
        render();
      }
      canvas.style.cursor = nearest(p) !== null ? (customAngleMode ? 'copy' : 'grab') : (hoveredAngle ? 'help' : (zoom > 1 ? 'zoom-in' : 'crosshair'));
      return;
    }
    hoveredAngle = null;
    keypoints[dragging].x = clamp(p.x, 0, canvas.width);
    keypoints[dragging].y = clamp(p.y, 0, canvas.height);
    render();
  });

  canvas.addEventListener('mouseup', function() {
    if (isPanning) { isPanning = false; canvas.style.cursor = 'crosshair'; return; }
    finishDrag();
  });

  canvas.addEventListener('mouseleave', function() {
    isPanning = false;
    dragging  = null;
    hoveredAngle = null;
    canvas.style.cursor = 'crosshair';
    render();
  });

  /* touch — same rule: no info.textContent inside touchmove */
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    var touched = nearest(getPos(e));
    if (customAngleMode) {
      if (touched !== null) pickCustomAngleKeypoint(touched);
      return;
    }
    dragging = touched;
    if (dragging !== null && info) info.textContent = names[dragging] + ' surukleniyor...';
  }, { passive: false });

  canvas.addEventListener('touchmove', function(e) {
    e.preventDefault();
    if (dragging === null) return;
    var p = getPos(e);
    keypoints[dragging].x = clamp(p.x, 0, canvas.width);
    keypoints[dragging].y = clamp(p.y, 0, canvas.height);
    render();
  }, { passive: false });

  canvas.addEventListener('touchend', function() {
    finishDrag();
  });

  window.addEventListener('mouseup', function() {
    if (isPanning) { isPanning = false; }
    if (dragging !== null) finishDrag();
  });

  window.addEventListener('touchend', function() {
    if (dragging !== null) finishDrag();
  }, { passive: true });

  /* buttons — clone each to remove any previous listeners from old initialisations */
  function rebind(sel, fn) {
    var old = element.querySelector(sel);
    if (!old) return;
    var fresh = old.cloneNode(true);
    old.parentNode.replaceChild(fresh, old);
    fresh.addEventListener('click', fn);
  }

  /* Expose getter so the Apply-Changes button JS can read fresh keypoints directly */
  var peWrap = element.querySelector('.pe-wrap');
  if (peWrap) {
    peWrap._peGetKps = function() {
      return JSON.stringify(editorStatePayload());
    };
  }

  rebind('[data-action="reset"]', function() {
    keypoints = JSON.parse(JSON.stringify(origKps));
    zoom = 1.0; panX = 0; panY = 0;
    customAnglePick = [];
    customAngles = [];
    render();
    emitKeypointsToHiddenOutput();
    if (info) info.textContent = 'Orijinal konumlar ve zoom sifirlandi.';
  });

  rebind('[data-action="names"]', function() {
    showNames = !showNames;
    render();
  });

  rebind('[data-action="save"]', function() {
    var a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = 'pose_edited.png';
    a.click();
    if (info) info.textContent = 'PNG kaydedildi.';
  });

  rebind('[data-action="angles"]', function() {
    showAngles = !showAngles;
    render();
    if (info) info.textContent = showAngles ? 'Acilar gosteriliyor.' : 'Acilar gizlendi.';
  });

  rebind('[data-action="custom-angle"]', function() {
    setCustomAngleMode(!customAngleMode);
  });

  rebind('[data-action="fullscreen"]', function() {
    var wrap = element.querySelector('.pe-wrap');
    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
      if (wrap.requestFullscreen) { wrap.requestFullscreen(); }
      else if (wrap.webkitRequestFullscreen) { wrap.webkitRequestFullscreen(); }
    } else {
      if (document.exitFullscreen) { document.exitFullscreen(); }
      else if (document.webkitExitFullscreen) { document.webkitExitFullscreen(); }
    }
  });

  rebind('[data-action="keypoints"]', function() {
    showKeypoints = !showKeypoints;
    render();
    if (info) info.textContent = showKeypoints ? 'Noktalar gosteriliyor.' : 'Noktalar gizlendi.';
  });

  function emitNavTrigger(elemId) {
    var container = document.querySelector(elemId);
    var el = container
      ? (container.querySelector('textarea') || container.querySelector('input[type="text"]') || container.querySelector('input'))
      : null;
    if (!el) { console.warn('Nav trigger not found:', elemId); return; }
    var proto = el.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    var nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (nativeSetter && nativeSetter.set) nativeSetter.set.call(el, String(Date.now()));
    else el.value = String(Date.now());
    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  rebind('[data-action="prev"]', function() {
    emitNavTrigger('#pe_prev_trigger');
    if (info) info.textContent = 'Onceki goruntüye geçiliyor...';
  });

  rebind('[data-action="next"]', function() {
    emitNavTrigger('#pe_next_trigger');
    if (info) info.textContent = 'Sonraki goruntüye geçiliyor...';
  });
}

/* run once on mount (value is empty → returns early) */
bootEditor();

/* MutationObserver: fires when Gradio re-renders the template with new value.
   Debounced so rapid DOM changes (e.g. CSS transitions) don't spam calls.
   The _peKey dedup guard inside bootEditor handles info.textContent mutations
   that occur during drag — they cause the observer to fire but bootEditor
   returns early without touching the canvas or its listeners. */
if (!element._peObserver) {
  element._peObserver = new MutationObserver(function() {
    clearTimeout(element._peTimer);
    element._peTimer = setTimeout(bootEditor, 80);
  });
  /* characterData:true catches Svelte's fine-grained text-node updates
     (when only pe-data textContent changes, childList alone won't fire). */
  element._peObserver.observe(element, {
    childList: true,
    subtree: true,
    characterData: true,
  });
}
"""


# ── Python helpers ───────────────────────────────────────────────────────────

def _img_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_editor_component() -> gr.HTML:
    """Return a configured gr.HTML component for the pose editor."""
    return gr.HTML(
        value="",
        html_template=EDITOR_HTML_TEMPLATE,
        css_template=EDITOR_CSS_TEMPLATE,
        js_on_load=EDITOR_JS_ON_LOAD,
        min_height=200,
        container=False,
        padding=False,
    )


def prepare_editor(
    image: Image.Image,
    json_file,
    parse_pose_json_fn,
) -> Tuple[str, str]:
    """Build editor payload — returns (base64-encoded JSON string, status)."""
    if image is None:
        return "", "Once bir goruntu yukle."
    if json_file is None:
        return "", "Bir JSON dosyasi yukle."

    from pathlib import Path
    json_path = Path(json_file.name)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    try:
        idx_to_name, kps = parse_pose_json_fn(data)
    except Exception as e:
        return "", f"JSON parse error: {e}"

    orig_w, orig_h = image.size
    cs = min(1.0, _CANVAS_MAX_PX / max(orig_w, orig_h))
    disp = image.convert("RGB")
    if cs < 1.0:
        disp = disp.resize((int(orig_w * cs), int(orig_h * cs)), Image.LANCZOS)

    canvas_kps = [
        {"id": i, "x": float(kps[i][1]) * cs, "y": float(kps[i][0]) * cs, "c": float(kps[i][2])}
        for i in range(len(kps))
    ]

    skeleton = joints_dict()["coco_25"]["skeleton"]
    kp_names = [idx_to_name.get(i, str(i)) for i in range(len(kps))]
    img_b64  = _img_to_b64(disp)

    payload = json.dumps({
        "img": f"data:image/jpeg;base64,{img_b64}",
        "kps": canvas_kps,
        "sk":  skeleton,
        "nm":  kp_names,
        "cs":  cs,
        "manual_angles": data.get("manual_angles", []),
    }, separators=(",", ":"))

    value = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return value, f"Editor hazir: {json_path.name}  |  scale={cs:.2f}"


def _build_anchor_points(h: int, w: int, step: int = 80) -> np.ndarray:
    """Generate anchor points along image edges to stabilise TPS warp."""
    pts = []
    # four corners
    for y in (0, h - 1):
        for x in (0, w - 1):
            pts.append([y, x])
    # edge samples
    for x in range(step, w - 1, step):
        pts.append([0, x])
        pts.append([h - 1, x])
    for y in range(step, h - 1, step):
        pts.append([y, 0])
        pts.append([y, w - 1])
    return np.array(pts, dtype=np.float64)


def tps_warp_image(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    grid_max: int = 400,
) -> np.ndarray:
    """Warp *img* so that pixels at *src_pts* move to *dst_pts*.

    Uses scipy's RBFInterpolator with thin-plate-spline kernel.
    Both point arrays are shape (N, 2) in **(row, col)** order.

    For performance, the warp map is computed on a downscaled grid
    (max *grid_max* px on the longer side) then upscaled to full
    resolution before applying cv2.remap.
    """
    h, w = img.shape[:2]

    # anchor points — identical in source & destination so edges stay fixed
    anchors = _build_anchor_points(h, w, step=80)
    all_src = np.vstack([src_pts, anchors]).astype(np.float64)
    all_dst = np.vstack([dst_pts, anchors]).astype(np.float64)

    # We need the REVERSE mapping: for every output pixel find where to
    # sample in the input.  So we fit  dst → src.
    interp_y = RBFInterpolator(all_dst, all_src[:, 0], kernel="thin_plate_spline", smoothing=0.0)
    interp_x = RBFInterpolator(all_dst, all_src[:, 1], kernel="thin_plate_spline", smoothing=0.0)

    # Compute warp map on a SMALLER grid for speed, then upscale
    scale = min(1.0, grid_max / max(h, w))
    gh, gw = max(1, int(h * scale)), max(1, int(w * scale))

    gy = np.linspace(0, h - 1, gh)
    gx = np.linspace(0, w - 1, gw)
    grid_y, grid_x = np.meshgrid(gy, gx, indexing="ij")
    query = np.column_stack([grid_y.ravel(), grid_x.ravel()]).astype(np.float64)

    small_map_y = interp_y(query).reshape(gh, gw).astype(np.float32)
    small_map_x = interp_x(query).reshape(gh, gw).astype(np.float32)

    # Upscale warp maps to full resolution
    if scale < 1.0:
        map_y = cv2.resize(small_map_y, (w, h), interpolation=cv2.INTER_LINEAR)
        map_x = cv2.resize(small_map_x, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        map_y, map_x = small_map_y, small_map_x

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    return warped


def apply_edited_keypoints(
    image: Image.Image,
    kps_json: str,
) -> Tuple[Optional[Image.Image], str]:
  """Redraw only skeleton/keypoints on top of the original image.

  The underlying photo pixels are preserved; only pose coordinates change.
  """
  if image is None:
    return None, "Goruntu yuklenmemis."
  if not kps_json or not kps_json.strip():
    return None, "Keypoint verisi yok - once canvas'ta 'Export Keypoints' tikla."

  try:
    payload = json.loads(kps_json)
  except Exception as e:
    return None, f"JSON parse error: {e}"

  kps_list = payload["keypoints"]
  canvas_scale = float(payload.get("canvas_scale", 1.0))

  # Build edited keypoint array in image-space coordinates.
  kps_arr = np.zeros((25, 3), dtype=np.float32)
  for kp in kps_list:
    idx = int(kp["id"])
    kps_arr[idx] = [kp["y"] / canvas_scale, kp["x"] / canvas_scale, kp["c"]]

  img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

  conf_thr = 0.09
  # Draw skeleton directly on the original image without geometric warp.
  skeleton = joints_dict()["coco_25"]["skeleton"]
  img_np = draw_points_and_skeleton(
    img_np,
    kps_arr,
    skeleton,
    person_index=0,
    points_color_palette="gist_rainbow",
    skeleton_color_palette="jet",
    points_palette_samples=10,
    confidence_threshold=conf_thr,
  )
  out = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
  return out, "Duzenlenen keypointler uygulandi (sadece iskelet koordinatlari guncellendi)"


# ── Path-based editor helpers ────────────────────────────────────────────────

def prepare_editor_from_path(
    original_img_path: str,
    json_path_str: str,
) -> Tuple[str, str]:
    """Build editor payload from file paths (no gr.File needed).

    Uses the **original** source image so the canvas background is clean.
    Keypoints come from the saved JSON.
    """
    original_img_path = (original_img_path or "").strip()
    if not original_img_path:
        return "", "Orijinal goruntu yolu belirtilmemis."

    json_path_str = (json_path_str or "").split("\n")[0].strip()
    if not json_path_str:
        return "", "JSON yolu belirtilmemis."

    from pathlib import Path
    img_path = Path(original_img_path)
    if not img_path.exists():
        return "", f"Goruntu dosyasi bulunamadi: {img_path}"

    json_path = Path(json_path_str)
    if not json_path.exists():
        return "", f"JSON dosyasi bulunamadi: {json_path}"

    image = Image.open(img_path).convert("RGB")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Inline JSON parse (same logic as app.parse_pose_json)
    idx_to_name = {int(k): v for k, v in data.get("skeleton", {}).items()}
    kp_outer = data.get("keypoints", [])
    if not kp_outer:
        return "", "JSON'da 'keypoints' bulunamadi."
    person_dict = kp_outer[0]
    if not isinstance(person_dict, dict) or not person_dict:
        return "", "JSON'da tespit edilmis kisi/keypoint yok."
    kp_list = person_dict.get("0")
    if kp_list is None:
        kp_list = person_dict[next(iter(person_dict.keys()))]
    if len(kp_list) != 25:
        return "", f"Beklenen 25 keypoint, gelen: {len(kp_list)}"
    # kp_list: [[row, col, c], ...]  (model output is y,x,c order)
    kps = [(float(a), float(b), float(c)) for a, b, c in kp_list]

    orig_w, orig_h = image.size
    cs = min(1.0, _CANVAS_MAX_PX / max(orig_w, orig_h))
    disp = image
    if cs < 1.0:
        disp = image.resize((int(orig_w * cs), int(orig_h * cs)), Image.LANCZOS)

    # kps[i] = (row, col, c)  → canvas x = col*cs, canvas y = row*cs
    canvas_kps = [
        {"id": i, "x": kps[i][1] * cs, "y": kps[i][0] * cs, "c": kps[i][2]}
        for i in range(len(kps))
    ]

    skeleton = joints_dict()["coco_25"]["skeleton"]
    kp_names = [idx_to_name.get(i, str(i)) for i in range(len(kps))]
    img_b64  = _img_to_b64(disp)

    payload = json.dumps({
        "img": f"data:image/jpeg;base64,{img_b64}",
        "kps": canvas_kps,
        "sk":  skeleton,
        "nm":  kp_names,
        "cs":  cs,
        "manual_angles": data.get("manual_angles", []),
    }, separators=(",", ":"))

    value = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return value, f"Editor hazir: {json_path.name}  |  scale={cs:.2f}"


def _extract_existing_json_path(text: str) -> str:
    """Pick the first existing .json path from a multiline textbox value."""
    from pathlib import Path

    for line in (text or "").splitlines():
        candidate = line.strip()
        if not candidate.lower().endswith(".json"):
            continue
        try:
            if Path(candidate).exists():
                return candidate
        except Exception:
            continue
    return ""


def _calc_angle_degrees_from_rc(kps_rc: list, ia: int, ib: int, ic: int) -> Optional[float]:
    """Return the 2D angle at keypoint B from image-space (row, col, conf) keypoints."""
    try:
        a = kps_rc[ia]
        b = kps_rc[ib]
        c = kps_rc[ic]
        if not (_valid_angle_keypoint_rc(a) and _valid_angle_keypoint_rc(b) and _valid_angle_keypoint_rc(c)):
            return None
        bax = float(a[1]) - float(b[1])
        bay = float(a[0]) - float(b[0])
        bcx = float(c[1]) - float(b[1])
        bcy = float(c[0]) - float(b[0])
        ma = float(np.hypot(bax, bay))
        mc = float(np.hypot(bcx, bcy))
        if ma < 1 or mc < 1:
            return None
        cosang = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (ma * mc)))
        return round(float(np.degrees(np.arccos(cosang))), 3)
    except Exception:
        return None


def _valid_angle_keypoint_rc(kp: Any) -> bool:
    try:
        y = float(kp[0])
        x = float(kp[1])
        conf = float(kp[2])
    except Exception:
        return False
    return (
        np.isfinite(y)
        and np.isfinite(x)
        and conf >= _ANGLE_CONFIDENCE_THRESHOLD
        and not (abs(x) < 1e-6 and abs(y) < 1e-6)
    )


def _manual_angle_indices(raw_angle: Any) -> Optional[list]:
    if isinstance(raw_angle, dict):
        raw_indices = raw_angle.get("keypoint_indices") or raw_angle.get("points")
    else:
        raw_indices = raw_angle
    if not isinstance(raw_indices, (list, tuple)) or len(raw_indices) != 3:
        return None
    try:
        indices = [int(raw_indices[0]), int(raw_indices[1]), int(raw_indices[2])]
    except Exception:
        return None
    if len(set(indices)) != 3 or any(i < 0 or i >= 25 for i in indices):
        return None
    return indices


def _build_manual_angle_records(
    raw_angles: Any,
    kps_rc: list,
    idx_to_name: Dict[int, str],
) -> list:
    records = []
    if not isinstance(raw_angles, list):
        return records

    seen = set()
    for n, raw_angle in enumerate(raw_angles, start=1):
        indices = _manual_angle_indices(raw_angle)
        if indices is None:
            continue
        key = tuple(indices)
        if key in seen:
            continue
        seen.add(key)

        ia, ib, ic = indices
        angle_degrees = _calc_angle_degrees_from_rc(kps_rc, ia, ib, ic)
        if angle_degrees is None:
            continue
        names = [idx_to_name.get(i, str(i)) for i in indices]
        records.append({
            "label": f"manual_angle_{len(records) + 1}",
            "keypoint_indices": indices,
            "keypoint_names": names,
            "vertex_index": ib,
            "vertex_name": idx_to_name.get(ib, str(ib)),
            "vectors": [
                {
                    "from_index": ib,
                    "from_name": idx_to_name.get(ib, str(ib)),
                    "to_index": ia,
                    "to_name": idx_to_name.get(ia, str(ia)),
                },
                {
                    "from_index": ib,
                    "from_name": idx_to_name.get(ib, str(ib)),
                    "to_index": ic,
                    "to_name": idx_to_name.get(ic, str(ic)),
                },
            ],
            "angle_degrees": angle_degrees,
            "source": "manual_canvas_selection",
            "method": "2D angle between vectors B->A and B->C; keypoint order is [A, B(vertex), C]",
        })
    return records


def _build_standard_angle_records(
    kps_rc: list,
    idx_to_name: Dict[int, str],
) -> list:
    records = []
    for ia, ib, ic, label in _STANDARD_ANGLE_DEFINITIONS:
        if ia >= len(kps_rc) or ib >= len(kps_rc) or ic >= len(kps_rc):
            continue
        angle_degrees = _calc_angle_degrees_from_rc(kps_rc, ia, ib, ic)
        if angle_degrees is None:
            continue
        names = [idx_to_name.get(i, str(i)) for i in (ia, ib, ic)]
        records.append({
            "label": label,
            "keypoint_indices": [ia, ib, ic],
            "keypoint_names": names,
            "vertex_index": ib,
            "vertex_name": idx_to_name.get(ib, str(ib)),
            "vectors": [
                {
                    "from_index": ib,
                    "from_name": idx_to_name.get(ib, str(ib)),
                    "to_index": ia,
                    "to_name": idx_to_name.get(ia, str(ia)),
                },
                {
                    "from_index": ib,
                    "from_name": idx_to_name.get(ib, str(ib)),
                    "to_index": ic,
                    "to_name": idx_to_name.get(ic, str(ic)),
                },
            ],
            "angle_degrees": angle_degrees,
            "source": "standard_canvas_angle",
            "method": "2D angle between vectors B->A and B->C; keypoint order is [A, B(vertex), C]",
        })
    return records


def _safe_filename_part(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_") or "pose"


def _manual_angle_schema() -> Dict[str, Any]:
    return {
        "version": 1,
        "coordinate_source": "keypoints",
        "keypoint_order": "[A, B(vertex), C]",
        "method": "2D angle between vectors B->A and B->C",
        "unit": "degrees",
    }


def _save_angles_sidecar_json(
    manual_angle_records: list,
    standard_angle_records: list,
    source_json_path: str,
    original_img_path: str,
) -> Optional[Path]:
    """Write manual angle records to easy_ViTPose/temp/açılar as a separate JSON file."""
    _ANGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_json_path) if source_json_path else None
    if source_path:
        base = f"{source_path.parent.name}_{source_path.stem}"
    else:
        base = Path(original_img_path or "pose").stem
    out_path = _ANGLE_OUTPUT_DIR / f"{_safe_filename_part(base)}_angles.json"

    out_data = {
        "source_json": str(source_path) if source_path else "",
        "source_image": original_img_path or "",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "manual_angle_schema": _manual_angle_schema(),
        "angles": standard_angle_records + manual_angle_records,
        "standard_angles": standard_angle_records,
        "manual_angles": manual_angle_records,
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _build_editor_payload_from_kps(
    original_img_path: str,
    kps_rc: list,
    idx_to_name: Optional[Dict[int, str]] = None,
    manual_angles: Optional[list] = None,
) -> Tuple[str, str]:
    """Create editor payload directly from (row, col, conf) keypoints."""
    img_path = Path((original_img_path or "").strip())
    if not img_path.exists():
        return "", f"Goruntu bulunamadi: {img_path}"

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    cs = min(1.0, _CANVAS_MAX_PX / max(orig_w, orig_h))
    disp = image if cs >= 1.0 else image.resize((int(orig_w * cs), int(orig_h * cs)), Image.LANCZOS)

    canvas_kps = [
        {"id": i, "x": float(kps_rc[i][1]) * cs, "y": float(kps_rc[i][0]) * cs, "c": float(kps_rc[i][2])}
        for i in range(len(kps_rc))
    ]

    kp_names = [idx_to_name.get(i, str(i)) if idx_to_name else str(i) for i in range(len(kps_rc))]
    payload = json.dumps({
        "img": f"data:image/jpeg;base64,{_img_to_b64(disp)}",
        "kps": canvas_kps,
        "sk": joints_dict()["coco_25"]["skeleton"],
        "nm": kp_names,
        "cs": cs,
        "manual_angles": manual_angles or [],
    }, separators=(",", ":"))

    return base64.b64encode(payload.encode("utf-8")).decode("ascii"), "OK"


def _build_editor_payload_from_canvas_kps(
    original_img_path: str,
    canvas_kps: list,
    canvas_scale: float,
    idx_to_name: Optional[Dict[int, str]] = None,
    manual_angles: Optional[list] = None,
) -> Tuple[str, str]:
    """Create editor payload from current canvas-space keypoints (x, y, c)."""
    img_path = Path((original_img_path or "").strip())
    if not img_path.exists():
        return "", f"Goruntu bulunamadi: {img_path}"

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    cs = min(1.0, _CANVAS_MAX_PX / max(orig_w, orig_h))
    disp = image if cs >= 1.0 else image.resize((int(orig_w * cs), int(orig_h * cs)), Image.LANCZOS)

    if canvas_scale <= 0:
        return "", "Gecersiz canvas scale"

    scale_ratio = cs / float(canvas_scale)
    payload_kps = [
        {
            "id": int(kp["id"]),
            "x": float(kp["x"]) * scale_ratio,
            "y": float(kp["y"]) * scale_ratio,
            "c": float(kp.get("c", 1.0)),
        }
        for kp in canvas_kps
    ]

    kp_names = [idx_to_name.get(i, str(i)) if idx_to_name else str(i) for i in range(len(payload_kps))]
    payload = json.dumps({
      "img": f"data:image/jpeg;base64,{_img_to_b64(disp)}",
      "kps": payload_kps,
      "sk": joints_dict()["coco_25"]["skeleton"],
      "nm": kp_names,
      "cs": cs,
      "manual_angles": manual_angles or [],
    }, separators=(",", ":"))

    return base64.b64encode(payload.encode("utf-8")).decode("ascii"), "OK"


def apply_and_save_keypoints(
    original_img_path: str,
    kps_json: str,
    json_path_str: str,
    current_payload: str = "",
) -> Tuple[str, str]:
    """Persist edited keypoints to JSON, then return a fresh canvas payload.

    Image pixels are NEVER warped — only the skeleton overlay changes.
    Returns (editor_html_payload, status_message).
    """
    from pathlib import Path

    original_img_path = (original_img_path or "").strip()
    if not original_img_path:
      return current_payload, "Orijinal goruntu yolu yok."

    if not kps_json or not kps_json.strip():
      return current_payload, "Keypoint verisi yok - once noktayi surukleyin."

    try:
      payload = json.loads(kps_json)
    except Exception as e:
      return current_payload, f"JSON parse error: {e}"

    img_path = Path(original_img_path)
    if not img_path.exists():
      return current_payload, f"Goruntu bulunamadi: {img_path}"

    kps_list = payload["keypoints"]
    canvas_scale = float(payload.get("canvas_scale", 1.0))
    raw_manual_angles = payload.get("manual_angles", [])
    save_standard_angles = bool(payload.get("show_standard_angles", False))

    # Convert canvas coords -> image coords (row, col, c)
    kps_for_json = [[0.0, 0.0, 0.0] for _ in range(25)]
    for kp in kps_list:
      idx = int(kp["id"])
      kps_for_json[idx] = [
        float(kp["y"]) / canvas_scale,
        float(kp["x"]) / canvas_scale,
        float(kp["c"]),
      ]

    status = "Goruntu guncellendi (JSON yolu bulunamadi)"
    idx_to_name: Dict[int, str] = {i: str(i) for i in range(25)}
    json_path = _extract_existing_json_path(json_path_str or "")

    manual_angle_records = _build_manual_angle_records(raw_manual_angles, kps_for_json, idx_to_name)
    standard_angle_records = _build_standard_angle_records(kps_for_json, idx_to_name) if save_standard_angles else []

    if json_path:
      try:
        p = Path(json_path)
        orig_data = json.loads(p.read_text(encoding="utf-8"))

        idx_to_name = {int(k): v for k, v in orig_data.get("skeleton", {}).items()} or idx_to_name
        manual_angle_records = _build_manual_angle_records(raw_manual_angles, kps_for_json, idx_to_name)
        standard_angle_records = _build_standard_angle_records(kps_for_json, idx_to_name) if save_standard_angles else []

        if not orig_data.get("keypoints") or not isinstance(orig_data["keypoints"], list):
          orig_data["keypoints"] = [{"0": kps_for_json}]
        else:
          person_dict = orig_data["keypoints"][0]
          if not isinstance(person_dict, dict) or not person_dict:
            person_dict = {"0": kps_for_json}
            orig_data["keypoints"][0] = person_dict
          person_key = "0" if "0" in person_dict else next(iter(person_dict.keys()))
          person_dict[person_key] = kps_for_json

        orig_data["angles"] = standard_angle_records + manual_angle_records
        orig_data["standard_angles"] = standard_angle_records
        orig_data["manual_angles"] = manual_angle_records
        orig_data["manual_angle_schema"] = _manual_angle_schema()

        p.write_text(json.dumps(orig_data, ensure_ascii=False), encoding="utf-8")
        angles_path = _save_angles_sidecar_json(manual_angle_records, standard_angle_records, str(p), original_img_path)
        status = f"Kaydedildi: {p.name} | aci: {len(standard_angle_records) + len(manual_angle_records)} | aci JSON: {angles_path}"
      except Exception as e:
        return current_payload, f"JSON kaydedilemedi: {e}"
    else:
      try:
        angles_path = _save_angles_sidecar_json(manual_angle_records, standard_angle_records, "", original_img_path)
        status = f"Goruntu guncellendi | aci: {len(standard_angle_records) + len(manual_angle_records)} | aci JSON: {angles_path}"
      except Exception as e:
        return current_payload, f"Aci JSON kaydedilemedi: {e}"

    new_payload, prep_status = _build_editor_payload_from_canvas_kps(
      original_img_path,
      kps_list,
      canvas_scale=canvas_scale,
      idx_to_name=idx_to_name,
      manual_angles=manual_angle_records,
    )
    if not new_payload:
      return current_payload, f"{status} | Canvas yenilenemedi: {prep_status}"
    return new_payload, status
