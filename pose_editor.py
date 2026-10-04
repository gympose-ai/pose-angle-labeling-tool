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
    <button class="pe-btn pe-angle-select" data-action="angle-select">&#128204; A&#231;&#305; Se&#231;</button>
    <button class="pe-btn pe-custom-angle" data-action="custom-angle">&#8736; &#214;zel A&#231;&#305;</button>
    <button class="pe-btn pe-delete-angle" data-action="delete-angle">&#9003; A&#231;&#305; Sil</button>
    <button class="pe-btn pe-fullscreen" data-action="fullscreen">&#x2922; Tam Ekran</button>
    <button class="pe-btn pe-keypoints" data-action="keypoints">&#x25CF; Noktalar</button>
    <span class="pe-info">Goruntu yukleniyor...</span>
    <div class="pe-nav-group">
      <button class="pe-btn pe-prev" data-action="prev">&#9664; Prev</button>
      <button class="pe-btn pe-next" data-action="next">Next &#9654;</button>
    </div>
  </div>
  <div class="pe-angle-picker" style="display:none"></div>
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
.pe-angle-select { background: #2f80ed; }
.pe-angle-select.is-active { outline: 2px solid #fff; box-shadow: 0 0 0 2px rgba(47,128,237,.45); }
.pe-custom-angle { background: #16a085; }
.pe-custom-angle.is-active { outline: 2px solid #fff; box-shadow: 0 0 0 2px rgba(22,160,133,.45); }
.pe-delete-angle { background: #b83280; }
.pe-delete-angle.is-active { outline: 2px solid #fff; box-shadow: 0 0 0 2px rgba(184,50,128,.45); }
.pe-fullscreen { background: #e67e22; }
.pe-keypoints { background: #d35400; }
.pe-info   { font-size: 12px; color: #aaa; font-family: monospace; }
.pe-nav-group { margin-left: auto; display: flex; gap: 6px; }
.pe-prev   { background: #555e6e; }
.pe-next   { background: #555e6e; }
.pe-angle-picker {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 6px;
  background: rgba(255,255,255,.06);
}
.pe-angle-group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.pe-angle-group-title {
  color: #d8dee9;
  font: 700 12px sans-serif;
  min-width: 86px;
}
.pe-angle-choice {
  border: 1px solid rgba(255,255,255,.22);
  background: rgba(255,255,255,.10);
  color: #fff;
  border-radius: 5px;
  padding: 5px 9px;
  font-size: 12px;
  cursor: pointer;
}
.pe-angle-choice.is-selected {
  background: #ffb000;
  border-color: #ffd166;
  color: #1e1e2e;
  font-weight: 700;
}

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
  var anglePicker = element.querySelector('.pe-angle-picker');

  var R = 7;
  var origKps   = JSON.parse(JSON.stringify(DATA.kps));
  var keypoints = JSON.parse(JSON.stringify(DATA.kps));
  var skeleton  = DATA.sk;
  var names     = DATA.nm;
  var csScale   = DATA.cs;
  var editorRole = DATA.editor_role || 'main';
  var outputId = DATA.output_id || 'kp_editor_output';
  var prevTriggerId = DATA.prev_trigger_id || 'pe_prev_trigger';
  var nextTriggerId = DATA.next_trigger_id || 'pe_next_trigger';
  var dragging  = null;
  var hoveredAngle = null;
  var angleHitboxes = [];
  var customAngleMode = false;
  var customAnglePick = [];
  var deleteAngleMode = false;
  var deletedStandardAngles = {};
  (DATA.deleted_standard_angles || []).forEach(function(label) {
    deletedStandardAngles[String(label)] = true;
  });
  var customAngles = (DATA.manual_angles || []).map(function(a) {
    var pts = Array.isArray(a) ? a : (a.keypoint_indices || a.points || []);
    return [Number(pts[0]), Number(pts[1]), Number(pts[2])];
  }).filter(function(a) {
    return a.length === 3 && a.every(function(v) { return Number.isInteger(v); });
  });
  var showNames  = true;
  var showAngles = false;
  var showSelectedAngles = Boolean(DATA.show_selected_angles);
  var selectedAngleKeys = {};
  (DATA.selected_angle_keys || []).forEach(function(key) {
    selectedAngleKeys[String(key)] = true;
  });
  var showKeypoints = true;

  var STANDARD_ANGLE_DEFS = [
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

  var DERIVED_ANGLE_LABELS = [
    'Bacak açısı',
    'Gövde açısı',
    'Diz fleksiyon açısı',
    'İki bacak arası açı',
    'Gövde-bacak açısı',
    'Kolların yatay açısı',
    'Parabolik/uçuş açısı',
    'Kalça-gövde eksantisyonu',
    'Bacak yatay sapması',
    'Pelvis açısı',
    'Kolların yanda açısı',
    'Gövde kalça fleksiyonu',
    'Kalça ekstansiyonu',
    'Kolların geriye gidişi',
    'Gövde Sapması',
    'Kol Sapması',
    'Bacak Sapması',
    'Diz ekstansiyonu',
    'GovdeSapma',
    'R.KolSapma',
    'L.KolSapma',
    'R.BacakSapma',
    'L.BacakSapma',
    'R.KolGeriGidis',
    'L.KolGeriGidis',
    'R.KalcaFleksExt',
    'L.KalcaFleksExt',
    'R.DizEkst',
    'L.DizEkst'
  ];

  function angleKey(source, label) {
    return source + ':' + label;
  }

  function selectedAngleCount() {
    return Object.keys(selectedAngleKeys).length;
  }

  function angleDisplayActive() {
    return showAngles || (showSelectedAngles && selectedAngleCount() > 0);
  }

  function shouldDrawAngle(source, label) {
    if (showAngles) return true;
    if (!showSelectedAngles) return false;
    return Boolean(selectedAngleKeys[angleKey(source, label)]);
  }

  /* ── zoom / pan state ── */
  var zoom      = 1.0, panX = 0, panY = 0;
  var isPanning = false;
  var panStart  = {x:0, y:0}, panOrigin = {x:0, y:0};
  var MIN_ZOOM  = 0.5,  MAX_ZOOM = 10.0;

  function emitKeypointsToHiddenOutput() {
    var payload = JSON.stringify(editorStatePayload());
    var container = document.querySelector('#' + outputId);
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
      show_selected_angles: showSelectedAngles,
      selected_angle_keys: Object.keys(selectedAngleKeys),
      deleted_standard_angles: Object.keys(deletedStandardAngles),
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
    drawDerivedMetrics();
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

  function midpointPoint(a, b) {
    if (!validAngleKeypoint(a) || !validAngleKeypoint(b)) return null;
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      c: Math.min(a.c, b.c)
    };
  }

  function bodyCenter(primaryIdx, leftIdx, rightIdx) {
    if (primaryIdx < keypoints.length && validAngleKeypoint(keypoints[primaryIdx])) {
      return { point: keypoints[primaryIdx], indices: [primaryIdx] };
    }
    if (leftIdx < keypoints.length && rightIdx < keypoints.length) {
      var mid = midpointPoint(keypoints[leftIdx], keypoints[rightIdx]);
      if (mid) return { point: mid, indices: [leftIdx, rightIdx] };
    }
    return null;
  }

  function vectorAngleDeviation(vx, vy, ax, ay) {
    var mv = Math.sqrt(vx * vx + vy * vy);
    var ma = Math.sqrt(ax * ax + ay * ay);
    if (mv < 1 || ma < 1) return null;
    var cosang = Math.abs((vx * ax + vy * ay) / (mv * ma));
    return Math.round(Math.acos(Math.max(-1, Math.min(1, cosang))) * 180 / Math.PI);
  }

  function directedAngleDegrees(refx, refy, vx, vy) {
    var mr = Math.sqrt(refx * refx + refy * refy);
    var mv = Math.sqrt(vx * vx + vy * vy);
    if (mr < 1 || mv < 1) return null;
    var deg = (Math.atan2(vy, vx) - Math.atan2(refy, refx)) * 180 / Math.PI;
    while (deg < 0) deg += 360;
    while (deg >= 360) deg -= 360;
    return Math.round(deg);
  }

  function pushMetric(metrics, label, angle, x, y, points, vertex, geometry) {
    if (angle === null || !Number.isFinite(angle)) return;
    metrics.push({
      label: label,
      angle: angle,
      x: x,
      y: y,
      points: points,
      vertex: vertex || 'Referans eksen',
      geometry: geometry || null
    });
  }

  function derivedMetrics() {
    var metrics = [];
    var shoulderCenter = bodyCenter(5, 6, 7);
    var hipCenter = bodyCenter(14, 12, 13);

    if (shoulderCenter && hipCenter) {
      var sx = shoulderCenter.point.x, sy = shoulderCenter.point.y;
      var hx = hipCenter.point.x, hy = hipCenter.point.y;
      pushMetric(
        metrics,
        'GovdeSapma',
        vectorAngleDeviation(sx - hx, sy - hy, 0, -1),
        (sx + hx) / 2,
        (sy + hy) / 2,
        ['shoulder_center', 'hip_center'],
        'Dikey eksen',
        {
          kind: 'axis',
          origin: { x: hx, y: hy },
          target: { x: sx, y: sy },
          axisX: 0,
          axisY: -1
        }
      );
    }

    function addLimbDeviation(label, startIdx, preferredEndIdx, fallbackEndIdx, axisName, axisX, axisY) {
      if (startIdx >= keypoints.length) return;
      var start = keypoints[startIdx];
      var endIdx = preferredEndIdx;
      var end = preferredEndIdx < keypoints.length ? keypoints[preferredEndIdx] : null;
      if (!validAngleKeypoint(end) && fallbackEndIdx < keypoints.length) {
        endIdx = fallbackEndIdx;
        end = keypoints[fallbackEndIdx];
      }
      if (!validAngleKeypoint(start) || !validAngleKeypoint(end)) return;
      pushMetric(
        metrics,
        label,
        vectorAngleDeviation(end.x - start.x, end.y - start.y, axisX, axisY),
        (start.x + end.x) / 2,
        (start.y + end.y) / 2,
        [
          nameForKeypoint(startIdx) + ' [' + startIdx + ']',
          nameForKeypoint(endIdx) + ' [' + endIdx + ']'
        ],
        axisName,
        {
          kind: 'axis',
          origin: { x: start.x, y: start.y },
          target: { x: end.x, y: end.y },
          axisX: axisX,
          axisY: axisY
        }
      );
    }

    addLimbDeviation('R.KolSapma', 7, 11, 9, 'Yatay eksen', 1, 0);
    addLimbDeviation('L.KolSapma', 6, 10, 8, 'Yatay eksen', 1, 0);
    addLimbDeviation('R.BacakSapma', 13, 18, 16, 'Yatay eksen', 1, 0);
    addLimbDeviation('L.BacakSapma', 12, 17, 15, 'Yatay eksen', 1, 0);

    function addArmSwing(label, shoulderIdx, wristIdx, elbowIdx, hipIdx) {
      if (shoulderIdx >= keypoints.length || hipIdx >= keypoints.length) return;
      var shoulder = keypoints[shoulderIdx];
      var hip = keypoints[hipIdx];
      var endIdx = wristIdx;
      var end = wristIdx < keypoints.length ? keypoints[wristIdx] : null;
      if (!validAngleKeypoint(end) && elbowIdx < keypoints.length) {
        endIdx = elbowIdx;
        end = keypoints[elbowIdx];
      }
      if (!validAngleKeypoint(shoulder) || !validAngleKeypoint(hip) || !validAngleKeypoint(end)) return;
      pushMetric(
        metrics,
        label,
        directedAngleDegrees(hip.x - shoulder.x, hip.y - shoulder.y, end.x - shoulder.x, end.y - shoulder.y),
        (shoulder.x + end.x) / 2,
        (shoulder.y + end.y) / 2,
        [
          nameForKeypoint(shoulderIdx) + ' [' + shoulderIdx + ']',
          nameForKeypoint(endIdx) + ' [' + endIdx + ']'
        ],
        'Govde referansi',
        {
          kind: 'directed',
          origin: { x: shoulder.x, y: shoulder.y },
          reference: { x: hip.x, y: hip.y },
          target: { x: end.x, y: end.y }
        }
      );
    }

    addArmSwing('R.KolGeriGidis', 7, 11, 9, 13);
    addArmSwing('L.KolGeriGidis', 6, 10, 8, 12);

    function addAliasAngle(label, ia, ib, ic) {
      if (ia >= keypoints.length || ib >= keypoints.length || ic >= keypoints.length) return;
      var ka = keypoints[ia], kb = keypoints[ib], kc = keypoints[ic];
      if (!validAngleKeypoint(ka) || !validAngleKeypoint(kb) || !validAngleKeypoint(kc)) return;
      var ang = calcAngle(ka, kb, kc);
      if (ang === null) return;
      pushMetric(
        metrics,
        label,
        ang,
        kb.x + 16 / zoom,
        kb.y + 16 / zoom,
        [
          nameForKeypoint(ia) + ' [' + ia + ']',
          nameForKeypoint(ib) + ' [' + ib + ']',
          nameForKeypoint(ic) + ' [' + ic + ']'
        ],
        nameForKeypoint(ib) + ' [' + ib + ']',
        {
          kind: 'joint',
          ia: ia,
          ib: ib,
          ic: ic
        }
      );
    }

    function firstValidPoint(indices) {
      for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        if (idx >= keypoints.length) continue;
        var kp = keypoints[idx];
        if (validAngleKeypoint(kp)) return { point: kp, idx: idx };
      }
      return null;
    }

    function addSplitAngleMetric() {
      var leftLeg = firstValidPoint([17, 15]);
      var rightLeg = firstValidPoint([18, 16]);
      if (!leftLeg || !rightLeg || !hipCenter) return;
      var vertex = { x: hipCenter.point.x, y: hipCenter.point.y, c: 1 };
      var ang = calcAngle(leftLeg.point, vertex, rightLeg.point);
      pushMetric(
        metrics,
        'İki bacak arası açı',
        ang,
        vertex.x + 16 / zoom,
        vertex.y + 16 / zoom,
        [
          nameForKeypoint(leftLeg.idx) + ' [' + leftLeg.idx + ']',
          'hip_center',
          nameForKeypoint(rightLeg.idx) + ' [' + rightLeg.idx + ']'
        ],
        'hip_center',
        {
          kind: 'jointPoints',
          a: { x: leftLeg.point.x, y: leftLeg.point.y, c: leftLeg.point.c },
          b: vertex,
          c: { x: rightLeg.point.x, y: rightLeg.point.y, c: rightLeg.point.c }
        }
      );
    }

    function addPelvisAngleMetric() {
      var leftHip = keypoints.length > 12 ? keypoints[12] : null;
      var rightHip = keypoints.length > 13 ? keypoints[13] : null;
      if (!validAngleKeypoint(leftHip) || !validAngleKeypoint(rightHip)) return;
      pushMetric(
        metrics,
        'Pelvis açısı',
        vectorAngleDeviation(rightHip.x - leftHip.x, rightHip.y - leftHip.y, 1, 0),
        (leftHip.x + rightHip.x) / 2,
        (leftHip.y + rightHip.y) / 2,
        [
          nameForKeypoint(12) + ' [12]',
          nameForKeypoint(13) + ' [13]'
        ],
        'Yatay eksen',
        {
          kind: 'axis',
          origin: { x: leftHip.x, y: leftHip.y },
          target: { x: rightHip.x, y: rightHip.y },
          axisX: 1,
          axisY: 0
        }
      );
    }

    addAliasAngle('R.KalcaFleksExt', 7, 13, 16);
    addAliasAngle('L.KalcaFleksExt', 6, 12, 15);
    addAliasAngle('R.DizEkst', 13, 16, 18);
    addAliasAngle('L.DizEkst', 12, 15, 17);
    addSplitAngleMetric();
    addPelvisAngleMetric();

    function averageNumbers(values) {
      var clean = values.filter(function(v) { return v !== null && Number.isFinite(v); });
      if (!clean.length) return null;
      return Math.round(clean.reduce(function(sum, v) { return sum + v; }, 0) / clean.length);
    }

    function circularMeanDegrees(values) {
      var clean = values.filter(function(v) { return v !== null && Number.isFinite(v); });
      if (!clean.length) return null;
      var sx = 0, sy = 0;
      for (var i = 0; i < clean.length; i++) {
        var rad = clean[i] * Math.PI / 180;
        sx += Math.cos(rad);
        sy += Math.sin(rad);
      }
      if (Math.sqrt(sx * sx + sy * sy) < 1e-6) return averageNumbers(clean);
      var deg = Math.atan2(sy, sx) * 180 / Math.PI;
      while (deg < 0) deg += 360;
      while (deg >= 360) deg -= 360;
      return Math.round(deg);
    }

    function itemLabelPoint(items) {
      var xs = [], ys = [];
      for (var i = 0; i < items.length; i++) {
        if (items[i] && Number.isFinite(items[i].x) && Number.isFinite(items[i].y)) {
          xs.push(items[i].x);
          ys.push(items[i].y);
        }
      }
      if (!xs.length) return null;
      return {
        x: xs.reduce(function(sum, v) { return sum + v; }, 0) / xs.length,
        y: ys.reduce(function(sum, v) { return sum + v; }, 0) / ys.length
      };
    }

    function addMetricFromItems(label, items, circular) {
      var validItems = items.filter(function(item) {
        return item && item.angle !== null && Number.isFinite(item.angle);
      });
      if (!validItems.length) return;
      var angle = circular ? circularMeanDegrees(validItems.map(function(item) { return item.angle; }))
                           : averageNumbers(validItems.map(function(item) { return item.angle; }));
      var labelPoint = itemLabelPoint(validItems);
      if (!labelPoint) return;
      var points = [];
      validItems.forEach(function(item) {
        (item.points || []).forEach(function(point) {
          if (points.indexOf(point) === -1) points.push(point);
        });
      });
      pushMetric(
        metrics,
        label,
        angle,
        labelPoint.x,
        labelPoint.y,
        points,
        validItems.map(function(item) { return item.vertex; }).join(' + '),
        {
          kind: 'multi',
          items: validItems.map(function(item) { return item.geometry; }).filter(Boolean)
        }
      );
    }

    function segmentDeviationItem(startIdx, preferredEndIdx, fallbackEndIdx, axisName, axisX, axisY, transform) {
      if (startIdx >= keypoints.length) return null;
      var start = keypoints[startIdx];
      var endIdx = preferredEndIdx;
      var end = preferredEndIdx < keypoints.length ? keypoints[preferredEndIdx] : null;
      if (!validAngleKeypoint(end) && fallbackEndIdx < keypoints.length) {
        endIdx = fallbackEndIdx;
        end = keypoints[fallbackEndIdx];
      }
      if (!validAngleKeypoint(start) || !validAngleKeypoint(end)) return null;
      var raw = vectorAngleDeviation(end.x - start.x, end.y - start.y, axisX, axisY);
      if (raw === null) return null;
      return {
        angle: transform ? transform(raw) : raw,
        x: (start.x + end.x) / 2,
        y: (start.y + end.y) / 2,
        points: [
          nameForKeypoint(startIdx) + ' [' + startIdx + ']',
          nameForKeypoint(endIdx) + ' [' + endIdx + ']'
        ],
        vertex: axisName,
        geometry: {
          kind: 'axis',
          origin: { x: start.x, y: start.y },
          target: { x: end.x, y: end.y },
          axisX: axisX,
          axisY: axisY
        }
      };
    }

    function jointAngleItem(ia, ib, ic, transform) {
      if (ia >= keypoints.length || ib >= keypoints.length || ic >= keypoints.length) return null;
      var ka = keypoints[ia], kb = keypoints[ib], kc = keypoints[ic];
      if (!validAngleKeypoint(ka) || !validAngleKeypoint(kb) || !validAngleKeypoint(kc)) return null;
      var raw = calcAngle(ka, kb, kc);
      if (raw === null) return null;
      return {
        angle: transform ? transform(raw) : raw,
        x: kb.x + 16 / zoom,
        y: kb.y + 16 / zoom,
        points: [
          nameForKeypoint(ia) + ' [' + ia + ']',
          nameForKeypoint(ib) + ' [' + ib + ']',
          nameForKeypoint(ic) + ' [' + ic + ']'
        ],
        vertex: nameForKeypoint(ib) + ' [' + ib + ']',
        geometry: {
          kind: 'joint',
          ia: ia,
          ib: ib,
          ic: ic
        }
      };
    }

    function armSwingItem(shoulderIdx, wristIdx, elbowIdx, hipIdx) {
      if (shoulderIdx >= keypoints.length || hipIdx >= keypoints.length) return null;
      var shoulder = keypoints[shoulderIdx];
      var hip = keypoints[hipIdx];
      var endIdx = wristIdx;
      var end = wristIdx < keypoints.length ? keypoints[wristIdx] : null;
      if (!validAngleKeypoint(end) && elbowIdx < keypoints.length) {
        endIdx = elbowIdx;
        end = keypoints[elbowIdx];
      }
      if (!validAngleKeypoint(shoulder) || !validAngleKeypoint(hip) || !validAngleKeypoint(end)) return null;
      return {
        angle: directedAngleDegrees(hip.x - shoulder.x, hip.y - shoulder.y, end.x - shoulder.x, end.y - shoulder.y),
        x: (shoulder.x + end.x) / 2,
        y: (shoulder.y + end.y) / 2,
        points: [
          nameForKeypoint(shoulderIdx) + ' [' + shoulderIdx + ']',
          nameForKeypoint(hipIdx) + ' [' + hipIdx + ']',
          nameForKeypoint(endIdx) + ' [' + endIdx + ']'
        ],
        vertex: 'Govde referansi',
        geometry: {
          kind: 'directed',
          origin: { x: shoulder.x, y: shoulder.y },
          reference: { x: hip.x, y: hip.y },
          target: { x: end.x, y: end.y }
        }
      };
    }

    function trunkDeviationItem(transform) {
      if (!shoulderCenter || !hipCenter) return null;
      var sx = shoulderCenter.point.x, sy = shoulderCenter.point.y;
      var hx = hipCenter.point.x, hy = hipCenter.point.y;
      var raw = vectorAngleDeviation(sx - hx, sy - hy, 0, -1);
      if (raw === null) return null;
      return {
        angle: transform ? transform(raw) : raw,
        x: (sx + hx) / 2,
        y: (sy + hy) / 2,
        points: ['shoulder_center', 'hip_center'],
        vertex: 'Dikey eksen',
        geometry: {
          kind: 'axis',
          origin: { x: hx, y: hy },
          target: { x: sx, y: sy },
          axisX: 0,
          axisY: -1
        }
      };
    }

    function bodyFlightItem() {
      if (!shoulderCenter || !hipCenter) return null;
      var leftLeg = firstValidPoint([17, 15]);
      var rightLeg = firstValidPoint([18, 16]);
      if (!leftLeg || !rightLeg) return null;
      var footMid = midpointPoint(leftLeg.point, rightLeg.point);
      if (!footMid) return null;
      var target = shoulderCenter.point;
      var raw = vectorAngleDeviation(target.x - footMid.x, target.y - footMid.y, 1, 0);
      if (raw === null) return null;
      return {
        angle: raw,
        x: (target.x + footMid.x) / 2,
        y: (target.y + footMid.y) / 2,
        points: ['shoulder_center', 'leg_endpoint_center'],
        vertex: 'Yatay eksen',
        geometry: {
          kind: 'axis',
          origin: { x: footMid.x, y: footMid.y },
          target: { x: target.x, y: target.y },
          axisX: 1,
          axisY: 0
        }
      };
    }

    addMetricFromItems('Bacak açısı', [
      segmentDeviationItem(13, 18, 16, 'Dikey eksen', 0, 1, function(raw) { return 180 - raw; }),
      segmentDeviationItem(12, 17, 15, 'Dikey eksen', 0, 1, function(raw) { return 180 - raw; })
    ]);
    addMetricFromItems('Gövde açısı', [trunkDeviationItem(function(raw) { return 180 - raw; })]);
    addMetricFromItems('Diz fleksiyon açısı', [
      jointAngleItem(13, 16, 18, function(raw) { return 180 - raw; }),
      jointAngleItem(12, 15, 17, function(raw) { return 180 - raw; })
    ]);
    addMetricFromItems('Gövde-bacak açısı', [
      jointAngleItem(7, 13, 16),
      jointAngleItem(6, 12, 15)
    ]);
    addMetricFromItems('Kolların yatay açısı', [
      segmentDeviationItem(7, 11, 9, 'Yatay eksen', 1, 0, function(raw) { return 180 - raw; }),
      segmentDeviationItem(6, 10, 8, 'Yatay eksen', 1, 0, function(raw) { return 180 - raw; })
    ]);
    addMetricFromItems('Parabolik/uçuş açısı', [bodyFlightItem()]);
    addMetricFromItems('Kalça-gövde eksantisyonu', [
      jointAngleItem(7, 13, 16, function(raw) { return Math.max(0, raw - 90); }),
      jointAngleItem(6, 12, 15, function(raw) { return Math.max(0, raw - 90); })
    ]);
    addMetricFromItems('Bacak yatay sapması', [
      segmentDeviationItem(13, 18, 16, 'Yatay eksen', 1, 0),
      segmentDeviationItem(12, 17, 15, 'Yatay eksen', 1, 0)
    ]);
    addMetricFromItems('Kolların yanda açısı', [
      jointAngleItem(9, 7, 13),
      jointAngleItem(8, 6, 12)
    ]);
    addMetricFromItems('Gövde kalça fleksiyonu', [
      jointAngleItem(7, 13, 16, function(raw) { return 180 - raw; }),
      jointAngleItem(6, 12, 15, function(raw) { return 180 - raw; })
    ]);
    addMetricFromItems('Kalça ekstansiyonu', [
      jointAngleItem(7, 13, 16),
      jointAngleItem(6, 12, 15)
    ]);
    addMetricFromItems('Kolların geriye gidişi', [
      armSwingItem(7, 11, 9, 13),
      armSwingItem(6, 10, 8, 12)
    ], true);
    addMetricFromItems('Gövde Sapması', [trunkDeviationItem()]);
    addMetricFromItems('Kol Sapması', [
      segmentDeviationItem(7, 11, 9, 'Yatay eksen', 1, 0),
      segmentDeviationItem(6, 10, 8, 'Yatay eksen', 1, 0)
    ]);
    addMetricFromItems('Bacak Sapması', [
      segmentDeviationItem(13, 18, 16, 'Dikey eksen', 0, 1),
      segmentDeviationItem(12, 17, 15, 'Dikey eksen', 0, 1)
    ]);
    addMetricFromItems('Diz ekstansiyonu', [
      jointAngleItem(13, 16, 18),
      jointAngleItem(12, 15, 17)
    ]);

    return metrics;
  }

  function drawDerivedMetrics() {
    if (!angleDisplayActive()) return;
    var metrics = derivedMetrics();
    ctx.save();
    ctx.font = 'bold ' + (11 / zoom) + 'px sans-serif';
    for (var i = 0; i < metrics.length; i++) {
      var m = metrics[i];
      if (!shouldDrawAngle('derived', m.label)) continue;
      drawDerivedMetricGeometry(m);
      var text = m.label + ': ' + m.angle + '\u00b0';
      var tx = m.x + 8 / zoom;
      var ty = m.y - 8 / zoom;
      ctx.lineWidth = 3 / zoom;
      ctx.strokeStyle = 'rgba(0,0,0,0.82)';
      ctx.strokeText(text, tx, ty);
      ctx.fillStyle = '#ffb000';
      ctx.fillText(text, tx, ty);
      angleHitboxes.push({
        x: tx,
        y: ty,
        r: Math.max(28 / zoom, ctx.measureText(text).width / 2),
        label: m.label,
        angle: m.angle,
        source: 'derived',
        standard_label: null,
        custom_index: null,
        points: m.points,
        vertex: m.vertex
      });
    }
    ctx.restore();
  }

  function drawDerivedMetricGeometry(metric) {
    if (!metric || !metric.geometry) return;
    var g = metric.geometry;
    if (g.kind === 'joint') {
      drawDerivedJointGeometry(g.ia, g.ib, g.ic);
    } else if (g.kind === 'jointPoints') {
      drawDerivedJointPointGeometry(g.a, g.b, g.c);
    } else if (g.kind === 'multi') {
      (g.items || []).forEach(function(item) {
        drawDerivedMetricGeometry({ geometry: item });
      });
    } else if (g.kind === 'axis') {
      drawDerivedAxisGeometry(g.origin, g.target, g.axisX, g.axisY);
    } else if (g.kind === 'directed') {
      drawDerivedDirectedGeometry(g.origin, g.reference, g.target, metric.angle);
    }
  }

  function drawDerivedJointPointGeometry(ka, kb, kc) {
    if (!ka || !kb || !kc) return;
    var dA = Math.sqrt((ka.x-kb.x)*(ka.x-kb.x) + (ka.y-kb.y)*(ka.y-kb.y));
    var dC = Math.sqrt((kc.x-kb.x)*(kc.x-kb.x) + (kc.y-kb.y)*(kc.y-kb.y));
    if (dA < 1 || dC < 1) return;
    var arcR = Math.max(12 / zoom, Math.min(30 / zoom, Math.min(dA, dC) * 0.35));
    var angA = Math.atan2(ka.y - kb.y, ka.x - kb.x);
    var angC = Math.atan2(kc.y - kb.y, kc.x - kb.x);
    var diff = angC - angA;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(kb.x, kb.y, arcR, angA, angC, diff < 0);
    ctx.strokeStyle = 'rgba(255,176,0,0.96)';
    ctx.lineWidth = 2.5 / zoom;
    ctx.stroke();

    ctx.setLineDash([4 / zoom, 3 / zoom]);
    ctx.beginPath();
    ctx.moveTo(kb.x, kb.y);
    ctx.lineTo(kb.x + arcR * Math.cos(angA), kb.y + arcR * Math.sin(angA));
    ctx.moveTo(kb.x, kb.y);
    ctx.lineTo(kb.x + arcR * Math.cos(angC), kb.y + arcR * Math.sin(angC));
    ctx.strokeStyle = 'rgba(255,176,0,0.65)';
    ctx.lineWidth = 1.5 / zoom;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawDerivedJointGeometry(ia, ib, ic) {
    if (ia >= keypoints.length || ib >= keypoints.length || ic >= keypoints.length) return;
    var ka = keypoints[ia], kb = keypoints[ib], kc = keypoints[ic];
    if (!validAngleKeypoint(ka) || !validAngleKeypoint(kb) || !validAngleKeypoint(kc)) return;
    var dA = Math.sqrt((ka.x-kb.x)*(ka.x-kb.x) + (ka.y-kb.y)*(ka.y-kb.y));
    var dC = Math.sqrt((kc.x-kb.x)*(kc.x-kb.x) + (kc.y-kb.y)*(kc.y-kb.y));
    if (dA < 1 || dC < 1) return;
    var arcR = Math.max(12 / zoom, Math.min(30 / zoom, Math.min(dA, dC) * 0.35));
    var angA = Math.atan2(ka.y - kb.y, ka.x - kb.x);
    var angC = Math.atan2(kc.y - kb.y, kc.x - kb.x);
    var diff = angC - angA;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(kb.x, kb.y, arcR, angA, angC, diff < 0);
    ctx.strokeStyle = 'rgba(255,176,0,0.96)';
    ctx.lineWidth = 2.5 / zoom;
    ctx.stroke();

    ctx.setLineDash([4 / zoom, 3 / zoom]);
    ctx.beginPath();
    ctx.moveTo(kb.x, kb.y);
    ctx.lineTo(kb.x + arcR * Math.cos(angA), kb.y + arcR * Math.sin(angA));
    ctx.moveTo(kb.x, kb.y);
    ctx.lineTo(kb.x + arcR * Math.cos(angC), kb.y + arcR * Math.sin(angC));
    ctx.strokeStyle = 'rgba(255,176,0,0.65)';
    ctx.lineWidth = 1.5 / zoom;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawDerivedAxisGeometry(origin, target, axisX, axisY) {
    if (!origin || !target) return;
    var vx = target.x - origin.x;
    var vy = target.y - origin.y;
    var len = Math.sqrt(vx * vx + vy * vy);
    var axisLen = Math.sqrt(axisX * axisX + axisY * axisY);
    if (len < 1 || axisLen < 1) return;
    var ax = axisX / axisLen;
    var ay = axisY / axisLen;
    if ((vx * ax + vy * ay) < 0) {
      ax = -ax;
      ay = -ay;
    }
    var refLen = Math.max(26 / zoom, Math.min(58 / zoom, len * 0.55));
    var ref = { x: origin.x + ax * refLen, y: origin.y + ay * refLen };

    drawDerivedSegment(origin, target, 'rgba(255,176,0,0.92)', false);
    drawDerivedSegment(origin, ref, 'rgba(255,176,0,0.62)', true);
    drawDerivedArc(origin, ref, target, Math.min(refLen, len) * 0.55, false);
  }

  function drawDerivedDirectedGeometry(origin, reference, target, angle) {
    if (!origin || !reference || !target) return;
    var dR = Math.sqrt((reference.x-origin.x)*(reference.x-origin.x) + (reference.y-origin.y)*(reference.y-origin.y));
    var dT = Math.sqrt((target.x-origin.x)*(target.x-origin.x) + (target.y-origin.y)*(target.y-origin.y));
    if (dR < 1 || dT < 1) return;
    drawDerivedSegment(origin, reference, 'rgba(255,176,0,0.58)', true);
    drawDerivedSegment(origin, target, 'rgba(255,176,0,0.96)', false);
    drawDerivedArc(origin, reference, target, Math.min(dR, dT) * 0.35, angle > 180);
  }

  function drawDerivedSegment(a, b, color, dashed) {
    ctx.setLineDash(dashed ? [5 / zoom, 4 / zoom] : []);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = (dashed ? 1.6 : 2.6) / zoom;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawDerivedArc(origin, refPoint, targetPoint, radius, clockwiseLong) {
    var arcR = Math.max(12 / zoom, Math.min(34 / zoom, radius || 24 / zoom));
    var angA = Math.atan2(refPoint.y - origin.y, refPoint.x - origin.x);
    var angB = Math.atan2(targetPoint.y - origin.y, targetPoint.x - origin.x);
    var diff = angB - angA;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    var anticlockwise = diff < 0;
    if (clockwiseLong) anticlockwise = true;

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, arcR, angA, angB, anticlockwise);
    ctx.strokeStyle = 'rgba(255,176,0,0.96)';
    ctx.lineWidth = 2.4 / zoom;
    ctx.stroke();

    ctx.setLineDash([4 / zoom, 3 / zoom]);
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(origin.x + arcR * Math.cos(angA), origin.y + arcR * Math.sin(angA));
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(origin.x + arcR * Math.cos(angB), origin.y + arcR * Math.sin(angB));
    ctx.strokeStyle = 'rgba(255,176,0,0.55)';
    ctx.lineWidth = 1.4 / zoom;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawAngles() {
    if (!angleDisplayActive()) return;
    /* [idxA, idxB(vertex), idxC, label]  — standard COCO-25 indices */
    var ANG = STANDARD_ANGLE_DEFS;
    ctx.save();
    for (var ai = 0; ai < ANG.length; ai++) {
      var ia = ANG[ai][0], ib = ANG[ai][1], ic = ANG[ai][2];
      if (deletedStandardAngles[ANG[ai][3]]) continue;
      if (!shouldDrawAngle('standard', ANG[ai][3])) continue;
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
        source: 'standard',
        standard_label: ANG[ai][3],
        custom_index: null,
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
      drawCustomAngle(def[0], def[1], def[2], 'Ozel Aci ' + (ai + 1), ai);
    }
    ctx.restore();
  }

  function drawCustomAngle(ia, ib, ic, label, customIndex) {
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
      source: 'manual',
      standard_label: null,
      custom_index: customIndex,
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
    if (!angleDisplayActive() && !customAngles.length) return null;
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

  function angleChoiceButton(source, label) {
    var key = angleKey(source, label);
    var selected = selectedAngleKeys[key] ? ' is-selected' : '';
    return '<button type="button" class="pe-angle-choice' + selected + '" data-angle-key="' +
      key + '">' + label + '</button>';
  }

  function renderAnglePicker() {
    if (!anglePicker) return;
    anglePicker.style.display = showSelectedAngles ? 'flex' : 'none';
    if (!showSelectedAngles) return;
    var standardButtons = STANDARD_ANGLE_DEFS.map(function(def) {
      return angleChoiceButton('standard', def[3]);
    }).join('');
    var derivedButtons = DERIVED_ANGLE_LABELS.map(function(label) {
      return angleChoiceButton('derived', label);
    }).join('');
    anglePicker.innerHTML =
      '<div class="pe-angle-group"><span class="pe-angle-group-title">Standart</span>' +
      standardButtons +
      '</div><div class="pe-angle-group"><span class="pe-angle-group-title">Teknik</span>' +
      derivedButtons +
      '</div>';
  }

  function setAngleSelectMode(active) {
    showSelectedAngles = active;
    if (showSelectedAngles) {
      showAngles = false;
      customAngleMode = false;
      deleteAngleMode = false;
      customAnglePick = [];
    }
    var btn = element.querySelector('[data-action="angle-select"]');
    if (btn) btn.classList.toggle('is-active', showSelectedAngles);
    var customBtn = element.querySelector('[data-action="custom-angle"]');
    if (customBtn) customBtn.classList.toggle('is-active', customAngleMode);
    var deleteBtn = element.querySelector('[data-action="delete-angle"]');
    if (deleteBtn) deleteBtn.classList.toggle('is-active', deleteAngleMode);
    renderAnglePicker();
    render();
    emitKeypointsToHiddenOutput();
    if (info) {
      info.textContent = showSelectedAngles
        ? 'Aci secimi acik: listeden acilari secin.'
        : 'Aci secimi kapatildi.';
    }
  }

  if (anglePicker) {
    anglePicker.onclick = function(e) {
      var btn = e.target && e.target.closest ? e.target.closest('.pe-angle-choice') : null;
      if (!btn) return;
      var key = btn.getAttribute('data-angle-key');
      if (!key) return;
      if (selectedAngleKeys[key]) delete selectedAngleKeys[key];
      else selectedAngleKeys[key] = true;
      showSelectedAngles = true;
      showAngles = false;
      renderAnglePicker();
      render();
      emitKeypointsToHiddenOutput();
      if (info) info.textContent = selectedAngleCount() + ' aci secildi.';
    };
  }
  renderAnglePicker();

  function setCustomAngleMode(active) {
    customAngleMode = active;
    if (customAngleMode) setAngleSelectMode(false);
    if (customAngleMode) setDeleteAngleMode(false);
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

  function setDeleteAngleMode(active) {
    deleteAngleMode = active;
    if (deleteAngleMode) {
      customAngleMode = false;
      customAnglePick = [];
      showAngles = true;
      showSelectedAngles = false;
      renderAnglePicker();
    }
    var deleteBtn = element.querySelector('[data-action="delete-angle"]');
    if (deleteBtn) deleteBtn.classList.toggle('is-active', deleteAngleMode);
    var customBtn = element.querySelector('[data-action="custom-angle"]');
    if (customBtn) customBtn.classList.toggle('is-active', customAngleMode);
    var selectBtn = element.querySelector('[data-action="angle-select"]');
    if (selectBtn) selectBtn.classList.toggle('is-active', showSelectedAngles);
    canvas.style.cursor = deleteAngleMode ? 'not-allowed' : (customAngleMode ? 'copy' : 'crosshair');
    render();
    if (info) {
      info.textContent = deleteAngleMode
        ? 'Aci silme: silmek istediginiz acinin yazisina veya yayina tiklayin.'
        : 'Aci silme kapatildi.';
    }
  }

  function deleteHoveredAngle(hit) {
    if (!hit) {
      if (info) info.textContent = 'Silmek icin once bir acinin uzerine tiklayin.';
      return;
    }
    if (hit.source === 'manual' && hit.custom_index !== null) {
      customAngles.splice(hit.custom_index, 1);
    } else if (hit.source === 'standard' && hit.standard_label) {
      deletedStandardAngles[hit.standard_label] = true;
    } else {
      if (info) info.textContent = hit.label + ' turetilmis metriktir; silinemez.';
      return;
    }
    hoveredAngle = null;
    render();
    emitKeypointsToHiddenOutput();
    if (info) info.textContent = hit.label + ' silindi. Kaydetmek icin Apply & Save kullanin.';
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
    if (deleteAngleMode && e.button === 0) {
      deleteHoveredAngle(findHoveredAngle(getPos(e)));
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
      canvas.style.cursor = deleteAngleMode
        ? (hoveredAngle ? 'not-allowed' : 'crosshair')
        : (nearest(p) !== null ? (customAngleMode ? 'copy' : 'grab') : (hoveredAngle ? 'help' : (zoom > 1 ? 'zoom-in' : 'crosshair')));
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
    if (deleteAngleMode) {
      deleteHoveredAngle(findHoveredAngle(getPos(e)));
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
    peWrap.dataset.editorRole = editorRole;
    peWrap._peGetKps = function() {
      return JSON.stringify(editorStatePayload());
    };
  }

  rebind('[data-action="reset"]', function() {
    keypoints = JSON.parse(JSON.stringify(origKps));
    zoom = 1.0; panX = 0; panY = 0;
    customAnglePick = [];
    customAngles = [];
    deletedStandardAngles = {};
    selectedAngleKeys = {};
    showSelectedAngles = false;
    deleteAngleMode = false;
    renderAnglePicker();
    var deleteBtn = element.querySelector('[data-action="delete-angle"]');
    if (deleteBtn) deleteBtn.classList.remove('is-active');
    var selectBtn = element.querySelector('[data-action="angle-select"]');
    if (selectBtn) selectBtn.classList.remove('is-active');
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
    if (showAngles) showSelectedAngles = false;
    renderAnglePicker();
    var selectBtn = element.querySelector('[data-action="angle-select"]');
    if (selectBtn) selectBtn.classList.toggle('is-active', showSelectedAngles);
    render();
    if (info) info.textContent = showAngles ? 'Acilar gosteriliyor.' : 'Acilar gizlendi.';
  });

  rebind('[data-action="angle-select"]', function() {
    setAngleSelectMode(!showSelectedAngles);
  });
  var initialSelectBtn = element.querySelector('[data-action="angle-select"]');
  if (initialSelectBtn) initialSelectBtn.classList.toggle('is-active', showSelectedAngles);

  rebind('[data-action="custom-angle"]', function() {
    setCustomAngleMode(!customAngleMode);
  });

  rebind('[data-action="delete-angle"]', function() {
    setDeleteAngleMode(!deleteAngleMode);
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
    emitNavTrigger('#' + prevTriggerId);
    if (info) info.textContent = 'Onceki goruntüye geçiliyor...';
  });

  rebind('[data-action="next"]', function() {
    emitNavTrigger('#' + nextTriggerId);
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
        "deleted_standard_angles": data.get("deleted_standard_angles", []),
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
    editor_role: str = "main",
    output_id: str = "kp_editor_output",
    prev_trigger_id: str = "pe_prev_trigger",
    next_trigger_id: str = "pe_next_trigger",
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
        "deleted_standard_angles": data.get("deleted_standard_angles", []),
        "editor_role": editor_role,
        "output_id": output_id,
        "prev_trigger_id": prev_trigger_id,
        "next_trigger_id": next_trigger_id,
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


def _point_xy_from_rc(kps_rc: list, idx: int) -> Optional[Tuple[float, float, list]]:
    if idx >= len(kps_rc) or not _valid_angle_keypoint_rc(kps_rc[idx]):
        return None
    kp = kps_rc[idx]
    return float(kp[1]), float(kp[0]), [idx]


def _midpoint_xy_from_rc(kps_rc: list, ia: int, ib: int) -> Optional[Tuple[float, float, list]]:
    pa = _point_xy_from_rc(kps_rc, ia)
    pb = _point_xy_from_rc(kps_rc, ib)
    if pa is None or pb is None:
        return None
    return (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, [ia, ib]


def _body_center_xy_from_rc(
    kps_rc: list,
    primary_idx: int,
    left_idx: int,
    right_idx: int,
) -> Optional[Tuple[float, float, list]]:
    primary = _point_xy_from_rc(kps_rc, primary_idx)
    if primary is not None:
        return primary
    return _midpoint_xy_from_rc(kps_rc, left_idx, right_idx)


def _vector_axis_deviation_degrees(vx: float, vy: float, ax: float, ay: float) -> Optional[float]:
    mag_v = float(np.hypot(vx, vy))
    mag_a = float(np.hypot(ax, ay))
    if mag_v < 1 or mag_a < 1:
        return None
    cosang = abs((vx * ax + vy * ay) / (mag_v * mag_a))
    return round(float(np.degrees(np.arccos(max(-1.0, min(1.0, cosang))))), 3)


def _directed_angle_degrees(refx: float, refy: float, vx: float, vy: float) -> Optional[float]:
    if float(np.hypot(refx, refy)) < 1 or float(np.hypot(vx, vy)) < 1:
        return None
    angle = float(np.degrees(np.arctan2(vy, vx) - np.arctan2(refy, refx)))
    while angle < 0:
        angle += 360.0
    while angle >= 360.0:
        angle -= 360.0
    return round(angle, 3)


def _calc_angle_degrees_between_xy(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Optional[float]:
    bax = float(a[0]) - float(b[0])
    bay = float(a[1]) - float(b[1])
    bcx = float(c[0]) - float(b[0])
    bcy = float(c[1]) - float(b[1])
    ma = float(np.hypot(bax, bay))
    mc = float(np.hypot(bcx, bcy))
    if ma < 1 or mc < 1:
        return None
    cosang = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (ma * mc)))
    return round(float(np.degrees(np.arccos(cosang))), 3)


def _metric_names(indices: list, idx_to_name: Dict[int, str]) -> list:
    return [idx_to_name.get(i, str(i)) for i in indices]


def _derived_metric_record(
    label: str,
    angle_degrees: Optional[float],
    keypoint_indices: list,
    idx_to_name: Dict[int, str],
    metric_type: str,
    reference: str,
    method: str,
) -> Optional[Dict[str, Any]]:
    if angle_degrees is None:
        return None
    unique_indices = list(dict.fromkeys(int(i) for i in keypoint_indices))
    return {
        "label": label,
        "keypoint_indices": unique_indices,
        "keypoint_names": _metric_names(unique_indices, idx_to_name),
        "angle_degrees": angle_degrees,
        "source": "derived_pose_metric",
        "metric_type": metric_type,
        "reference": reference,
        "method": method,
    }


def _add_segment_deviation_metric(
    records: list,
    kps_rc: list,
    idx_to_name: Dict[int, str],
    label: str,
    start_idx: int,
    preferred_end_idx: int,
    fallback_end_idx: int,
    axis_name: str,
    axis_x: float,
    axis_y: float,
) -> None:
    start = _point_xy_from_rc(kps_rc, start_idx)
    end = _point_xy_from_rc(kps_rc, preferred_end_idx)
    end_idx = preferred_end_idx
    if end is None:
        end = _point_xy_from_rc(kps_rc, fallback_end_idx)
        end_idx = fallback_end_idx
    if start is None or end is None:
        return
    angle = _vector_axis_deviation_degrees(end[0] - start[0], end[1] - start[1], axis_x, axis_y)
    record = _derived_metric_record(
        label,
        angle,
        [start_idx, end_idx],
        idx_to_name,
        "axis_deviation",
        axis_name,
        "2D segment deviation from the named image-plane axis; 0 degrees means aligned with the axis",
    )
    if record:
        records.append(record)


def _add_arm_swing_metric(
    records: list,
    kps_rc: list,
    idx_to_name: Dict[int, str],
    label: str,
    shoulder_idx: int,
    wrist_idx: int,
    elbow_idx: int,
    hip_idx: int,
) -> None:
    shoulder = _point_xy_from_rc(kps_rc, shoulder_idx)
    hip = _point_xy_from_rc(kps_rc, hip_idx)
    end = _point_xy_from_rc(kps_rc, wrist_idx)
    end_idx = wrist_idx
    if end is None:
        end = _point_xy_from_rc(kps_rc, elbow_idx)
        end_idx = elbow_idx
    if shoulder is None or hip is None or end is None:
        return
    angle = _directed_angle_degrees(
        hip[0] - shoulder[0],
        hip[1] - shoulder[1],
        end[0] - shoulder[0],
        end[1] - shoulder[1],
    )
    record = _derived_metric_record(
        label,
        angle,
        [shoulder_idx, hip_idx, end_idx],
        idx_to_name,
        "directed_segment_angle",
        "trunk line shoulder->hip",
        "Clockwise 2D directed angle from shoulder->hip trunk reference to shoulder->wrist arm segment; elbow is used if wrist is unavailable",
    )
    if record:
        records.append(record)


def _add_joint_alias_metric(
    records: list,
    kps_rc: list,
    idx_to_name: Dict[int, str],
    label: str,
    ia: int,
    ib: int,
    ic: int,
    reference: str,
) -> None:
    angle = _calc_angle_degrees_from_rc(kps_rc, ia, ib, ic)
    record = _derived_metric_record(
        label,
        angle,
        [ia, ib, ic],
        idx_to_name,
        "joint_angle_alias",
        reference,
        "2D angle between vectors B->A and B->C; keypoint order is [A, B(vertex), C]",
    )
    if record:
        record["vertex_index"] = ib
        record["vertex_name"] = idx_to_name.get(ib, str(ib))
        records.append(record)


def _average_degrees(values: list) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 3)


def _circular_mean_degrees(values: list) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    sx = sum(float(np.cos(np.radians(v))) for v in clean)
    sy = sum(float(np.sin(np.radians(v))) for v in clean)
    if float(np.hypot(sx, sy)) < 1e-6:
        return _average_degrees(clean)
    angle = float(np.degrees(np.arctan2(sy, sx)))
    while angle < 0:
        angle += 360.0
    while angle >= 360.0:
        angle -= 360.0
    return round(angle, 3)


def _merge_metric_indices(items: list) -> list:
    indices = []
    for item in items:
        for idx in item.get("keypoint_indices", []):
            if idx not in indices:
                indices.append(idx)
    return indices


def _add_composite_metric_record(
    records: list,
    idx_to_name: Dict[int, str],
    label: str,
    items: list,
    metric_type: str,
    reference: str,
    method: str,
    circular: bool = False,
) -> None:
    valid_items = [item for item in items if item and item.get("angle_degrees") is not None]
    if not valid_items:
        return
    values = [item["angle_degrees"] for item in valid_items]
    angle = _circular_mean_degrees(values) if circular else _average_degrees(values)
    record = _derived_metric_record(
        label,
        angle,
        _merge_metric_indices(valid_items),
        idx_to_name,
        metric_type,
        reference,
        method,
    )
    if record:
        record["component_metrics"] = valid_items
        records.append(record)


def _segment_deviation_item(
    kps_rc: list,
    idx_to_name: Dict[int, str],
    start_idx: int,
    preferred_end_idx: int,
    fallback_end_idx: int,
    axis_name: str,
    axis_x: float,
    axis_y: float,
    transform=None,
) -> Optional[Dict[str, Any]]:
    start = _point_xy_from_rc(kps_rc, start_idx)
    end = _point_xy_from_rc(kps_rc, preferred_end_idx)
    end_idx = preferred_end_idx
    if end is None:
        end = _point_xy_from_rc(kps_rc, fallback_end_idx)
        end_idx = fallback_end_idx
    if start is None or end is None:
        return None
    raw = _vector_axis_deviation_degrees(end[0] - start[0], end[1] - start[1], axis_x, axis_y)
    if raw is None:
        return None
    angle = transform(raw) if transform else raw
    return {
        "angle_degrees": round(float(angle), 3),
        "keypoint_indices": [start_idx, end_idx],
        "keypoint_names": _metric_names([start_idx, end_idx], idx_to_name),
        "reference": axis_name,
    }


def _joint_angle_item(
    kps_rc: list,
    idx_to_name: Dict[int, str],
    ia: int,
    ib: int,
    ic: int,
    transform=None,
) -> Optional[Dict[str, Any]]:
    raw = _calc_angle_degrees_from_rc(kps_rc, ia, ib, ic)
    if raw is None:
        return None
    angle = transform(raw) if transform else raw
    return {
        "angle_degrees": round(float(angle), 3),
        "keypoint_indices": [ia, ib, ic],
        "keypoint_names": _metric_names([ia, ib, ic], idx_to_name),
        "vertex_index": ib,
        "vertex_name": idx_to_name.get(ib, str(ib)),
    }


def _arm_swing_item(
    kps_rc: list,
    idx_to_name: Dict[int, str],
    shoulder_idx: int,
    wrist_idx: int,
    elbow_idx: int,
    hip_idx: int,
) -> Optional[Dict[str, Any]]:
    shoulder = _point_xy_from_rc(kps_rc, shoulder_idx)
    hip = _point_xy_from_rc(kps_rc, hip_idx)
    end = _point_xy_from_rc(kps_rc, wrist_idx)
    end_idx = wrist_idx
    if end is None:
        end = _point_xy_from_rc(kps_rc, elbow_idx)
        end_idx = elbow_idx
    if shoulder is None or hip is None or end is None:
        return None
    angle = _directed_angle_degrees(
        hip[0] - shoulder[0],
        hip[1] - shoulder[1],
        end[0] - shoulder[0],
        end[1] - shoulder[1],
    )
    if angle is None:
        return None
    indices = [shoulder_idx, hip_idx, end_idx]
    return {
        "angle_degrees": angle,
        "keypoint_indices": indices,
        "keypoint_names": _metric_names(indices, idx_to_name),
        "reference": "trunk line shoulder->hip",
    }


def _trunk_deviation_item(kps_rc: list, idx_to_name: Dict[int, str], transform=None) -> Optional[Dict[str, Any]]:
    shoulder_center = _body_center_xy_from_rc(kps_rc, 5, 6, 7)
    hip_center = _body_center_xy_from_rc(kps_rc, 14, 12, 13)
    if shoulder_center is None or hip_center is None:
        return None
    raw = _vector_axis_deviation_degrees(
        shoulder_center[0] - hip_center[0],
        shoulder_center[1] - hip_center[1],
        0.0,
        -1.0,
    )
    if raw is None:
        return None
    angle = transform(raw) if transform else raw
    indices = shoulder_center[2] + hip_center[2]
    return {
        "angle_degrees": round(float(angle), 3),
        "keypoint_indices": indices,
        "keypoint_names": _metric_names(indices, idx_to_name),
        "reference": "vertical_axis",
    }


def _body_flight_item(kps_rc: list, idx_to_name: Dict[int, str]) -> Optional[Dict[str, Any]]:
    shoulder_center = _body_center_xy_from_rc(kps_rc, 5, 6, 7)
    left_leg = _first_valid_xy_from_rc(kps_rc, [17, 15])
    right_leg = _first_valid_xy_from_rc(kps_rc, [18, 16])
    if shoulder_center is None or left_leg is None or right_leg is None:
        return None
    foot_mid = ((left_leg[0] + right_leg[0]) / 2, (left_leg[1] + right_leg[1]) / 2)
    angle = _vector_axis_deviation_degrees(shoulder_center[0] - foot_mid[0], shoulder_center[1] - foot_mid[1], 1.0, 0.0)
    if angle is None:
        return None
    indices = shoulder_center[2] + left_leg[2] + right_leg[2]
    return {
        "angle_degrees": angle,
        "keypoint_indices": indices,
        "keypoint_names": _metric_names(indices, idx_to_name),
        "reference": "horizontal_axis",
    }


def _first_valid_xy_from_rc(kps_rc: list, indices: list) -> Optional[Tuple[float, float, list]]:
    for idx in indices:
        point = _point_xy_from_rc(kps_rc, idx)
        if point is not None:
            return point
    return None


def _add_split_angle_metric(records: list, kps_rc: list, idx_to_name: Dict[int, str]) -> None:
    left_leg = _first_valid_xy_from_rc(kps_rc, [17, 15])
    right_leg = _first_valid_xy_from_rc(kps_rc, [18, 16])
    hip_center = _body_center_xy_from_rc(kps_rc, 14, 12, 13)
    if left_leg is None or right_leg is None or hip_center is None:
        return
    angle = _calc_angle_degrees_between_xy(
        (left_leg[0], left_leg[1]),
        (hip_center[0], hip_center[1]),
        (right_leg[0], right_leg[1]),
    )
    record = _derived_metric_record(
        "İki bacak arası açı",
        angle,
        left_leg[2] + hip_center[2] + right_leg[2],
        idx_to_name,
        "synthetic_joint_angle",
        "hip_center",
        "2D angle between left and right leg endpoints at the hip center; ankles are preferred and knees are used if ankles are unavailable",
    )
    if record:
        record["vertex_name"] = "hip_center"
        records.append(record)


def _add_pelvis_angle_metric(records: list, kps_rc: list, idx_to_name: Dict[int, str]) -> None:
    left_hip = _point_xy_from_rc(kps_rc, 12)
    right_hip = _point_xy_from_rc(kps_rc, 13)
    if left_hip is None or right_hip is None:
        return
    angle = _vector_axis_deviation_degrees(right_hip[0] - left_hip[0], right_hip[1] - left_hip[1], 1.0, 0.0)
    record = _derived_metric_record(
        "Pelvis açısı",
        angle,
        [12, 13],
        idx_to_name,
        "axis_deviation",
        "horizontal_axis",
        "2D pelvis-line deviation from the horizontal axis using left and right hip keypoints",
    )
    if record:
        records.append(record)


def _build_derived_metric_records(kps_rc: list, idx_to_name: Dict[int, str]) -> list:
    records = []

    shoulder_center = _body_center_xy_from_rc(kps_rc, 5, 6, 7)
    hip_center = _body_center_xy_from_rc(kps_rc, 14, 12, 13)
    if shoulder_center is not None and hip_center is not None:
        angle = _vector_axis_deviation_degrees(
            shoulder_center[0] - hip_center[0],
            shoulder_center[1] - hip_center[1],
            0.0,
            -1.0,
        )
        record = _derived_metric_record(
            "GovdeSapma",
            angle,
            shoulder_center[2] + hip_center[2],
            idx_to_name,
            "axis_deviation",
            "vertical_axis",
            "2D trunk-line deviation from vertical; neck is preferred for shoulder center and hip midpoint is used if hip center is unavailable",
        )
        if record:
            records.append(record)

    _add_segment_deviation_metric(records, kps_rc, idx_to_name, "R.KolSapma", 7, 11, 9, "horizontal_axis", 1.0, 0.0)
    _add_segment_deviation_metric(records, kps_rc, idx_to_name, "L.KolSapma", 6, 10, 8, "horizontal_axis", 1.0, 0.0)
    _add_segment_deviation_metric(records, kps_rc, idx_to_name, "R.BacakSapma", 13, 18, 16, "horizontal_axis", 1.0, 0.0)
    _add_segment_deviation_metric(records, kps_rc, idx_to_name, "L.BacakSapma", 12, 17, 15, "horizontal_axis", 1.0, 0.0)

    _add_arm_swing_metric(records, kps_rc, idx_to_name, "R.KolGeriGidis", 7, 11, 9, 13)
    _add_arm_swing_metric(records, kps_rc, idx_to_name, "L.KolGeriGidis", 6, 10, 8, 12)

    _add_joint_alias_metric(records, kps_rc, idx_to_name, "R.KalcaFleksExt", 7, 13, 16, "right hip flexion/extension")
    _add_joint_alias_metric(records, kps_rc, idx_to_name, "L.KalcaFleksExt", 6, 12, 15, "left hip flexion/extension")
    _add_joint_alias_metric(records, kps_rc, idx_to_name, "R.DizEkst", 13, 16, 18, "right knee extension")
    _add_joint_alias_metric(records, kps_rc, idx_to_name, "L.DizEkst", 12, 15, 17, "left knee extension")
    _add_split_angle_metric(records, kps_rc, idx_to_name)
    _add_pelvis_angle_metric(records, kps_rc, idx_to_name)

    _add_composite_metric_record(
        records,
        idx_to_name,
        "Bacak açısı",
        [
            _segment_deviation_item(kps_rc, idx_to_name, 13, 18, 16, "vertical_axis", 0.0, 1.0, lambda raw: 180.0 - raw),
            _segment_deviation_item(kps_rc, idx_to_name, 12, 17, 15, "vertical_axis", 0.0, 1.0, lambda raw: 180.0 - raw),
        ],
        "bilateral_segment_angle",
        "vertical_axis",
        "Average right/left leg segment anatomical angle from the vertical axis; ankles are preferred and knees are used if ankles are unavailable",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Gövde açısı",
        [_trunk_deviation_item(kps_rc, idx_to_name, lambda raw: 180.0 - raw)],
        "trunk_angle",
        "vertical_axis",
        "Trunk anatomical angle from vertical, computed as 180 - trunk vertical deviation",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Diz fleksiyon açısı",
        [
            _joint_angle_item(kps_rc, idx_to_name, 13, 16, 18, lambda raw: 180.0 - raw),
            _joint_angle_item(kps_rc, idx_to_name, 12, 15, 17, lambda raw: 180.0 - raw),
        ],
        "bilateral_joint_flexion",
        "hip-knee-ankle",
        "Average right/left knee flexion, computed as 180 - knee extension angle",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Gövde-bacak açısı",
        [
            _joint_angle_item(kps_rc, idx_to_name, 7, 13, 16),
            _joint_angle_item(kps_rc, idx_to_name, 6, 12, 15),
        ],
        "bilateral_joint_angle",
        "shoulder-hip-knee",
        "Average right/left angle between trunk and upper leg",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Kolların yatay açısı",
        [
            _segment_deviation_item(kps_rc, idx_to_name, 7, 11, 9, "horizontal_axis", 1.0, 0.0, lambda raw: 180.0 - raw),
            _segment_deviation_item(kps_rc, idx_to_name, 6, 10, 8, "horizontal_axis", 1.0, 0.0, lambda raw: 180.0 - raw),
        ],
        "bilateral_segment_angle",
        "horizontal_axis",
        "Average right/left arm anatomical angle from the horizontal axis",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Parabolik/uçuş açısı",
        [_body_flight_item(kps_rc, idx_to_name)],
        "single_frame_body_flight_axis",
        "horizontal_axis",
        "Single-frame body flight axis angle from leg endpoint center to shoulder center; true parabolic trajectory requires multiple frames",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Kalça-gövde eksantisyonu",
        [
            _joint_angle_item(kps_rc, idx_to_name, 7, 13, 16, lambda raw: max(0.0, raw - 90.0)),
            _joint_angle_item(kps_rc, idx_to_name, 6, 12, 15, lambda raw: max(0.0, raw - 90.0)),
        ],
        "bilateral_hip_extension_offset",
        "shoulder-hip-knee",
        "Average right/left hip-trunk extension amount above 90 degrees",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Bacak yatay sapması",
        [
            _segment_deviation_item(kps_rc, idx_to_name, 13, 18, 16, "horizontal_axis", 1.0, 0.0),
            _segment_deviation_item(kps_rc, idx_to_name, 12, 17, 15, "horizontal_axis", 1.0, 0.0),
        ],
        "bilateral_axis_deviation",
        "horizontal_axis",
        "Average right/left leg deviation from the horizontal axis",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Kolların yanda açısı",
        [
            _joint_angle_item(kps_rc, idx_to_name, 9, 7, 13),
            _joint_angle_item(kps_rc, idx_to_name, 8, 6, 12),
        ],
        "bilateral_shoulder_angle",
        "elbow/wrist-shoulder-hip",
        "Average right/left shoulder side angle",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Gövde kalça fleksiyonu",
        [
            _joint_angle_item(kps_rc, idx_to_name, 7, 13, 16, lambda raw: 180.0 - raw),
            _joint_angle_item(kps_rc, idx_to_name, 6, 12, 15, lambda raw: 180.0 - raw),
        ],
        "bilateral_hip_flexion",
        "shoulder-hip-knee",
        "Average right/left hip flexion, computed as 180 - trunk-leg angle",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Kalça ekstansiyonu",
        [
            _joint_angle_item(kps_rc, idx_to_name, 7, 13, 16),
            _joint_angle_item(kps_rc, idx_to_name, 6, 12, 15),
        ],
        "bilateral_hip_extension_angle",
        "shoulder-hip-knee",
        "Average right/left hip extension/anatomical opening angle",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Kolların geriye gidişi",
        [
            _arm_swing_item(kps_rc, idx_to_name, 7, 11, 9, 13),
            _arm_swing_item(kps_rc, idx_to_name, 6, 10, 8, 12),
        ],
        "bilateral_directed_segment_angle",
        "trunk line shoulder->hip",
        "Circular mean of right/left directed arm swing angles",
        circular=True,
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Gövde Sapması",
        [_trunk_deviation_item(kps_rc, idx_to_name)],
        "trunk_axis_deviation",
        "vertical_axis",
        "Trunk-line deviation from vertical",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Kol Sapması",
        [
            _segment_deviation_item(kps_rc, idx_to_name, 7, 11, 9, "horizontal_axis", 1.0, 0.0),
            _segment_deviation_item(kps_rc, idx_to_name, 6, 10, 8, "horizontal_axis", 1.0, 0.0),
        ],
        "bilateral_axis_deviation",
        "horizontal_axis",
        "Average right/left arm deviation from the horizontal axis",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Bacak Sapması",
        [
            _segment_deviation_item(kps_rc, idx_to_name, 13, 18, 16, "vertical_axis", 0.0, 1.0),
            _segment_deviation_item(kps_rc, idx_to_name, 12, 17, 15, "vertical_axis", 0.0, 1.0),
        ],
        "bilateral_axis_deviation",
        "vertical_axis",
        "Average right/left leg deviation from the vertical axis",
    )
    _add_composite_metric_record(
        records,
        idx_to_name,
        "Diz ekstansiyonu",
        [
            _joint_angle_item(kps_rc, idx_to_name, 13, 16, 18),
            _joint_angle_item(kps_rc, idx_to_name, 12, 15, 17),
        ],
        "bilateral_knee_extension",
        "hip-knee-ankle",
        "Average right/left knee extension angle",
    )

    return records


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
    deleted_standard_angles: Optional[list] = None,
) -> list:
    records = []
    deleted_labels = {str(label) for label in (deleted_standard_angles or [])}
    for ia, ib, ic, label in _STANDARD_ANGLE_DEFINITIONS:
        if label in deleted_labels:
            continue
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


def _angle_record_key(record: Dict[str, Any]) -> str:
    source = "derived" if record.get("source") == "derived_pose_metric" else "standard"
    return f"{source}:{record.get('label', '')}"


def _filter_angle_records_by_keys(records: list, selected_keys: set) -> list:
    if not selected_keys:
        return []
    return [record for record in records if _angle_record_key(record) in selected_keys]


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
    derived_metric_records: list,
    source_json_path: str,
    original_img_path: str,
    deleted_standard_angles: Optional[list] = None,
    selected_angle_keys: Optional[list] = None,
    show_selected_angles: bool = False,
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
        "angles": standard_angle_records + derived_metric_records + manual_angle_records,
        "standard_angles": standard_angle_records,
        "derived_metrics": derived_metric_records,
        "manual_angles": manual_angle_records,
        "deleted_standard_angles": deleted_standard_angles or [],
        "selected_angle_keys": selected_angle_keys or [],
        "show_selected_angles": show_selected_angles,
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _build_editor_payload_from_kps(
    original_img_path: str,
    kps_rc: list,
    idx_to_name: Optional[Dict[int, str]] = None,
    manual_angles: Optional[list] = None,
    deleted_standard_angles: Optional[list] = None,
    selected_angle_keys: Optional[list] = None,
    show_selected_angles: bool = False,
    editor_role: str = "main",
    output_id: str = "kp_editor_output",
    prev_trigger_id: str = "pe_prev_trigger",
    next_trigger_id: str = "pe_next_trigger",
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
        "deleted_standard_angles": deleted_standard_angles or [],
        "selected_angle_keys": selected_angle_keys or [],
        "show_selected_angles": show_selected_angles,
        "editor_role": editor_role,
        "output_id": output_id,
        "prev_trigger_id": prev_trigger_id,
        "next_trigger_id": next_trigger_id,
    }, separators=(",", ":"))

    return base64.b64encode(payload.encode("utf-8")).decode("ascii"), "OK"


def _build_editor_payload_from_canvas_kps(
    original_img_path: str,
    canvas_kps: list,
    canvas_scale: float,
    idx_to_name: Optional[Dict[int, str]] = None,
    manual_angles: Optional[list] = None,
    deleted_standard_angles: Optional[list] = None,
    selected_angle_keys: Optional[list] = None,
    show_selected_angles: bool = False,
    editor_role: str = "main",
    output_id: str = "kp_editor_output",
    prev_trigger_id: str = "pe_prev_trigger",
    next_trigger_id: str = "pe_next_trigger",
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
      "deleted_standard_angles": deleted_standard_angles or [],
      "selected_angle_keys": selected_angle_keys or [],
      "show_selected_angles": show_selected_angles,
      "editor_role": editor_role,
      "output_id": output_id,
      "prev_trigger_id": prev_trigger_id,
      "next_trigger_id": next_trigger_id,
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
    deleted_standard_angles = [
      str(label) for label in payload.get("deleted_standard_angles", [])
      if str(label)
    ]
    editor_role = str(payload.get("editor_role") or "main")
    output_id = str(payload.get("output_id") or "kp_editor_output")
    prev_trigger_id = str(payload.get("prev_trigger_id") or "pe_prev_trigger")
    next_trigger_id = str(payload.get("next_trigger_id") or "pe_next_trigger")
    save_standard_angles = bool(payload.get("show_standard_angles", False))
    selected_angle_keys = [
      str(key) for key in payload.get("selected_angle_keys", [])
      if str(key)
    ]
    selected_angle_key_set = set(selected_angle_keys)
    save_selected_angles = bool(payload.get("show_selected_angles", False)) and bool(selected_angle_key_set)
    save_system_angles = save_standard_angles or save_selected_angles

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
    standard_angle_records = _build_standard_angle_records(kps_for_json, idx_to_name, deleted_standard_angles) if save_system_angles else []
    derived_metric_records = _build_derived_metric_records(kps_for_json, idx_to_name) if save_system_angles else []
    if save_selected_angles and not save_standard_angles:
      standard_angle_records = _filter_angle_records_by_keys(standard_angle_records, selected_angle_key_set)
      derived_metric_records = _filter_angle_records_by_keys(derived_metric_records, selected_angle_key_set)

    if json_path:
      try:
        p = Path(json_path)
        orig_data = json.loads(p.read_text(encoding="utf-8"))

        idx_to_name = {int(k): v for k, v in orig_data.get("skeleton", {}).items()} or idx_to_name
        manual_angle_records = _build_manual_angle_records(raw_manual_angles, kps_for_json, idx_to_name)
        standard_angle_records = _build_standard_angle_records(kps_for_json, idx_to_name, deleted_standard_angles) if save_system_angles else []
        derived_metric_records = _build_derived_metric_records(kps_for_json, idx_to_name) if save_system_angles else []
        if save_selected_angles and not save_standard_angles:
          standard_angle_records = _filter_angle_records_by_keys(standard_angle_records, selected_angle_key_set)
          derived_metric_records = _filter_angle_records_by_keys(derived_metric_records, selected_angle_key_set)

        if not orig_data.get("keypoints") or not isinstance(orig_data["keypoints"], list):
          orig_data["keypoints"] = [{"0": kps_for_json}]
        else:
          person_dict = orig_data["keypoints"][0]
          if not isinstance(person_dict, dict) or not person_dict:
            person_dict = {"0": kps_for_json}
            orig_data["keypoints"][0] = person_dict
          person_key = "0" if "0" in person_dict else next(iter(person_dict.keys()))
          person_dict[person_key] = kps_for_json

        orig_data["angles"] = standard_angle_records + derived_metric_records + manual_angle_records
        orig_data["standard_angles"] = standard_angle_records
        orig_data["derived_metrics"] = derived_metric_records
        orig_data["manual_angles"] = manual_angle_records
        orig_data["deleted_standard_angles"] = deleted_standard_angles
        orig_data["selected_angle_keys"] = selected_angle_keys
        orig_data["show_selected_angles"] = save_selected_angles
        orig_data["manual_angle_schema"] = _manual_angle_schema()

        p.write_text(json.dumps(orig_data, ensure_ascii=False), encoding="utf-8")
        angles_path = _save_angles_sidecar_json(
          manual_angle_records,
          standard_angle_records,
          derived_metric_records,
          str(p),
          original_img_path,
          deleted_standard_angles,
          selected_angle_keys,
          save_selected_angles,
        )
        total_angle_count = len(standard_angle_records) + len(derived_metric_records) + len(manual_angle_records)
        status = f"Kaydedildi: {p.name} | aci: {total_angle_count} | aci JSON: {angles_path}"
      except Exception as e:
        return current_payload, f"JSON kaydedilemedi: {e}"
    else:
      try:
        angles_path = _save_angles_sidecar_json(
          manual_angle_records,
          standard_angle_records,
          derived_metric_records,
          "",
          original_img_path,
          deleted_standard_angles,
          selected_angle_keys,
          save_selected_angles,
        )
        total_angle_count = len(standard_angle_records) + len(derived_metric_records) + len(manual_angle_records)
        status = f"Goruntu guncellendi | aci: {total_angle_count} | aci JSON: {angles_path}"
      except Exception as e:
        return current_payload, f"Aci JSON kaydedilemedi: {e}"

    new_payload, prep_status = _build_editor_payload_from_canvas_kps(
      original_img_path,
      kps_list,
      canvas_scale=canvas_scale,
      idx_to_name=idx_to_name,
      manual_angles=manual_angle_records,
      deleted_standard_angles=deleted_standard_angles,
      selected_angle_keys=selected_angle_keys,
      show_selected_angles=save_selected_angles,
      editor_role=editor_role,
      output_id=output_id,
      prev_trigger_id=prev_trigger_id,
      next_trigger_id=next_trigger_id,
    )
    if not new_payload:
      return current_payload, f"{status} | Canvas yenilenemedi: {prep_status}"
    return new_payload, status
