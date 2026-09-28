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
<div class="pe-wrap pe-empty">
  <div class="pe-data" style="display:none">${value}</div>
  <div class="pe-empty-state">Görsel veya video yüklediğinizde pose editörü burada açılır.</div>
  <canvas class="pe-canvas" tabindex="0"></canvas>
  <div class="pe-bar">
    <button class="pe-btn pe-reset"  data-action="reset">&#8617; Reset</button>
    <button class="pe-btn pe-names"  data-action="names">&#128065; Names</button>
    <button class="pe-btn pe-save"   data-action="save">&#128190; Save PNG</button>
    <button class="pe-btn pe-angles" data-action="angles">&#128208; A&#231;&#305;lar</button>
    <button class="pe-btn pe-custom-angle" data-action="custom-angle">&#8736; &#214;zel A&#231;&#305;</button>
    <button class="pe-btn pe-delete-angle" data-action="delete-angle">&#9003; A&#231;&#305; Sil</button>
    <button class="pe-btn pe-fullscreen" data-action="fullscreen">&#x2922; Tam Ekran</button>
    <button class="pe-btn pe-keypoints" data-action="keypoints">&#x25CF; Noktalar</button>
    <button class="pe-btn pe-undo" data-action="undo" title="Ctrl+Z">&#8630; Geri Al</button>
    <button class="pe-btn pe-redo" data-action="redo" title="Ctrl+Y / Ctrl+Shift+Z">&#8631; Yinele</button>
    <span class="pe-info">Goruntu yukleniyor...</span>
    <div class="pe-nav-group">
      <button class="pe-btn pe-prev" data-action="prev">&#9664; Prev</button>
      <div class="pe-frame-control">
        <input class="pe-frame-slider" type="range" min="0" max="0" value="0" step="1" aria-label="Video frame" />
        <span class="pe-frame-counter">1 / 1</span>
      </div>
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
.pe-empty-state {
  display: none; min-height: 180px; align-items: center; justify-content: center;
  padding: 24px; border: 2px dashed #555e6e; border-radius: 7px;
  color: #aeb6c7; font-size: 15px; text-align: center;
}
.pe-wrap.pe-empty .pe-empty-state { display: flex; }
.pe-wrap.pe-empty .pe-canvas,
.pe-wrap.pe-empty .pe-bar { display: none; }
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
.pe-delete-angle { background: #b83280; }
.pe-delete-angle.is-active { outline: 2px solid #fff; box-shadow: 0 0 0 2px rgba(184,50,128,.45); }
.pe-fullscreen { background: #e67e22; }
.pe-keypoints { background: #d35400; }
.pe-undo, .pe-redo { background: #4b6584; }
.pe-canvas:focus { outline: 2px solid rgba(79,140,255,.7); outline-offset: 2px; }
.pe-info   { font-size: 12px; color: #aaa; font-family: monospace; }
.pe-nav-group {
  margin-left: auto; display: flex; gap: 8px; align-items: center;
  flex: 1 1 440px; justify-content: flex-end;
}
.pe-prev, .pe-next {
  background: #555e6e;
  width: 108px;
  min-width: 108px;
  flex: 0 0 108px;
  box-sizing: border-box;
  text-align: center;
}
.pe-btn:disabled { opacity: .45; cursor: not-allowed; }
.pe-frame-control {
  display: none; align-items: center; gap: 8px;
  flex: 1 1 260px; min-width: 180px; max-width: 420px;
}
.pe-frame-slider { flex: 1 1 auto; min-width: 120px; cursor: pointer; accent-color: #4f8cff; }
.pe-frame-counter {
  min-width: 58px; text-align: right; white-space: nowrap;
  color: #c8cad3; font: 600 12px/1.2 monospace;
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
  if (!raw || raw.length < 20) {
    if (element._peKey === '') return;
    element._peKey = '';
    element._peCurrentPayload = '';
    var emptyWrap = element.querySelector('.pe-wrap');
    if (emptyWrap) {
      emptyWrap.classList.add('pe-empty');
      delete emptyWrap._peGetKps;
      delete emptyWrap._peGetPayload;
      delete emptyWrap._peLoadPayload;
    }
    var emptyCanvas = element.querySelector('.pe-canvas');
    if (emptyCanvas) {
      emptyCanvas.width = 1;
      emptyCanvas.height = 1;
      emptyCanvas.getContext('2d').clearRect(0, 0, 1, 1);
    }
    return;
  }

  /* same payload as last init → nothing to do (e.g. info.textContent mutation) */
  if (element._peKey === raw) return;
  element._peKey = raw;

  var DATA;
  try { DATA = JSON.parse(atob(raw)); } catch(e) { return; }

  var activeWrap = element.querySelector('.pe-wrap');
  if (activeWrap) activeWrap.classList.remove('pe-empty');

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
  var editorRole = DATA.editor_role || 'main';
  var outputId = DATA.output_id || 'kp_editor_output';
  var prevTriggerId = DATA.prev_trigger_id || 'pe_prev_trigger';
  var nextTriggerId = DATA.next_trigger_id || 'pe_next_trigger';
  var frameTriggerId = DATA.frame_trigger_id || '';
  var frameIndex = Math.max(0, Number(DATA.frame_index) || 0);
  var frameCount = Math.max(1, Number(DATA.frame_count) || 1);
  var requestedFrameIndex = frameIndex;
  element._peCurrentPayload = raw;
  var dragging  = null;
  var dragStartSnapshot = null;
  var undoStack = [];
  var redoStack = [];
  var HISTORY_LIMIT = 100;
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
  var showKeypoints = true;

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

  function snapshotKeypoints() {
    return JSON.parse(JSON.stringify(keypoints));
  }

  function sameKeypoints(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  function updateHistoryControls() {
    var undoBtn = element.querySelector('[data-action="undo"]');
    var redoBtn = element.querySelector('[data-action="redo"]');
    if (undoBtn) undoBtn.disabled = undoStack.length === 0;
    if (redoBtn) redoBtn.disabled = redoStack.length === 0;
  }

  function recordKeypointChange(before) {
    if (!before || sameKeypoints(before, keypoints)) return false;
    undoStack.push(before);
    if (undoStack.length > HISTORY_LIMIT) undoStack.shift();
    redoStack = [];
    updateHistoryControls();
    return true;
  }

  function restoreKeypoints(snapshot, message) {
    keypoints = JSON.parse(JSON.stringify(snapshot));
    dragging = null;
    dragStartSnapshot = null;
    hoveredAngle = null;
    render();
    emitKeypointsToHiddenOutput();
    updateHistoryControls();
    if (info) info.textContent = message;
  }

  function undoKeypointMove() {
    if (!undoStack.length) return;
    redoStack.push(snapshotKeypoints());
    restoreKeypoints(undoStack.pop(), 'Son keypoint hareketi geri alindi.');
  }

  function redoKeypointMove() {
    if (!redoStack.length) return;
    undoStack.push(snapshotKeypoints());
    restoreKeypoints(redoStack.pop(), 'Keypoint hareketi yeniden uygulandi.');
  }

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
      deleted_standard_angles: Object.keys(deletedStandardAngles),
      manual_angles: customAngles.map(function(a) {
        return { keypoint_indices: [a[0], a[1], a[2]] };
      }),
      editor_role: editorRole,
      output_id: outputId,
      prev_trigger_id: prevTriggerId,
      next_trigger_id: nextTriggerId,
      frame_trigger_id: frameTriggerId,
      frame_index: frameIndex,
      frame_count: frameCount
    };
  }

  function updateFrameControls() {
    var navGroup = element.querySelector('.pe-nav-group');
    var frameControl = element.querySelector('.pe-frame-control');
    var slider = element.querySelector('.pe-frame-slider');
    var counter = element.querySelector('.pe-frame-counter');
    var prevBtn = element.querySelector('[data-action="prev"]');
    var nextBtn = element.querySelector('[data-action="next"]');
    var videoNavigation = editorRole === 'video' && !!frameTriggerId;
    var controlIndex = Math.max(0, Math.min(requestedFrameIndex, frameCount - 1));

    if (navGroup) navGroup.style.display = videoNavigation ? 'flex' : 'none';
    if (frameControl) frameControl.style.display = videoNavigation ? 'flex' : 'none';
    if (slider) {
      slider.min = '0';
      slider.max = String(Math.max(0, frameCount - 1));
      slider.value = String(controlIndex);
      slider.disabled = !videoNavigation || frameCount <= 1;
    }
    if (counter) counter.textContent = (controlIndex + 1) + ' / ' + frameCount;
    if (videoNavigation) {
      if (prevBtn) prevBtn.disabled = controlIndex <= 0;
      if (nextBtn) nextBtn.disabled = controlIndex >= frameCount - 1;
    } else {
      if (prevBtn) prevBtn.disabled = false;
      if (nextBtn) nextBtn.disabled = false;
    }
  }

  /* load image then size + draw canvas */
  var img = new window.Image();
  img.onload = function() {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    render();
    emitKeypointsToHiddenOutput();
    updateFrameControls();
    if (info) info.textContent =
      'Noktalari surukleleyin  (' + keypoints.length + ' keypoint)';
  };
  img.onerror = function() {
    if (info) info.textContent = 'Goruntu yuklenemedi!';
  };
  img.src = DATA.img;

  function loadFramePayload(nextRaw) {
    if (!nextRaw || nextRaw.length < 20) return;
    var nextData;
    try { nextData = JSON.parse(atob(nextRaw)); } catch(e) {
      if (info) info.textContent = 'Frame verisi okunamadi!';
      return;
    }

    var incomingFrameIndex = Math.max(0, Number(nextData.frame_index) || 0);
    if (editorRole === 'video' && frameTriggerId && incomingFrameIndex !== requestedFrameIndex) {
      return;
    }

    DATA = nextData;
    element._peCurrentPayload = nextRaw;
    origKps = JSON.parse(JSON.stringify(nextData.kps || []));
    keypoints = JSON.parse(JSON.stringify(nextData.kps || []));
    skeleton = nextData.sk || [];
    names = nextData.nm || [];
    csScale = Number(nextData.cs) || 1.0;
    editorRole = nextData.editor_role || editorRole;
    outputId = nextData.output_id || outputId;
    prevTriggerId = nextData.prev_trigger_id || prevTriggerId;
    nextTriggerId = nextData.next_trigger_id || nextTriggerId;
    frameTriggerId = nextData.frame_trigger_id || frameTriggerId;
    frameIndex = Math.max(0, Number(nextData.frame_index) || 0);
    frameCount = Math.max(1, Number(nextData.frame_count) || 1);
    requestedFrameIndex = frameIndex;

    deletedStandardAngles = {};
    (nextData.deleted_standard_angles || []).forEach(function(label) {
      deletedStandardAngles[String(label)] = true;
    });
    customAngles = (nextData.manual_angles || []).map(function(a) {
      var pts = Array.isArray(a) ? a : (a.keypoint_indices || a.points || []);
      return [Number(pts[0]), Number(pts[1]), Number(pts[2])];
    }).filter(function(a) {
      return a.length === 3 && a.every(function(v) { return Number.isInteger(v); });
    });
    dragging = null;
    dragStartSnapshot = null;
    undoStack = [];
    redoStack = [];
    hoveredAngle = null;
    angleHitboxes = [];
    customAngleMode = false;
    customAnglePick = [];
    deleteAngleMode = false;
    var customBtn = element.querySelector('[data-action="custom-angle"]');
    var deleteBtn = element.querySelector('[data-action="delete-angle"]');
    if (customBtn) customBtn.classList.remove('is-active');
    if (deleteBtn) deleteBtn.classList.remove('is-active');

    var nextImg = new window.Image();
    nextImg.onload = function() {
      img = nextImg;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      updateFrameControls();
      updateHistoryControls();
      render();
      emitKeypointsToHiddenOutput();
      if (info) info.textContent =
        'Frame ' + (frameIndex + 1) + '/' + frameCount +
        '  |  Zoom: ' + Math.round(zoom * 100) + '%';
    };
    nextImg.onerror = function() {
      if (info) info.textContent = 'Yeni frame yuklenemedi!';
    };
    nextImg.src = nextData.img;
  }

  function overlayMetrics() {
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var i = 0; i < keypoints.length; i++) {
      var kp = keypoints[i];
      if (!kp || kp.c < 0.1) continue;
      minX = Math.min(minX, kp.x);
      minY = Math.min(minY, kp.y);
      maxX = Math.max(maxX, kp.x);
      maxY = Math.max(maxY, kp.y);
    }

    var poseSpan = Number.isFinite(minX)
      ? Math.max(maxX - minX, maxY - minY)
      : 0;
    var subjectRadius = poseSpan > 0
      ? clamp(poseSpan * 0.012, 2.75, R)
      : R;
    var radiusScreen = clamp(subjectRadius * zoom, 2.75, R);
    var lineScreen = clamp(radiusScreen * 0.34, 0.9, 2.5);
    var strokeScreen = clamp(radiusScreen * 0.22, 0.75, 1.5);

    return {
      radiusScreen: radiusScreen,
      radiusWorld: radiusScreen / zoom,
      hitRadiusWorld: (radiusScreen + strokeScreen / 2) / zoom,
      lineWorld: lineScreen / zoom,
      strokeWorld: strokeScreen / zoom,
      dragStrokeWorld: Math.max(strokeScreen, 1.75) / zoom,
      fontWorld: clamp(radiusScreen * 1.65, 7, 11) / zoom
    };
  }

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(img, 0, 0);

    /* Scale overlays with the visible subject, capped for readability. */
    var metrics = overlayMetrics();
    var lw = metrics.lineWorld;
    var rr = metrics.radiusWorld;
    var fs = metrics.fontWorld;

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
        ctx.lineWidth = (dragging === i)
          ? metrics.dragStrokeWorld
          : metrics.strokeWorld;
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

  function pushMetric(metrics, label, angle, x, y, points, vertex) {
    if (angle === null || !Number.isFinite(angle)) return;
    metrics.push({
      label: label,
      angle: angle,
      x: x,
      y: y,
      points: points,
      vertex: vertex || 'Referans eksen'
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
        'Dikey eksen'
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
        axisName
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
        'Govde referansi'
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
        nameForKeypoint(ib) + ' [' + ib + ']'
      );
    }

    addAliasAngle('R.KalcaFleksExt', 7, 13, 16);
    addAliasAngle('L.KalcaFleksExt', 6, 12, 15);
    addAliasAngle('R.DizEkst', 13, 16, 18);
    addAliasAngle('L.DizEkst', 12, 15, 17);

    return metrics;
  }

  function drawDerivedMetrics() {
    if (!showAngles) return;
    var metrics = derivedMetrics();
    ctx.save();
    ctx.font = 'bold ' + (11 / zoom) + 'px sans-serif';
    for (var i = 0; i < metrics.length; i++) {
      var m = metrics[i];
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
      if (deletedStandardAngles[ANG[ai][3]]) continue;
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
    /* Hit area is exactly the adaptive visible dot plus its stroke. */
    var best = null, bestD = overlayMetrics().hitRadiusWorld;
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
    }
    var deleteBtn = element.querySelector('[data-action="delete-angle"]');
    if (deleteBtn) deleteBtn.classList.toggle('is-active', deleteAngleMode);
    var customBtn = element.querySelector('[data-action="custom-angle"]');
    if (customBtn) customBtn.classList.toggle('is-active', customAngleMode);
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
    var moved = dragging !== null && recordKeypointChange(dragStartSnapshot);
    if (dragging !== null && info)
      info.textContent = names[dragging] + '  (' +
        Math.round(keypoints[dragging].x / csScale) + ', ' +
        Math.round(keypoints[dragging].y / csScale) + ')';
    if (dragging !== null && moved) emitKeypointsToHiddenOutput();
    dragging = null;
    dragStartSnapshot = null;
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
    try { canvas.focus({preventScroll: true}); } catch (_) { canvas.focus(); }
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
    if (kpIdx !== null && e.button === 0) {
      dragging = kpIdx;
      dragStartSnapshot = snapshotKeypoints();
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
    if (dragging !== null) finishDrag();
    hoveredAngle = null;
    canvas.style.cursor = 'crosshair';
    render();
  });

  /* touch — same rule: no info.textContent inside touchmove */
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    try { canvas.focus({preventScroll: true}); } catch (_) { canvas.focus(); }
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
    dragStartSnapshot = dragging !== null ? snapshotKeypoints() : null;
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
    peWrap._peGetPayload = function() {
      return element._peCurrentPayload || raw;
    };
    peWrap._peLoadPayload = loadFramePayload;
  }

  rebind('[data-action="reset"]', function() {
    var beforeReset = snapshotKeypoints();
    keypoints = JSON.parse(JSON.stringify(origKps));
    recordKeypointChange(beforeReset);
    zoom = 1.0; panX = 0; panY = 0;
    customAnglePick = [];
    customAngles = [];
    deletedStandardAngles = {};
    deleteAngleMode = false;
    var deleteBtn = element.querySelector('[data-action="delete-angle"]');
    if (deleteBtn) deleteBtn.classList.remove('is-active');
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

  rebind('[data-action="undo"]', undoKeypointMove);
  rebind('[data-action="redo"]', redoKeypointMove);

  var historyKeyHandler = function(e) {
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    var target = e.target;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    var key = String(e.key || '').toLowerCase();
    if (key === 'z' && e.shiftKey) {
      e.preventDefault();
      redoKeypointMove();
    } else if (key === 'z') {
      e.preventDefault();
      undoKeypointMove();
    } else if (key === 'y') {
      e.preventDefault();
      redoKeypointMove();
    }
  };
  if (element._peHistoryKeyHandler) {
    element.removeEventListener('keydown', element._peHistoryKeyHandler);
  }
  element._peHistoryKeyHandler = historyKeyHandler;
  element.addEventListener('keydown', historyKeyHandler);
  updateHistoryControls();

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

  function emitFrameTrigger(targetIndex) {
    if (!frameTriggerId || frameCount <= 0) return;
    var target = Math.max(0, Math.min(Number(targetIndex) || 0, frameCount - 1));
    if (target === requestedFrameIndex) return;
    requestedFrameIndex = target;
    updateFrameControls();
    var container = document.querySelector('#' + frameTriggerId);
    var el = container
      ? (container.querySelector('textarea') || container.querySelector('input[type="text"]') || container.querySelector('input'))
      : null;
    if (!el) { console.warn('Frame trigger not found:', frameTriggerId); return; }
    var proto = el.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    var nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (nativeSetter && nativeSetter.set) nativeSetter.set.call(el, String(target));
    else el.value = String(target);
    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function bindFrameSlider() {
    var old = element.querySelector('.pe-frame-slider');
    if (!old) return;
    var slider = old.cloneNode(true);
    old.parentNode.replaceChild(slider, old);
    function requestFrame() {
      var target = Math.max(0, Math.min(Number(slider.value) || 0, frameCount - 1));
      var counter = element.querySelector('.pe-frame-counter');
      if (counter) counter.textContent = (target + 1) + ' / ' + frameCount;
      emitFrameTrigger(target);
    }

    slider.addEventListener('input', requestFrame);
    slider.addEventListener('change', requestFrame);
  }

  rebind('[data-action="prev"]', function() {
    if (editorRole === 'video' && frameTriggerId) emitFrameTrigger(requestedFrameIndex - 1);
    else emitNavTrigger('#' + prevTriggerId);
    if (info) info.textContent = 'Onceki goruntuye geciliyor...';
  });

  rebind('[data-action="next"]', function() {
    if (editorRole === 'video' && frameTriggerId) emitFrameTrigger(requestedFrameIndex + 1);
    else emitNavTrigger('#' + nextTriggerId);
    if (info) info.textContent = 'Sonraki goruntuye geciliyor...';
  });
  bindFrameSlider();
  updateFrameControls();
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
    frame_trigger_id: str = "",
    frame_index: int = 0,
    frame_count: int = 1,
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
    person_dict = kp_outer[0] if kp_outer else None
    has_detected_pose = isinstance(person_dict, dict) and bool(person_dict)
    if has_detected_pose:
        kp_list = person_dict.get("0")
        if kp_list is None:
            kp_list = person_dict[next(iter(person_dict.keys()))]
    else:
        # Keep no-pose video frames navigable in the same editor. Confidence 0
        # prevents the placeholder points and skeleton from being rendered.
        kp_list = [[0.0, 0.0, 0.0] for _ in range(25)]
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
        "frame_trigger_id": frame_trigger_id,
        "frame_index": max(0, int(frame_index)),
        "frame_count": max(1, int(frame_count)),
    }, separators=(",", ":"))

    value = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    pose_note = "" if has_detected_pose else "  |  pose bulunamadi"
    return value, f"Editor hazir: {json_path.name}  |  scale={cs:.2f}{pose_note}"


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
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _build_editor_payload_from_kps(
    original_img_path: str,
    kps_rc: list,
    idx_to_name: Optional[Dict[int, str]] = None,
    manual_angles: Optional[list] = None,
    deleted_standard_angles: Optional[list] = None,
    editor_role: str = "main",
    output_id: str = "kp_editor_output",
    prev_trigger_id: str = "pe_prev_trigger",
    next_trigger_id: str = "pe_next_trigger",
    frame_trigger_id: str = "",
    frame_index: int = 0,
    frame_count: int = 1,
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
        "editor_role": editor_role,
        "output_id": output_id,
        "prev_trigger_id": prev_trigger_id,
        "next_trigger_id": next_trigger_id,
        "frame_trigger_id": frame_trigger_id,
        "frame_index": max(0, int(frame_index)),
        "frame_count": max(1, int(frame_count)),
    }, separators=(",", ":"))

    return base64.b64encode(payload.encode("utf-8")).decode("ascii"), "OK"


def _build_editor_payload_from_canvas_kps(
    original_img_path: str,
    canvas_kps: list,
    canvas_scale: float,
    idx_to_name: Optional[Dict[int, str]] = None,
    manual_angles: Optional[list] = None,
    deleted_standard_angles: Optional[list] = None,
    editor_role: str = "main",
    output_id: str = "kp_editor_output",
    prev_trigger_id: str = "pe_prev_trigger",
    next_trigger_id: str = "pe_next_trigger",
    frame_trigger_id: str = "",
    frame_index: int = 0,
    frame_count: int = 1,
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
      "editor_role": editor_role,
      "output_id": output_id,
      "prev_trigger_id": prev_trigger_id,
      "next_trigger_id": next_trigger_id,
      "frame_trigger_id": frame_trigger_id,
      "frame_index": max(0, int(frame_index)),
      "frame_count": max(1, int(frame_count)),
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
    frame_trigger_id = str(payload.get("frame_trigger_id") or "")
    frame_index = max(0, int(payload.get("frame_index") or 0))
    frame_count = max(1, int(payload.get("frame_count") or 1))
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
    standard_angle_records = _build_standard_angle_records(kps_for_json, idx_to_name, deleted_standard_angles) if save_standard_angles else []
    derived_metric_records = _build_derived_metric_records(kps_for_json, idx_to_name) if save_standard_angles else []

    if json_path:
      try:
        p = Path(json_path)
        orig_data = json.loads(p.read_text(encoding="utf-8"))

        idx_to_name = {int(k): v for k, v in orig_data.get("skeleton", {}).items()} or idx_to_name
        manual_angle_records = _build_manual_angle_records(raw_manual_angles, kps_for_json, idx_to_name)
        standard_angle_records = _build_standard_angle_records(kps_for_json, idx_to_name, deleted_standard_angles) if save_standard_angles else []
        derived_metric_records = _build_derived_metric_records(kps_for_json, idx_to_name) if save_standard_angles else []

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
        orig_data["manual_angle_schema"] = _manual_angle_schema()

        p.write_text(json.dumps(orig_data, ensure_ascii=False), encoding="utf-8")
        angles_path = _save_angles_sidecar_json(manual_angle_records, standard_angle_records, derived_metric_records, str(p), original_img_path, deleted_standard_angles)
        total_angle_count = len(standard_angle_records) + len(derived_metric_records) + len(manual_angle_records)
        status = f"Kaydedildi: {p.name} | aci: {total_angle_count} | aci JSON: {angles_path}"
      except Exception as e:
        return current_payload, f"JSON kaydedilemedi: {e}"
    else:
      try:
        angles_path = _save_angles_sidecar_json(manual_angle_records, standard_angle_records, derived_metric_records, "", original_img_path, deleted_standard_angles)
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
      editor_role=editor_role,
      output_id=output_id,
      prev_trigger_id=prev_trigger_id,
      next_trigger_id=next_trigger_id,
      frame_trigger_id=frame_trigger_id,
      frame_index=frame_index,
      frame_count=frame_count,
    )
    if not new_payload:
      return current_payload, f"{status} | Canvas yenilenemedi: {prep_status}"
    return new_payload, status
