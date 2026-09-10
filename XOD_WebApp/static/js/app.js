/* XOD — X-ray Object Detection (client)
 *
 * Responsibilities that live in the browser (the server only runs the model):
 *   - decode + display the X-ray on a pan/zoom canvas
 *   - real-time enhancement (brightness/contrast/gamma/negative/sharpen/edges/
 *     equalize/false-colour) computed on an offscreen buffer
 *   - draw detection boxes as an overlay with show/hide, per-class toggles and
 *     a confidence threshold
 *   - export the annotated image and the raw detection JSON
 *   - admin model (.pt) upload
 *
 * Coordinate model: box coordinates from the server are in ORIGINAL image
 * pixels. The offscreen "work" buffer may be downscaled for performance, but
 * everything is drawn through a single (offset + zoom) transform expressed in
 * original-image space, so boxes and pixels always line up.
 */
(function () {
  "use strict";

  var MAX_WORK_SIDE = 2200;   // cap the enhancement buffer's longest side
  var MAX_ZOOM = 14;
  var ENH_DEBOUNCE = 70;      // ms

  // ---- element cache ------------------------------------------------------
  var $ = function (id) { return document.getElementById(id); };
  var els = {};
  [
    "view", "stage", "stageEmpty", "stageBusy", "hudZoom", "hudPos",
    "zoomIn", "zoomOut", "zoomFit",
    "imageInput", "detectBtn", "exportPngBtn", "exportJsonBtn",
    "modelChip", "modelDot", "modelName", "modelClasses",
    "detCount", "showBoxes", "confSlider", "confVal",
    "detTiming", "tInf", "tPrePost", "tTotal", "detList", "detEmpty",
    "classSection", "classLegend", "toggleAllClasses",
    "brightness", "brightnessVal", "contrast", "contrastVal", "gamma", "gammaVal",
    "resetEnh", "paletteSwatches", "hist",
    "modelAdmin", "modelInput", "modelFileName", "adminKeyRow", "adminKey",
    "uploadModelBtn", "adminStatus"
  ].forEach(function (id) { els[id] = $(id); });

  // ---- state --------------------------------------------------------------
  var state = {
    image: null, imgW: 0, imgH: 0,
    workW: 0, workH: 0,
    baseImageData: null,       // original pixels at work resolution
    enhancedCanvas: null,      // enhanced result at work resolution

    view: { zoom: 1, minZoom: 0.05, offsetX: 0, offsetY: 0 },

    enh: {
      brightness: 0, contrast: 0, gamma: 1.0,
      invert: false, sharpen: false, edges: false, equalize: false,
      palette: "gray"
    },

    detections: [],
    lastResult: null,
    lastFile: null,
    modelClasses: [],
    classColors: {},           // class name -> "#rrggbb"
    classVisible: {},          // class name -> bool
    confThreshold: 0.25,
    serverFloor: 0.10,
    showBoxes: true,
    selectedId: null,
    busy: false
  };

  // ---- categorical class palette (stable, distinct on dark bg) ------------
  var CLASS_PALETTE = [
    "#f2a541", "#46c0d9", "#ec5b56", "#57c78a", "#b18cf0", "#f06fae",
    "#8fd14f", "#ffd23f", "#5a9bff", "#ff8f4d", "#4dd6c0", "#d98cff",
    "#c0d94d", "#ff6f6f"
  ];

  // =========================================================================
  //  Palettes (256-entry [r,g,b] LUTs) for false-colour
  // =========================================================================
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function buildPalette(fn) {
    var lut = new Array(256);
    for (var i = 0; i < 256; i++) { lut[i] = fn(i / 255); }
    return lut;
  }
  var PALETTES = {
    hot: buildPalette(function (t) {
      return [clamp01(t / 0.365) * 255 | 0,
              clamp01((t - 0.365) / 0.365) * 255 | 0,
              clamp01((t - 0.73) / 0.27) * 255 | 0];
    }),
    jet: buildPalette(function (t) {
      return [clamp01(1.5 - Math.abs(4 * t - 3)) * 255 | 0,
              clamp01(1.5 - Math.abs(4 * t - 2)) * 255 | 0,
              clamp01(1.5 - Math.abs(4 * t - 1)) * 255 | 0];
    }),
    ice: buildPalette(function (t) {
      return [clamp01((t - 0.5) / 0.5) * 255 | 0,
              clamp01((t - 0.25) / 0.5) * 255 | 0,
              clamp01(t / 0.5) * 255 | 0];
    })
  };

  // =========================================================================
  //  Enhancement pipeline
  // =========================================================================
  function computeLUT() {
    var e = state.enh;
    var cf = (259 * (e.contrast + 255)) / (255 * (259 - e.contrast));
    var invGamma = 1 / e.gamma;
    var lut = new Uint8ClampedArray(256);
    for (var i = 0; i < 256; i++) {
      var v = 255 * Math.pow(i / 255, invGamma); // gamma
      v = cf * (v - 128) + 128;                  // contrast
      v = v + e.brightness * 1.28;               // brightness (-100..100 -> ~-128..128)
      if (e.invert) v = 255 - v;                 // negative
      lut[i] = v;
    }
    return lut;
  }

  function equalize(img) {
    var d = img.data, N = img.width * img.height;
    var hist = new Float64Array(256);
    var lum = new Float64Array(N);
    for (var p = 0, i = 0; p < N; p++, i += 4) {
      var l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      lum[p] = l;
      var b = l | 0; if (b < 0) b = 0; else if (b > 255) b = 255;
      hist[b]++;
    }
    var cdf = new Float64Array(256), acc = 0, cdfMin = 0, seen = false;
    for (var k = 0; k < 256; k++) {
      acc += hist[k]; cdf[k] = acc;
      if (!seen && acc > 0) { cdfMin = acc; seen = true; }
    }
    var denom = (N - cdfMin) || 1;
    var map = new Float64Array(256);
    for (var m = 0; m < 256; m++) map[m] = Math.round((cdf[m] - cdfMin) / denom * 255);
    var out = new Uint8ClampedArray(d.length);
    for (var q = 0, j = 0; q < N; q++, j += 4) {
      var oldL = lum[q];
      var bi = oldL | 0; if (bi < 0) bi = 0; else if (bi > 255) bi = 255;
      var scale = oldL > 1 ? map[bi] / oldL : 0;
      out[j] = d[j] * scale; out[j + 1] = d[j + 1] * scale; out[j + 2] = d[j + 2] * scale; out[j + 3] = 255;
    }
    return new ImageData(out, img.width, img.height);
  }

  function convolve3(img, k) {
    var w = img.width, h = img.height, s = img.data;
    var out = new Uint8ClampedArray(s.length);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var r = 0, g = 0, b = 0;
        for (var ky = -1; ky <= 1; ky++) {
          for (var kx = -1; kx <= 1; kx++) {
            var px = x + kx; if (px < 0) px = 0; else if (px >= w) px = w - 1;
            var py = y + ky; if (py < 0) py = 0; else if (py >= h) py = h - 1;
            var idx = (py * w + px) * 4;
            var kv = k[(ky + 1) * 3 + (kx + 1)];
            r += s[idx] * kv; g += s[idx + 1] * kv; b += s[idx + 2] * kv;
          }
        }
        var o = (y * w + x) * 4;
        out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
      }
    }
    return new ImageData(out, w, h);
  }

  function sobel(img) {
    var w = img.width, h = img.height, s = img.data;
    var gray = new Float64Array(w * h);
    for (var p = 0, i = 0; p < w * h; p++, i += 4)
      gray[p] = 0.299 * s[i] + 0.587 * s[i + 1] + 0.114 * s[i + 2];
    var gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    var gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
    var out = new Uint8ClampedArray(s.length);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var sx = 0, sy = 0;
        for (var ky = -1; ky <= 1; ky++) {
          for (var kx = -1; kx <= 1; kx++) {
            var qx = x + kx; if (qx < 0) qx = 0; else if (qx >= w) qx = w - 1;
            var qy = y + ky; if (qy < 0) qy = 0; else if (qy >= h) qy = h - 1;
            var v = gray[qy * w + qx];
            var ki = (ky + 1) * 3 + (kx + 1);
            sx += v * gx[ki]; sy += v * gy[ki];
          }
        }
        var mag = Math.sqrt(sx * sx + sy * sy);
        if (mag > 255) mag = 255;
        var o = (y * w + x) * 4;
        out[o] = out[o + 1] = out[o + 2] = mag; out[o + 3] = 255;
      }
    }
    return new ImageData(out, w, h);
  }

  function applyPalette(img, lut) {
    var d = img.data, out = new Uint8ClampedArray(d.length);
    for (var i = 0; i < d.length; i += 4) {
      var l = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      if (l < 0) l = 0; else if (l > 255) l = 255;
      var c = lut[l];
      out[i] = c[0]; out[i + 1] = c[1]; out[i + 2] = c[2]; out[i + 3] = 255;
    }
    return new ImageData(out, img.width, img.height);
  }

  var SHARPEN = [0, -1, 0, -1, 5, -1, 0, -1, 0];

  function renderEnhanced() {
    if (!state.baseImageData) return;
    var src = state.baseImageData.data;
    var out = new Uint8ClampedArray(src.length);
    var lut = computeLUT();
    for (var i = 0; i < src.length; i += 4) {
      out[i] = lut[src[i]]; out[i + 1] = lut[src[i + 1]]; out[i + 2] = lut[src[i + 2]]; out[i + 3] = 255;
    }
    var img = new ImageData(out, state.workW, state.workH);

    if (state.enh.equalize) img = equalize(img);
    if (state.enh.edges) img = sobel(img);
    else if (state.enh.sharpen) img = convolve3(img, SHARPEN);
    if (state.enh.palette !== "gray") img = applyPalette(img, PALETTES[state.enh.palette]);

    state.enhancedCanvas.getContext("2d").putImageData(img, 0, 0);
    drawHistogram(img);
    draw();
  }

  var enhTimer = null;
  function scheduleEnhance() {
    if (enhTimer) clearTimeout(enhTimer);
    enhTimer = setTimeout(renderEnhanced, ENH_DEBOUNCE);
  }

  // =========================================================================
  //  Histogram
  // =========================================================================
  function drawHistogram(img) {
    var cv = els.hist, ctx = cv.getContext("2d");
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (!img) return;
    var bins = new Float64Array(256), d = img.data;
    for (var i = 0; i < d.length; i += 4) {
      var l = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      if (l < 0) l = 0; else if (l > 255) l = 255;
      bins[l]++;
    }
    var max = 0;
    for (var k = 0; k < 256; k++) if (bins[k] > max) max = bins[k];
    if (max <= 0) return;
    var logMax = Math.log(max + 1);
    ctx.fillStyle = "rgba(154,166,180,0.85)";
    var bw = W / 256;
    for (var b = 0; b < 256; b++) {
      var hgt = (Math.log(bins[b] + 1) / logMax) * (H - 2);
      ctx.fillRect(b * bw, H - hgt, Math.max(0.5, bw - 0.3), hgt);
    }
  }

  // =========================================================================
  //  Viewer (pan / zoom / draw)
  // =========================================================================
  function stageSize() {
    var r = els.stage.getBoundingClientRect();
    return { w: r.width, h: r.height };
  }

  function resizeCanvas() {
    var cv = els.view, s = stageSize();
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(s.w * dpr));
    cv.height = Math.max(1, Math.round(s.h * dpr));
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
  }

  function fitView() {
    if (!state.image) return;
    var s = stageSize(), pad = 26;
    var z = Math.min((s.w - 2 * pad) / state.imgW, (s.h - 2 * pad) / state.imgH);
    if (!isFinite(z) || z <= 0) z = 1;
    state.view.zoom = z;
    state.view.minZoom = Math.min(z, z * 0.5) * 0.6;
    state.view.offsetX = (s.w - state.imgW * z) / 2;
    state.view.offsetY = (s.h - state.imgH * z) / 2;
    updateHud();
  }

  function draw() {
    var cv = els.view, ctx = cv.getContext("2d");
    var s = stageSize();
    ctx.clearRect(0, 0, s.w, s.h);
    if (!state.image || !state.enhancedCanvas) return;

    var v = state.view;
    ctx.imageSmoothingEnabled = v.zoom < 3.5;
    ctx.drawImage(
      state.enhancedCanvas, 0, 0, state.workW, state.workH,
      v.offsetX, v.offsetY, state.imgW * v.zoom, state.imgH * v.zoom
    );

    if (state.showBoxes) drawBoxes(ctx);
  }

  function filteredDetections() {
    return state.detections.filter(function (d) {
      return d.confidence >= state.confThreshold &&
             state.classVisible[d["class"]] !== false;
    });
  }

  function drawBoxes(ctx) {
    var v = state.view;
    var dets = filteredDetections();
    for (var n = 0; n < dets.length; n++) {
      var d = dets[n];
      var color = state.classColors[d["class"]] || "#f2a541";
      var x = v.offsetX + d.bbox[0] * v.zoom;
      var y = v.offsetY + d.bbox[1] * v.zoom;
      var w = (d.bbox[2] - d.bbox[0]) * v.zoom;
      var h = (d.bbox[3] - d.bbox[1]) * v.zoom;
      var sel = d.id === state.selectedId;

      if (sel) {
        ctx.save();
        ctx.shadowColor = color; ctx.shadowBlur = 12;
      }
      ctx.lineWidth = sel ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      if (sel) ctx.restore();

      // label
      var label = d["class"] + "  " + Math.round(d.confidence * 100) + "%";
      ctx.font = "600 12px " + "-apple-system, Segoe UI, Roboto, sans-serif";
      var tw = ctx.measureText(label).width;
      var lh = 17, ly = y - lh;
      if (ly < 0) ly = y + 1; // flip inside if it would clip off the top
      ctx.fillStyle = color;
      ctx.fillRect(x, ly, tw + 12, lh);
      ctx.fillStyle = "#12100a";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + 6, ly + lh / 2 + 0.5);
    }
  }

  function updateHud() {
    els.hudZoom.textContent = Math.round(state.view.zoom * 100) + "%";
  }

  // ---- pan / zoom events --------------------------------------------------
  function screenToImage(cx, cy) {
    return {
      x: (cx - state.view.offsetX) / state.view.zoom,
      y: (cy - state.view.offsetY) / state.view.zoom
    };
  }

  function zoomAt(cx, cy, factor) {
    if (!state.image) return;
    var v = state.view;
    var before = screenToImage(cx, cy);
    var z = v.zoom * factor;
    if (z < v.minZoom) z = v.minZoom;
    if (z > MAX_ZOOM) z = MAX_ZOOM;
    v.zoom = z;
    v.offsetX = cx - before.x * z;
    v.offsetY = cy - before.y * z;
    updateHud(); draw();
  }

  function wireViewer() {
    var cv = els.view;

    cv.addEventListener("wheel", function (e) {
      if (!state.image) return;
      e.preventDefault();
      var r = els.stage.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.12 : 1 / 1.12);
    }, { passive: false });

    var dragging = false, moved = false, lastX = 0, lastY = 0, downX = 0, downY = 0;
    cv.addEventListener("mousedown", function (e) {
      if (!state.image) return;
      dragging = true; moved = false;
      lastX = downX = e.clientX; lastY = downY = e.clientY;
      cv.classList.add("panning");
    });
    window.addEventListener("mousemove", function (e) {
      // cursor position readout
      if (state.image) {
        var r = els.stage.getBoundingClientRect();
        if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
          var p = screenToImage(e.clientX - r.left, e.clientY - r.top);
          if (p.x >= 0 && p.y >= 0 && p.x <= state.imgW && p.y <= state.imgH)
            els.hudPos.textContent = (p.x | 0) + ", " + (p.y | 0) + " px";
        }
      }
      if (!dragging) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 3) moved = true;
      state.view.offsetX += dx; state.view.offsetY += dy;
      lastX = e.clientX; lastY = e.clientY;
      draw();
    });
    window.addEventListener("mouseup", function (e) {
      if (dragging && !moved) handleStageClick(e);
      dragging = false; cv.classList.remove("panning");
    });

    els.zoomIn.addEventListener("click", function () { var s = stageSize(); zoomAt(s.w / 2, s.h / 2, 1.25); });
    els.zoomOut.addEventListener("click", function () { var s = stageSize(); zoomAt(s.w / 2, s.h / 2, 1 / 1.25); });
    els.zoomFit.addEventListener("click", function () { fitView(); draw(); });

    window.addEventListener("resize", function () { resizeCanvas(); draw(); });

    // keyboard: + / - zoom, 0 fit, b toggle boxes (ignored while typing)
    window.addEventListener("keydown", function (e) {
      var tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target && e.target.isContentEditable)) return;
      var s = stageSize();
      switch (e.key) {
        case "+": case "=": if (state.image) zoomAt(s.w / 2, s.h / 2, 1.25); break;
        case "-": case "_": if (state.image) zoomAt(s.w / 2, s.h / 2, 1 / 1.25); break;
        case "0": if (state.image) { fitView(); draw(); } break;
        case "b": case "B":
          state.showBoxes = !state.showBoxes;
          els.showBoxes.checked = state.showBoxes;
          draw();
          break;
        default: return;
      }
      e.preventDefault();
    });
  }

  // click on a box selects the matching detection
  function handleStageClick(e) {
    var r = els.stage.getBoundingClientRect();
    var p = screenToImage(e.clientX - r.left, e.clientY - r.top);
    var hit = null;
    var dets = filteredDetections();
    for (var i = 0; i < dets.length; i++) {
      var d = dets[i], bb = d.bbox;
      if (p.x >= bb[0] && p.x <= bb[2] && p.y >= bb[1] && p.y <= bb[3]) {
        if (!hit || (bb[2] - bb[0]) * (bb[3] - bb[1]) < (hit.bbox[2] - hit.bbox[0]) * (hit.bbox[3] - hit.bbox[1]))
          hit = d; // prefer the smallest box under the cursor
      }
    }
    selectDetection(hit ? hit.id : null);
  }

  // =========================================================================
  //  Image loading + detection
  // =========================================================================
  function loadImageFile(file) {
    if (!file) return;
    state.lastFile = file;
    var url = URL.createObjectURL(file);
    var im = new Image();
    im.onload = function () {
      URL.revokeObjectURL(url);
      setImage(im);
      runDetection(file);
    };
    im.onerror = function () {
      URL.revokeObjectURL(url);
      setAdminNote("");
      alert("Could not open that image.");
    };
    im.src = url;
  }

  function setImage(im) {
    state.image = im;
    state.imgW = im.naturalWidth;
    state.imgH = im.naturalHeight;

    var longest = Math.max(state.imgW, state.imgH);
    var ws = longest > MAX_WORK_SIDE ? MAX_WORK_SIDE / longest : 1;
    state.workW = Math.max(1, Math.round(state.imgW * ws));
    state.workH = Math.max(1, Math.round(state.imgH * ws));

    var tc = document.createElement("canvas");
    tc.width = state.workW; tc.height = state.workH;
    var tctx = tc.getContext("2d");
    tctx.drawImage(im, 0, 0, state.workW, state.workH);
    state.baseImageData = tctx.getImageData(0, 0, state.workW, state.workH);

    state.enhancedCanvas = document.createElement("canvas");
    state.enhancedCanvas.width = state.workW;
    state.enhancedCanvas.height = state.workH;

    els.stageEmpty.hidden = true;
    [els.zoomIn, els.zoomOut, els.zoomFit, els.detectBtn, els.exportPngBtn].forEach(function (b) { b.disabled = false; });

    resizeCanvas();
    fitView();
    renderEnhanced(); // draws
  }

  function setBusy(on) {
    state.busy = on;
    els.stageBusy.hidden = !on;
    els.detectBtn.disabled = on || !state.image;
  }

  function runDetection(file) {
    if (!file) file = state.lastFile;
    if (!file) return;
    setBusy(true);
    var fd = new FormData();
    fd.append("image", file);
    fd.append("conf", String(state.serverFloor));
    fetch("/api/detect", { method: "POST", body: fd })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        var j = res.j;
        if (!j || !j.ok) {
          showDetError((j && j.error) || "Detection failed.");
          return;
        }
        state.detections = j.detections || [];
        state.lastResult = j;
        // ensure any newly-seen classes are visible + coloured
        state.detections.forEach(function (d) {
          if (state.classVisible[d["class"]] === undefined) state.classVisible[d["class"]] = true;
          if (!state.classColors[d["class"]]) assignColorForClass(d["class"], d.class_id);
        });
        updateTiming(j.processing_time_ms);
        renderDetList();
        renderClassLegend();
        els.exportJsonBtn.disabled = false;
        draw();
      })
      .catch(function () { showDetError("Request failed — is the server reachable?"); })
      .finally(function () { setBusy(false); });
  }

  function showDetError(msg) {
    els.detEmpty.hidden = false;
    els.detEmpty.textContent = msg;
    els.detList.innerHTML = "";
    els.detCount.textContent = "0";
    els.detTiming.hidden = true;
  }

  function updateTiming(t) {
    if (!t) { els.detTiming.hidden = true; return; }
    els.detTiming.hidden = false;
    els.tInf.textContent = t.inference.toFixed(1) + " ms";
    els.tPrePost.textContent = t.preprocess.toFixed(1) + " + " + t.postprocess.toFixed(1) + " ms";
    els.tTotal.textContent = t.total.toFixed(1) + " ms";
  }

  // =========================================================================
  //  Detection list + class legend
  // =========================================================================
  function assignColorForClass(name, classId) {
    var idx = (typeof classId === "number" ? classId : Object.keys(state.classColors).length);
    state.classColors[name] = CLASS_PALETTE[idx % CLASS_PALETTE.length];
  }

  function renderDetList() {
    var dets = filteredDetections();
    els.detCount.textContent = String(dets.length);
    els.detCount.classList.toggle("hot", dets.length > 0);

    els.detList.innerHTML = "";
    if (state.detections.length === 0) {
      els.detEmpty.hidden = false;
      els.detEmpty.textContent = "No objects detected.";
      return;
    }
    if (dets.length === 0) {
      els.detEmpty.hidden = false;
      els.detEmpty.textContent = "All detections are below the confidence threshold.";
      return;
    }
    els.detEmpty.hidden = true;

    dets.forEach(function (d) {
      var li = document.createElement("li");
      li.className = "det-row" + (d.id === state.selectedId ? " selected" : "");
      li.dataset.id = d.id;
      var color = state.classColors[d["class"]] || "#f2a541";
      var pct = Math.round(d.confidence * 100);
      li.innerHTML =
        '<span class="det-swatch" style="background:' + color + '"></span>' +
        '<span class="det-name">' + escapeHtml(d["class"]) + '</span>' +
        '<span class="det-conf">' + pct + '%</span>' +
        '<span class="det-bar"><i style="width:' + pct + '%;background:' + color + '"></i></span>';
      li.addEventListener("click", function () { selectDetection(d.id); });
      els.detList.appendChild(li);
    });
  }

  function renderClassLegend() {
    if (!state.modelClasses.length) { els.classSection.hidden = true; return; }
    els.classSection.hidden = false;
    var counts = (state.lastResult && state.lastResult.counts) || {};
    els.classLegend.innerHTML = "";
    state.modelClasses.forEach(function (name, idx) {
      if (!state.classColors[name]) assignColorForClass(name, idx);
      var on = state.classVisible[name] !== false;
      var chip = document.createElement("button");
      chip.className = "class-chip " + (on ? "on" : "off");
      var c = counts[name];
      chip.innerHTML =
        '<span class="cc-dot" style="background:' + state.classColors[name] + '"></span>' +
        '<span>' + escapeHtml(name) + '</span>' +
        (c ? '<span class="cc-count">' + c + '</span>' : '');
      chip.addEventListener("click", function () {
        state.classVisible[name] = !(state.classVisible[name] !== false);
        renderClassLegend(); renderDetList(); draw();
      });
      els.classLegend.appendChild(chip);
    });
  }

  function selectDetection(id) {
    state.selectedId = (state.selectedId === id) ? null : id;
    renderDetList();
    draw();
    if (state.selectedId !== null) {
      var row = els.detList.querySelector('[data-id="' + state.selectedId + '"]');
      if (row && row.scrollIntoView) row.scrollIntoView({ block: "nearest" });
    }
  }

  // =========================================================================
  //  Export
  // =========================================================================
  function download(blob, filename) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 100);
  }

  function exportPng() {
    if (!state.image) return;
    var cv = document.createElement("canvas");
    cv.width = state.imgW; cv.height = state.imgH;
    var ctx = cv.getContext("2d");
    ctx.drawImage(state.enhancedCanvas, 0, 0, state.workW, state.workH, 0, 0, state.imgW, state.imgH);

    if (state.showBoxes) {
      var lw = Math.max(2, Math.round(state.imgW / 600));
      var fs = Math.max(12, Math.round(state.imgW / 90));
      ctx.font = "600 " + fs + "px -apple-system, Segoe UI, Roboto, sans-serif";
      ctx.textBaseline = "middle";
      filteredDetections().forEach(function (d) {
        var color = state.classColors[d["class"]] || "#f2a541";
        var x = d.bbox[0], y = d.bbox[1], w = d.bbox[2] - d.bbox[0], h = d.bbox[3] - d.bbox[1];
        ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.strokeRect(x, y, w, h);
        var label = d["class"] + "  " + Math.round(d.confidence * 100) + "%";
        var tw = ctx.measureText(label).width;
        var lh = fs + 8, ly = y - lh; if (ly < 0) ly = y;
        ctx.fillStyle = color; ctx.fillRect(x, ly, tw + 12, lh);
        ctx.fillStyle = "#12100a"; ctx.fillText(label, x + 6, ly + lh / 2);
      });
    }
    cv.toBlob(function (b) { download(b, "xod-annotated.png"); }, "image/png");
  }

  function exportJson() {
    if (!state.lastResult) return;
    var payload = JSON.stringify(state.lastResult, null, 2);
    download(new Blob([payload], { type: "application/json" }), "xod-detections.json");
  }

  // =========================================================================
  //  Model admin upload
  // =========================================================================
  function setAdminStatus(msg, cls) {
    els.adminStatus.textContent = msg || "";
    els.adminStatus.className = "admin-status" + (cls ? " " + cls : "");
  }
  function setAdminNote() { /* reserved */ }

  function uploadModel() {
    var file = els.modelInput.files && els.modelInput.files[0];
    if (!file) { setAdminStatus("Choose a .pt file first.", "err"); return; }
    var fd = new FormData();
    fd.append("model", file);
    if (els.adminKey && els.adminKey.value) fd.append("admin_key", els.adminKey.value);

    els.uploadModelBtn.disabled = true;
    setAdminStatus("Uploading & validating…", "working");
    fetch("/api/model", { method: "POST", body: fd })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        var j = res.j;
        if (!j || !j.ok) { setAdminStatus((j && j.error) || "Upload failed.", "err"); return; }
        setAdminStatus("Active model: " + j.model.name + " (" + j.model.num_classes + " classes)", "ok");
        applyModelInfo(j.model, true);
      })
      .catch(function () { setAdminStatus("Upload failed — file may be too large for the server buffer; try SFTP.", "err"); })
      .finally(function () { els.uploadModelBtn.disabled = false; });
  }

  // =========================================================================
  //  Status / model info
  // =========================================================================
  function applyModelInfo(model, loaded) {
    state.modelClasses = model.classes || [];
    state.modelClasses.forEach(function (name, idx) { if (!state.classColors[name]) assignColorForClass(name, idx); });
    els.modelChip.classList.toggle("loaded", !!loaded);
    els.modelChip.classList.toggle("error", !loaded);
    els.modelName.textContent = model.name || (loaded ? "model" : "no model");
    if (state.modelClasses.length) els.modelClasses.textContent = state.modelClasses.length + " classes";
    else els.modelClasses.textContent = "—";
    renderClassLegend();
  }

  function loadStatus() {
    fetch("/api/status").then(function (r) { return r.json(); }).then(function (j) {
      state.serverFloor = (j.defaults && j.defaults.conf_floor) != null ? j.defaults.conf_floor : 0.10;
      var conf = (j.defaults && j.defaults.conf) != null ? j.defaults.conf : 0.25;
      state.confThreshold = conf;
      els.confSlider.value = String(Math.round(conf * 100));
      els.confVal.textContent = Math.round(conf * 100) + "%";

      if (j.admin_locked) els.adminKeyRow.hidden = false;

      if (j.model && j.model.loaded) {
        applyModelInfo({ name: j.model.name, classes: j.model.classes, num_classes: j.model.num_classes }, true);
      } else {
        els.modelChip.classList.add("error");
        els.modelName.textContent = "no model loaded";
        els.modelClasses.textContent = "—";
        if (j.model && j.model.error) els.modelChip.title = j.model.error;
      }
    }).catch(function () {
      els.modelChip.classList.add("error");
      els.modelName.textContent = "status unavailable";
    });
  }

  // =========================================================================
  //  Wiring
  // =========================================================================
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function wireControls() {
    // upload / detect / export
    els.imageInput.addEventListener("change", function (e) {
      if (e.target.files && e.target.files[0]) loadImageFile(e.target.files[0]);
    });
    els.detectBtn.addEventListener("click", function () { runDetection(state.lastFile); });
    els.exportPngBtn.addEventListener("click", exportPng);
    els.exportJsonBtn.addEventListener("click", exportJson);

    // drag & drop onto the stage
    ["dragenter", "dragover"].forEach(function (ev) {
      els.stage.addEventListener(ev, function (e) { e.preventDefault(); els.stage.classList.add("dragover"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      els.stage.addEventListener(ev, function (e) { e.preventDefault(); if (ev === "dragleave" && e.target !== els.stage) return; els.stage.classList.remove("dragover"); });
    });
    els.stage.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) loadImageFile(f);
    });

    // detection controls
    els.showBoxes.addEventListener("change", function () { state.showBoxes = els.showBoxes.checked; draw(); });
    els.confSlider.addEventListener("input", function () {
      state.confThreshold = parseInt(els.confSlider.value, 10) / 100;
      els.confVal.textContent = els.confSlider.value + "%";
      renderDetList(); renderClassLegend(); draw();
    });
    els.toggleAllClasses.addEventListener("click", function () {
      var anyOn = state.modelClasses.some(function (n) { return state.classVisible[n] !== false; });
      state.modelClasses.forEach(function (n) { state.classVisible[n] = !anyOn; });
      renderClassLegend(); renderDetList(); draw();
    });

    // enhancement sliders
    els.brightness.addEventListener("input", function () {
      state.enh.brightness = parseInt(els.brightness.value, 10);
      els.brightnessVal.textContent = els.brightness.value;
      scheduleEnhance();
    });
    els.contrast.addEventListener("input", function () {
      state.enh.contrast = parseInt(els.contrast.value, 10);
      els.contrastVal.textContent = els.contrast.value;
      scheduleEnhance();
    });
    els.gamma.addEventListener("input", function () {
      state.enh.gamma = parseInt(els.gamma.value, 10) / 100;
      els.gammaVal.textContent = state.enh.gamma.toFixed(2);
      scheduleEnhance();
    });

    // enhancement toggles
    document.querySelectorAll(".chip[data-enh]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var key = btn.dataset.enh;
        state.enh[key] = !state.enh[key];
        btn.classList.toggle("active", state.enh[key]);
        // edges and sharpen are mutually exclusive in the pipeline
        if (key === "edges" && state.enh.edges && state.enh.sharpen) {
          state.enh.sharpen = false;
          var sh = document.querySelector('.chip[data-enh="sharpen"]'); if (sh) sh.classList.remove("active");
        }
        if (key === "sharpen" && state.enh.sharpen && state.enh.edges) {
          state.enh.edges = false;
          var ed = document.querySelector('.chip[data-enh="edges"]'); if (ed) ed.classList.remove("active");
        }
        renderEnhanced();
      });
    });

    // palette swatches
    els.paletteSwatches.querySelectorAll(".swatch").forEach(function (sw) {
      sw.addEventListener("click", function () {
        els.paletteSwatches.querySelectorAll(".swatch").forEach(function (s) { s.classList.remove("active"); });
        sw.classList.add("active");
        state.enh.palette = sw.dataset.palette;
        renderEnhanced();
      });
    });

    // reset enhancement
    els.resetEnh.addEventListener("click", function () {
      state.enh = { brightness: 0, contrast: 0, gamma: 1.0, invert: false, sharpen: false, edges: false, equalize: false, palette: "gray" };
      els.brightness.value = 0; els.brightnessVal.textContent = "0";
      els.contrast.value = 0; els.contrastVal.textContent = "0";
      els.gamma.value = 100; els.gammaVal.textContent = "1.00";
      document.querySelectorAll(".chip[data-enh]").forEach(function (b) { b.classList.remove("active"); });
      els.paletteSwatches.querySelectorAll(".swatch").forEach(function (s) { s.classList.toggle("active", s.dataset.palette === "gray"); });
      renderEnhanced();
    });

    // model admin
    els.modelInput.addEventListener("change", function () {
      var f = els.modelInput.files && els.modelInput.files[0];
      els.modelFileName.textContent = f ? f.name : "no file";
      els.uploadModelBtn.disabled = !f;
    });
    els.uploadModelBtn.addEventListener("click", uploadModel);
  }

  // ---- init ---------------------------------------------------------------
  function init() {
    resizeCanvas();
    drawHistogram(null);
    wireViewer();
    wireControls();
    loadStatus();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
