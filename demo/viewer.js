/*
 * CAM3R project page — live point-cloud viewers.
 *
 * three.js r164.1, pinned, loaded from jsDelivr via the import map in index.html.
 *
 * Design notes
 * ------------
 * A WebGL context is a scarce resource: browsers keep roughly 8-16 alive per
 * page and silently drop the oldest ones beyond that.  Rather than instantiate
 * one renderer per example (which would need 33), each results section owns a
 * single viewer rig and swaps the point clouds when a tab is selected.  The
 * page therefore holds seven contexts in total: two for the two-view
 * comparison, one for the zero-shot example and four for the multi-view
 * comparison.  It also means a visitor downloads one set of clouds at a time
 * instead of every example up front.
 *
 * Both comparisons use side-by-side canvases driven by one master camera: the
 * two-view section with two panels, the multi-view section with four.  Every
 * panel carries its own OrbitControls bound to that same camera and shares the
 * one target object, so a drag, zoom or pan started in any panel moves all of
 * them; the slave cameras copy the master pose every frame while keeping their
 * own aspect ratio.  This is more reliable than a single-canvas toggle because
 * the methods stay visible simultaneously, and it cannot desynchronise: there
 * is only ever one source of truth for the pose.
 *
 * Clouds are normalised at render time (bounding sphere recentred, radius
 * scaled to 1) so that a prediction recovered up to an unknown global scale
 * can be compared with metric ground truth in the same frame.  The .ply files
 * on disk keep their original coordinates.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const DEFAULT_POINT_SIZE = 0.010;

/* The four panels of the multi-view comparison, in display order: the method
   key used to build the .ply path, and the label used while loading. */
const MULTIVIEW_METHODS = [
  ['gt', 'ground truth'],
  ['vggt', 'VGGT'],
  ['pi3', '\u03c0\u00b3'],
  ['cam3r', 'CAM3R']
];

/* ------------------------------------------------------------------ panel */

function makePanel(el) {
  const canvas = el.querySelector('canvas');
  const overlay = el.querySelector('.overlay');
  const overlayText = overlay.querySelector('.msg');

  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, powerPreference: 'high-performance'
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e13);

  const camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.01, 200);
  camera.up.set(0, -1, 0);          // pointmaps are in camera coords: +y is down
  camera.position.set(0, 0, -3);

  return {
    el, canvas, renderer, scene, camera, overlay, overlayText,
    points: null,
    status(kind, text) {
      if (kind === 'hidden') { overlay.hidden = true; return; }
      overlay.hidden = false;
      overlay.classList.toggle('error', kind === 'error');
      overlayText.textContent = text;
    },
    setCloud(geometry, size) {
      this.clear();
      geometry.computeBoundingSphere();
      const bs = geometry.boundingSphere;
      const r = bs && isFinite(bs.radius) && bs.radius > 1e-6 ? bs.radius : 1;
      const material = new THREE.PointsMaterial({
        size: size, vertexColors: true, sizeAttenuation: true
      });
      if (!geometry.hasAttribute('color')) {
        material.vertexColors = false;
        material.color = new THREE.Color(0xbfc9d4);
      }
      const pts = new THREE.Points(geometry, material);
      // normalise: centre on the bounding sphere, scale its radius to 1
      pts.scale.setScalar(1 / r);
      pts.position.copy(bs.center).multiplyScalar(-1 / r);
      this.points = pts;
      this.scene.add(pts);
    },
    clear() {
      if (!this.points) return;
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    },
    resize() {
      const w = Math.max(1, this.el.clientWidth);
      const h = Math.max(1, this.el.clientHeight);
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      return { w, h };
    }
  };
}

/* ------------------------------------------------------------------- rig */

/**
 * A viewer rig owns one or more panels, a shared master camera and a render
 * loop that only runs while the rig is on screen.
 */
class Rig {
  constructor(root) {
    this.root = root;
    this.panels = Array.from(root.querySelectorAll('.panel')).map(makePanel);
    this.master = this.panels[0].camera;
    this.pointSize = DEFAULT_POINT_SIZE;
    this.loader = new PLYLoader();
    this.token = 0;
    this.visible = false;
    this.running = false;
    this.pending = null;

    this.controls = this.panels.map((p) => {
      const c = new OrbitControls(this.master, p.canvas);
      c.enableDamping = false;      // several controls on one camera must not fight
      c.enablePan = true;
      c.screenSpacePanning = true;
      c.rotateSpeed = 0.9;
      c.zoomSpeed = 0.9;
      c.minDistance = 0.15;
      c.maxDistance = 40;
      return c;
    });
    // every control shares one target object, so panning stays in lockstep
    for (let i = 1; i < this.controls.length; i++) {
      this.controls[i].target = this.controls[0].target;
    }

    this.home();

    const ro = new ResizeObserver(() => this.resize());
    ro.observe(root);
    window.addEventListener('resize', () => this.resize());

    const io = new IntersectionObserver((entries) => {
      this.visible = entries.some((e) => e.isIntersecting);
      if (!this.visible) return;
      this.start();
      if (this.pending) {
        const q = this.pending;
        this.pending = null;
        this.show(q.urls, q.labels);
      }
    }, { rootMargin: '200px' });
    io.observe(root);

    this.resize();
    this.render();
  }

  home() {
    this.master.position.set(0, 0, -2.9);
    this.master.up.set(0, -1, 0);
    this.controls[0].target.set(0, 0, 0);
    this.master.lookAt(0, 0, 0);
    this.controls.forEach((c) => c.update());
  }

  setPointSize(v) {
    this.pointSize = v;
    this.panels.forEach((p) => { if (p.points) p.points.material.size = v; });
  }

  /**
   * Ask for a set of clouds. Nothing is fetched until the rig scrolls into
   * view, so opening the page does not pull every example at once.
   */
  request(urls, labels) {
    if (this.visible) return this.show(urls, labels);
    this.token++;
    this.pending = { urls, labels };
    this.panels.forEach((p, i) => {
      p.clear();
      p.status('loading', 'Loading ' + labels[i] + '\u2026');
    });
    return Promise.resolve([]);
  }

  /** Load a set of urls (one per panel) and reframe once they are in. */
  async show(urls, labels) {
    this.token++;
    const mine = this.token;
    const results = await Promise.all(
      urls.map((u, i) => new Promise((resolve) => {
        const panel = this.panels[i];
        panel.clear();
        panel.status('loading', 'Loading ' + labels[i] + '…');
        this.loader.load(
          u,
          (geometry) => {
            if (mine !== this.token) { geometry.dispose(); resolve(false); return; }
            try {
              panel.setCloud(geometry, this.pointSize);
              panel.status('hidden');
              resolve(true);
            } catch (err) {
              console.error('[cam3r] build failed: ' + u, err);
              panel.status('error', 'Could not display this point cloud.');
              resolve(false);
            }
          },
          (xhr) => {
            if (mine !== this.token) return;
            if (xhr && xhr.lengthComputable && xhr.total) {
              const pct = Math.round((xhr.loaded / xhr.total) * 100);
              panel.status('loading', 'Loading ' + labels[i] + '… ' + pct + '%');
            }
          },
          (err) => {
            console.error('[cam3r] load failed: ' + u, err);
            if (mine !== this.token) { resolve(false); return; }
            panel.status('error',
              'Could not load this point cloud (' + u.split('/').pop() +
              '). Try reloading the page.');
            resolve(false);
          }
        );
      }))
    );
    if (mine === this.token && results.some(Boolean)) this.home();
    return results;
  }

  resize() {
    this.panels.forEach((p, i) => {
      const { w, h } = p.resize();
      if (i === 0) {
        this.master.aspect = w / h;
        this.master.updateProjectionMatrix();
      }
    });
  }

  start() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      requestAnimationFrame(tick);
      if (!this.visible) return;
      this.controls.forEach((c) => c.update());
      this.render();
    };
    requestAnimationFrame(tick);
  }

  render() {
    this.panels.forEach((p, i) => {
      if (i > 0) {
        // slave cameras track the master pose but keep their own aspect
        p.camera.position.copy(this.master.position);
        p.camera.quaternion.copy(this.master.quaternion);
        p.camera.up.copy(this.master.up);
        p.camera.zoom = this.master.zoom;
        p.camera.updateProjectionMatrix();
      }
      p.renderer.render(p.scene, p.camera);
    });
  }
}

/* ------------------------------------------------------- section wiring */

function wireSection(rootId, examples, urlsFor, labelsFor, onSelect) {
  const root = document.getElementById(rootId);
  if (!root) return null;

  const stage = root.querySelector('.stage');
  let rig;
  try {
    rig = new Rig(stage);
  } catch (err) {
    // one section failing (no context left, driver bug) must not take the
    // other one down with it
    console.error('[cam3r] could not start the viewer in #' + rootId, err);
    root.querySelectorAll('.panel .overlay').forEach((o) => {
      o.hidden = false;
      o.classList.add('error');
      o.querySelector('.msg').textContent =
        'The live viewer could not start in this browser. The figures above ' +
        'show the same reconstructions.';
    });
    return null;
  }

  const sizeInput = root.querySelector('input[type="range"]');
  if (sizeInput) {
    sizeInput.value = String(DEFAULT_POINT_SIZE);
    sizeInput.addEventListener('input', () => rig.setPointSize(parseFloat(sizeInput.value)));
  }
  const resetBtn = root.querySelector('button[data-action="reset"]');
  if (resetBtn) resetBtn.addEventListener('click', () => rig.home());

  wireTabs(root, (i) => {
    onSelect(root, examples[i]);
    rig.request(urlsFor(examples[i]), labelsFor(examples[i]));
  });
  return rig;
}

/**
 * Wire a .tabs strip inside root: click and left/right arrow selection, with
 * aria-selected kept in sync and the first tab chosen on load.
 */
function wireTabs(root, onIndex) {
  const tabs = Array.from(root.querySelectorAll('.tabs button'));
  let current = -1;

  function select(i) {
    if (i === current) return;
    current = i;
    tabs.forEach((b, j) => b.setAttribute('aria-selected', String(j === i)));
    onIndex(i);
  }

  tabs.forEach((b, i) => {
    b.addEventListener('click', () => select(i));
    b.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const n = tabs.length;
      const j = (i + (e.key === 'ArrowRight' ? 1 : n - 1)) % n;
      tabs[j].focus();
      select(j);
    });
  });

  select(0);
}

function checkWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext &&
      (c.getContext('webgl2') || c.getContext('webgl')));
  } catch (e) {
    return false;
  }
}

/* ------------------------------------------------------------------ data */

export function boot(TWOVIEW, ZEROSHOT, MULTIVIEW) {
  const guard = (fn) => { try { fn(); } catch (err) { console.error('[cam3r]', err); } };

  if (!checkWebGL()) {
    document.querySelectorAll('.panel .overlay').forEach((o) => {
      o.hidden = false;
      o.classList.add('error');
      o.querySelector('.msg').textContent =
        'This browser does not have WebGL available, so the live viewers ' +
        'cannot run. The figures above still show the same reconstructions.';
    });
    return;
  }

  guard(() => wireSection(
    'twoview-widget', TWOVIEW,
    (ex) => [ex.pred, ex.gt],
    () => ['prediction', 'ground truth'],
    (root, ex) => {
      root.querySelector('[data-slot="name"]').textContent = ex.name;
      root.querySelector('[data-slot="src"]').textContent = ex.source;
      const img = root.querySelector('[data-slot="strip"]');
      img.src = ex.strip;
      img.alt = ex.name + ': input views with mutual-nearest-neighbour matches and predicted confidence maps';
      const m = root.querySelector('[data-slot="metrics"]');
      m.innerHTML = '';
      ex.metrics.forEach(([k, v]) => {
        const d = document.createElement('div');
        d.className = 'metric';
        d.innerHTML = '<span class="k"></span><span class="v"></span>';
        d.querySelector('.k').textContent = k;
        d.querySelector('.v').textContent = v;
        m.appendChild(d);
      });
    }
  ));

  guard(() => wireSection(
    'zeroshot-widget', ZEROSHOT,
    (ex) => [ex.pred],
    () => ['prediction'],
    (root, ex) => {
      root.querySelector('[data-slot="name"]').textContent = ex.name;
      root.querySelector('[data-slot="src"]').textContent = ex.source;
      const i0 = root.querySelector('[data-slot="view0"]');
      const i1 = root.querySelector('[data-slot="view1"]');
      i0.src = ex.view0; i0.alt = ex.name + ', first input view';
      i1.src = ex.view1; i1.alt = ex.name + ', second input view';
      const mm = root.querySelector('[data-slot="matches"]');
      mm.src = ex.matches;
      mm.alt = ex.name + ': mutual-nearest-neighbour match quality for the pair';
    }
  ));

  // Four panels, one per method, all driven by the rig's single master camera.
  guard(() => wireSection(
    'multiview-widget', MULTIVIEW,
    (ex) => MULTIVIEW_METHODS.map(
      ([key]) => 'assets/multiview_clouds/' + ex.key + '_' + key + '.ply'),
    () => MULTIVIEW_METHODS.map(([, label]) => label),
    (root, ex) => {
      root.querySelector('[data-slot="name"]').textContent = ex.name;
      root.querySelector('[data-slot="src"]').textContent = ex.source;
      const note = root.querySelector('[data-slot="views"]');
      note.textContent = ex.views || '';
      note.hidden = !ex.views;
    }
  ));
}
