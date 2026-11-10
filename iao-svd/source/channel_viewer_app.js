import * as THREE from "__THREE_URL__";
import { OrbitControls } from "__ORBIT_URL__";

const B = window.__BUNDLE__;
const PAIRS = B.pairs;

// Palette, lighting and material response are taken from the accepted v2/v3
// q214 viewers so this instrument reads as the same family of figures.
const THEMES = {
  aurora: { label: "极光", atoms: { Ru: 0x315a74, O: 0xef784c, H: 0xf7fbff, other: 0xb8c4ce }, target: 0xf4c659, targetEmissive: 0x593700, bond: 0x7891a5, cell: 0x91abc0, negative: 0xa855e6, positive: 0x20b99b, hemi: [0xffffff, 0xb8c8d8, 2.1], key: [0xffffff, 2.35], fill: [0xbfeef0, 1.1], rim: [0xaac7ff, 0.85], exposure: 1.12, atomRoughness: 0.28, atomMetalness: 0.1, surfaceRoughness: 0.18, clearcoat: 0.92, surfaceGlow: 0.055, page: "linear-gradient(160deg,#eef5fb 0%,#dce9f6 55%,#cfe0f2 100%)" },
  porcelain: { label: "瓷白", atoms: { Ru: 0x38658f, O: 0xd96b4e, H: 0xf9fbfc, other: 0xb9c4ce }, target: 0xd6aa51, targetEmissive: 0x4c3408, bond: 0x8496a5, cell: 0x9eb1c0, negative: 0x8158d8, positive: 0x27a98e, hemi: [0xffffff, 0xc4d2df, 1.95], key: [0xfffdf8, 2.25], fill: [0xc7e1f7, 0.95], rim: [0x8bb7df, 0.72], exposure: 1.08, atomRoughness: 0.34, atomMetalness: 0.08, surfaceRoughness: 0.2, clearcoat: 0.88, surfaceGlow: 0.05, page: "linear-gradient(160deg,#f7fafc 0%,#e8eef4 60%,#dde6ee 100%)" },
  celadon: { label: "青瓷", atoms: { Ru: 0x416d68, O: 0xce725d, H: 0xf8f7f2, other: 0xb6c4bd }, target: 0xd0a75d, targetEmissive: 0x4a340c, bond: 0x7f9188, cell: 0x9db4a8, negative: 0x8a5bd5, positive: 0x2d9ed6, hemi: [0xfcfff9, 0xbccdc3, 1.9], key: [0xfff8ea, 2.15], fill: [0xbfe2d4, 0.9], rim: [0x9ecfe7, 0.8], exposure: 1.06, atomRoughness: 0.38, atomMetalness: 0.05, surfaceRoughness: 0.23, clearcoat: 0.72, surfaceGlow: 0.045, page: "linear-gradient(160deg,#f2f8f4 0%,#e2eee7 60%,#d6e7dd 100%)" },
  prism: { label: "棱镜", atoms: { Ru: 0x4052a8, O: 0xeda15a, H: 0xf7f3ff, other: 0xb5b6d6 }, target: 0x60cfd4, targetEmissive: 0x0e4345, bond: 0x8e91b7, cell: 0xa9aade, negative: 0xdd4fa7, positive: 0x36bde9, hemi: [0xfbf5ff, 0xc4c8eb, 2.15], key: [0xffffff, 2.4], fill: [0xb8fff4, 1.25], rim: [0xe6b9ff, 1.2], exposure: 1.15, atomRoughness: 0.22, atomMetalness: 0.14, surfaceRoughness: 0.14, clearcoat: 1, surfaceGlow: 0.082, page: "linear-gradient(160deg,#f4f2fd 0%,#e6e6f8 60%,#dcdcf4 100%)" },
  dawn: { label: "晨曦", atoms: { Ru: 0x4d6175, O: 0xdc7751, H: 0xfffaf5, other: 0xbfc1c2 }, target: 0xe1b75f, targetEmissive: 0x55380a, bond: 0x8e8e8a, cell: 0xb2aaa4, negative: 0x9856d6, positive: 0x2fa6a7, hemi: [0xfffbf4, 0xc7cbd0, 1.95], key: [0xffe3c7, 2.2], fill: [0xbfe1e2, 0.9], rim: [0xf0b28f, 0.85], exposure: 1.08, atomRoughness: 0.32, atomMetalness: 0.07, surfaceRoughness: 0.21, clearcoat: 0.82, surfaceGlow: 0.052, page: "linear-gradient(160deg,#fdf6ef 0%,#f2e9e1 60%,#e9ded4 100%)" },
};
const RADIUS = { Ru: 0.58, O: 0.34, H: 0.22, X: 0.3 };
const FIELD_LABEL = {
  bond: "成键轨道 φ+", anti: "反键轨道 φ−",
  pa: "A 伙伴振幅", pb: "B 伙伴振幅", rho: "相干交叉项",
};

const state = {
  pair: 0, channel: 1, field: "bond", theme: "aurora",
  isoRel: 0.30,
  atomSize: 0.55, opacity: 0.72, showAll: false, showSum: false,
  atoms: true, bonds: true, cell: true, mode: "rotate", framePlaying: false,
};

// ---------------------------------------------------------------- shell
document.documentElement.style.height = "100%";
document.body.style.cssText =
  "margin:0;height:100%;font:13px/1.55 'Segoe UI',system-ui,sans-serif;color:#243447;overflow:hidden";
document.body.innerHTML = `
<div id="stage" style="position:fixed;inset:0"></div>
<div id="side" style="position:fixed;left:18px;top:18px;bottom:18px;width:372px;overflow-y:auto;
     padding:18px 20px;box-sizing:border-box;background:rgba(255,255,255,.86);
     backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.9);border-radius:16px;
     box-shadow:0 18px 48px rgba(40,70,110,.16)"></div>
<div id="themes" style="position:fixed;left:50%;top:18px;transform:translateX(-50%);display:flex;gap:4px;
     padding:5px;background:rgba(255,255,255,.84);backdrop-filter:blur(14px);border-radius:999px;
     box-shadow:0 8px 24px rgba(40,70,110,.14)"></div>
<div id="nav" style="position:fixed;right:18px;bottom:18px;display:flex;gap:5px;padding:5px;
     background:rgba(255,255,255,.86);backdrop-filter:blur(14px);border-radius:999px;
     box-shadow:0 8px 24px rgba(40,70,110,.14)"></div>
<div id="frames" style="position:fixed;left:410px;right:190px;bottom:16px;min-width:330px;
     padding:7px 10px;background:rgba(255,255,255,.86);backdrop-filter:blur(14px);
     border:1px solid rgba(255,255,255,.9);border-radius:14px;
     box-shadow:0 8px 24px rgba(40,70,110,.14)"></div>
<div id="hint" style="position:fixed;left:410px;bottom:76px;font-size:11.5px;color:#5d7186"></div>`;
const side = document.getElementById("side");
const stage = document.getElementById("stage");

function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}
const CELL = "padding:3px 2px;border-bottom:1px solid #e5edf5";
const NUM = CELL + ";text-align:right;font-variant-numeric:tabular-nums";
const HEAD = "margin:16px 0 6px;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:#7b8ea3";
const BTN = (on) => `padding:6px 12px;border-radius:999px;cursor:pointer;font-size:12px;border:1px solid ${on ? "#2a9d8f" : "#d5e0ea"};background:${on ? "#e4f5f2" : "#fff"};color:${on ? "#1d7d71" : "#4a5c6e"}`;

// ---------------------------------------------------------------- three
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
stage.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 800);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
const hemi = new THREE.HemisphereLight();
const key = new THREE.DirectionalLight();
const fill = new THREE.DirectionalLight();
const rim = new THREE.DirectionalLight();
key.position.set(8, 10, 12); fill.position.set(-9, -4, 6); rim.position.set(4, -10, 2);
scene.add(hemi, key, fill, rim);
const atomGroup = new THREE.Group(), bondGroup = new THREE.Group();
const cellGroup = new THREE.Group(), orbitalGroup = new THREE.Group();
const modelGroup = new THREE.Group();
modelGroup.add(atomGroup, bondGroup, cellGroup, orbitalGroup);
scene.add(modelGroup);
const clock = new THREE.Clock();
const savedUp = new THREE.Vector3(0, 0, 1);
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();



function resize() {
  // Keep the CSS viewport size independent from the high-DPI drawing buffer.
  // With updateStyle=false a pixelRatio=2 canvas retains 2W x 2H CSS layout
  // dimensions and only its upper-left quarter is visible.
  renderer.setSize(innerWidth, innerHeight, true);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

// ------------------------------------------------- marching tetrahedra
// Six tetrahedra per voxel, sixteen cases each. Small enough to verify by eye,
// and free of the ambiguous-face holes a hand-transcribed 256-entry
// marching-cubes table tends to produce.
const TETRA = [[0,5,1,6],[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6]];
const CORNER = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
const TET_EDGE = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
const TET_TABLE = [
  [], [0,1,2], [0,3,4], [1,2,4,1,4,3],
  [1,3,5], [0,3,5,0,5,2], [0,1,5,0,5,4], [2,4,5],
  [2,4,5], [0,1,5,0,5,4], [0,3,5,0,5,2], [1,3,5],
  [1,2,4,1,4,3], [0,3,4], [0,1,2], [],
];

function isosurface(volume, n, half, level) {
  const positions = [], normals = [];
  const step = (2 * half) / (n - 1);
  const at = (i, j, k) => volume[(i * n + j) * n + k];
  const clamp = (v) => (v < 0 ? 0 : v > n - 1 ? n - 1 : v);
  // Central-difference gradient of the scalar field. The surface normal is
  // -grad f, interpolated along the same edge parameter as the vertex. Without
  // this the geometry is non-indexed, so computeVertexNormals() would hand back
  // one normal per triangle and the surface would render visibly faceted no
  // matter how fine the sampling grid is.
  const gradient = (i, j, k, out) => {
    out[0] = (at(clamp(i + 1), j, k) - at(clamp(i - 1), j, k));
    out[1] = (at(i, clamp(j + 1), k) - at(i, clamp(j - 1), k));
    out[2] = (at(i, j, clamp(k + 1)) - at(i, j, clamp(k - 1)));
  };
  // interior of a positive lobe is where f is large, so outward is -grad f;
  // for a negative lobe the interior is where f is most negative, so it flips
  const orient = level >= 0 ? -1 : 1;
  const corner = new Array(8), value = new Array(8), grad = new Array(8);
  for (let c = 0; c < 8; c++) { corner[c] = [0, 0, 0]; grad[c] = [0, 0, 0]; }
  const index = new Array(8);
  for (let i = 0; i < n - 1; i++)
    for (let j = 0; j < n - 1; j++)
      for (let k = 0; k < n - 1; k++) {
        let below = 0;
        for (let c = 0; c < 8; c++) {
          const d = CORNER[c];
          const v = at(i + d[0], j + d[1], k + d[2]);
          value[c] = v;
          if (v < level) below++;
        }
        if (below === 8 || below === 0) continue;
        for (let c = 0; c < 8; c++) {
          const d = CORNER[c];
          index[c] = [i + d[0], j + d[1], k + d[2]];
          corner[c][0] = -half + index[c][0] * step;
          corner[c][1] = -half + index[c][1] * step;
          corner[c][2] = -half + index[c][2] * step;
          gradient(index[c][0], index[c][1], index[c][2], grad[c]);
        }
        for (const tet of TETRA) {
          let mask = 0;
          for (let t = 0; t < 4; t++) if (value[tet[t]] >= level) mask |= 1 << t;
          const tri = TET_TABLE[mask];
          for (let e = 0; e < tri.length; e++) {
            const edge = TET_EDGE[tri[e]];
            const ia = tet[edge[0]], ib = tet[edge[1]];
            let t = (level - value[ia]) / (value[ib] - value[ia]);
            if (!isFinite(t)) t = 0.5;
            t = t < 0 ? 0 : t > 1 ? 1 : t;
            positions.push(
              corner[ia][0] + t * (corner[ib][0] - corner[ia][0]),
              corner[ia][1] + t * (corner[ib][1] - corner[ia][1]),
              corner[ia][2] + t * (corner[ib][2] - corner[ia][2]));
            const nx = orient * (grad[ia][0] + t * (grad[ib][0] - grad[ia][0]));
            const ny = orient * (grad[ia][1] + t * (grad[ib][1] - grad[ia][1]));
            const nz = orient * (grad[ia][2] + t * (grad[ib][2] - grad[ia][2]));
            const length = Math.hypot(nx, ny, nz) || 1;
            normals.push(nx / length, ny / length, nz / length);
          }
        }
      }
  // Marching tetrahedra emits triangles whose winding depends on the case, so
  // about half end up wound against the outward normal. Under DoubleSide the
  // renderer then flips the shading normal on those fragments and the surface
  // reads as alternating dark/light wedges. Make winding agree with the normal.
  for (let t = 0; t + 8 < positions.length; t += 9) {
    const ux = positions[t + 3] - positions[t], uy = positions[t + 4] - positions[t + 1], uz = positions[t + 5] - positions[t + 2];
    const vx = positions[t + 6] - positions[t], vy = positions[t + 7] - positions[t + 1], vz = positions[t + 8] - positions[t + 2];
    const fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
    if (fx * normals[t] + fy * normals[t + 1] + fz * normals[t + 2] < 0) {
      for (let c = 0; c < 3; c++) {
        let tmp = positions[t + 3 + c]; positions[t + 3 + c] = positions[t + 6 + c]; positions[t + 6 + c] = tmp;
        tmp = normals[t + 3 + c]; normals[t + 3 + c] = normals[t + 6 + c]; normals[t + 6 + c] = tmp;
      }
    }
  }
  return refine(positions, normals);
}

/** Weld coincident vertices, Taubin-smooth, and recompute averaged normals. */
function refine(positions, normals) {
  const geometry = new THREE.BufferGeometry();
  if (!positions.length) {
    geometry.setAttribute("position", new THREE.Float32BufferAttribute([], 3));
    return geometry;
  }
  const TOL = 1e-4;
  const map = new Map(), unique = [], index = new Array(positions.length / 3);
  for (let i = 0, v = 0; i < positions.length; i += 3, v++) {
    const key = Math.round(positions[i] / TOL) + "," +
                Math.round(positions[i + 1] / TOL) + "," +
                Math.round(positions[i + 2] / TOL);
    let j = map.get(key);
    if (j === undefined) {
      j = unique.length / 3;
      map.set(key, j);
      unique.push(positions[i], positions[i + 1], positions[i + 2]);
    }
    index[v] = j;
  }
  const P = Float32Array.from(unique);
  const count = P.length / 3;

  const neighbours = new Array(count);
  for (let v = 0; v < count; v++) neighbours[v] = new Set();
  for (let t = 0; t + 2 < index.length; t += 3) {
    const a = index[t], b = index[t + 1], c = index[t + 2];
    neighbours[a].add(b); neighbours[a].add(c);
    neighbours[b].add(a); neighbours[b].add(c);
    neighbours[c].add(a); neighbours[c].add(b);
  }
  const lists = neighbours.map((s) => Array.from(s));
  const scratch = new Float32Array(P.length);
  // Taubin lambda/mu smoothing: the negative mu pass undoes the volume
  // shrinkage that repeated plain Laplacian passes would cause, so the
  // isosurface keeps its size while the facets disappear.
  const pass = (factor) => {
    for (let v = 0; v < count; v++) {
      const list = lists[v];
      if (list.length < 3) {
        scratch[3 * v] = P[3 * v]; scratch[3 * v + 1] = P[3 * v + 1]; scratch[3 * v + 2] = P[3 * v + 2];
        continue;
      }
      let sx = 0, sy = 0, sz = 0;
      for (const w of list) { sx += P[3 * w]; sy += P[3 * w + 1]; sz += P[3 * w + 2]; }
      const inv = 1 / list.length;
      scratch[3 * v] = P[3 * v] + factor * (sx * inv - P[3 * v]);
      scratch[3 * v + 1] = P[3 * v + 1] + factor * (sy * inv - P[3 * v + 1]);
      scratch[3 * v + 2] = P[3 * v + 2] + factor * (sz * inv - P[3 * v + 2]);
    }
    P.set(scratch);
  };
  for (let i = 0; i < 3; i++) { pass(0.55); pass(-0.58); }

  geometry.setAttribute("position", new THREE.BufferAttribute(P, 3));
  geometry.setIndex(index);
  geometry.computeVertexNormals();

  // Smoothing moved the vertices, so a few triangles can end up wound against
  // their averaged normal again. Re-check once here, after the final positions
  // and normals exist, so no dark facet survives.
  const N = geometry.getAttribute("normal").array;
  for (let t = 0; t + 2 < index.length; t += 3) {
    const a = index[t], b = index[t + 1], c = index[t + 2];
    const ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
    const vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
    const fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
    if (fx * N[3 * a] + fy * N[3 * a + 1] + fz * N[3 * a + 2] < 0) {
      index[t + 1] = c; index[t + 2] = b;
    }
  }
  geometry.setIndex(index);
  return geometry;
}

const cache = new Map();
function volumeFor(pair, key) {
  const id = pair.id + "/" + key;
  if (!cache.has(id)) {
    const entry = pair.volumes[key], raw = atob(entry.data), n = pair.grid.n;
    const out = new Float32Array(n * n * n);
    for (let i = 0; i < out.length; i++) {
      let b = raw.charCodeAt(i);
      if (b > 127) b -= 256;
      out[i] = (b / 127) * entry.scale;
    }
    cache.set(id, out);
  }
  return cache.get(id);
}

// ---------------------------------------------------------------- scene
function cylinderBetween(a, b, radius, colour) {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
  const dir = vb.clone().sub(va);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, dir.length(), 20),
    new THREE.MeshPhysicalMaterial({ color: colour, roughness: 0.42, metalness: 0.05 }));
  mesh.position.copy(va).add(vb).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  return mesh;
}

function buildStructure(pair, preserveView) {
  const theme = THEMES[state.theme];
  [atomGroup, bondGroup, cellGroup].forEach((g) => {
    while (g.children.length) { const c = g.children.pop(); c.geometry?.dispose(); c.material?.dispose(); }
  });
  const sphere = new THREE.SphereGeometry(1, 40, 28);
  pair.atoms.forEach((atom, index) => {
    const isTarget = pair.targetIndex.includes(index);
    const mesh = new THREE.Mesh(sphere, new THREE.MeshPhysicalMaterial({
      color: isTarget ? theme.target : (theme.atoms[atom.el] ?? theme.atoms.other),
      roughness: theme.atomRoughness, metalness: theme.atomMetalness,
      clearcoat: 0.5, clearcoatRoughness: 0.22,
      emissive: isTarget ? theme.targetEmissive : 0x000000,
      emissiveIntensity: isTarget ? 0.18 : 0,
    }));
    mesh.userData = { el: atom.el, target: isTarget, base: (RADIUS[atom.el] ?? RADIUS.X) * (isTarget ? 1.18 : 1) };
    mesh.position.set(...atom.xyz);
    atomGroup.add(mesh);
  });
  updateAtomSize();
  pair.bonds.forEach(([i, j]) => {
    const isTarget = pair.targetIndex.includes(i) && pair.targetIndex.includes(j);
    bondGroup.add(cylinderBetween(pair.atoms[i].xyz, pair.atoms[j].xyz,
      isTarget ? 0.095 : 0.048, isTarget ? theme.target : theme.bond));
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pair.cellEdges), 3));
  cellGroup.add(new THREE.LineSegments(geometry,
    new THREE.LineBasicMaterial({ color: theme.cell, transparent: true, opacity: 0.42 })));
  orbitalGroup.position.set(...pair.fieldOffset);
  atomGroup.visible = state.atoms; bondGroup.visible = state.bonds; cellGroup.visible = state.cell;

  if (!preserveView) {
    const span = pair.grid.half * 2.6;
    camera.up.set(0, 0, 1); savedUp.copy(camera.up);
    camera.position.set(span * 1.12, -span * 1.36, span * 0.85);
    controls.target.set(0, 0, 0);
    controls.minDistance = span * 0.12;
    controls.maxDistance = span * 12;
    camera.far = Math.max(1000, span * 40);
    camera.updateProjectionMatrix();
    controls.update(); controls.saveState();
  }
}

function updateAtomSize() {
  atomGroup.children.forEach((m) => m.scale.setScalar(m.userData.base * (state.atomSize / 0.55)));
}

function rebuild() {
  const pair = PAIRS[state.pair], theme = THEMES[state.theme];
  while (orbitalGroup.children.length) {
    const c = orbitalGroup.children.pop(); c.geometry.dispose(); c.material.dispose();
  }
  const keys = state.showSum ? ["rhoSum"] : (
    state.showAll ? pair.channels.map((c) => state.field + c.i) : [state.field + state.channel]
  );
  keys.forEach((key) => {
    const volume = volumeFor(pair, key);
    const level = state.isoRel * pair.volumes[key].scale;
    [["positive", +1], ["negative", -1]].forEach(([phase, sign]) => {
      const geometry = isosurface(volume, pair.grid.n, pair.grid.half, sign * level);
      if (geometry.getAttribute("position").count === 0) { geometry.dispose(); return; }
      const colour = theme[phase];
      // v3 "depthfix": a colour-less depth pre-pass so far faces of the same
      // surface cannot show through the transparent near faces.
      const depth = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        side: THREE.DoubleSide, colorWrite: false, depthWrite: true, depthTest: true }));
      depth.userData = { depthOnly: true }; depth.renderOrder = 0;
      orbitalGroup.add(depth);
      const surface = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial({
        color: colour, emissive: colour, emissiveIntensity: theme.surfaceGlow,
        side: THREE.DoubleSide, transparent: true, opacity: state.opacity,
        roughness: theme.surfaceRoughness, metalness: 0.02,
        clearcoat: theme.clearcoat, clearcoatRoughness: 0.12,
        flatShading: false, depthWrite: false, depthTest: true }));
      surface.userData = { depthOnly: false, phase }; surface.renderOrder = 1;
      orbitalGroup.add(surface);
    });
  });
}

function applyTheme(name) {
  state.theme = name;
  const theme = THEMES[name];
  document.getElementById("stage").style.background = theme.page;
  renderer.toneMappingExposure = theme.exposure;
  hemi.color.setHex(theme.hemi[0]); hemi.groundColor.setHex(theme.hemi[1]); hemi.intensity = theme.hemi[2];
  key.color.setHex(theme.key[0]); key.intensity = theme.key[1];
  fill.color.setHex(theme.fill[0]); fill.intensity = theme.fill[1];
  rim.color.setHex(theme.rim[0]); rim.intensity = theme.rim[1];
  buildStructure(PAIRS[state.pair], true);
  rebuild(); renderThemes(); renderPanel();
}

// ------------------------------------------------- VESTA-style rotation
let mode = "rotate", dragId = null, dragX = 0, dragY = 0;
function setMode(next) {
  mode = state.mode = next;
  dragId = null;
  controls.enableRotate = false;
  controls.enablePan = true;
  controls.mouseButtons.LEFT = next === "rotate" ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
  controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
  controls.touches.ONE = next === "rotate" ? THREE.TOUCH.ROTATE : THREE.TOUCH.PAN;
  renderer.domElement.style.cursor = next === "rotate" ? "grab" : "move";
  document.getElementById("hint").textContent =
    (next === "rotate" ? "旋转模式 · 左键上下/左右旋转晶胞 · 右键拖动" : "拖动模式 · 左/右键拖动") +
    " · 滚轮缩放 · 双击聚焦 · R 复位 · ←/→ 切通道 · ↑/↓ 切帧";
  renderNav();
}
function beginRotate(event) {
  if (mode !== "rotate") return;
  if (event.pointerType === "mouse" && event.button !== 0) return;
  controls.autoRotate = false;
  dragId = event.pointerId; dragX = event.clientX; dragY = event.clientY;
  renderer.domElement.style.cursor = "grabbing"; event.preventDefault();
}
function moveRotate(event) {
  if (event.pointerId !== dragId) return;
  const dx = event.clientX - dragX, dy = event.clientY - dragY;
  dragX = event.clientX; dragY = event.clientY;
  if (!dx && !dy) return;
  camera.updateMatrixWorld();
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const axis = up.multiplyScalar(dx).add(right.multiplyScalar(dy)).normalize();
  const rotation = new THREE.Quaternion().setFromAxisAngle(axis, Math.hypot(dx, dy) * 0.005);
  modelGroup.quaternion.premultiply(rotation);
  event.preventDefault();
}
function endRotate(event) {
  if (event.pointerId !== dragId) return;
  dragId = null;
  renderer.domElement.style.cursor = mode === "rotate" ? "grab" : "move";
}
function resetView() {
  modelGroup.quaternion.identity();
  camera.up.copy(savedUp); controls.reset(); camera.lookAt(controls.target); controls.update();
}
function zoomView(factor) {
  const offset = camera.position.clone().sub(controls.target);
  const distance = THREE.MathUtils.clamp(offset.length() * factor, controls.minDistance, controls.maxDistance);
  camera.position.copy(controls.target).add(offset.setLength(distance));
  controls.update();
}
function focusDouble(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
              -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(atomGroup.children, false)[0];
  if (!hit) return;
  const offset = camera.position.clone().sub(controls.target);
  const focus = hit.object.getWorldPosition(new THREE.Vector3());
  controls.target.copy(focus);
  camera.position.copy(focus).add(offset);
  controls.update();
}
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());
renderer.domElement.addEventListener("pointerdown", beginRotate);
renderer.domElement.addEventListener("pointermove", moveRotate);
renderer.domElement.addEventListener("pointerup", endRotate);
renderer.domElement.addEventListener("pointercancel", endRotate);
renderer.domElement.addEventListener("dblclick", focusDouble);
addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "BUTTON", "TEXTAREA"].includes(e.target?.tagName)) return;
  const pair = PAIRS[state.pair];
  if (e.key === "ArrowLeft") { state.channel = ((state.channel - 2 + pair.channels.length) % pair.channels.length) + 1; state.showAll = state.showSum = false; renderPanel(); rebuild(); }
  if (e.key === "ArrowRight") { state.channel = (state.channel % pair.channels.length) + 1; state.showAll = state.showSum = false; renderPanel(); rebuild(); }
  if (e.key === "ArrowUp") { selectPair((state.pair - 1 + PAIRS.length) % PAIRS.length); }
  if (e.key === "ArrowDown") { selectPair((state.pair + 1) % PAIRS.length); }
  if (e.key.toLowerCase() === "r") resetView();
  if (e.key === "+" || e.key === "=") zoomView(0.9);
  if (e.key === "-") zoomView(1.1);
});

function selectPair(index) {
  // preserve the camera: comparing the same bond across structures is the whole
  // point, and a view reset destroys the visual A/B
  state.pair = index;
  state.channel = Math.min(state.channel, PAIRS[index].channels.length);
  buildStructure(PAIRS[index], true); renderPanel(); rebuild();
}

let frameTimer = null;
function frameLabel(pair) {
  const match = pair.structure.match(/q[_=]?(\d+)p(\d+)/i);
  return match ? `q=${match[1]}.${match[2]}` : pair.structure;
}
function setFramePlaying(on) {
  state.framePlaying = on;
  if (frameTimer !== null) clearInterval(frameTimer);
  frameTimer = on ? setInterval(() => selectPair((state.pair + 1) % PAIRS.length), 1600) : null;
  renderFrameBar();
}
function renderFrameBar() {
  const box = document.getElementById("frames");
  if (!box) return;
  box.innerHTML = "";
  const row = h(`<div style="display:grid;grid-template-columns:auto auto 1fr auto auto;gap:8px;align-items:center"></div>`);
  const play = h(`<button title="播放/暂停结构帧" style="${BTN(state.framePlaying)};width:34px;padding:5px 0">${state.framePlaying ? "Ⅱ" : "▶"}</button>`);
  play.onclick = () => setFramePlaying(!state.framePlaying);
  const prev = h(`<button title="上一帧" style="${BTN(false)};width:32px;padding:5px 0">‹</button>`);
  prev.onclick = () => selectPair((state.pair - 1 + PAIRS.length) % PAIRS.length);
  const slider = h(`<input aria-label="结构/CUBE 帧" type="range" min="0" max="${PAIRS.length - 1}" step="1" value="${state.pair}" style="width:100%;accent-color:#2a9d8f">`);
  slider.oninput = () => selectPair(+slider.value);
  const next = h(`<button title="下一帧" style="${BTN(false)};width:32px;padding:5px 0">›</button>`);
  next.onclick = () => selectPair((state.pair + 1) % PAIRS.length);
  const label = h(`<span style="min-width:92px;text-align:right;color:#315a74;font:600 11.5px ui-monospace,Consolas">${frameLabel(PAIRS[state.pair])} · ${state.pair + 1}/${PAIRS.length}</span>`);
  row.append(play, prev, slider, next, label);
  box.appendChild(row);
  const ticks = h(`<div style="display:flex;justify-content:space-between;margin:2px 45px 0 82px;color:#7b8ea3;font:9px ui-monospace,Consolas"></div>`);
  PAIRS.forEach((pair) => ticks.appendChild(h(`<span>${frameLabel(pair).replace("q=", "")}</span>`)));
  box.appendChild(ticks);
}

// ---------------------------------------------------------------- chrome
function renderThemes() {
  const box = document.getElementById("themes");
  box.innerHTML = "";
  Object.entries(THEMES).forEach(([id, theme]) => {
    const on = id === state.theme;
    const b = h(`<button style="${BTN(on)};display:flex;align-items:center;gap:6px">
      <span style="width:9px;height:9px;border-radius:50%;background:#${theme.positive.toString(16).padStart(6, "0")}"></span>${theme.label}</button>`);
    b.onclick = () => applyTheme(id);
    box.appendChild(b);
  });
}
function renderNav() {
  const box = document.getElementById("nav");
  box.innerHTML = "";
  const add = (label, on, fn) => { const b = h(`<button style="${BTN(on)}">${label}</button>`); b.onclick = fn; box.appendChild(b); };
  add("↻ 旋转", mode === "rotate", () => setMode("rotate"));
  add("✥ 拖动", mode === "pan", () => setMode("pan"));
  add("◇ 巡航", controls.autoRotate, () => { controls.autoRotate = !controls.autoRotate; controls.autoRotateSpeed = 1.1; renderNav(); });
  add("−", false, () => zoomView(1.22));
  add("+", false, () => zoomView(0.82));
  add("⌂", false, resetView);
}

function pct(x) { return (100 * x).toFixed(2) + "%"; }

function renderPanel() {
  const pair = PAIRS[state.pair];
  const active = pair.channels.find((c) => c.i === state.channel) || pair.channels[0];
  side.innerHTML = "";
  side.appendChild(h(`<div style="display:flex;align-items:center;gap:7px">
    <span style="width:9px;height:9px;border-radius:50%;background:#2a9d8f"></span>
    <span style="font-size:10.5px;letter-spacing:.11em;color:#5d7186">RUO₂ · PRINCIPAL DENSITY-COHERENCE CHANNELS</span></div>`));
  side.appendChild(h(`<div style="font-size:19px;font-weight:600;margin-top:6px;color:#1c2c3e">
      ${pair.scope} · ${state.showSum ? "通道和" : `通道 ${active.i}`}</div>`));
  side.appendChild(h(`<div style="color:#5d7186;font-size:12px;margin-top:3px">
      ${pair.structure} · R = ${pair.R} Å · ${state.showSum ? "逐点 Σρᵢ" : `占比 ${pct(active.share)}`} · ${FIELD_LABEL[state.field]}${pair.bvkClosed ? ` · BvK ${pair.bvkMesh.join("×")}` : ""}</div>`));
  const isProduct = state.showSum || state.field === "rho";
  const isChemicalOrbital = state.field === "bond" || state.field === "anti";
  side.appendChild(h(`<div style="margin-top:8px;padding:7px 9px;border-radius:8px;background:#fff4df;
      color:#775816;font-size:11px">${isProduct
        ? "当前是带符号的密度相干交叉项 proxy，不是电子密度，也不可积分。"
        : isChemicalOrbital
          ? `当前是密度导出的${state.field === "bond" ? "成键" : "反键"}通道轨道振幅；正负色表示轨道瓣相位。`
          : "当前是归一化伙伴振幅，不是密度；正负色显示固定 gauge 下的轨道瓣相位。"}</div>`));
  const theme = THEMES[state.theme];
  side.appendChild(h(`<div style="display:flex;gap:14px;align-items:center;margin-top:8px;font-size:11.5px;color:#53677b">
    <span>${isProduct ? "交叉项符号" : isChemicalOrbital ? "轨道相位" : "伙伴振幅相位"}</span>
    <span><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#${theme.positive.toString(16).padStart(6, "0")};margin-right:4px"></i>正（+）</span>
    <span><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#${theme.negative.toString(16).padStart(6, "0")};margin-right:4px"></i>负（−）</span>
  </div>`));
  if (!isProduct && active.gaugeFragile) {
    side.appendChild(h(`<div style="margin-top:7px;padding:7px 9px;border-radius:8px;background:#fdf5e3;
      border:1px solid #f0dcae;color:#7a5c14;font-size:11px">该通道的符号 gauge margin 仅 ${active.gaugeMargin.toFixed(4)}；
      单个伙伴的整体正负跨结构可能翻转，成对乘积不受影响。</div>`));
  }

  side.appendChild(h(`<div style="${HEAD}">键</div>`));
  const select = h(`<select style="width:100%;padding:7px;border-radius:8px;border:1px solid #d5e0ea;background:#fff;color:#243447"></select>`);
  PAIRS.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${p.structure} · ${p.scope} · ${p.R} Å`;
    if (i === state.pair) o.selected = true;
    select.appendChild(o);
  });
  select.onchange = () => selectPair(+select.value);
  side.appendChild(select);

  side.appendChild(h(`<div style="${HEAD}">整键统计</div>`));
  side.appendChild(h(`<table style="width:100%;border-collapse:collapse">
    <tr><td style="${CELL}">键长</td><td style="${NUM}">${pair.R} Å</td></tr>
    <tr><td style="${CELL}">W = Σs²</td><td style="${NUM}"><b>${pair.W.toFixed(5)}</b></td></tr>
    <tr><td style="${CELL}">selected A / B star</td><td style="${NUM}">${pair.WSelectedA.toFixed(4)} / ${pair.WSelectedB == null ? "—" : pair.WSelectedB.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">full A / B star</td><td style="${NUM}">${pair.WFullA == null ? "—" : pair.WFullA.toFixed(4)} / ${pair.WFullB == null ? "—" : pair.WFullB.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">canonical 独立 pair spectrum</td><td style="${NUM}">${pair.canonicalSpectrum.length ? pair.canonicalSpectrum.map((x) => x.toFixed(4)).join(", ") : "—"}</td></tr>
    <tr><td style="${CELL}">canonical support rank A / B</td><td style="${NUM}">${pair.canonicalSupportRanks[0] ?? "—"} / ${pair.canonicalSupportRanks[1] ?? "—"}</td></tr>
    <tr><td style="${CELL}">IAO 维数</td><td style="${NUM}">${pair.nIAO}</td></tr></table>`));

  side.appendChild(h(`<div style="${HEAD}">通道</div>`));
  const table = h(`<table style="width:100%;border-collapse:collapse;font-size:12px"></table>`);
  table.appendChild(h(`<tr style="color:#7b8ea3;font-size:10.5px">
    <td style="${CELL}">ch</td><td style="${NUM}">s</td><td style="${NUM}">s²</td>
    <td style="${NUM}">占比</td></tr>`));
  pair.channels.forEach((c) => {
    const row = h(`<tr style="cursor:pointer">
      <td style="${CELL}">${c.i}${c.degenerate ? ' <span title="与相邻通道近简并：只有二维子空间有意义" style="color:#e0a020">⚠</span>' : ""}${c.gaugeFragile ? ' <span title="伙伴相位 gauge 较脆弱" style="color:#c07828">◆</span>' : ""}</td>
      <td style="${NUM}">${c.s.toFixed(4)}</td><td style="${NUM}">${c.s2.toFixed(5)}</td>
      <td style="${NUM}">${pct(c.share)}</td></tr>`);
    if (c.i === state.channel && !state.showAll && !state.showSum) row.style.background = "#e4f5f2";
    row.onclick = () => { state.channel = c.i; state.showAll = state.showSum = false; renderPanel(); rebuild(); };
    table.appendChild(row);
  });
  side.appendChild(table);

  side.appendChild(h(`<div style="${HEAD}">通道 ${active.i} 明细</div>`));
  side.appendChild(h(`<table style="width:100%;border-collapse:collapse;font-size:12px">
    <tr><td style="${CELL}">2s（密度权重）</td><td style="${NUM}">${active.twoS.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">本键内占比</td><td style="${NUM}">${pct(active.share)}</td></tr>
    <tr><td style="${CELL}">selected A / B star 内占比</td><td style="${NUM}">${pct(active.ctxShareA)} / ${active.ctxShareB == null ? "—" : pct(active.ctxShareB)}</td></tr>
    <tr><td style="${CELL}">n(${pair.labelA})</td><td style="${NUM}">${active.nA.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">n(${pair.labelB})</td><td style="${NUM}">${active.nB.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">n(φ+) 成键占据</td><td style="${NUM}"><b>${active.nPlus.toFixed(4)}</b></td></tr>
    <tr><td style="${CELL}">n(φ−) 反键占据</td><td style="${NUM}">${active.nMinus.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">normalized coherence c</td><td style="${NUM}">${active.coherence.toFixed(4)}</td></tr>
    <tr><td style="${CELL}">环境泄漏 A / B</td><td style="${NUM}">${active.leakA.toFixed(3)} / ${active.leakB.toFixed(3)}</td></tr>
    <tr><td style="${CELL}">相对谱间隙</td><td style="${NUM}">${active.gap == null ? "—" : active.gap.toFixed(4)}</td></tr></table>`));
  side.appendChild(h(`<div style="font-size:11px;color:#7b8ea3;margin-top:4px">符号 gauge margin：${active.gaugeMargin.toFixed(4)}${active.gaugeFragile ? "（fragile）" : ""}</div>`));

  // A near-degenerate pair has no individually meaningful members: any rotation
  // inside the spanned subspace is an equally valid pair of channels. Only the
  // group sum is an observable, so it is reported explicitly.
  // shown whenever the bond HAS a degenerate group, not only when one of its
  // members happens to be selected -- the flag in the table needs explaining
  // exactly when the user can see it
  if (pair.channels.some((c) => c.degenerate)) {
    const group = pair.channels.filter((c) => c.degenerate);
    const sum = group.reduce((a, c) => a + c.s2, 0);
    const shareSum = group.reduce((a, c) => a + c.share, 0);
    side.appendChild(h(`<div style="margin-top:9px;padding:9px 11px;border-radius:9px;
        background:#fdf5e3;border:1px solid #f0dcae;color:#7a5c14;font-size:11.5px;line-height:1.55">
      <b>通道 ${group.map((c) => c.i).join(" 与 ")} 近简并</b>（相对谱间隙 ${(100 * Math.min(...group.map((c) => c.gap))).toFixed(1)}%）。
      单个通道的取向在这个二维子空间里是任意的，<b>不要分别解读</b>；只有整组是可观测量：<br>
      <span style="font-size:13px">Σs² = <b>${sum.toFixed(5)}</b> &nbsp; 占本键 <b>${(100 * shareSum).toFixed(2)}%</b></span><br>
      对 Ru=O 这类氧代键，这正是两条等价 π 通道的预期特征。跨结构跟踪时必须整组做 Procrustes 对齐。</div>`));
  }

  side.appendChild(h(`<div style="${HEAD}">表示</div>`));
  const fieldRow = h(`<div style="display:flex;gap:5px;flex-wrap:wrap"></div>`);
  (pair.availableFields || Object.keys(FIELD_LABEL)).forEach((k) => {
    const label = FIELD_LABEL[k];
    const b = h(`<button style="${BTN(state.field === k)};flex:1 1 31%;padding:6px 4px;font-size:11px">${label}</button>`);
    b.onclick = () => { state.field = k; state.showSum = false; renderPanel(); rebuild(); };
    fieldRow.appendChild(b);
  });
  side.appendChild(fieldRow);
  const allBox = h(`<label style="display:block;margin-top:8px;color:#4a5c6e;font-size:12px;cursor:pointer">
      <input type="checkbox" ${state.showAll ? "checked" : ""}> 叠加全部通道</label>`);
  allBox.querySelector("input").onchange = (e) => { state.showAll = e.target.checked; if (state.showAll) state.showSum = false; renderPanel(); rebuild(); };
  side.appendChild(allBox);
  if (pair.hasChannelSum) {
    const sumBox = h(`<label style="display:block;margin-top:5px;color:#4a5c6e;font-size:12px;cursor:pointer">
        <input type="checkbox" ${state.showSum ? "checked" : ""}> 显示通道和 CUBE（Σρᵢ）</label>`);
    sumBox.querySelector("input").onchange = (e) => { state.showSum = e.target.checked; if (state.showSum) { state.showAll = false; state.field = "rho"; } renderPanel(); rebuild(); };
    side.appendChild(sumBox);
  }

  // ---- isovalue -------------------------------------------------------
  side.appendChild(h(`<div style="${HEAD}">等值面</div>`));
  const isoRow = h(`<div style="display:flex;align-items:center;gap:10px"></div>`);
  isoRow.appendChild(h(`<span style="font-size:12px;color:#4a5c6e;white-space:nowrap">相对等值面</span>`));
  const slider = h(`<input type="range" min="0.02" max="0.9" step="0.01" value="${state.isoRel}" style="flex:1;accent-color:#2a9d8f">`);
  const readout = h(`<span style="font-size:11.5px;color:#5d7186;white-space:nowrap;min-width:96px;text-align:right">
      ${state.isoRel.toFixed(2)} × max|field|</span>`);
  slider.oninput = () => {
    state.isoRel = +slider.value;
    readout.textContent = state.isoRel.toFixed(2) + " × max|field|";
  };
  slider.onchange = () => rebuild();
  isoRow.appendChild(slider); isoRow.appendChild(readout);
  side.appendChild(isoRow);
  side.appendChild(h(`<div style="font-size:11px;color:#7b8ea3;margin-top:4px">按本场自身峰值归一，只比较形状。</div>`));

  // ---- atom size ------------------------------------------------------
  const sizeRow = h(`<div style="display:flex;align-items:center;gap:10px;margin-top:12px"></div>`);
  sizeRow.appendChild(h(`<span style="font-size:12px;color:#4a5c6e;white-space:nowrap">原子大小</span>`));
  const sizeSlider = h(`<input type="range" min="0.15" max="1.4" step="0.01" value="${state.atomSize}" style="flex:1;accent-color:#2a9d8f">`);
  const sizeOut = h(`<span style="font-size:11.5px;color:#5d7186;min-width:96px;text-align:right">${state.atomSize.toFixed(2)}×</span>`);
  sizeSlider.oninput = () => { state.atomSize = +sizeSlider.value; sizeOut.textContent = state.atomSize.toFixed(2) + "×"; updateAtomSize(); };
  sizeRow.appendChild(sizeSlider); sizeRow.appendChild(sizeOut);
  side.appendChild(sizeRow);

  // ---- toggles + opacity ---------------------------------------------
  const toggles = h(`<div style="display:flex;align-items:center;gap:10px;margin-top:12px;flex-wrap:wrap"></div>`);
  [["原子", "atoms", atomGroup], ["键", "bonds", bondGroup], [pair.cellLabel || "晶胞", "cell", cellGroup]].forEach(([label, flag, group]) => {
    const l = h(`<label style="display:flex;align-items:center;gap:5px;font-size:12px;color:#4a5c6e;cursor:pointer">
        <input type="checkbox" ${state[flag] ? "checked" : ""} style="accent-color:#2a9d8f"> ${label}</label>`);
    l.querySelector("input").onchange = (e) => { state[flag] = e.target.checked; group.visible = e.target.checked; };
    toggles.appendChild(l);
  });
  side.appendChild(toggles);
  const opacityRow = h(`<div style="display:flex;align-items:center;gap:10px;margin-top:10px"></div>`);
  opacityRow.appendChild(h(`<span style="font-size:12px;color:#4a5c6e;white-space:nowrap">透明度</span>`));
  const opacitySlider = h(`<input type="range" min="0.15" max="1" step="0.01" value="${state.opacity}" style="flex:1;accent-color:#2a9d8f">`);
  opacitySlider.oninput = () => {
    state.opacity = +opacitySlider.value;
    orbitalGroup.children.forEach((m) => { if (!m.userData.depthOnly) m.material.opacity = state.opacity; });
  };
  opacityRow.appendChild(opacitySlider);
  side.appendChild(opacityRow);
  const resetRow = h(`<div style="display:flex;justify-content:flex-end;margin-top:10px"></div>`);
  const resetButton = h(`<button style="${BTN(false)};padding:7px 18px">复位</button>`);
  resetButton.onclick = () => {
    Object.assign(state, { isoRel: 0.30, atomSize: 0.55, opacity: 0.72,
      atoms: true, bonds: true, cell: true, showAll: false, showSum: false });
    atomGroup.visible = bondGroup.visible = cellGroup.visible = true;
    resetView(); updateAtomSize(); renderPanel(); rebuild();
  };
  resetRow.appendChild(resetButton);
  side.appendChild(resetRow);

  side.appendChild(h(`<div style="${HEAD}">邻域（多中心）</div>`));
  const ctx = h(`<table style="width:100%;border-collapse:collapse;font-size:12px"></table>`);
  pair.contextPairs.slice(0, 8).forEach((c) => ctx.appendChild(h(
    `<tr><td style="${CELL}">${pair.labelA}–${c.label}</td>
     <td style="${NUM}">${c.W.toFixed(4)}</td><td style="${NUM}">${pct(c.frac)}</td></tr>`)));
  side.appendChild(ctx);

  side.appendChild(h(`<div style="${HEAD}">全部键对照</div>`));
  const compare = h(`<table style="width:100%;border-collapse:collapse;font-size:11.5px"></table>`);
  compare.appendChild(h(`<tr style="color:#7b8ea3;font-size:10.5px">
      <td style="${CELL}">结构</td><td style="${CELL}">键</td><td style="${NUM}">R</td><td style="${NUM}">W</td></tr>`));
  PAIRS.forEach((p, i) => {
    const row = h(`<tr style="cursor:pointer"><td style="${CELL}">${p.structure}</td>
      <td style="${CELL}">${p.scope}</td><td style="${NUM}">${p.R.toFixed(3)}</td>
      <td style="${NUM}">${p.W.toFixed(4)}</td></tr>`);
    if (i === state.pair) row.style.background = "#e4f5f2";
    row.onclick = () => selectPair(i);
    compare.appendChild(row);
  });
  side.appendChild(compare);

  side.appendChild(h(`<div style="${HEAD}">口径与边界</div>`));
  side.appendChild(h(`<div style="color:#6b7f93;font-size:11px;line-height:1.6">
    ${isProduct
      ? "颜色表示 2s·Re[Â*B̂] 交叉项的正负，不是电荷正负，也不是单个轨道的相位。"
      : isChemicalOrbital
        ? `φ+ = (A+B)/√2，φ− = (A−B)/√2；它们是由密度矩阵主通道得到的 natural-bond-like 轨道，不是单条 KS 本征轨道。当前显示 ${state.field === "bond" ? "φ+ 成键轨道" : "φ− 反键轨道"}。`
        : "颜色表示伙伴振幅在固定 gauge 下的正负轨道瓣；整体同时反号不改变物理量。"}<br><br>
    <b style="color:#b07d18">边界</b>：s² 是活性空间 IAO Wiberg 型相干指标，不是键级、不是键能。通道数被 O 2p 的三维子空间锁死为 3，三条通道不构成三重键证据。proxy <b>不可积分、不可由峰值或等值面体积推导 population，也不可跨结构比较绝对强度</b>。<br>
    本键表示偏差 ${pair.gridChecks.cross_gram_max_abs_deviation.toFixed(4)}；${pair.bvkClosed ? `场已在 ${pair.bvkMesh.join("×")} Born–von Kármán 超胞闭合，页面显示目标键局部裁剪。` : "多 k primitive-cell proxy 不具备定量 Fourier/Born–von Kármán 约定。"}
  </div>`));
  renderFrameBar();
}

window.__PDC__ = { state, scene, surfaces: orbitalGroup, molecule: atomGroup, renderer,
  isosurface, volumeFor, controls, camera,
  // geometry is indexed, so triangles come from the index, not the vertex count
  triangleCount: () => orbitalGroup.children.filter((m) => !m.userData.depthOnly)
    .reduce((n, m) => n + (m.geometry.getIndex()?.count ?? m.geometry.getAttribute("position").count) / 3, 0) };

// ---------------------------------------------------------------- start
resize();
buildStructure(PAIRS[0], false);
applyTheme("aurora");
setMode("rotate");
renderPanel();
renderThemes();
renderNav();
renderer.setAnimationLoop(() => {
  controls.update();
  const theme = THEMES[state.theme], t = clock.getElapsedTime();
  orbitalGroup.children.forEach((m) => {
    if (!m.userData.depthOnly) {
      m.material.emissiveIntensity = theme.surfaceGlow *
        (0.86 + 0.14 * Math.sin(t * 1.25 + (m.userData.phase === "positive" ? 1.6 : 0)));
    }
  });
  renderer.render(scene, camera);
});
