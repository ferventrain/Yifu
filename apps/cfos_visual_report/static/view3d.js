import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

const SIGNAL_COLOR = 0xc44e52;
const REGION_SURFACE_COLOR = 0x3182bd;
const ROOT_SURFACE_COLOR = 0xa8bace;
const BRAIN_OUTLINE_COLOR = 0x64748b;
const DEFAULT_RESOLUTION_UM = [25, 25, 25];

let circleTexture = null;

function getCircleTexture() {
  if (circleTexture) return circleTexture;
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.5, "rgba(255,255,255,0.9)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  circleTexture = new THREE.CanvasTexture(canvas);
  return circleTexture;
}

function atlasPosition(point, shape, resolutionUm) {
  const res = resolutionUm?.length === 3 ? resolutionUm : DEFAULT_RESOLUTION_UM;
  const dvUm = Number(point.dv) * res[0];
  const apUm = Number(point.ap) * res[1];
  const mlUm = Number(point.ml) * res[2];
  const maxUm = Math.max(
    Math.max(Number(shape[0]) - 1, 1) * res[0],
    Math.max(Number(shape[1]) - 1, 1) * res[1],
    Math.max(Number(shape[2]) - 1, 1) * res[2],
  );
  return [
    (mlUm / maxUm - 0.5) * 2.0,
    (apUm / maxUm - 0.5) * 2.0,
    (dvUm / maxUm - 0.5) * 2.0,
  ];
}

function disposeObject3D(object) {
  if (!object) return;
  object.geometry?.dispose();
  if (Array.isArray(object.material)) {
    object.material.forEach((material) => material.dispose());
  } else {
    object.material?.dispose();
  }
}

export class PointsViewer3D {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xe8edf4);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.container.replaceChildren(this.renderer.domElement);

    // Trackball rotation feels closer to brainrender/vedo than constrained orbit controls.
    this.controls = new TrackballControls(this.camera, this.renderer.domElement);
    this.controls.rotateSpeed = 1.35;
    this.controls.zoomSpeed = 0.42;
    this.controls.panSpeed = 0.55;
    this.controls.staticMoving = false;
    this.controls.dynamicDampingFactor = 0.14;
    this.controls.noZoom = true;
    this._minCameraDistance = 0.35;
    this._maxCameraDistance = 7.5;
    this._sceneCenter = new THREE.Vector3(0, 0, 0);
    this._bindWheelZoomFix();

    const ambient = new THREE.AmbientLight(0xffffff, 0.58);
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.82);
    keyLight.position.set(2, 3, 4);
    this.scene.add(ambient, keyLight);

    this.signalPoints = null;
    this.rootSurface = null;
    this.regionSurface = null;
    this.brainOutlineSurface = null;
    this.fullPayload = null;
    this.focusRegionIds = null;
    this.atlasShape = [456, 528, 320];
    this.atlasResolutionUm = DEFAULT_RESOLUTION_UM;
    this.sampleId = null;
    this._cameraInitialized = false;
    this._animationFrame = null;
    this._raycaster = new THREE.Raycaster();
    this._raycaster.params.Points = { threshold: 0.04 };
    this._visiblePointRecords = [];
    this.onRegionPick = null;
    this._pickDragged = false;
    this.controls.addEventListener("start", () => {
      this._pickDragged = false;
    });
    this.controls.addEventListener("change", () => {
      this._pickDragged = true;
    });
    this._onCanvasClick = this._handleCanvasClick.bind(this);
    this.renderer.domElement.addEventListener("click", this._onCanvasClick);
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(this.container);
    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    this.animate();
  }

  _bindWheelZoomFix() {
    const canvas = this.renderer.domElement;
    this._onWheel = (event) => {
      event.preventDefault();
      const delta = Math.sign(event.deltaY);
      if (!delta) return;
      const direction = new THREE.Vector3().subVectors(this.camera.position, this._sceneCenter);
      const distance = direction.length();
      if (distance <= 0) return;
      const scale = delta > 0 ? 1.04 : 0.96;
      const nextDistance = THREE.MathUtils.clamp(distance * scale, this._minCameraDistance, this._maxCameraDistance);
      direction.setLength(nextDistance);
      this.camera.position.copy(this._sceneCenter).add(direction);
      this.controls.update();
    };
    canvas.addEventListener("wheel", this._onWheel, { passive: false });
  }

  _clampCameraDistance() {
    const direction = new THREE.Vector3().subVectors(this.camera.position, this._sceneCenter);
    const distance = direction.length();
    if (distance <= 0) return;
    const clamped = THREE.MathUtils.clamp(distance, this._minCameraDistance, this._maxCameraDistance);
    if (Math.abs(clamped - distance) > 1e-6) {
      direction.setLength(clamped);
      this.camera.position.copy(this._sceneCenter).add(direction);
    }
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.controls.handleResize();
  }

  animate() {
    this._animationFrame = requestAnimationFrame(() => this.animate());
    this.controls.update();
    this._sceneCenter.copy(this.controls.target);
    this._clampCameraDistance();
    this.renderer.render(this.scene, this.camera);
  }

  _applySpatialMetadata(payload) {
    if (payload?.atlas_shape_dv_ap_ml) {
      this.atlasShape = payload.atlas_shape_dv_ap_ml;
    }
    if (payload?.atlas_resolution_um_dv_ap_ml) {
      this.atlasResolutionUm = payload.atlas_resolution_um_dv_ap_ml;
    }
  }

  clearSignalPoints() {
    if (this.signalPoints) {
      this.scene.remove(this.signalPoints);
      disposeObject3D(this.signalPoints);
      this.signalPoints = null;
    }
  }

  _clearMeshObject(meshRefName) {
    const mesh = this[meshRefName];
    if (!mesh) return;
    this.scene.remove(mesh);
    disposeObject3D(mesh);
    this[meshRefName] = null;
  }

  clearRegionSurface() {
    this._clearMeshObject("regionSurface");
  }

  clearBrainOutlineSurface() {
    this._clearMeshObject("brainOutlineSurface");
  }

  clearRootSurface() {
    this._clearMeshObject("rootSurface");
  }

  clearPoints() {
    this.clearSignalPoints();
    this.clearRegionSurface();
    this.clearBrainOutlineSurface();
    this.clearRootSurface();
    this.fullPayload = null;
    this.focusRegionIds = null;
    this._cameraInitialized = false;
  }

  _sceneBoundsObject() {
    return this.regionSurface || this.brainOutlineSurface || this.rootSurface || this.signalPoints;
  }

  fitCameraToScene() {
    const target = this._sceneBoundsObject();
    if (!target) {
      this.resetCamera();
      return;
    }
    const box = new THREE.Box3().setFromObject(target);
    if (box.isEmpty()) {
      this.resetCamera();
      return;
    }
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    this._sceneCenter.copy(center);
    this.controls.target.copy(center);
    const radius = Math.max(size.x, size.y, size.z, 0.2) * 0.55;
    // brainrender "three_quarters" style diagonal view
    this.camera.position.set(
      center.x + radius * 2.15,
      center.y + radius * 1.35,
      center.z + radius * 2.35,
    );
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(center);
    this.camera.near = Math.max(radius / 100, 0.05);
    this.camera.far = Math.max(radius * 30, 50);
    this.camera.updateProjectionMatrix();
    this._minCameraDistance = Math.max(radius * 0.35, 0.25);
    this._maxCameraDistance = Math.max(radius * 6.5, 4.0);
    this.controls.update();
    this._cameraInitialized = true;
  }

  _pointSize(count) {
    return Math.max(0.008, Math.min(0.022, 28 / Math.cbrt(Math.max(count, 1))));
  }

  _buildPointCloud(points, { color, size, opacity = 0.88 }) {
    const positions = new Float32Array(points.length * 3);
    points.forEach((point, index) => {
      const [x, y, z] = atlasPosition(point, this.atlasShape, this.atlasResolutionUm);
      const base = index * 3;
      positions[base] = x;
      positions[base + 1] = y;
      positions[base + 2] = z;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color,
      size,
      map: getCircleTexture(),
      alphaTest: 0.15,
      transparent: true,
      opacity,
      sizeAttenuation: true,
      depthWrite: true,
    });
    return new THREE.Points(geometry, material);
  }

  _buildSurfaceMesh(surfacePayload, { color, opacity, shininess = 40 }) {
    const vertices = surfacePayload.vertices || [];
    const faces = surfacePayload.faces || [];
    if (!vertices.length || !faces.length) return null;

    this._applySpatialMetadata(surfacePayload);

    const positions = new Float32Array(vertices.length * 3);
    vertices.forEach((vertex, index) => {
      const [x, y, z] = atlasPosition(vertex, this.atlasShape, this.atlasResolutionUm);
      const base = index * 3;
      positions[base] = x;
      positions[base + 1] = y;
      positions[base + 2] = z;
    });

    const indices = new Uint32Array(faces.length * 3);
    faces.forEach((face, index) => {
      const base = index * 3;
      indices[base] = face[0];
      indices[base + 1] = face[1];
      indices[base + 2] = face[2];
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      color,
      transparent: true,
      opacity,
      shininess,
      specular: 0x142028,
      flatShading: false,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = opacity <= 0.2 ? 0 : 1;
    return mesh;
  }

  setRootSurface(surfacePayload) {
    this.clearRootSurface();
    if (!surfacePayload?.available) return false;
    this.rootSurface = this._buildSurfaceMesh(surfacePayload, {
      color: ROOT_SURFACE_COLOR,
      opacity: 0.14,
      shininess: 20,
    });
    if (this.rootSurface) {
      this.scene.add(this.rootSurface);
      this._syncWholeBrainContext();
      return true;
    }
    return false;
  }

  _syncWholeBrainContext() {
    const inRegionFocus = Boolean(this.focusRegionIds?.size);
    if (this.rootSurface) {
      // Root (997) is the resting-state whole-brain mesh; hide it during region focus
      // so it does not look like a second brain outline when outline is toggled off.
      this.rootSurface.visible = !inRegionFocus;
    }
  }

  setRegionSurface(surfacePayload) {
    this.clearRegionSurface();
    if (!surfacePayload?.available) return false;
    this.regionSurface = this._buildSurfaceMesh(surfacePayload, {
      color: REGION_SURFACE_COLOR,
      opacity: 0.34,
      shininess: 48,
    });
    if (this.regionSurface) {
      this.scene.add(this.regionSurface);
      return true;
    }
    return false;
  }

  setBrainOutlineSurface(surfacePayload) {
    this.clearBrainOutlineSurface();
    if (!surfacePayload?.available) return false;
    this.brainOutlineSurface = this._buildSurfaceMesh(surfacePayload, {
      color: BRAIN_OUTLINE_COLOR,
      opacity: 0.12,
      shininess: 16,
    });
    if (this.brainOutlineSurface) {
      this.scene.add(this.brainOutlineSurface);
      return true;
    }
    return false;
  }

  updateBrainOutline(surfacePayload) {
    if (!surfacePayload?.available) {
      this.clearBrainOutlineSurface();
      return false;
    }
    const ok = this.setBrainOutlineSurface(surfacePayload);
    this._syncWholeBrainContext();
    return ok;
  }

  _filteredSignalPoints() {
    if (!this.fullPayload?.points?.length) return [];
    const points = this.fullPayload.points;
    if (!this.focusRegionIds || this.focusRegionIds.size === 0) {
      return points;
    }
    return points.filter((point) => this.focusRegionIds.has(Number(point.region_id || 0)));
  }

  _handleCanvasClick(event) {
    if (this._pickDragged) return;
    if (typeof this.onRegionPick !== "function" || !this.signalPoints || !this._visiblePointRecords.length) {
      return;
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    this._raycaster.setFromCamera(mouse, this.camera);
    const hits = this._raycaster.intersectObject(this.signalPoints);
    if (!hits.length) return;
    const point = this._visiblePointRecords[hits[0].index];
    const regionId = Number(point?.region_id || 0);
    if (regionId > 0) this.onRegionPick(regionId);
  }

  _renderSignalPoints() {
    this.clearSignalPoints();
    const visiblePoints = this._filteredSignalPoints();
    this._visiblePointRecords = visiblePoints;
    if (visiblePoints.length) {
      this.signalPoints = this._buildPointCloud(visiblePoints, {
        color: SIGNAL_COLOR,
        size: this._pointSize(visiblePoints.length),
      });
      this.scene.add(this.signalPoints);
    }
    return Boolean(visiblePoints.length || this._sceneBoundsObject());
  }

  loadPayload(payload, sampleId, rootSurfacePayload = null) {
    this.sampleId = sampleId;
    this.fullPayload = payload;
    this.focusRegionIds = null;
    this.clearRegionSurface();
    this.clearBrainOutlineSurface();
    if (!payload?.available || !payload.points?.length) {
      this.clearPoints();
      return false;
    }

    this._applySpatialMetadata(payload);
    this.resize();
    if (rootSurfacePayload) {
      this.setRootSurface(rootSurfacePayload);
    }
    const ok = this._renderSignalPoints();
    if (!this._cameraInitialized) {
      this.fitCameraToScene();
      this._cameraInitialized = true;
    }
    this.resize();
    return ok;
  }

  setRegionFocus(memberRegionIds, regionSurfacePayload = null, brainOutlinePayload = undefined) {
    if (!this.fullPayload) return false;

    this.clearRegionSurface();
    if (brainOutlinePayload !== undefined) {
      if (brainOutlinePayload) {
        this.setBrainOutlineSurface(brainOutlinePayload);
      } else {
        this.clearBrainOutlineSurface();
      }
    }

    if (!memberRegionIds || memberRegionIds.length === 0) {
      this.focusRegionIds = null;
      this._syncWholeBrainContext();
      return this._renderSignalPoints();
    }

    this.focusRegionIds = new Set(memberRegionIds.map((value) => Number(value)));
    if (regionSurfacePayload) {
      this.setRegionSurface(regionSurfacePayload);
    }
    this._syncWholeBrainContext();
    return this._renderSignalPoints();
  }

  clearRegionFocus() {
    return this.setRegionFocus(null, null, null);
  }

  resetCamera() {
    this.fitCameraToScene();
  }

  viewsStorageKey(sampleId) {
    return `cfos_report_views_${sampleId || "default"}`;
  }

  listSavedViews(sampleId = this.sampleId) {
    const raw = localStorage.getItem(this.viewsStorageKey(sampleId));
    if (!raw) return [];
    try {
      const payload = JSON.parse(raw);
      return Array.isArray(payload?.views) ? payload.views : [];
    } catch {
      return [];
    }
  }

  _writeSavedViews(sampleId, views) {
    localStorage.setItem(
      this.viewsStorageKey(sampleId),
      JSON.stringify({ views }),
    );
  }

  _cameraPayload() {
    this._sceneCenter.copy(this.controls.target);
    return {
      position: this.camera.position.toArray(),
      quaternion: this.camera.quaternion.toArray(),
      up: this.camera.up.toArray(),
      sceneCenter: this._sceneCenter.toArray(),
    };
  }

  _applyCameraPayload(payload) {
    if (!payload) return false;
    this.camera.position.fromArray(payload.position);
    if (payload.quaternion) {
      this.camera.quaternion.fromArray(payload.quaternion);
    }
    if (payload.up) {
      this.camera.up.fromArray(payload.up);
    }
    if (payload.sceneCenter) {
      this._sceneCenter.fromArray(payload.sceneCenter);
      this.controls.target.copy(this._sceneCenter);
    } else if (payload.target) {
      this._sceneCenter.fromArray(payload.target);
      this.controls.target.copy(this._sceneCenter);
    }
    this.controls.update();
    this._cameraInitialized = true;
    return true;
  }

  nextViewName(sampleId = this.sampleId) {
    const existing = new Set(this.listSavedViews(sampleId).map((view) => view.name));
    let index = 1;
    while (existing.has(`view${index}`)) index += 1;
    return `view${index}`;
  }

  saveNamedView(name, sampleId = this.sampleId) {
    const viewName = String(name || this.nextViewName(sampleId)).trim();
    if (!viewName) return null;
    const views = this.listSavedViews(sampleId).filter((view) => view.name !== viewName);
    views.push({
      name: viewName,
      saved_at: new Date().toISOString(),
      camera: this._cameraPayload(),
    });
    this._writeSavedViews(sampleId, views);
    return viewName;
  }

  loadNamedView(name, sampleId = this.sampleId) {
    const view = this.listSavedViews(sampleId).find((item) => item.name === name);
    if (!view?.camera) return false;
    return this._applyCameraPayload(view.camera);
  }

  deleteNamedView(name, sampleId = this.sampleId) {
    const views = this.listSavedViews(sampleId).filter((view) => view.name !== name);
    this._writeSavedViews(sampleId, views);
  }

  cameraStorageKey(sampleId) {
    return `cfos_report_camera_${sampleId || "default"}`;
  }

  saveCamera(sampleId = this.sampleId) {
    return this.saveNamedView(this.nextViewName(sampleId), sampleId);
  }

  loadCamera(sampleId = this.sampleId) {
    const views = this.listSavedViews(sampleId);
    if (!views.length) {
      const raw = localStorage.getItem(this.cameraStorageKey(sampleId));
      if (!raw) return false;
      try {
        return this._applyCameraPayload(JSON.parse(raw));
      } catch {
        return false;
      }
    }
    return this.loadNamedView(views[views.length - 1].name, sampleId);
  }

  screenshot() {
    this.resize();
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  dispose() {
    if (this._animationFrame) cancelAnimationFrame(this._animationFrame);
    window.removeEventListener("resize", this._onResize);
    this.renderer.domElement.removeEventListener("wheel", this._onWheel);
    this.renderer.domElement.removeEventListener("click", this._onCanvasClick);
    this._resizeObserver.disconnect();
    this.clearPoints();
    this.renderer.dispose();
    this.controls.dispose();
  }
}
