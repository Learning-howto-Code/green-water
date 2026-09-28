import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type PreviewLayer = { index: number; paths: number[][] };

// Colour runs from blue-violet (bottom) through teal to yellow (top).
const LOW = new THREE.Color("#3b2f8f");
const MID = new THREE.Color("#1f9e89");
const HIGH = new THREE.Color("#f2d024");

function layerColor(t: number, out: THREE.Color) {
  return t < 0.5 ? out.lerpColors(LOW, MID, t * 2) : out.lerpColors(MID, HIGH, (t - 0.5) * 2);
}

export class Viewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private lines: THREE.LineSegments | null = null;
  private grid: THREE.GridHelper | null = null;
  private bounds = new THREE.Box3();

  constructor(private el: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
    this.camera.up.set(0, 0, 1); // printer Z is up
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    loop();
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.el;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  show(layers: PreviewLayer[], totalLayers: number, refit: boolean) {
    let count = 0;
    for (const layer of layers) for (const p of layer.paths) count += Math.max(0, p.length / 3 - 1);

    const positions = new Float32Array(count * 6);
    const colors = new Float32Array(count * 6);
    const color = new THREE.Color();
    let i = 0;
    for (const layer of layers) {
      layerColor(layer.index / Math.max(1, totalLayers - 1), color);
      for (const p of layer.paths) {
        for (let k = 0; k + 5 < p.length; k += 3) {
          positions.set(p.slice(k, k + 6), i * 6);
          colors.set([color.r, color.g, color.b, color.r, color.g, color.b], i * 6);
          i++;
        }
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingBox();

    if (this.lines) {
      this.lines.geometry.dispose();
      this.lines.geometry = geometry;
    } else {
      this.lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }));
      this.scene.add(this.lines);
    }
    this.bounds.copy(geometry.boundingBox!);
    this.updateGrid();
    if (refit) this.fit();
  }

  private updateGrid() {
    const size = this.bounds.getSize(new THREE.Vector3());
    const extent = Math.ceil((Math.max(size.x, size.y) + 20) / 10) * 10;
    if (this.grid && (this.grid.userData.extent as number) === extent) return;
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
    }
    this.grid = new THREE.GridHelper(extent, extent / 10, 0x8894a8, 0x8894a8);
    (this.grid.material as THREE.Material).opacity = 0.35;
    (this.grid.material as THREE.Material).transparent = true;
    this.grid.rotation.x = Math.PI / 2; // lie flat in XY
    this.grid.userData.extent = extent;
    this.scene.add(this.grid);
  }

  fit() {
    const center = this.bounds.getCenter(new THREE.Vector3());
    const radius = this.bounds.getBoundingSphere(new THREE.Sphere()).radius || 50;
    const dist = radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.1;
    const dir = new THREE.Vector3(1, -1.4, 0.9).normalize();
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.controls.target.copy(center);
    this.controls.update();
  }
}
