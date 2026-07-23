"""Web-based robot visualization using viser with MuJoCo XML support."""

from __future__ import annotations

import math
import os
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
import trimesh.visual

from legged_gym import LEGGED_GYM_ROOT_DIR

try:
    import viser
    import viser.transforms as vtf
    HAS_VISER = True
except ImportError:
    HAS_VISER = False


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


@dataclass
class Body:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    parent: Optional[str]
    joint_name: Optional[str]
    joint_axis: Optional[np.ndarray]
    joint_range: Optional[Tuple[float, float]]


@dataclass
class Geom:
    body_name: str
    mesh_path: Optional[str]
    geom_pos: np.ndarray
    geom_quat: np.ndarray
    size: Optional[np.ndarray]
    rgba: Optional[np.ndarray]
    material: Optional[str]


class MjcfKinematicModel:
    def __init__(self, xml_path: str, dof_names: Optional[List[str]] = None):
        self.xml_path = xml_path
        self.xml_dir = os.path.dirname(os.path.abspath(xml_path))
        self._dof_names = dof_names
        
        self.bodies: Dict[str, Body] = {}
        self.geoms: List[Geom] = []
        self.meshes: Dict[str, str] = {}
        self.materials: Dict[str, np.ndarray] = {}
        self._dof_indices: Dict[str, int] = {}
        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self._default_joint_axis: Dict[str, np.ndarray] = {}
        
        self._parse_xml()
        self._assign_dof_indices()
        
    def _parse_xyz(self, s: Optional[str]) -> np.ndarray:
        if s is None:
            return np.zeros(3)
        return np.array([float(x) for x in s.split()])

    def _parse_rpy(self, s: Optional[str]) -> np.ndarray:
        if s is None:
            return np.zeros(3)
        return np.array([float(x) for x in s.split()])

    def _rpy_to_quat(self, rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])

    def _parse_xml(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        compiler = root.find('compiler')
        self.meshdir = compiler.get('meshdir', 'meshes') if compiler is not None else 'meshes'

        self._parse_defaults(root)
        self._parse_assets(root)
        self._parse_worldbody(root)
        
    def _parse_defaults(self, root: ET.Element) -> None:
        default = root.find('default')
        if default is None:
            return
        
        parent_axis = np.array([0.0, 1.0, 0.0])
        
        def parse_default_class(elem: ET.Element, parent_joint_axis: np.ndarray, class_prefix: str = "") -> None:
            class_name = elem.get('class', class_prefix)
            
            joint = elem.find('joint')
            if joint is not None:
                axis_str = joint.get('axis')
                if axis_str:
                    current_axis = self._parse_xyz(axis_str)
                else:
                    current_axis = parent_joint_axis.copy()
                self._default_joint_axis[class_name] = current_axis
            elif class_name:
                self._default_joint_axis[class_name] = parent_joint_axis.copy()
            
            for child in elem.findall('default'):
                parse_default_class(child, 
                    self._default_joint_axis.get(class_name, parent_joint_axis), 
                    child.get('class', ''))
        
        for child in default.findall('default'):
            parse_default_class(child, parent_axis)
        
    def _parse_assets(self, root: ET.Element) -> None:
        asset = root.find('asset')
        if asset is None:
            return
        
        for material in asset.findall('material'):
            name = material.get('name')
            rgba_str = material.get('rgba')
            if name and rgba_str:
                self.materials[name] = self._parse_xyz(rgba_str)

        for mesh in asset.findall('mesh'):
            name = mesh.get('name')
            filename = mesh.get('file')
            if name and filename:
                self.meshes[name] = filename

    def _parse_worldbody(self, root: ET.Element) -> None:
        worldbody = root.find('worldbody')
        if worldbody is None:
            return
        
        for body_elem in worldbody.findall('body'):
            self._parse_body(body_elem, parent_name=None)

    def _parse_body(self, body_elem: ET.Element, parent_name: Optional[str]) -> None:
        name = body_elem.get('name')
        if name is None:
            return
        
        pos = self._parse_xyz(body_elem.get('pos', '0 0 0'))
        
        euler_str = body_elem.get('euler')
        quat_str = body_elem.get('quat')
        
        if quat_str is not None:
            quat_parts = np.array([float(x) for x in quat_str.split()])
            if len(quat_parts) == 3:
                quat = np.array([1, quat_parts[0], quat_parts[1], quat_parts[2]])
            else:
                quat = quat_parts
        elif euler_str is not None:
            quat = self._rpy_to_quat(self._parse_rpy(euler_str))
        else:
            quat = np.array([1, 0, 0, 0])

        joint_elem = body_elem.find('joint')
        joint_name = joint_elem.get('name') if joint_elem is not None else None
        joint_axis = None
        joint_range = None
        
        if joint_elem is not None:
            axis_str = joint_elem.get('axis')
            if axis_str:
                joint_axis = self._parse_xyz(axis_str)
            else:
                class_name = joint_elem.get('class', '')
                joint_axis = self._default_joint_axis.get(class_name)
                if joint_axis is None:
                    for key in self._default_joint_axis:
                        if key in class_name or class_name in key:
                            joint_axis = self._default_joint_axis[key]
                            break
            
            range_str = joint_elem.get('range')
            if range_str:
                parts = range_str.split()
                joint_range = (float(parts[0]), float(parts[1]))

        self.bodies[name] = Body(
            name=name,
            pos=pos,
            quat=quat,
            parent=parent_name,
            joint_name=joint_name,
            joint_axis=joint_axis,
            joint_range=joint_range,
        )

        for geom in body_elem.findall('geom'):
            self._parse_geom(geom, name)

        for child_body in body_elem.findall('body'):
            self._parse_body(child_body, parent_name=name)

    def _parse_geom(self, geom_elem: ET.Element, body_name: str) -> None:
        mesh_name = geom_elem.get('mesh')
        
        mesh_path = None
        if mesh_name and mesh_name in self.meshes:
            mesh_path = os.path.join(self.meshdir, self.meshes[mesh_name])

        pos = self._parse_xyz(geom_elem.get('pos', '0 0 0'))
        
        euler_str = geom_elem.get('euler')
        quat_str = geom_elem.get('quat')
        
        if quat_str is not None:
            quat_parts = np.array([float(x) for x in quat_str.split()])
            if len(quat_parts) == 3:
                quat = np.array([1, quat_parts[0], quat_parts[1], quat_parts[2]])
            else:
                quat = quat_parts
        elif euler_str is not None:
            quat = self._rpy_to_quat(self._parse_rpy(euler_str))
        else:
            quat = np.array([1, 0, 0, 0])

        size_str = geom_elem.get('size')
        size = self._parse_xyz(size_str) if size_str else None

        rgba = None
        rgba_str = geom_elem.get('rgba')
        if rgba_str:
            rgba = self._parse_xyz(rgba_str)
        else:
            material_name = geom_elem.get('material')
            if material_name and material_name in self.materials:
                rgba = self.materials[material_name]

        self.geoms.append(Geom(
            body_name=body_name,
            mesh_path=mesh_path,
            geom_pos=pos,
            geom_quat=quat,
            size=size,
            rgba=rgba,
            material=geom_elem.get('material'),
        ))

    def _assign_dof_indices(self) -> None:
        if self._dof_names is None:
            return
        for i, name in enumerate(self._dof_names):
            self._dof_indices[name] = i

    def _quat_to_matrix(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def load_body_meshes(self) -> Dict[str, trimesh.Trimesh]:
        body_meshes: Dict[str, List[trimesh.Trimesh]] = {}
        
        for geom in self.geoms:
            if geom.mesh_path is None:
                continue
                
            full_path = os.path.join(self.xml_dir, geom.mesh_path)
            if not os.path.exists(full_path):
                continue
            
            if full_path in self._mesh_cache:
                mesh = self._mesh_cache[full_path].copy()
            else:
                try:
                    mesh = trimesh.load(full_path, force='mesh')
                    self._mesh_cache[full_path] = mesh.copy()
                except Exception:
                    continue
            
            T_geom = np.eye(4)
            T_geom[:3, :3] = self._quat_to_matrix(geom.geom_quat)
            T_geom[:3, 3] = geom.geom_pos
            mesh.apply_transform(T_geom)
            
            if geom.rgba is not None:
                # use a matte (non-metallic, fully rough) PBR material instead of
                # vertex colors - otherwise the mesh picks up the environment map's
                # default glossy reflectance and looks washed out / overexposed
                base_color = np.clip(geom.rgba, 0, 1).astype(np.float64)
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorFactor=tuple(base_color),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                )
            
            if geom.body_name not in body_meshes:
                body_meshes[geom.body_name] = []
            body_meshes[geom.body_name].append(mesh)
        
        result = {}
        for body_name, meshes in body_meshes.items():
            if meshes:
                result[body_name] = trimesh.util.concatenate(meshes)
        return result

    def _axis_angle_to_quat(self, axis: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])

    def forward_kinematics(
        self,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        world_pos = {}
        world_quat = {}
        
        root_bodies = [n for n, b in self.bodies.items() if b.parent is None]
        if not root_bodies:
            root_bodies = [list(self.bodies.keys())[0]] if self.bodies else []
        
        for root in root_bodies:
            world_pos[root] = base_pos.copy()
            world_quat[root] = base_quat_wxyz.copy()
        
        changed = True
        while changed:
            changed = False
            for name, body in self.bodies.items():
                if name in world_pos or body.parent is None:
                    continue
                if body.parent not in world_pos:
                    continue
                
                parent_pos = world_pos[body.parent]
                parent_quat = world_quat[body.parent]
                parent_rot = self._quat_to_matrix(parent_quat)
                
                if body.joint_name and body.joint_axis is not None:
                    dof_idx = self._dof_indices.get(body.joint_name, -1)
                    if dof_idx >= 0 and dof_idx < len(dof_pos):
                        joint_angle = float(dof_pos[dof_idx])
                        joint_quat = self._axis_angle_to_quat(body.joint_axis, joint_angle)
                        effective_quat = self._quat_mul(joint_quat, body.quat)
                    else:
                        effective_quat = body.quat
                else:
                    effective_quat = body.quat
                
                world_quat[name] = self._quat_mul(parent_quat, effective_quat)
                
                # body.pos is in PARENT frame, only apply parent rotation
                world_pos[name] = parent_pos + parent_rot @ body.pos
                
                changed = True
        
        return {name: (world_pos[name], world_quat[name]) for name in world_pos}


class ViserViewer:
    def __init__(
        self,
        xml_path: str,
        dof_names: Optional[List[str]] = None,
        num_envs: int = 1,
        server: Optional[object] = None,
        port: int = 8080,
    ):
        if not HAS_VISER:
            raise ImportError("viser is required. Install: pip install viser")

        self.num_envs = num_envs
        self.xml_path = xml_path

        self.kin_model = MjcfKinematicModel(xml_path, dof_names)
        self.body_meshes = self.kin_model.load_body_meshes()

        if server is not None:
            self.server = server
        else:
            self.server = viser.ViserServer(port=port)

        self._body_handles: Dict[str, object] = {}
        self._env_frames: List[object] = []

        self._build_scene()

    def _build_scene(self) -> None:
        for env_idx in range(self.num_envs):
            prefix = f"/env_{env_idx}" if self.num_envs > 1 else ""
            frame = self.server.scene.add_frame(f"{prefix}/robot", show_axes=False)
            self._env_frames.append(frame)

            for body_name, mesh in self.body_meshes.items():
                path = f"{prefix}/robot/{body_name}"
                handle = self.server.scene.add_mesh_trimesh(
                    path,
                    mesh,
                    cast_shadow=True,
                    receive_shadow=True,
                )
                self._body_handles[(env_idx, body_name)] = handle

        self._grid_handle = self.server.scene.add_grid(
            "/ground",
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=0.2,
            plane_opacity=0.4,
        )

        # viser's built-in default lights are on by default and were washing out
        # the robot's materials regardless of environment map / added light
        # intensity - disable them and light the scene explicitly instead
        self.server.scene.configure_default_lights(enabled=False)

        # use the HDRI only for lighting (reflections/shading on the robot's
        # materials) - background=False keeps the studio backdrop (walls, floor,
        # light stands) from being rendered, so we get a plain black sky instead
        self.server.scene.configure_environment_map(
            hdri="studio",
            background=False,
            environment_intensity=0.35,
        )
        self.server.scene.set_background_image(np.zeros((2, 2, 3), dtype=np.uint8))
        # Dimmed sun: the terrain top faces are lit directly from above, so a
        # bright sun washed the dark-green ground out to a pale green when viewed
        # from the top.  1.0 keeps the robot readable while letting the terrain
        # stay dark.
        self.server.scene.add_light_directional(
            "/sun",
            color=(255, 255, 255),
            intensity=1.0,
            cast_shadow=True,
            position=(3.0, -3.0, 5.0),
        )

        self._setup_camera()
        self._setup_camera_gui()
        self._setup_keyboard_control()
        self._setup_live_telemetry()
        self._setup_respawn_button()

    def _setup_camera(self) -> None:
        self._camera_offset = np.array([1.3, 1.3, 0.9])
        self._camera_look_at_offset = np.array([0.0, 0.0, 0.3])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = self._camera_offset.copy()
            client.camera.look_at = self._camera_look_at_offset.copy()
            client.camera.fov = np.radians(60.0)

    def _setup_camera_gui(self) -> None:
        """Add camera tracking and FOV controls."""
        self._camera_tracking_enabled = False
        self._camera_fov_degrees = 60.0

        with self.server.gui.add_folder("Camera"):
            cb_tracking = self.server.gui.add_checkbox(
                "Track robot",
                initial_value=self._camera_tracking_enabled,
            )

            @cb_tracking.on_update
            def _(_) -> None:
                self._camera_tracking_enabled = cb_tracking.value

            slider_fov = self.server.gui.add_slider(
                "FOV (\u00b0)",
                min=30.0,
                max=120.0,
                step=1.0,
                initial_value=self._camera_fov_degrees,
            )

            @slider_fov.on_update
            def _(_) -> None:
                self._camera_fov_degrees = slider_fov.value
                for client in self.server.get_clients().values():
                    client.camera.fov = np.radians(slider_fov.value)

    def _setup_keyboard_control(self) -> None:
        """TFGH/RY keyboard driving of the velocity command.

        T/G = forward/back, F/H = strafe left/right, R/Y = turn left/right,
        Space = hard stop.  (WASD is deliberately avoided - viser's own 3D
        fly-controls already bind those keys; T/F/G/H keep the same inverted-T
        layout on free keys.)

        Viser only forwards key-*down* (no key-up).  Hold is inferred from
        browser key-repeat.  Browsers only auto-repeat the *last* key, so a
        second key used to drop the first; we promote already-active keys to
        **sticky latches** when a new key is pressed so multi-axis chords work
        (e.g. hold T then tap R keeps forward).  Sticky keys clear on a second
        press of the same key or Space.
        """
        # command envelope in the robot base frame (m/s, m/s, rad/s)
        self._max_vel = {"fwd": 1.5, "back": 1.0, "lat": 0.7, "yaw": 1.5}
        self._tau_accel = 0.10   # ramp-up time constant (s)
        self._tau_decel = 0.30   # coast-down time constant (s)

        # "Held" is inferred from browser key-repeat.  A held key emits one
        # keydown, then (after the OS *initial* repeat delay, ~0.25-0.6 s) a fast
        # stream of repeats.  We bridge that first gap with a long grace, but as
        # soon as repeats are confirmed (two triggers closer than
        # ``_repeat_gap_max``) we switch to a short grace so releasing is
        # detected almost immediately and the command coasts to zero promptly.
        self._repeat_gap_max = 0.15   # max gap counted as auto-repeat (s)
        self._grace_initial = 0.6     # release window before repeats confirmed
        self._grace_repeat = 0.18     # release window once repeating (slightly > typical gap)
        # When a new key is pressed, any other key seen this recently is promoted
        # to sticky (bridges the gap left by browser single-key-repeat).
        self._multi_key_promote_window = 0.40

        keys = ("T", "F", "G", "H", "R", "Y")
        self._drive_keys = keys
        self._key_last_ts: Dict[str, float] = {k: -1e9 for k in keys}
        self._key_repeating: Dict[str, bool] = {k: False for k in keys}
        # Sticky latches: stay on without key-repeat (multi-key chords).
        self._key_sticky: Dict[str, bool] = {k: False for k in keys}
        self._cmd_vel = np.zeros(3, dtype=float)  # ramped [vx, vy, yaw]
        self._get_command_last_t: Optional[float] = None

        key_labels = {
            "T": "Forward (T)",
            "G": "Backward (G)",
            "F": "Strafe left (F)",
            "H": "Strafe right (H)",
            "R": "Turn left (R)",
            "Y": "Turn right (Y)",
        }

        with self.server.gui.add_folder("Keyboard Drive (TFGH/RY)", expand_by_default=True):
            self.server.gui.add_markdown(
                "**Hold** a key to move; combining keys keeps prior axes on "
                "(sticky). Press the same key again or **Space** to clear.\n\n"
                "- **T / G** forward / back\n"
                "- **F / H** strafe left / right\n"
                "- **R / Y** turn left / right\n"
                "- **Space** hard stop (clears all axes)\n\n"
                "_Click the 3D view first so it has keyboard focus._"
            )

            def _make_press(key: str):
                def _(_event) -> None:
                    self._on_drive_key_event(key, now=time.perf_counter())
                return _

            for key, label in key_labels.items():
                cmd = self.server.gui.add_command(label, hotkey=key)
                cmd.on_trigger(_make_press(key))

            stop_cmd = self.server.gui.add_command("Stop (Space)", hotkey="space")

            def _stop(_event) -> None:
                self.clear_drive_command()

            stop_cmd.on_trigger(_stop)

    def clear_drive_command(self) -> None:
        """Zero sticky/held keyboard drive state and the ramped command."""
        if not hasattr(self, "_key_last_ts"):
            return
        for k in self._key_last_ts:
            self._key_last_ts[k] = -1e9
            self._key_repeating[k] = False
            self._key_sticky[k] = False
        self._cmd_vel[:] = 0.0

    def _is_drive_key_active(self, key: str, now: Optional[float] = None) -> bool:
        """True if key contributes to the command (sticky or grace-held)."""
        if not hasattr(self, "_key_last_ts") or key not in self._key_last_ts:
            return False
        if self._key_sticky.get(key, False):
            return True
        if now is None:
            now = time.perf_counter()
        elapsed = now - self._key_last_ts[key]
        grace = self._grace_repeat if self._key_repeating[key] else self._grace_initial
        return elapsed <= grace

    def _was_recently_active(self, key: str, now: float) -> bool:
        """Broader window used when deciding multi-key sticky promotion."""
        if self._key_sticky.get(key, False):
            return True
        if self._key_last_ts[key] < 0:
            return False
        return (now - self._key_last_ts[key]) <= self._multi_key_promote_window

    def _on_drive_key_event(self, key: str, now: Optional[float] = None) -> None:
        """Handle a single key-down / auto-repeat event for multi-key drive.

        Pure control-flow (no GUI); unit-tested.  When a *new* key is pressed
        while others are active, those others become sticky so the browser's
        single-key-repeat does not zero their axes.
        """
        if key not in self._key_last_ts:
            return
        if now is None:
            now = time.perf_counter()
        last = self._key_last_ts[key]
        is_repeat = (last > 0.0) and ((now - last) <= self._repeat_gap_max)

        if is_repeat:
            self._key_repeating[key] = True
            self._key_last_ts[key] = now
            return

        # Fresh press (not auto-repeat).  Second press on a sticky key clears it.
        if self._key_sticky.get(key, False):
            self._key_sticky[key] = False
            self._key_last_ts[key] = -1e9
            self._key_repeating[key] = False
            return

        # Multi-key: promote other recently-active keys to sticky latches.
        promote_window = getattr(self, "_multi_key_promote_window", 0.40)
        for k in self._key_last_ts:
            if k == key:
                continue
            if self._was_recently_active(k, now):
                self._key_sticky[k] = True

        self._key_last_ts[key] = now
        self._key_repeating[key] = False

    def _setup_live_telemetry(self) -> None:
        """Read-out of commanded vs. achieved base velocity, as numbers and as
        two small live time-series plots (forward speed and yaw rate)."""
        self._plot_window = 200   # samples kept (~a few seconds at 60 Hz)
        self._plot_stride = 3     # push new plot data every N frames (~20 Hz)
        self._plot_counter = 0
        self._telemetry_t0 = time.perf_counter()

        self._hist = {
            k: deque(maxlen=self._plot_window)
            for k in ("t", "vx_cmd", "vx_act", "yaw_cmd", "yaw_act")
        }
        # Seed with two points so uPlot has a valid initial range to draw.
        for _ in range(2):
            for k in self._hist:
                self._hist[k].append(0.0)

        with self.server.gui.add_folder("Command vs Achieved", expand_by_default=True):
            self._telemetry_fields = {
                "cmd": self.server.gui.add_text(
                    "Command  (vx, vy, yaw)",
                    initial_value="vx=+0.00  vy=+0.00  yaw=+0.00",
                    disabled=True,
                ),
                "act": self.server.gui.add_text(
                    "Achieved (vx, vy, yaw)",
                    initial_value="vx=+0.00  vy=+0.00  yaw=+0.00",
                    disabled=True,
                ),
            }

            import viser.uplot as uplot

            t0 = np.asarray(self._hist["t"], dtype=np.float64)
            cmd_series = uplot.Series(
                label="cmd", stroke="#f59e0b", width=2, dash=(4.0, 4.0)
            )
            act_series = uplot.Series(label="act", stroke="#3b82f6", width=2)
            x_series = uplot.Series()

            # x is elapsed *seconds*, not wall-clock: disable uPlot's time scale
            # (otherwise the axis renders as "1/1/70 2:00am" style timestamps).
            scales = {"x": uplot.Scale(time=False)}
            x_axis = uplot.Axis(label="t (s)")

            self._plot_vx = self.server.gui.add_uplot(
                data=(t0, np.asarray(self._hist["vx_cmd"]), np.asarray(self._hist["vx_act"])),
                series=(x_series, cmd_series, act_series),
                title="Forward speed vx (m/s)",
                aspect=2.2,
                scales=scales,
                axes=(x_axis, uplot.Axis()),
                legend=uplot.Legend(show=True),
            )
            self._plot_yaw = self.server.gui.add_uplot(
                data=(t0, np.asarray(self._hist["yaw_cmd"]), np.asarray(self._hist["yaw_act"])),
                series=(x_series, cmd_series, act_series),
                title="Yaw rate (rad/s)",
                aspect=2.2,
                scales=scales,
                axes=(x_axis, uplot.Axis()),
                legend=uplot.Legend(show=True),
            )

    def update_live_telemetry(self, linear_velocity, yaw_rate, command=None) -> None:
        """Publish commanded (``command`` = [vx, vy, yaw]) and achieved base
        velocity as numbers and append them to the live plots."""
        linear = np.asarray(linear_velocity, dtype=float)
        yaw = float(yaw_rate)
        cmd = np.zeros(3) if command is None else np.asarray(command, dtype=float)

        self._telemetry_fields["cmd"].value = (
            f"vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  yaw={cmd[2]:+.2f}"
        )
        self._telemetry_fields["act"].value = (
            f"vx={linear[0]:+.2f}  vy={linear[1]:+.2f}  yaw={yaw:+.2f}"
        )

        t = time.perf_counter() - self._telemetry_t0
        self._hist["t"].append(t)
        self._hist["vx_cmd"].append(float(cmd[0]))
        self._hist["vx_act"].append(float(linear[0]))
        self._hist["yaw_cmd"].append(float(cmd[2]))
        self._hist["yaw_act"].append(yaw)

        self._plot_counter += 1
        if self._plot_counter % self._plot_stride == 0:
            ts = np.asarray(self._hist["t"], dtype=np.float64)
            self._plot_vx.data = (
                ts,
                np.asarray(self._hist["vx_cmd"]),
                np.asarray(self._hist["vx_act"]),
            )
            self._plot_yaw.data = (
                ts,
                np.asarray(self._hist["yaw_cmd"]),
                np.asarray(self._hist["yaw_act"]),
            )

    def _setup_respawn_button(self) -> None:
        """Add a manual "Respawn" button that triggers self._respawn_callback."""
        self._respawn_callback: Optional[object] = None
        self._taxonomy_spawn_callback: Optional[object] = None

        with self.server.gui.add_folder("Reset", expand_by_default=True):
            respawn_button = self.server.gui.add_button("Respawn")

            @respawn_button.on_click
            def _(_) -> None:
                if self._respawn_callback is not None:
                    self._respawn_callback()

    def set_respawn_callback(self, callback) -> None:
        """Register a callback invoked when the "Respawn" button is clicked."""
        self._respawn_callback = callback

    def setup_taxonomy_spawn_panel(
        self,
        type_names: Optional[Sequence[str]] = None,
        num_levels: int = 4,
        initial_level: int = 0,
        initial_type: int = 0,
    ) -> None:
        """GUI to teleport the play robot onto a taxonomy tile (type × level).

        Call ``set_taxonomy_spawn_callback(fn)`` with
        ``fn(level: int, type_idx: int)`` to wire the teleport.
        """
        from legged_gym.utils.terrain import TAXONOMY_TYPE_NAMES, TAXONOMY_NUM_LEVELS

        names = list(type_names) if type_names is not None else list(TAXONOMY_TYPE_NAMES)
        n_levels = int(num_levels) if num_levels else TAXONOMY_NUM_LEVELS
        level_options = [f"L{i}" for i in range(n_levels)]
        init_level = int(max(0, min(initial_level, n_levels - 1)))
        init_type = int(max(0, min(initial_type, len(names) - 1)))

        with self.server.gui.add_folder("Spawn (taxonomy)", expand_by_default=True):
            self.server.gui.add_markdown(
                "Teleport the robot to a **type × level** tile of the taxonomy "
                "showcase grid."
            )
            dd_type = self.server.gui.add_dropdown(
                "Terrain type",
                options=names,
                initial_value=names[init_type],
            )
            dd_level = self.server.gui.add_dropdown(
                "Difficulty",
                options=level_options,
                initial_value=level_options[init_level],
            )
            status = self.server.gui.add_text(
                "Last spawn",
                initial_value="(none yet)",
                disabled=True,
            )
            btn = self.server.gui.add_button("Teleport", color="green")

            @btn.on_click
            def _(_) -> None:
                try:
                    type_idx = names.index(dd_type.value)
                except ValueError:
                    type_idx = 0
                try:
                    level = level_options.index(dd_level.value)
                except ValueError:
                    level = 0
                ok = False
                if self._taxonomy_spawn_callback is not None:
                    ok = bool(self._taxonomy_spawn_callback(level, type_idx))
                label = f"{names[type_idx]} L{level}"
                status.value = f"{'OK' if ok else 'FAILED'}: {label}"
                print(f"[viser] taxonomy spawn → {label} (ok={ok})")

        self._taxonomy_spawn_gui = {
            "type": dd_type,
            "level": dd_level,
            "status": status,
        }

    def set_taxonomy_spawn_callback(self, callback) -> None:
        """Register ``callback(level: int, type_idx: int) -> bool`` for Teleport."""
        self._taxonomy_spawn_callback = callback

    def get_command(self) -> np.ndarray:
        """Current velocity command [lin_vel_x, lin_vel_y, ang_vel_yaw] derived
        from the TFGH/RY keys.

        Each active key defines a per-axis *target*; the returned command ramps
        towards that target (fast when accelerating, slower when coasting) so
        the robot eases in and, on release, decelerates smoothly to zero rather
        than snapping.  Active = sticky latch or grace-held (see
        ``_on_drive_key_event``).
        """
        if not hasattr(self, "_key_last_ts"):
            return np.array([0.0, 0.0, 0.0])

        now = time.perf_counter()
        if self._get_command_last_t is None:
            dt = 0.0
        else:
            dt = now - self._get_command_last_t
        self._get_command_last_t = now
        dt = float(min(max(dt, 0.0), 0.1))

        def held(key: str) -> bool:
            if self._key_sticky.get(key, False):
                return True
            elapsed = now - self._key_last_ts[key]
            grace = self._grace_repeat if self._key_repeating[key] else self._grace_initial
            is_held = elapsed <= grace
            # Only clear the auto-repeat latch after the *long* timeout, so that
            # crossing the short repeat-grace doesn't snap the window back open
            # and re-trigger the hold.  A new press then re-primes from scratch.
            if elapsed > self._grace_initial:
                self._key_repeating[key] = False
            return is_held

        m = self._max_vel
        target = np.array([
            (m["fwd"] if held("T") else 0.0) - (m["back"] if held("G") else 0.0),
            (m["lat"] if held("F") else 0.0) - (m["lat"] if held("H") else 0.0),
            (m["yaw"] if held("R") else 0.0) - (m["yaw"] if held("Y") else 0.0),
        ])

        for i in range(3):
            tgt = target[i]
            cur = self._cmd_vel[i]
            # accelerate quickly toward a larger magnitude, coast down slowly
            tau = self._tau_accel if abs(tgt) > abs(cur) else self._tau_decel
            alpha = 1.0 - math.exp(-dt / tau) if (dt > 0.0 and tau > 1e-6) else 1.0
            self._cmd_vel[i] = cur + (tgt - cur) * alpha

        return self._cmd_vel.copy()

    @staticmethod
    def _ensure_lit_material(mesh: trimesh.Trimesh) -> None:
        """Guarantee the terrain exports with a matte, non-metallic PBR material
        and vertex normals.  glTF's default material (used when a mesh has none,
        e.g. a plain ColorVisuals mesh) is fully metallic, which renders pitch
        black when the environment map is used for lighting only.  We keep any
        existing per-vertex colors as a COLOR_0 attribute so height/terrain
        shading survives, but force a lit material like the robot meshes use."""
        visual = getattr(mesh, 'visual', None)
        material = getattr(visual, 'material', None)
        if not isinstance(material, trimesh.visual.material.PBRMaterial):
            existing_colors = None
            if isinstance(visual, trimesh.visual.ColorVisuals) and \
                    visual.vertex_colors is not None and len(visual.vertex_colors):
                existing_colors = np.asarray(visual.vertex_colors)
            pbr = trimesh.visual.material.PBRMaterial(
                baseColorFactor=(1.0, 1.0, 1.0, 1.0),
                metallicFactor=0.0,
                roughnessFactor=1.0,
                doubleSided=True,
            )
            mesh.visual = trimesh.visual.TextureVisuals(material=pbr)
            if existing_colors is not None:
                mesh.visual.vertex_attributes['color'] = existing_colors
        # process=False leaves vertex normals empty; glTF then renders unlit/dark.
        _ = mesh.vertex_normals

    def set_terrain_mesh(self, terrain_mesh: Optional[trimesh.Trimesh]) -> None:
        if terrain_mesh is None:
            return

        self._ensure_lit_material(terrain_mesh)

        if hasattr(self, '_terrain_handle') and self._terrain_handle is not None:
            self._terrain_handle.remove()

        self._terrain_handle = self.server.scene.add_mesh_trimesh(
            "/terrain",
            terrain_mesh,
            cast_shadow=True,
            receive_shadow=True,
        )
        if hasattr(self, '_grid_handle') and self._grid_handle is not None:
            self._grid_handle.visible = False

    def set_terrain_from_heightfield(
        self,
        height_samples: np.ndarray,
        horizontal_scale: float = 0.1,
        vertical_scale: float = 0.005,
        return_mesh: bool = False,
        stride: int = 1,
    ):
        # Full-resolution heightfields (e.g. 1200x1200 ~ 2.9M triangles) are too
        # heavy for the browser and get silently dropped, so the terrain looks
        # flat.  Downsampling by ``stride`` keeps world alignment (x/y scale by
        # stride) while cutting the triangle count to something renderable.
        #
        # Indexing must match ``Terrain`` / Genesis: height_samples[x_idx, y_idx]
        # where axis-0 is world X (terrain rows / length) and axis-1 is world Y
        # (terrain cols / width).  The old meshgrid default swapped X/Y, so the
        # visual mesh sat at the wrong place and robots looked buried in the mesh
        # even when physics feet were on the real heightfield.
        if stride > 1:
            height_samples = height_samples[::stride, ::stride]
            horizontal_scale = horizontal_scale * stride
        hfield = height_samples.astype(np.float64) * vertical_scale
        nx, ny = hfield.shape

        x = np.arange(nx) * horizontal_scale
        y = np.arange(ny) * horizontal_scale
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = hfield

        vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        xi, yi = np.mgrid[:nx - 1, :ny - 1]
        i0 = (xi * ny + yi).ravel()
        faces = np.column_stack([
            i0, i0 + 1, i0 + ny + 1,
            i0, i0 + ny + 1, i0 + ny,
        ]).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        heights = vertices[:, 2]
        h_min, h_max = heights.min(), heights.max()
        if h_max - h_min > 1e-6:
            normalized = (heights - h_min) / (h_max - h_min)
        else:
            normalized = np.zeros_like(heights)
        # Dark-green terrain: keep a subtle height gradient but stay in a deep,
        # muted green range (roughly RGB (9,28,14) .. (21,50,23)) rather than the
        # old bright pastel green.  Even lit from directly above by the sun this
        # reads as a dark forest green.
        colors = np.zeros((len(heights), 4), dtype=np.uint8)
        colors[:, 0] = (9 + 12 * normalized).astype(np.uint8)
        colors[:, 1] = (28 + 22 * (1 - normalized)).astype(np.uint8)
        colors[:, 2] = (14 + 9 * normalized).astype(np.uint8)
        colors[:, 3] = 255

        # A bare ColorVisuals mesh exports to glTF with NO material, so the
        # viewer falls back to glTF's default material - which is fully metallic
        # (metallicFactor=1.0).  With the environment map used for lighting only
        # (background=False) there's almost nothing for a metal surface to
        # reflect, so the terrain rendered pitch black.  Mirror the robot meshes:
        # attach a matte, non-metallic PBR material (baseColor white) and keep
        # the per-vertex height colors as a COLOR_0 attribute, which three.js
        # multiplies with the white base color - so we get the height gradient
        # AND correct lighting.  Also trigger vertex-normal computation
        # (process=False leaves them empty, which glTF renders unlit / dark).
        # doubleSided: the heightfield triangle winding makes the lit front face
        # point *down*, so a single-sided material renders the terrain black when
        # viewed from above (back faces get culled -> you see the void) and only
        # green from below.  Render both sides so the ground is visible and lit
        # from any camera angle.
        pbr = trimesh.visual.material.PBRMaterial(
            baseColorFactor=(1.0, 1.0, 1.0, 1.0),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=pbr)
        mesh.visual.vertex_attributes['color'] = colors
        _ = mesh.vertex_normals

        if return_mesh:
            return mesh
        self.set_terrain_mesh(mesh)

    def update(
        self,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        dof_pos: np.ndarray,
        env_idx: int = 0,
        flush: bool = True,
    ) -> None:
        fk_results = self.kin_model.forward_kinematics(
            base_pos, base_quat_wxyz, dof_pos
        )

        with self.server.atomic():
            for body_name, (pos, quat) in fk_results.items():
                handle = self._body_handles.get((env_idx, body_name))
                if handle is not None:
                    handle.position = pos
                    handle.wxyz = quat

            if self._camera_tracking_enabled:
                for client in self.server.get_clients().values():
                    client.camera.position = base_pos + self._camera_offset
                    client.camera.look_at = base_pos + self._camera_look_at_offset

        if flush:
            self.server.flush()

    def update_from_simulator(
        self,
        env,
        env_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Publish the *current* state of selected training environments.

        This reads state directly from the active simulator, rather than from a
        checkpoint or a second playback scene.  Transfer all selected robots in
        three batched GPU-to-CPU copies, then flush a single Viser update.
        """
        if env_indices is None:
            env_indices = [0]
        indices = list(env_indices)[:self.num_envs]
        if not indices:
            return

        device = env.simulator.base_pos.device
        idx = torch.as_tensor(indices, device=device, dtype=torch.long)
        base_pos = env.simulator.base_pos[idx].detach().cpu().numpy()
        base_quat_xyzw = env.simulator.base_quat[idx].detach().cpu().numpy()
        dof_pos = env.simulator.dof_pos[idx].detach().cpu().numpy()

        for viewer_idx, (pos, quat_xyzw, joints) in enumerate(
            zip(base_pos, base_quat_xyzw, dof_pos)
        ):
            self.update(
                pos,
                _xyzw_to_wxyz(quat_xyzw),
                joints,
                env_idx=viewer_idx,
                flush=False,
            )
        self.server.flush()

    def set_taxonomy_labels(self, label_map) -> None:
        """Place per-tile text labels (~1.5 m above origins) in the Viser scene.

        Genesis has no in-world text draw API; Viser ``add_label`` is the
        interactive fallback so each showcase tile shows its type+level name.
        """
        if not label_map:
            return
        # Clear previous labels if re-called.
        for handle in getattr(self, "_taxonomy_label_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        self._taxonomy_label_handles = []
        # Colored marker spheres at the same height as a visual type key.
        for entry in label_map:
            pos = np.asarray(entry["position"], dtype=np.float64)
            label = str(entry["label"])
            row = int(entry.get("row", 0))
            col = int(entry.get("col", 0))
            try:
                handle = self.server.scene.add_label(
                    f"/taxonomy/label_{row}_{col}",
                    text=label,
                    position=pos,
                    font_size_mode="scene",
                    font_scene_height=0.35,
                    anchor="center-center",
                )
                self._taxonomy_label_handles.append(handle)
            except Exception as e:
                print(f"[viser_viewer] taxonomy label failed ({label}): {e}")
            color = entry.get("color", (1.0, 0.0, 0.0, 1.0))
            rgba = tuple(int(max(0, min(255, c * 255))) for c in color[:3])
            try:
                sphere = self.server.scene.add_icosphere(
                    f"/taxonomy/marker_{row}_{col}",
                    radius=0.12,
                    position=pos,
                    color=rgba,
                )
                self._taxonomy_label_handles.append(sphere)
            except Exception:
                pass

    def stop(self) -> None:
        if hasattr(self, 'server'):
            self.server.stop()


def create_viser_viewer(
    env,
    port: int = 8080,
    env_indices: Optional[Sequence[int]] = None,
) -> ViserViewer:
    xml_path = env.cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    
    dof_names = env.cfg.asset.dof_names

    if env_indices is None:
        env_indices = [0]

    viewer = ViserViewer(
        xml_path=xml_path,
        dof_names=dof_names,
        num_envs=len(env_indices),
        port=port,
    )

    try:
        terrain = getattr(env.simulator, '_terrain', None)
        if terrain is None:
            return viewer

        mesh_type = env.cfg.terrain.mesh_type

        if mesh_type == 'plane':
            pass
        elif mesh_type == 'trimesh' and hasattr(terrain, 'terrain_mesh'):
            import copy
            terrain_mesh = copy.deepcopy(terrain.terrain_mesh)
            transform = np.zeros((3,))
            transform[0] = -env.cfg.terrain.border_size
            transform[1] = -env.cfg.terrain.border_size
            transform[2] = 0.0
            translation = trimesh.transformations.translation_matrix(transform)
            terrain_mesh.apply_transform(translation)
            viewer.set_terrain_mesh(terrain_mesh)
        elif mesh_type == 'heightfield' and hasattr(terrain, 'heightsamples'):
            transform = np.zeros((3,))
            transform[0] = -env.cfg.terrain.border_size
            transform[1] = -env.cfg.terrain.border_size
            transform[2] = 0.0
            # Pick a stride that keeps the mesh under a triangle budget the
            # browser can actually render.  A fixed stride=3 is fine for a small
            # play grid, but a full training grid (e.g. 1200x1200 samples) still
            # yields ~300k+ triangles at stride=3, which Viser/three.js silently
            # drops - the ground then vanishes and the robot looks like it is
            # sinking into a void.  Scale the stride up with the sample count.
            nx, ny = terrain.heightsamples.shape
            max_verts = 90_000  # ~180k triangles, comfortably renderable
            stride = max(3, int(np.ceil(np.sqrt((nx * ny) / max_verts))))
            mesh = viewer.set_terrain_from_heightfield(
                terrain.heightsamples,
                horizontal_scale=env.cfg.terrain.horizontal_scale,
                vertical_scale=env.cfg.terrain.vertical_scale,
                return_mesh=True,
                stride=stride,
            )
            if mesh is not None:
                translation = trimesh.transformations.translation_matrix(transform)
                mesh.apply_transform(translation)
                viewer.set_terrain_mesh(mesh)

        # Taxonomy showcase labels (Viser text; Genesis has no draw_debug_text).
        from legged_gym.utils.terrain import (
            build_taxonomy_label_map,
            is_taxonomy_terrain_cfg,
            TAXONOMY_LABEL_Z_OFFSET,
            TAXONOMY_TYPE_NAMES,
        )
        if is_taxonomy_terrain_cfg(env.cfg.terrain):
            label_map = getattr(terrain, "taxonomy_labels", None)
            if not label_map and hasattr(terrain, "env_origins"):
                label_map = build_taxonomy_label_map(
                    terrain.env_origins, z_offset=TAXONOMY_LABEL_Z_OFFSET
                )
            if label_map:
                viewer.set_taxonomy_labels(label_map)
                # Elevated oblique default looking at the grid centre.
                length = float(env.cfg.terrain.terrain_length)
                width = float(env.cfg.terrain.terrain_width)
                cx = 0.5 * env.cfg.terrain.num_rows * length
                cy = 0.5 * env.cfg.terrain.num_cols * width
                viewer._camera_offset = np.array([cx - 18.0, cy - 22.0, 32.0]) - np.array([cx, cy, 0.0])
                # Absolute initial pose for new clients (not robot-relative).
                @viewer.server.on_client_connect
                def _taxonomy_cam(client, _cx=cx, _cy=cy) -> None:
                    client.camera.position = np.array([_cx - 18.0, _cy - 22.0, 32.0])
                    client.camera.look_at = np.array([_cx, _cy, 0.0])
                    client.camera.fov = np.radians(55.0)
            viewer.setup_taxonomy_spawn_panel(
                type_names=TAXONOMY_TYPE_NAMES,
                num_levels=int(env.cfg.terrain.num_rows),
                initial_level=0,
                initial_type=0,
            )
    except Exception as e:
        print(f"[viser_viewer] Warning: Could not load terrain: {e}")

    return viewer
