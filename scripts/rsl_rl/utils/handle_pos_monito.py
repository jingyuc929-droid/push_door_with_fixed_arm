#------------ Used for print the real-time pose of the handle in the GUI ----------------------------------
import time
import omni
import omni.ui as ui
import omni.usd
from omni.isaac.dynamic_control import _dynamic_control

dc = _dynamic_control.acquire_dynamic_control_interface()

def _auto_find_door_path():
    """Try to auto-detect an articulation prim path for env_0 door."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    candidates = []
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        name = prim.GetName().lower()
        # heuristic: look for env_0 and prim name containing door
        if "/env_0" in p and ("door" in name):
            candidates.append(p)
    # prefer shorter (closer to articulation root)
    candidates.sort(key=len)
    return candidates[0] if candidates else None

class HandleJointMonitor:
    def __init__(self, door_prim_path: str = None, dof_name: str = "handle_joint", update_hz: float = 20.0):
        self.door_prim_path = door_prim_path
        self.dof_name = dof_name
        self.dt = 1.0 / max(1e-3, float(update_hz))
        self._last_t = 0.0

        self._art = None
        self._dof = None

        # UI
        self.win = ui.Window("Handle Joint Monitor", width=320, height=140)
        with self.win.frame:
            with ui.VStack(spacing=6, height=0):
                self.lbl_status = ui.Label("Status: init...")
                self.lbl_path = ui.Label("Door: (unknown)")
                self.lbl_pos = ui.Label("pos: --")
                self.lbl_vel = ui.Label("vel: --")

        stream = omni.kit.app.get_app().get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update)

    def _resolve(self):
        # auto-detect path if not provided
        if not self.door_prim_path:
            self.door_prim_path = _auto_find_door_path()

        if not self.door_prim_path:
            self.lbl_status.text = "Status: waiting for stage/door prim..."
            return False

        self.lbl_path.text = f"Door: {self.door_prim_path}  |  DOF: {self.dof_name}"

        art = dc.get_articulation(self.door_prim_path)
        if art == 0:
            self.lbl_status.text = "Status: articulation not ready (path wrong or not spawned yet)"
            return False

        dof = dc.find_articulation_dof(art, self.dof_name)
        if dof == -1:
            self.lbl_status.text = "Status: DOF not found (check dof_name)"
            return False

        self._art = art
        self._dof = dof
        self.lbl_status.text = "Status: OK"
        return True

    def _on_update(self, _evt):
        now = time.time()
        if now - self._last_t < self.dt:
            return
        self._last_t = now

        if (self._dof is None) or (self._art is None):
            if not self._resolve():
                return

        # read dof state from PhysX
        st = dc.get_dof_state(self._dof, _dynamic_control.STATE_ALL)
        self.lbl_pos.text = f"pos: {st.pos:.6f}"
        self.lbl_vel.text = f"vel: {st.vel:.6f}"

_monitor = None

def start_handle_joint_monitor(door_prim_path: str = None, dof_name: str = "handle_joint", update_hz: float = 20.0):
    """Call once after IsaacSim app is up and the stage is loading."""
    global _monitor
    if _monitor is None:
        _monitor = HandleJointMonitor(door_prim_path=door_prim_path, dof_name=dof_name, update_hz=update_hz)
    return _monitor