class MujocoViewer:
    def __init__(self, plant, enabled=True):
        self.viewer = None
        self.enabled = bool(enabled)
        if not self.enabled or not hasattr(plant, "mujoco"):
            return
        try:
            import mujoco.viewer
            try:
                self.viewer = mujoco.viewer.launch_passive(
                    plant.model, plant.data, show_left_ui=False, show_right_ui=False
                )
            except TypeError:
                self.viewer = mujoco.viewer.launch_passive(plant.model, plant.data)
        except Exception as exc:
            raise RuntimeError("MuJoCo viewer launch failed; use --headless") from exc

    def sync(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
