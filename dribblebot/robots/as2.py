import os

from isaacgym import gymapi

from dribblebot import MINI_GYM_ROOT_DIR

from .robot import Robot


class As2(Robot):
    """Isaac Gym asset adapter for the 12-DOF AS2 quadruped."""

    EXPECTED_ACTUATED_DOFS = 12

    def initialize(self):
        asset_path = self.env.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        if not os.path.isfile(asset_path):
            raise FileNotFoundError(f"AS2 URDF was not found: {asset_path}")

        asset_config = self.env.cfg.asset
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_config.default_dof_drive_mode
        asset_options.collapse_fixed_joints = asset_config.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = asset_config.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = asset_config.flip_visual_attachments
        asset_options.fix_base_link = asset_config.fix_base_link
        asset_options.density = asset_config.density
        asset_options.angular_damping = asset_config.angular_damping
        asset_options.linear_damping = asset_config.linear_damping
        asset_options.max_angular_velocity = asset_config.max_angular_velocity
        asset_options.max_linear_velocity = asset_config.max_linear_velocity
        asset_options.armature = asset_config.armature
        asset_options.thickness = asset_config.thickness
        asset_options.disable_gravity = asset_config.disable_gravity

        # AS2 already uses primitive collision geometry. VHACD is unnecessary
        # and would make asset loading substantially slower.
        asset_options.vhacd_enabled = False

        asset = self.env.gym.load_asset(self.env.sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError(f"Isaac Gym failed to load the AS2 URDF: {asset_path}")

        self.num_dof = self.env.gym.get_asset_dof_count(asset)
        if self.num_dof != self.EXPECTED_ACTUATED_DOFS:
            raise ValueError(
                "AS2 must expose exactly 12 actuated hip/thigh/calf joints after import; "
                f"Isaac Gym reported {self.num_dof}."
            )

        self.num_actuated_dof = self.num_dof
        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)
        dof_props_asset = self.env.gym.get_asset_dof_properties(asset)
        rigid_shape_props_asset = self.env.gym.get_asset_rigid_shape_properties(asset)

        return asset, dof_props_asset, rigid_shape_props_asset
