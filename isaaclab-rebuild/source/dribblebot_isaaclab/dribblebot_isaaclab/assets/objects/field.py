"""IsaacLab-native soccer-field and goalpost visual assets."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from ..paths import asset_path


FIELD_TEXTURE_PATH = asset_path("textures/field.png")


@configclass
class TexturedFieldCfg(SpawnerCfg):
    """Configuration for one UV-mapped, non-colliding field plane."""

    func = None
    texture_file: str = MISSING
    size: tuple[float, float] = (8.5333336, 5.533679)
    roughness: float = 0.8


@clone
def spawn_textured_field(
    prim_path: str,
    cfg: TexturedFieldCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a textured USD mesh without adding a second contact surface."""

    del kwargs
    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    half_x, half_y = 0.5 * float(cfg.size[0]), 0.5 * float(cfg.size[1])
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-half_x, -half_y, 0.0),
            Gf.Vec3f(half_x, -half_y, 0.0),
            Gf.Vec3f(half_x, half_y, 0.0),
            Gf.Vec3f(-half_x, half_y, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(
        [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)]
    )

    material = UsdShade.Material.Define(stage, f"{prim_path}/FieldMaterial")
    surface = UsdShade.Shader.Define(stage, f"{prim_path}/FieldMaterial/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(cfg.roughness))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    texture = UsdShade.Shader.Define(stage, f"{prim_path}/FieldMaterial/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(cfg.texture_file))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    primvar = UsdShade.Shader.Define(stage, f"{prim_path}/FieldMaterial/Primvar")
    primvar.CreateIdAttr("UsdPrimvarReader_float2")
    primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    primvar.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        primvar.ConnectableAPI(), "result"
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return stage.GetPrimAtPath(prim_path)


SOCCER_FIELD_VISUAL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/SoccerFieldVisual",
    spawn=TexturedFieldCfg(func=spawn_textured_field, texture_file=str(FIELD_TEXTURE_PATH)),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002)),
)


def _goal_bar_cfg(prim_name: str, size: tuple[float, float, float], pos: tuple[float, float, float]):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.35, dynamic_friction=0.35, restitution=0.85
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.88)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


GOAL_EAST_NORTH_CFG = _goal_bar_cfg("GoalEastNorth", (0.04, 0.08, 1.0), (4.01, 1.04, 0.5))
GOAL_EAST_SOUTH_CFG = _goal_bar_cfg("GoalEastSouth", (0.04, 0.08, 1.0), (4.01, -1.04, 0.5))
GOAL_EAST_CROSSBAR_CFG = _goal_bar_cfg("GoalEastCrossbar", (0.04, 2.16, 0.04), (4.01, 0.0, 1.02))
GOAL_WEST_NORTH_CFG = _goal_bar_cfg("GoalWestNorth", (0.04, 0.08, 1.0), (-4.01, 1.04, 0.5))
GOAL_WEST_SOUTH_CFG = _goal_bar_cfg("GoalWestSouth", (0.04, 0.08, 1.0), (-4.01, -1.04, 0.5))
GOAL_WEST_CROSSBAR_CFG = _goal_bar_cfg("GoalWestCrossbar", (0.04, 2.16, 0.04), (-4.01, 0.0, 1.02))


__all__ = [
    "SOCCER_FIELD_VISUAL_CFG",
    "GOAL_EAST_NORTH_CFG",
    "GOAL_EAST_SOUTH_CFG",
    "GOAL_EAST_CROSSBAR_CFG",
    "GOAL_WEST_NORTH_CFG",
    "GOAL_WEST_SOUTH_CFG",
    "GOAL_WEST_CROSSBAR_CFG",
]
