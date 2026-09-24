"""USD field and goal assets sourced from the same URDFs as Isaac Gym."""

from dataclasses import MISSING
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from ..paths import asset_path


FIELD_TEXTURE_PATH = asset_path("textures/field.png")
FIELD_URDF_PATH = asset_path("objects/soccer_field/soccer_field.urdf")
GOAL_URDF_PATH = asset_path("objects/goalpost/goalpost.urdf")
_FIELD_BOX = tuple(float(v) for v in ET.parse(FIELD_URDF_PATH).find(".//collision/geometry/box").get("size").split())
_GOAL_LINK = ET.parse(GOAL_URDF_PATH).find("link")


@configclass
class TexturedFieldCfg(SpawnerCfg):
    """Textured field with the legacy thin box contact surface."""

    func = None
    texture_file: str = MISSING
    size: tuple[float, float] = _FIELD_BOX[:2]
    thickness: float = _FIELD_BOX[2]
    roughness: float = 0.8


@clone
def spawn_textured_field(
    prim_path: str,
    cfg: TexturedFieldCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Match Gym's field box: visual/contact top at z=0.002, ground at z=0."""

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
    collider = sim_utils.CuboidCfg(
        size=(*cfg.size, cfg.thickness),
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="average",
        ),
    )
    collider.func(f"{prim_path}/Collider", collider, translation=(0.0, 0.0, -0.5 * cfg.thickness))
    return stage.GetPrimAtPath(prim_path)


SOCCER_FIELD_VISUAL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/SoccerFieldVisual",
    spawn=TexturedFieldCfg(func=spawn_textured_field, texture_file=str(FIELD_TEXTURE_PATH)),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002)),
)


@configclass
class GoalVisualCfg(SpawnerCfg):
    """Original goal net/frame mesh, separate from the open-mouth collisions."""

    func = None


@clone
def spawn_goal_visual(prim_path, cfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    del cfg, kwargs
    visual = _GOAL_LINK.find("visual")
    geometry = visual.find("geometry/mesh")
    origin = visual.find("origin")
    source = trimesh.load(GOAL_URDF_PATH.parent / geometry.get("filename"), force="mesh", process=False)
    source.apply_scale([float(v) for v in geometry.get("scale").split()])
    transform = trimesh.transformations.euler_matrix(*[float(v) for v in origin.get("rpy").split()])
    transform[:3, 3] = [float(v) for v in origin.get("xyz").split()]
    source.apply_transform(transform)
    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(source.vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(source.faces))
    mesh.CreateFaceVertexIndicesAttr(source.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    color = tuple(float(v) for v in visual.find("material/color").get("rgba").split()[:3])
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    material.func(f"{prim_path}/Material", material)
    sim_utils.bind_visual_material(prim_path, f"{prim_path}/Material")
    # No CollisionAPI: convexifying the net would close the goal mouth.
    return stage.GetPrimAtPath(prim_path)


GOAL_VISUAL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/GoalVisual",
    spawn=GoalVisualCfg(func=spawn_goal_visual),
)


def _goal_bar_cfg(prim_name: str, collision_name: str):
    collision = _GOAL_LINK.find(f"collision[@name='{collision_name}']")
    size = tuple(float(v) for v in collision.find("geometry/box").get("size").split())
    pos = tuple(float(v) for v in collision.find("origin").get("xyz").split())
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="average",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


GOAL_EAST_NORTH_CFG = _goal_bar_cfg("GoalEastNorth", "positive_left_post")
GOAL_EAST_SOUTH_CFG = _goal_bar_cfg("GoalEastSouth", "positive_right_post")
GOAL_EAST_CROSSBAR_CFG = _goal_bar_cfg("GoalEastCrossbar", "positive_crossbar")
GOAL_WEST_NORTH_CFG = _goal_bar_cfg("GoalWestNorth", "negative_left_post")
GOAL_WEST_SOUTH_CFG = _goal_bar_cfg("GoalWestSouth", "negative_right_post")
GOAL_WEST_CROSSBAR_CFG = _goal_bar_cfg("GoalWestCrossbar", "negative_crossbar")


__all__ = [
    "GOAL_VISUAL_CFG",
    "SOCCER_FIELD_VISUAL_CFG",
    "GOAL_EAST_NORTH_CFG",
    "GOAL_EAST_SOUTH_CFG",
    "GOAL_EAST_CROSSBAR_CFG",
    "GOAL_WEST_NORTH_CFG",
    "GOAL_WEST_SOUTH_CFG",
    "GOAL_WEST_CROSSBAR_CFG",
]
