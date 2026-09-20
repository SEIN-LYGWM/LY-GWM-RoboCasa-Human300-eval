"""
Compatibility shim created by M445G_R1A_REPAIR_ROBOMIMIC_OBS_CORE_COMPAT_FAST.

Reason:
  Diffusion Policy RoboCasa policy imports:
      import robomimic.models.obs_core as rmoc

Some pip-installed robomimic layouts expose equivalent observation encoder classes
under robomimic.models.obs_nets instead of robomimic.models.obs_core.

Boundary:
  This shim only re-exports existing robomimic.models.obs_nets symbols.
  It does not fake model behavior and does not train/evaluate anything.
"""

from robomimic.models.obs_nets import *  # noqa: F401,F403

# ===== M445G_R1D_R2 VisualCoreLanguageConditioned wrapper start =====

class VisualCoreLanguageConditioned(VisualCore):
    """
    Minimal compatibility wrapper created by M445G_R1D_R2.

    Why:
      RoboCasa Diffusion Policy switches rgb core_class to
      "VisualCoreLanguageConditioned" when lang_emb exists, but the current
      robomimic installation exposes VisualCore only.

    Boundary:
      This wrapper only makes the robomimic obs encoder registry able to
      instantiate the configured class during workspace dryrun.
      It does not claim full official language-conditioned semantics.
      If the selected backbone_class is ResNet*ConvFiLM and that backbone
      exists, VisualCore will instantiate it through normal robomimic logic.
    """
    pass

try:
    # expose class in obs_core globals
    globals()["VisualCoreLanguageConditioned"] = VisualCoreLanguageConditioned
except Exception:
    pass

# ===== M445G_R1D_R2 VisualCoreLanguageConditioned wrapper end =====
