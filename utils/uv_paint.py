"""UV paint definitions and color helpers."""
from dataclasses import dataclass
from typing import Tuple

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string to an RGB tuple.

    @param h: Hex color string.
    @return: (r, g, b) tuple.
    """
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

@dataclass(frozen=True)
class UVPaint:
    """Paired daylight vs UV-on paint definition."""
    name: str
    day_hex: str            # how it looks in daylight (pre-activation)
    active_hex: str         # how it looks under UV (activated)
    translucent: bool = True
    day_alpha: float = 0.35
    active_alpha: float = 1.0

    @property
    def day_rgb(self) -> Tuple[int, int, int]:
        return hex_to_rgb(self.day_hex)

    @property
    def active_rgb(self) -> Tuple[int, int, int]:
        return hex_to_rgb(self.active_hex)

# Example paint definitions
VIOLET_GLOW = UVPaint(
    "VioletGlow",
    day_hex="#C9C9C9",      # neutral gray (subtle before activation)
    active_hex="#7B00FF",   # vivid violet under UV
    translucent=True,
    day_alpha=0.25,
    active_alpha=1.0,
)

GREEN_GLOW = UVPaint(
    "GreenGlow",
    day_hex="#D0D0D0",
    active_hex="#00FF7F",   # spring green-ish UV glow
    translucent=True,
    day_alpha=0.25,
    active_alpha=1.0,
)

# Additional variants
BLUE_GLOW = UVPaint(
    "BlueGlow",
    day_hex="#D5D5D5",      # faint gray pre-activation
    active_hex="#00BFFF",   # deep sky blue glow under UV
    translucent=True,
    day_alpha=0.25,
    active_alpha=1.0,
)

YELLOW_GLOW = UVPaint(
    "YellowGlow",
    day_hex="#D8D8D8",
    active_hex="#FFD400",   # bright yellow glow under UV
    translucent=True,
    day_alpha=0.25,
    active_alpha=1.0,
)

# Expanded palette for debugging/sweeps
RED_GLOW = UVPaint("RedGlow", day_hex="#D0D0D0", active_hex="#FF3030", translucent=True, day_alpha=0.25, active_alpha=1.0)
ORANGE_GLOW = UVPaint("OrangeGlow", day_hex="#D0D0D0", active_hex="#FF8C00", translucent=True, day_alpha=0.25, active_alpha=1.0)
CYAN_GLOW = UVPaint("CyanGlow", day_hex="#D0D0D0", active_hex="#00FFD5", translucent=True, day_alpha=0.25, active_alpha=1.0)
MAGENTA_GLOW = UVPaint("MagentaGlow", day_hex="#D0D0D0", active_hex="#FF00B7", translucent=True, day_alpha=0.25, active_alpha=1.0)
PINK_GLOW = UVPaint("PinkGlow", day_hex="#D0D0D0", active_hex="#FF6FAE", translucent=True, day_alpha=0.25, active_alpha=1.0)
LIME_GLOW = UVPaint("LimeGlow", day_hex="#D0D0D0", active_hex="#9BFF00", translucent=True, day_alpha=0.25, active_alpha=1.0)
CHARTREUSE_GLOW = UVPaint("ChartreuseGlow", day_hex="#D0D0D0", active_hex="#7FFF00", translucent=True, day_alpha=0.25, active_alpha=1.0)
SKY_GLOW = UVPaint("SkyGlow", day_hex="#D0D0D0", active_hex="#87CEEB", translucent=True, day_alpha=0.25, active_alpha=1.0)
NAVY_GLOW = UVPaint("NavyGlow", day_hex="#D0D0D0", active_hex="#001F5C", translucent=True, day_alpha=0.25, active_alpha=1.0)
GOLD_GLOW = UVPaint("GoldGlow", day_hex="#D0D0D0", active_hex="#FFD700", translucent=True, day_alpha=0.25, active_alpha=1.0)
AMBER_GLOW = UVPaint("AmberGlow", day_hex="#D0D0D0", active_hex="#FFBF00", translucent=True, day_alpha=0.25, active_alpha=1.0)
CORAL_GLOW = UVPaint("CoralGlow", day_hex="#D0D0D0", active_hex="#FF7F50", translucent=True, day_alpha=0.25, active_alpha=1.0)
MAROON_GLOW = UVPaint("MaroonGlow", day_hex="#D0D0D0", active_hex="#800000", translucent=True, day_alpha=0.25, active_alpha=1.0)
OLIVE_GLOW = UVPaint("OliveGlow", day_hex="#D0D0D0", active_hex="#808000", translucent=True, day_alpha=0.25, active_alpha=1.0)
MINT_GLOW = UVPaint("MintGlow", day_hex="#D0D0D0", active_hex="#98FF98", translucent=True, day_alpha=0.25, active_alpha=1.0)
AQUA_GLOW = UVPaint("AquaGlow", day_hex="#D0D0D0", active_hex="#00FFFF", translucent=True, day_alpha=0.25, active_alpha=1.0)
PURPLE_GLOW = UVPaint("PurpleGlow", day_hex="#D0D0D0", active_hex="#8000FF", translucent=True, day_alpha=0.25, active_alpha=1.0)
INDIGO_GLOW = UVPaint("IndigoGlow", day_hex="#D0D0D0", active_hex="#4B0082", translucent=True, day_alpha=0.25, active_alpha=1.0)
SLATE_GLOW = UVPaint("SlateGlow", day_hex="#D0D0D0", active_hex="#708090", translucent=True, day_alpha=0.25, active_alpha=1.0)
BROWN_GLOW = UVPaint("BrownGlow", day_hex="#D0D0D0", active_hex="#8B4513", translucent=True, day_alpha=0.25, active_alpha=1.0)
TAN_GLOW = UVPaint("TanGlow", day_hex="#D0D0D0", active_hex="#D2B48C", translucent=True, day_alpha=0.25, active_alpha=1.0)
PEACH_GLOW = UVPaint("PeachGlow", day_hex="#D0D0D0", active_hex="#FFDAB9", translucent=True, day_alpha=0.25, active_alpha=1.0)
STEEL_GLOW = UVPaint("SteelGlow", day_hex="#D0D0D0", active_hex="#4682B4", translucent=True, day_alpha=0.25, active_alpha=1.0)
LAVENDER_GLOW = UVPaint("LavenderGlow", day_hex="#D0D0D0", active_hex="#B57EDC", translucent=True, day_alpha=0.25, active_alpha=1.0)

# Focused UV palette for combo testing
NEON_YELLOW_GLOW = UVPaint("NeonYellowGlow", day_hex="#D0D0D0", active_hex="#E6FF00", translucent=True, day_alpha=0.25, active_alpha=1.0)
BLUE_LIGHT_GLOW = UVPaint("BlueLightGlow", day_hex="#D0D0D0", active_hex="#7FDBFF", translucent=True, day_alpha=0.25, active_alpha=1.0)
BLUE_MED_GLOW = UVPaint("BlueMedGlow", day_hex="#D0D0D0", active_hex="#0074D9", translucent=True, day_alpha=0.25, active_alpha=1.0)
BLUE_DARK_GLOW = UVPaint("BlueDarkGlow", day_hex="#D0D0D0", active_hex="#001F3F", translucent=True, day_alpha=0.25, active_alpha=1.0)
