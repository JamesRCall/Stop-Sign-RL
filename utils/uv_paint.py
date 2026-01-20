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
    active_alpha: float = 0.80

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
