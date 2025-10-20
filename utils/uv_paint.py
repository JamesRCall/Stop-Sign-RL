# utils/uv_paint.py
from dataclasses import dataclass
from typing import Tuple

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch*2 for ch in h)
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

@dataclass(frozen=True)
class UVPaint:
    name: str
    day_hex: str            # how it looks in daylight (pre-activation)
    active_hex: str         # how it looks under UV
    translucent: bool = True
    day_alpha: float = 0.35
    active_alpha: float = 0.80

    @property
    def day_rgb(self) -> Tuple[int,int,int]:
        return hex_to_rgb(self.day_hex)

    @property
    def active_rgb(self) -> Tuple[int,int,int]:
        return hex_to_rgb(self.active_hex)

# EXAMPLES — define only the pairs you’ll allow
VIOLET_GLOW = UVPaint("VioletGlow", day_hex="#C9C9C9", active_hex="#7B00FF", translucent=True, day_alpha=0.30, active_alpha=0.85)
GREEN_GLOW  = UVPaint("GreenGlow",  day_hex="#D0D0D0", active_hex="#00FF7F", translucent=True, day_alpha=0.25, active_alpha=0.85)

# You can also define “non-UV” solid colors as pairs with same day/active:
DEEP_RED    = UVPaint("DeepRed",    day_hex="#8B0000", active_hex="#8B0000", translucent=False)
CHARCOAL    = UVPaint("Charcoal",   day_hex="#333333", active_hex="#333333", translucent=False)
