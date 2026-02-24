"""
ui/theme.py
===========
Defines color palettes for dark and light themes.
Returns a dictionary of CSS variable values based on current mode.
Imported by app.py and used to inject theme into the CSS file.
"""


def get_theme_colors(dark_mode: bool) -> dict:
    """
    Returns a color palette dictionary based on the current theme mode.

    Args:
        dark_mode: True for dark theme, False for light theme

    Returns:
        dict of color variable names to hex/rgba values
    """
    if dark_mode:
        return {
            "BG_PRIMARY":     "#0a0a0f",
            "BG_SECONDARY":   "#111118",
            "BG_CARD":        "#16161f",
            "BG_INPUT":       "#1a1a24",
            "BORDER":         "#2a2a3a",
            "TEXT_PRIMARY":   "#f0eeea",
            "TEXT_SECONDARY": "#8b8a9a",
            "TEXT_MUTED":     "#4a4a5a",
            "ACCENT":         "#6c63ff",
            "ACCENT_LIGHT":   "#8b85ff",
            "ACCENT_GLOW":    "rgba(108,99,255,0.15)",
            "GREEN":          "#00d97e",
            "GREEN_BG":       "rgba(0,217,126,0.08)",
            "AMBER":          "#ffb340",
            "AMBER_BG":       "rgba(255,179,64,0.08)",
            "RED":            "#ff5e5e",
            "USER_BG":        "#1a1a2e",
            "USER_BORDER":    "#6c63ff",
            "BOT_BG":         "#0f1a16",
            "BOT_BORDER":     "#00d97e",
            "METRIC_BG":      "#13131e",
            "SIDEBAR_BG":     "#0d0d14",
            "SHADOW":         "rgba(0,0,0,0.5)",
        }
    else:
        return {
            "BG_PRIMARY":     "#f5f4f7",
            "BG_SECONDARY":   "#eeedf2",
            "BG_CARD":        "#ffffff",
            "BG_INPUT":       "#ffffff",
            "BORDER":         "#dddbe5",
            "TEXT_PRIMARY":   "#1a1825",
            "TEXT_SECONDARY": "#6b6880",
            "TEXT_MUTED":     "#a09dba",
            "ACCENT":         "#6c63ff",
            "ACCENT_LIGHT":   "#5a52e0",
            "ACCENT_GLOW":    "rgba(108,99,255,0.12)",
            "GREEN":          "#00a85e",
            "GREEN_BG":       "rgba(0,168,94,0.08)",
            "AMBER":          "#d4820a",
            "AMBER_BG":       "rgba(212,130,10,0.08)",
            "RED":            "#d93636",
            "USER_BG":        "#eeecff",
            "USER_BORDER":    "#6c63ff",
            "BOT_BG":         "#edfaf4",
            "BOT_BORDER":     "#00a85e",
            "METRIC_BG":      "#ffffff",
            "SIDEBAR_BG":     "#f0eff5",
            "SHADOW":         "rgba(100,90,160,0.12)",
        }
