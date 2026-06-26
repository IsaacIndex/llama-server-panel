from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from panel_gui import (  # noqa: E402
    CANVAS,
    BODY_FONT,
    FIELD_LABEL_WIDTH,
    HEADER_COPY_WRAP,
    IBM_BLUE,
    INK,
    INK_MUTED,
    INK_SUBTLE,
    INVERSE_INK,
    MONO_FONT,
    SURFACE_1,
    configure_carbon_style,
)


class RootStub:
    def __init__(self) -> None:
        self.background = ""
        self.options: dict[str, object] = {}

    def configure(self, **kwargs: object) -> None:
        self.background = str(kwargs["bg"])

    def option_add(self, key: str, value: object) -> None:
        self.options[key] = value


class StyleStub:
    def __init__(self) -> None:
        self.configured: dict[str, dict[str, object]] = {}
        self.mapped: dict[str, dict[str, object]] = {}

    def configure(self, style_name: str, **kwargs: object) -> None:
        self.configured[style_name] = kwargs

    def map(self, style_name: str, **kwargs: object) -> None:
        self.mapped[style_name] = kwargs


def relative_luminance(hex_color: str) -> float:
    rgb = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]

    def channel(value: float) -> float:
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = [channel(value) for value in rgb]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(relative_luminance(foreground), relative_luminance(background))
    darker = min(relative_luminance(foreground), relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


class PanelGuiStyleTest(unittest.TestCase):
    def test_style_contract_includes_readable_control_variants(self) -> None:
        root = RootStub()
        style = StyleStub()

        configure_carbon_style(root, style)

        self.assertEqual(root.background, CANVAS)
        self.assertIn("TButton", style.configured)
        self.assertIn("Secondary.TButton", style.configured)
        self.assertIn("Accent.TButton", style.configured)
        self.assertIn("Segment.TButton", style.configured)
        self.assertIn("SelectedSegment.TButton", style.configured)
        self.assertIn("Field.TLabel", style.configured)
        self.assertIn("TEntry", style.configured)
        self.assertEqual(style.configured["TButton"]["foreground"], INK)
        self.assertEqual(style.configured["Secondary.TButton"]["foreground"], INK)
        self.assertEqual(style.configured["Accent.TButton"]["foreground"], INVERSE_INK)
        self.assertEqual(style.configured["SelectedSegment.TButton"]["foreground"], INVERSE_INK)
        self.assertEqual(style.configured["Field.TLabel"]["background"], SURFACE_1)

    def test_status_bar_and_health_styles_are_registered(self) -> None:
        root = RootStub()
        style = StyleStub()

        configure_carbon_style(root, style)

        for name in (
            "StatusBar.TFrame",
            "StatusBarText.TLabel",
            "StatusBarMuted.TLabel",
            "HealthOk.TLabel",
            "HealthWarn.TLabel",
            "HealthBad.TLabel",
            "HealthIdle.TLabel",
        ):
            self.assertIn(name, style.configured)

    def test_core_color_pairs_clear_accessibility_contrast(self) -> None:
        self.assertGreaterEqual(contrast_ratio(INK, CANVAS), 12.0)
        self.assertGreaterEqual(contrast_ratio(INK_MUTED, CANVAS), 7.0)
        self.assertGreaterEqual(contrast_ratio(INK_SUBTLE, CANVAS), 4.5)
        self.assertGreaterEqual(contrast_ratio(INVERSE_INK, IBM_BLUE), 4.5)

    def test_layout_constants_leave_room_for_desktop_utility_text(self) -> None:
        self.assertGreaterEqual(HEADER_COPY_WRAP, 560)
        self.assertLessEqual(HEADER_COPY_WRAP, 760)
        self.assertGreaterEqual(FIELD_LABEL_WIDTH, 12)
        self.assertLess(MONO_FONT[1], BODY_FONT[1])


if __name__ == "__main__":
    unittest.main()
