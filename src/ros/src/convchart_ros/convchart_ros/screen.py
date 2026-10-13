"""The screen the GUI windows open on: its size, and the scale at which a canvas fills it."""
from __future__ import annotations

import ctypes


def screen_size() -> tuple[int, int] | None:
    """The X screen's size in pixels, or None without a display."""
    try:
        x11 = ctypes.CDLL("libX11.so.6")
    except OSError:
        return None
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XDefaultScreen.argtypes = x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XDisplayWidth.argtypes = x11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
    display = x11.XOpenDisplay(None)
    if not display:
        return None
    screen = x11.XDefaultScreen(display)
    size = x11.XDisplayWidth(display, screen), x11.XDisplayHeight(display, screen)
    x11.XCloseDisplay(display)
    return size


def fill_scale(width: int, height: int, screen: tuple[int, int] | None) -> float:
    """The scale at which a width x height canvas fills the screen, short of the desktop's bars and the title bar."""
    if screen is None:
        return 1.0
    return max(0.5, min((screen[0] - 40) / width, (screen[1] - 130) / height))
