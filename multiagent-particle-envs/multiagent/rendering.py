"""
2D rendering framework using Pyglet 2.0+
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import math
import numpy as np
import pyglet
from pyglet.gl import *


class Viewer:
    """
    A window for rendering 2D graphics
    """
    
    def __init__(self, width: int, height: int, display: Optional[str] = None):
        """
        Initialize a viewer with given dimensions.
        
        Args:
            width: The width of the window in pixels
            height: The height of the window in pixels
            display: The display specification (usually None)
        """
        self.width = width
        self.height = height
        self.display = pyglet.canvas.get_display()
        
        # Create window
        self.window = pyglet.window.Window(
            width=width, 
            height=height, 
            display=self.display,
            vsync=True,
            resizable=True
        )
        
        # Set window callbacks
        self.window.on_close = self.window_closed_by_user
        
        # Track geoms and one-time geoms
        self.geoms: List = []
        self.onetime_geoms: List = []
        
        # Set up transform
        self.transform = Transform()
        
        # Set up OpenGL
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        
    def close(self) -> None:
        """Close the viewer window"""
        self.window.close()
        
    def window_closed_by_user(self) -> None:
        """Handle window close event"""
        self.close()
        
    def set_bounds(self, left: float, right: float, bottom: float, top: float) -> None:
        """
        Set the bounds for the view transformation.
        
        Args:
            left: The left bound of the visible area
            right: The right bound of the visible area
            bottom: The bottom bound of the visible area
            top: The top bound of the visible area
        """
        assert right > left and top > bottom
        
        # Calculate scaling factors
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        
        # Update transform
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley),
            scale=(scalex, scaley)
        )
        
    def add_geom(self, geom) -> None:
        """
        Add a geom to the viewer (persists until removed)
        
        Args:
            geom: The geom to add
        """
        self.geoms.append(geom)
        
    def add_onetime(self, geom) -> None:
        """
        Add a geom to the viewer for one frame only
        
        Args:
            geom: The geom to add temporarily
        """
        self.onetime_geoms.append(geom)
        
    def render(self, return_rgb_array: bool = False) -> Optional[np.ndarray]:
        """
        Render the scene
        
        Args:
            return_rgb_array: Whether to return the rendered scene as an RGB array
            
        Returns:
            An RGB array if return_rgb_array is True, otherwise None
        """
        # Clear the screen
        glClearColor(1.0, 1.0, 1.0, 1.0)
        self.window.clear()
        
        # Set up viewing area
        self.window.switch_to()
        self.window.dispatch_events()
        
        # Draw all objects
        self.transform.enable()
        
        # Draw permanent geoms
        for geom in self.geoms:
            geom.render()
            
        # Draw onetime geoms
        for geom in self.onetime_geoms:
            geom.render()
            
        self.transform.disable()
        
        # Return pixel array if requested
        arr = None
        if return_rgb_array:
            # Get pixel data from buffer
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            
            # Convert to numpy array
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            
            # Convert RGBA to RGB and flip vertically
            arr = arr[::-1, :, 0:3]
            
        # Swap buffers
        self.window.flip()
        
        # Clear onetime geoms
        self.onetime_geoms = []
        
        return arr
        
    # Convenience drawing methods
    def draw_circle(self, radius: float = 10, res: int = 30, filled: bool = True, **attrs) -> 'FilledPolygon':
        """Draw a circle geom"""
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom
        
    def draw_polygon(self, v: List[Tuple[float, float]], filled: bool = True, **attrs) -> Union['FilledPolygon', 'PolyLine']:
        """Draw a polygon geom"""
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom
        
    def draw_polyline(self, v: List[Tuple[float, float]], **attrs) -> 'PolyLine':
        """Draw a polyline geom"""
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom
        
    def draw_line(self, start: Tuple[float, float], end: Tuple[float, float], **attrs) -> 'Line':
        """Draw a line geom"""
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom


class Geom:
    """Base class for all geometric objects"""
    
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
        
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
            
    def render1(self):
        raise NotImplementedError
        
    def add_attr(self, attr):
        self.attrs.append(attr)
        
    def set_color(self, r: float, g: float, b: float, alpha: float = 1.0):
        self._color.vec4 = (r, g, b, alpha)


class Attr:
    """Base class for attributes like color and transform"""
    
    def enable(self):
        raise NotImplementedError
        
    def disable(self):
        pass


class Transform(Attr):
    """Transform attribute for geoms"""
    
    def __init__(
        self, 
        translation: Tuple[float, float] = (0.0, 0.0), 
        rotation: float = 0.0, 
        scale: Tuple[float, float] = (1, 1)
    ):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
        
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)
        glRotatef(math.degrees(self.rotation), 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
        
    def disable(self):
        glPopMatrix()
        
    def set_translation(self, newx: float, newy: float):
        self.translation = (float(newx), float(newy))
        
    def set_rotation(self, new: float):
        self.rotation = float(new)
        
    def set_scale(self, newx: float, newy: float):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    """Color attribute for geoms"""
    
    def __init__(self, vec4: Tuple[float, float, float, float]):
        self.vec4 = vec4
        
    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    """Line style attribute for geoms"""
    
    def __init__(self, style: int):
        self.style = style
        
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
        
    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    """Line width attribute for geoms"""
    
    def __init__(self, stroke: float):
        self.stroke = stroke
        
    def enable(self):
        glLineWidth(self.stroke)


class Point(Geom):
    """Point geom"""
    
    def __init__(self):
        super().__init__()
        
    def render1(self):
        glBegin(GL_POINTS)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    """Filled polygon geom"""
    
    def __init__(self, v: List[Tuple[float, float]]):
        super().__init__()
        self.v = v
        
    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
            
        for p in self.v:
            glVertex3f(p[0], p[1], 0)
            
        glEnd()
        
        # Draw outline with darker color
        color = (
            self._color.vec4[0] * 0.5,
            self._color.vec4[1] * 0.5,
            self._color.vec4[2] * 0.5,
            self._color.vec4[3] * 0.5
        )
        glColor4f(*color)
        
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)
        glEnd()


class PolyLine(Geom):
    """Polyline geom"""
    
    def __init__(self, v: List[Tuple[float, float]], close: bool):
        super().__init__()
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
        
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)
        glEnd()
        
    def set_linewidth(self, x: float):
        self.linewidth.stroke = x


class Line(Geom):
    """Line geom"""
    
    def __init__(self, start: Tuple[float, float] = (0.0, 0.0), end: Tuple[float, float] = (0.0, 0.0)):
        super().__init__()
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
        
    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


# Helper functions to create geoms
def make_circle(radius: float = 10, res: int = 30, filled: bool = True) -> Union[FilledPolygon, PolyLine]:
    """Create a circle geom"""
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
        
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v: List[Tuple[float, float]], filled: bool = True) -> Union[FilledPolygon, PolyLine]:
    """Create a polygon geom"""
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v: List[Tuple[float, float]]) -> PolyLine:
    """Create a polyline geom"""
    return PolyLine(v, False)


def _add_attrs(geom: Geom, attrs: Dict[str, Any]) -> None:
    """Add attributes to a geom"""
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])