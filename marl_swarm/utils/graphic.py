import numpy as np
from OpenGL.GL import (
    GL_LINES,
    GL_QUADS,
    glBegin,
    glColor3f,
    glColor4f,
    glEnd,
    glTranslatef,
    glVertex3f,
    glVertex3fv,
    glPushMatrix,
    glPopMatrix,
)
from OpenGL.GLU import gluNewQuadric
from OpenGL.raw.GLU import gluSphere


def axes():
    """Draw axes on the opengl simulation view for [0,size] coordinate system."""
    glBegin(GL_LINES)

    # Z axis (blue)
    glColor3f(0, 0, 1.0)
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 0, 1))

    # X axis (green)
    glColor3f(0, 1.0, 0)
    glVertex3fv((0, 0, 0))
    glVertex3fv((1, 0, 0))

    # Y axis (red)
    glColor3f(1.0, 0, 0)
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 1, 0))

    glEnd()


def field(size):
    """Draw the field on the opengl simulation view for [0,size] coordinate system.

    Args:
        size: int the size of the side field
    """
    # Draw ground plane at z=0
    glBegin(GL_QUADS)
    glColor4f(0.2, 0.2, 0.2, 0.5)  # Dark gray ground
    glVertex3f(0, 0, 0)
    glVertex3f(size, 0, 0)
    glVertex3f(size, size, 0)
    glVertex3f(0, size, 0)
    glEnd()

    # Draw grid lines
    glColor3f(0.7, 0.7, 0.7)  # Light gray grid
    glBegin(GL_LINES)
    for i in range(size + 1):  # Draw lines from 0 to size
        # Lines along X axis
        glVertex3f(0, i, 0.01)
        glVertex3f(size, i, 0.01)
        
        # Lines along Y axis
        glVertex3f(i, 0, 0.01)
        glVertex3f(i, size, 0.01)
    glEnd()


def point(point):
    """Draw the drone as a little red dot with a stick to visualize better the projection on the grid.

    Args:
        point: tuple x,y,z position
    """
    sphere = gluNewQuadric()
    # glTranslatef(-point[1], point[0], point[2] - 2)
    glTranslatef(point[0], point[1], point[2])
    glColor4f(0.0, 0.0, 1.0, 1)
    gluSphere(sphere, 0.1, 32, 16)

    glBegin(GL_LINES)
    # glColor4f(0.5, 0.2, 0.2, 0.3)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -2 - point[2])
    glEnd()

def terminated_point(point):
    """Draw the terminated drone as a green dot with a stick to visualize better the projection on the grid.

    Args:
        point: tuple x,y,z position
    """
    sphere = gluNewQuadric()
    # glTranslatef(-point[1], point[0], point[2] - 2)
    glTranslatef(point[0], point[1], point[2])
    glColor4f(0.0, 1.0, 0.0, 1)  # Green color for terminated agents
    gluSphere(sphere, 0.1, 32, 16)

    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -2 - point[2])
    glEnd()


# def target_point(point):
#     """Draw the target point as a bigger yellow dot with a stick to visualize better the projection on the grid.
# 
#     Args:
#         point: tuple x,y,z position
#     """
#     sphere = gluNewQuadric()
#     glTranslatef(-point[1], point[0], point[2] - 2)
#     glColor4f(0.6, 0.6, 0, 0.7)
#     gluSphere(sphere, 0.2, 32, 16)
# 
#     glBegin(GL_LINES)
#     glColor4f(0.7, 0.7, 0, 0.3)
#     glVertex3f(0, 0, 0)
#     glVertex3f(0, 0, -2 - point[2])
#     glEnd()

# def box_obstacle(point, size):
#     """Draw a box obstacle on the opengl simulation view.
#
#     Args:
#         point: tuple x,y,z position
#         size: int the size of the box
#     """
#     glBegin(GL_QUADS)
#     glVertex3f(point[0] - size, point[1] - size, -2)
#     glVertex3f(point[0] + size, point[1] - size, -2)
#     glVertex3f(point[0] + size, point[1] + size, -2)
#     glVertex3f(point[0] - size, point[1] + size, -2)
#     glEnd()
#
#     glColor3f(1.0, 1.0, 1.0)
#     glBegin(GL_LINES)
#     for i in np.arange(-size, size, size):
#         glVertex3f(point[0] - size, point[1] + i, -1.99)
#         glVertex3f(point[0] + size, point[1] + i, -1.99)
#
#         glVertex3f(point[0] + i, point[1] + size, -1.99)
#         glVertex3f(point[0] + i, point[1] - size, -1.99)
#
#     glEnd()

# def box_obstacle(point, size):
#     """Draw a box obstacle on the opengl simulation view.

#     Args:
#         point: tuple x,y,z position
#         size: int the size of the box
#     """

#     size = size/2

#     # Draw the faces of the box in red
#     glColor3f(1.0, 0.0, 0.0)
#     glBegin(GL_QUADS)
#     # Bottom face
#     glVertex3f(point[0] - size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] - size)

#     # Top face
#     glVertex3f(point[0] - size, point[1] - size, point[2] + size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] + size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] + size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] + size)

#     # Front face
#     glVertex3f(point[0] - size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] + size)
#     glVertex3f(point[0] - size, point[1] - size, point[2] + size)

#     # Back face
#     glVertex3f(point[0] - size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] + size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] + size)

#     # Left face
#     glVertex3f(point[0] - size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] + size)
#     glVertex3f(point[0] - size, point[1] - size, point[2] + size)

#     # Right face
#     glVertex3f(point[0] + size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] + size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] + size)
#     glEnd()

#     # Draw edges in black
#     glColor3f(0, 0, 0)
#     glBegin(GL_LINES)
#     # Bottom edges
#     glVertex3f(point[0] - size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] - size)

#     glVertex3f(point[0] + size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] - size)

#     glVertex3f(point[0] + size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] - size)

#     glVertex3f(point[0] - size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] - size, point[2] - size)

#     # Top edges
#     glVertex3f(point[0] - size, point[1] - size, point[2] + size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] + size)

#     glVertex3f(point[0] + size, point[1] - size, point[2] + size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] + size)

#     glVertex3f(point[0] + size, point[1] + size, point[2] + size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] + size)

#     glVertex3f(point[0] - size, point[1] + size, point[2] + size)
#     glVertex3f(point[0] - size, point[1] - size, point[2] + size)

#     # Vertical edges
#     glVertex3f(point[0] - size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] - size, point[2] + size)

#     glVertex3f(point[0] + size, point[1] - size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] - size, point[2] + size)

#     glVertex3f(point[0] + size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] + size, point[1] + size, point[2] + size)

#     glVertex3f(point[0] - size, point[1] + size, point[2] - size)
#     glVertex3f(point[0] - size, point[1] + size, point[2] + size)
#     glEnd()


def box_obstacle(point, size):
    """Draw a box obstacle on the opengl simulation view.

    Args:
        point: tuple x,y,z position
        size: int the size of the box
    """
    half_size = size / 2
    glPushMatrix()
    # Apply same transformation as the drone; adjust if needed.
    # glTranslatef(-point[1], point[0], point[2] - 2)
    glTranslatef(point[0], point[1], point[2])

    # Draw faces in red
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_QUADS)
    # Bottom face
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(-half_size, half_size, -half_size)

    # Top face
    glVertex3f(-half_size, -half_size, half_size)
    glVertex3f(half_size, -half_size, half_size)
    glVertex3f(half_size, half_size, half_size)
    glVertex3f(-half_size, half_size, half_size)

    # Front face
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, half_size)
    glVertex3f(-half_size, -half_size, half_size)

    # Back face
    glVertex3f(-half_size, half_size, -half_size)
    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(half_size, half_size, half_size)
    glVertex3f(-half_size, half_size, half_size)

    # Left face
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(-half_size, half_size, -half_size)
    glVertex3f(-half_size, half_size, half_size)
    glVertex3f(-half_size, -half_size, half_size)

    # Right face
    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(half_size, half_size, half_size)
    glVertex3f(half_size, -half_size, half_size)
    glEnd()

    # Draw edges in black
    glColor3f(0, 0, 0)
    glBegin(GL_LINES)
    # Bottom edges
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, -half_size)

    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, half_size, -half_size)

    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(-half_size, half_size, -half_size)

    glVertex3f(-half_size, half_size, -half_size)
    glVertex3f(-half_size, -half_size, -half_size)

    # Top edges
    glVertex3f(-half_size, -half_size, half_size)
    glVertex3f(half_size, -half_size, half_size)

    glVertex3f(half_size, -half_size, half_size)
    glVertex3f(half_size, half_size, half_size)

    glVertex3f(half_size, half_size, half_size)
    glVertex3f(-half_size, half_size, half_size)

    glVertex3f(-half_size, half_size, half_size)
    glVertex3f(-half_size, -half_size, half_size)

    # Vertical edges
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(-half_size, -half_size, half_size)

    glVertex3f(half_size, -half_size, -half_size)
    glVertex3f(half_size, -half_size, half_size)

    glVertex3f(half_size, half_size, -half_size)
    glVertex3f(half_size, half_size, half_size)

    glVertex3f(-half_size, half_size, -half_size)
    glVertex3f(-half_size, half_size, half_size)
    glEnd()
    glPopMatrix()