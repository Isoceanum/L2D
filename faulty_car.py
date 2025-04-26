"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov
"""

import math
import random

import Box2D
import numpy as np

from gymnasium.error import DependencyNotInstalled

from ray_cast_callback import RayCastCallback

from constants import *

try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef, circleShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


class Car:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world: Box2D.b2World = world
        
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY1]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY2]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY3]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY4]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=circleShape(pos=(0, 0), radius=0.1 * SIZE),  # small circle at center
                    density=0.0,       # 0 density so it doesn't affect mass/inertia
                    isSensor=False      # Optional: set to True if you don’t want collisions
                )
            ],
        )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        #self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * SIZE, y * front_k * SIZE)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
            
            self.l2d_center_fixture = self.hull.fixtures[-1]
        

        self.particles = []
        self.target_steer = 0.0  # what the user/agent wants
        self.prev_steer = 0.0
        
        self.l2d_rays = {
            "front": 0.0,
            "left_45": 0.0,
            "right_45": 0.0,
            "left_90": 0.0,
            "right_90": 0.0,
        }

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation
        """
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side
        """
        
        """Control: set a target steering value"""
        self.prev_steer = self.target_steer
        self.target_steer = s
        #self.wheels[0].steer = s
        #self.wheels[1].steer = s

    def step(self, dt):    
        self.l2d_cast_rays()  
 
        for i, w in enumerate(self.wheels):
            if i in [0, 1]:  # Only front wheels should steer
                delta = self.target_steer - w.steer
                w.steer += np.clip(delta, -STEERING_INERTIA * dt, STEERING_INERTIA * dt)
                
        x, y = self.hull.position  # Updated position after physics step
        
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * ENGINE_POWER
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            #self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )
            # Apply passive friction to the car hull if no gas is applied (simulates coasting drag)
            if all(w.gas < 1e-4 for w in self.wheels[2:4]):  # rear wheels = driven
                drag_force = -1.0 * np.array(self.hull.linearVelocity)
                self.hull.ApplyForceToCenter(drag_force, wake=True)   
    
                
    def _get_transformed_path(self, fixture, angle, zoom, translation):
        import pygame.draw
        trans = fixture.body.transform
        path = [trans * v for v in fixture.shape.vertices]
        path = [pygame.math.Vector2(p).rotate_rad(angle) for p in path]
        return [
            (p[0] * zoom + translation[0], p[1] * zoom + translation[1])
            for p in path
        ]


    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (c[0] * zoom + translation[0], c[1] * zoom + translation[1])
                    for c in poly
                ]
                pygame.draw.lines(surface, color=p.color, points=poly, width=2, closed=False)
                
                
                
        self.l2d_draw_rays(surface, zoom, translation, angle)          
        self.l2d_draw_hull(surface, zoom, translation, angle)
        self.l2d_draw_wheels(surface, zoom, translation, angle)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass

        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
        
        
# Learn 2 Drive specific functions 
        
    def l2d_draw_hull(self, surface, zoom, translation, angle):
        import pygame.draw
        color = [int(c * 255) for c in self.hull.color]
        for f in list(self.hull.fixtures)[:-1]: # ignore the last centre fixture 
            if hasattr(f.shape, "vertices"):
                path = self.l2d_get_transformed_path(f, angle, zoom, translation)
                pygame.draw.polygon(surface, color=color, points=path) 
       
            
    def l2d_draw_wheels(self, surface, zoom, translation, angle):
        import pygame.draw
        for wheel in self.wheels:
            color = [int(c * 255) for c in wheel.color]
            for f in wheel.fixtures:
                if hasattr(f.shape, "vertices"):
                    path = self.l2d_get_transformed_path(f, angle, zoom, translation)
                    pygame.draw.polygon(surface, color=color, points=path)

            # ABS overlay (outside the fixture loop, like in your version)
            if not hasattr(wheel, "phase"):
                continue

            a1 = wheel.phase
            a2 = wheel.phase + 1.2
            s1, s2 = math.sin(a1), math.sin(a2)
            c1, c2 = math.cos(a1), math.cos(a2)

            if s1 > 0 and s2 > 0:
                continue
            if s1 > 0: c1 = np.sign(c1)
            if s2 > 0: c2 = np.sign(c2)

            white_poly = [
                (-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
            ]

            white_poly = [wheel.transform * v for v in white_poly]
            white_poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly]
            white_poly = [
                (c[0] * zoom + translation[0], c[1] * zoom + translation[1])
                for c in white_poly
            ]
            pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)


    def l2d_draw_rays(self, surface, zoom, translation, angle):
        import pygame.draw
        fixture = self.l2d_center_fixture
        body = fixture.body
        shape = fixture.shape

        # Car center in world coordinates
        center_world = body.transform * shape.pos
        forward = body.GetWorldVector(Box2D.b2Vec2(0, 1))

        # Define angles (in radians)
        ray_angles = {
            "front": 0.0,
            "left_45": math.radians(45),
            "right_45": math.radians(-45),
            "left_90": math.radians(90),
            "right_90": math.radians(-90),
        }

        for label, offset_rad in ray_angles.items():
            # Get rotated direction
            cos_a = math.cos(offset_rad)
            sin_a = math.sin(offset_rad)
            dir_x = forward[0] * cos_a - forward[1] * sin_a
            dir_y = forward[0] * sin_a + forward[1] * cos_a
            direction = (dir_x, dir_y)

            # Get ray length from precomputed values
            length = self.l2d_rays[label]

            # Compute end point in world coordinates
            end_world = (
                center_world[0] + direction[0] * length,
                center_world[1] + direction[1] * length,
            )

            # Rotate and project to screen space
            center_screen = pygame.math.Vector2(center_world).rotate_rad(angle)
            end_screen = pygame.math.Vector2(end_world).rotate_rad(angle)

            center_screen = (
                center_screen[0] * zoom + translation[0],
                center_screen[1] * zoom + translation[1],
            )
            end_screen = (
                end_screen[0] * zoom + translation[0],
                end_screen[1] * zoom + translation[1],
            )

            # Draw green line and dot
            pygame.draw.line(surface, (0, 255, 0), center_screen, end_screen, width=1)
            pygame.draw.circle(surface, (0, 255, 0), end_screen, radius=5)        
        
    def l2d_cast_rays(self):
        # Car forward direction in world space
        forward = self.hull.GetWorldVector(Box2D.b2Vec2(0, 1))
        center = self.hull.transform * self.l2d_center_fixture.shape.pos

        # Define angles (in radians)
        ray_angles = {
            "front": 0.0,
            "left_45": math.radians(45),
            "right_45": math.radians(-45),
            "left_90": math.radians(90),
            "right_90": math.radians(-90),
        }

        for label, offset_rad in ray_angles.items():
            # Manually rotate forward vector
            cos_a = math.cos(offset_rad)
            sin_a = math.sin(offset_rad)
            direction = (
                forward[0] * cos_a - forward[1] * sin_a,
                forward[0] * sin_a + forward[1] * cos_a
            )

            # Cast and store
            distance = self.l2d_cast_ray(center, direction, L2D_RAY_LENGTH)
            
            # --- Add noise ---
            #distance += np.random.normal(0, L2D_RAY_NOISE_STD_DEV)

            # --- Round to lower resolution ---
            distance = round(distance, L2D_RAY_ROUND_DIGITS)  # 1 decimal precision
            
            self.l2d_rays[label] = distance
            
    def l2d_cast_ray(self, start, direction, length):
        direction = self.l2d_normalize_direction(direction)
        end = (start[0] + direction[0] * length, start[1] + direction[1] * length)
        ray_callback = RayCastCallback(length)
        self.world.RayCast(ray_callback, start, end)
        return ray_callback.distance
    

    def l2d_get_transformed_path(self, fixture, angle, zoom, translation):
        import pygame.draw
        trans = fixture.body.transform
        path = [trans * v for v in fixture.shape.vertices]
        path = [pygame.math.Vector2(p).rotate_rad(angle) for p in path]
        return [
            (p[0] * zoom + translation[0], p[1] * zoom + translation[1])
            for p in path
        ]

    def l2d_normalize_direction(self, direction):
        length = math.sqrt(direction[0]**2 + direction[1]**2)
        return (direction[0] / length, direction[1] / length) if length != 0 else (0, 0)
    
