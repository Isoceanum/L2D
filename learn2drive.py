__credits__ = ["Andrea PIERRÉ"]

import math
import random
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from faulty_car import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

from rewards import l2d_calculate_step_reward
from constants import *
from config import *
from friction_detector import FrictionDetector


try:
    import Box2D
    from Box2D.b2 import fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


class Learn2Drive(gym.Env, EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        
        # Custome props for l2d
        self.l2d_walls = []
        self.l2d_last_segment_idx = 0
        self.l2d_time_since_last_tile = 0.0
        
        # values used for fault injection
        self.l2d_step_count = 0
        self.l2d_inject_fault = random.random() < FAULT_PROBABILITY
        self.fault_step = random.randint(*FAULT_STEP_RANGE) if self.l2d_inject_fault else None
        self.l2d_fault_active = False
        self.l2d_fault_location = None
        
        self.l2d_tile_reward = 0.0
        self.l2d_steps_on_grass = 0
        


        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake
            
            
        # l2d temp state space
        self.observation_space = spaces.Box(
            low=-1000.0, high=np.inf, shape=(L2D_OBSERVATION_SIZE,), dtype=np.float32
        )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        
        for w in self.l2d_walls:
            self.world.DestroyBody(w)
        self.l2d_walls = []

        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False
        

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            
            self.l2d_create_track_barrier(track[i], track[i - 1], i)
            
        self.track = track

        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
            
        # values used for fault injection
        self.l2d_step_count = 0
        self.l2d_inject_fault = random.random() < FAULT_PROBABILITY
        self.fault_step = random.randint(*FAULT_STEP_RANGE) if self.l2d_inject_fault else None
        self.l2d_fault_active = False
        self.l2d_fault_location = None
        self.l2d_steps_on_grass = 0
        self.l2d_last_segment_idx = 0
        
        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])
        

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        self.l2d_step_count += 1
        
        assert self.car is not None
        
        if self.fault_step is not None and self.fault_step <= self.l2d_step_count < self.fault_step + FAULT_DURATION:
            self.l2d_fault_active = True
            self.car.l2d_activate_fault()
        else:
            self.l2d_fault_active = False
            self.car.l2d_deactivate_fault()
            
            
        if self.car.outside_track:
            self.l2d_steps_on_grass += 1
        else:
            self.l2d_steps_on_grass = 0
            
            
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                self.car.steer(-action[0])
                
                if THROTTLE_BRAKE_RATIO is not None:
                    action[1] = 0.1
                    action[2] = 0.1 * THROTTLE_BRAKE_RATIO
                
                self.car.gas(action[1])
                self.car.brake(action[2])
                
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        #self.state = self._render("state_pixels")
        self.state = self.l2d_get_observation()
        step_reward = l2d_calculate_step_reward(REWARD_FUNCTIONS, self, action)
        
        terminated = False
        truncated = False
        info = {}
        
        self.l2d_time_since_last_tile += 1.0 / FPS
        
        if action is not None:  # First step without action, called from reset() 
                           

            #step_reward = self.reward - self.prev_reward
            #self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Termination due to finishing lap
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                
            if self.l2d_steps_on_grass > FPS * L2D_GRASS_TIMEOUT:
                terminated = True
                step_reward -= L2D_GRASS_TIMEOUT_PENALTY  # strong signal
                info["terminated_reason"] = "grass_timeout"
                
    
        if self.render_mode == "human":
            self.render()
            
        info["tiles_visited"] = self.tile_visited_count
        info["total_tiles"] = len(self.track)
        return self.state, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self.l2d_render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
            
        self.l2d_render_center_line(zoom, translation, angle)
            
    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()
            
            
# Learn 2 Drive specific functions  
    
    def l2d_render_indicators(self, W, H):
        import pygame
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)
        
        if self.l2d_fault_active:
            location_names = {
                0: "FRONT_LEFT",
                1: "FRONT_RIGHT",
                2: "BACK_LEFT",
                3: "BACK_RIGHT",
            }
            
            location_name = location_names.get(self.car.fault_location, "UNKNOWN")
            fault_font = pygame.font.SysFont("Courier", 16, bold=True)
            fault_text = f"FAULT: {FAULT_TYPE} | loc={location_name} | range={FAULT_STEP_RANGE}"
            fault_label = fault_font.render(fault_text, True, (255, 100, 100))  # light red
            self.surf.blit(fault_label, (10, H - 30))  # bottom-left corner
            
        # Draw labeled observation vector (3 logical columns)
        font = pygame.font.SysFont("Courier", 16)
        obs = self.l2d_get_observation()

        signal_names = [
            "ray_front",
            "ray_l_45",
            "ray_r_45", 
            "ray_l_90",
            "ray_r_90",
            "speed", 
            "ang_vel",
            "steer", 
            "gas", 
            "brake",
        ]

        # Logical grouping
        rays = signal_names[0:5]
        controls = signal_names[5:L2D_OBSERVATION_SIZE]

        # Column x-positions
        col_rays = W - 360
        col_ctrl = W - 180

        start_y = H - 5 * h + 10
        spacing = 18

        # Draw each group in its own column
        for i, name in enumerate(rays):
            value = obs[i]
            label = font.render(f"{name}: {value:.2f}", True, (255, 255, 255))
            self.surf.blit(label, (col_rays, start_y + i * spacing))

        for i, name in enumerate(controls):
            value = obs[i + 5]
            label = font.render(f"{name}: {value:.2f}", True, (255, 255, 255))
            self.surf.blit(label, (col_ctrl, start_y + i * spacing))

    def l2d_get_observation(self):
        obs = []

        # 1–5: Ray distances
        obs.append(round(self.car.l2d_rays["front"], 0))
        obs.append(round(self.car.l2d_rays["left_45"], 0))
        obs.append(round(self.car.l2d_rays["right_45"],0))
        
        obs.append(0.0 if L2D_DISABLE_SIDE_RAYS else round(self.car.l2d_rays["left_90"],0))
        obs.append(0.0 if L2D_DISABLE_SIDE_RAYS else round(self.car.l2d_rays["right_90"], 0))

        # 6: Speed (magnitude of linear velocity)
        speed = np.linalg.norm(self.car.hull.linearVelocity)
        obs.append(speed)

        # 7: Angular velocity
        obs.append(self.car.hull.angularVelocity)

        # 8: Steering angle (only need one front wheel — left or right)
        obs.append(self.car.wheels[0].joint.angle)

        # 9: Gas input (rear left wheel as representative)
        obs.append(self.car.wheels[2].gas)

        # 10: Brake input
        obs.append(self.car.wheels[2].brake)

        return np.array(obs, dtype=np.float32)  

    def l2d_create_track_barrier(self, track, prev_track, i):
        """
            Create left and right physical wall barriers for a given track tile.
            Parameters:
                x, y: Center of the road tile
                beta: Orientation angle of the road at this tile
        """
        
        # Unpack track data
        alpha1, beta1, x1, y1 = track
        alpha2, beta2, x2, y2 = prev_track
        
        # Barrier creation logic (always create both left and right barriers)
        for side in [-1.0, 1.0]:  # -1.0 for left side, 1.0 for right side
            # Calculate the barrier positions for this side
            barrier_start_left = (
                x1 + side * TRACK_WIDTH * math.cos(beta1),
                y1 + side * TRACK_WIDTH * math.sin(beta1),
            )
            barrier_start_right = (
                x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            barrier_end_left = (
                x2 + side * TRACK_WIDTH * math.cos(beta2),
                y2 + side * TRACK_WIDTH * math.sin(beta2),
            )
            barrier_end_right = (
                x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )

            # Create barrier vertices
            barrier_vertices = [barrier_start_left, barrier_start_right, barrier_end_right, barrier_end_left]

            # Create the physical barrier body
            barrier_fd = fixtureDef(
                shape=polygonShape(vertices=barrier_vertices),
                friction=0,
                density=0,
                filter=Box2D.b2Filter(categoryBits=L2D_CATEGORY_WALL, maskBits=0 if not L2D_HARD_BARRIER else 0xFFFF) 
            )
            
            barrier_body = self.world.CreateStaticBody(fixtures=barrier_fd)
            
            # Alternate the colors for both barriers (white/red or white/green)
            barrier_color = (255, 255, 255) if i % 2 == 0 else (255, 0, 0)
            barrier_body.color = barrier_color
            
            # Add the physical barrier to the list to remove it later
            self.l2d_walls.append(barrier_body)

            # Add the barrier to the visual rendering list
            self.road_poly.append((barrier_vertices, barrier_body.color))
                

    def l2d_render_center_line(self, zoom, translation, angle):
        """Draw a dashed yellow centerline over the track"""
        dash_color = (255, 255, 0)  # Yellow like real roads
        dash_length = 10  # pixels
        gap_length = 8

        for i in range(1, len(self.track)):
            x1, y1 = self.track[i - 1][2:4]
            x2, y2 = self.track[i][2:4]

            p1 = pygame.math.Vector2((x1, y1)).rotate_rad(angle)
            p2 = pygame.math.Vector2((x2, y2)).rotate_rad(angle)

            p1 = (p1[0] * zoom + translation[0], p1[1] * zoom + translation[1])
            p2 = (p2[0] * zoom + translation[0], p2[1] * zoom + translation[1])

            # Draw dashed line between p1 and p2
            line = pygame.math.Vector2(p2) - pygame.math.Vector2(p1)
            length = line.length()
            direction = line.normalize() if length != 0 else pygame.math.Vector2(0, 0)

            num_dashes = int(length // (dash_length + gap_length))

            for j in range(num_dashes):
                start = pygame.math.Vector2(p1) + direction * (j * (dash_length + gap_length))
                end = start + direction * dash_length
                pygame.draw.line(self.surf, dash_color, start, end, 2)

