import numpy as np
import random


class FlappyBirdEnv:
    def __init__(self, live_reward=0.1, 
                 pass_reward=3, death_reward=-3):
        self.width = 400
        self.height = 600

        self.gravity = 0.5
        self.jump_velocity = -8
        self.max_velocity = 10

        self.pipe_gap = 150
        self.pipe_width = 60
        self.pipe_speed = 3

        self.bird_x = 100

        self.live_reward = live_reward
        self.pass_reward = pass_reward
        self.death_reward = death_reward

        self.reset()

    def reset(self):
        self.bird_y = self.height / 2
        self.bird_velocity = 0

        self.score = 0
        self.steps = 0

        self.pipes = []
        self._add_pipe()

        return self._get_state()

    def step(self, action):
        reward += self.live_reward
        terminated = False

        self.steps += 1

        # jump
        if action == 1:
            self.bird_velocity = self.jump_velocity

        # physics
        self.bird_velocity += self.gravity
        self.bird_velocity = np.clip(self.bird_velocity, -10, 10)

        self.bird_y += self.bird_velocity

        # pipes
        self._update_pipes()

        # collision
        if self._check_collision():
            reward -=self.death_reward
            terminated = True
            truncated = True

        # pass pipe
        pipe = self.pipes[0]
        if pipe["x"] + self.pipe_width < self.bird_x and not pipe["passed"]:
            pipe["passed"] = True
            reward += self.pass_reward
            self.score += 1

        state = self._get_state()

        return state, reward, terminated, truncated, self.score

    def _add_pipe(self):
        pipe_top = random.randint(50, 450)

        pipe = {
            "x": self.width,
            "top": pipe_top,
            "bottom": pipe_top + self.pipe_gap,
            "passed": False
        }

        self.pipes.append(pipe)

    def _update_pipes(self):
        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed

        if self.pipes[0]["x"] < -self.pipe_width:
            self.pipes.pop(0)
            self._add_pipe()

    def _get_state(self):
        pipe = self.pipes[0]

        pipe_center = (pipe["top"] + pipe["bottom"]) / 2

        state = np.array([
            self.bird_y / self.height,
            self.bird_velocity / self.max_velocity,
            (pipe["x"] - self.bird_x) / self.width,
            pipe_center / self.height
        ], dtype=np.float32)

        return state

    def _check_collision(self):
        # ground / ceiling
        if self.bird_y < 0 or self.bird_y > self.height:
            return True

        pipe = self.pipes[0]

        # pipe collision
        if pipe["x"] < self.bird_x < pipe["x"] + self.pipe_width:

            if self.bird_y < pipe["top"] or self.bird_y > pipe["bottom"]:
                return True

        return False