import numpy as np
import random

class FlappyBirdEnv:

    def __init__(self):

        self.width = 400
        self.height = 600

        self.gravity = 0.5
        self.jump_velocity = -8

        self.pipe_gap = 150
        self.pipe_width = 60
        self.pipe_speed = 3

        self.reset()

    def reset(self):

        self.bird_y = self.height // 2
        self.bird_velocity = 0

        self.bird_x = 100

        self.pipes = []
        self.score = 0

        self._add_pipe()

        return self._get_state()

    def step(self, action):

        reward = 0.1
        done = False

        if action == 1:
            self.bird_velocity = self.jump_velocity

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        self._update_pipes()

        if self._check_collision():

            reward = -10
            done = True

        pipe = self.pipes[0]

        if pipe["x"] + self.pipe_width < self.bird_x and not pipe["passed"]:
            pipe["passed"] = True
            reward = 10
            self.score += 1

        state = self._get_state()

        return state, reward, done

    def _add_pipe(self):

        pipe_height = random.randint(100, 400)

        pipe = {
            "x": self.width,
            "top": pipe_height,
            "bottom": pipe_height + self.pipe_gap,
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

        state = np.array([
            self.bird_y / self.height,
            self.bird_velocity / 10,
            (pipe["x"] - self.bird_x) / self.width,
            pipe["top"] / self.height,
            pipe["bottom"] / self.height
        ])

        return state

    def _check_collision(self):

        if self.bird_y < 0 or self.bird_y > self.height:
            return True

        pipe = self.pipes[0]

        if pipe["x"] < self.bird_x < pipe["x"] + self.pipe_width:

            if self.bird_y < pipe["top"] or self.bird_y > pipe["bottom"]:
                return True

        return False