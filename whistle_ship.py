#!/usr/bin/python
import sys
import pygame
import time

import pyaudio
import scipy
import struct
import numpy

import threading
import time, datetime
import math
import random


class Gate:
    def __init__(self, start_x, start_gap_y):
        self.x = start_x
        self.y = 0
        self.width = 50

        self.gap = 250
        self.opening_y = start_gap_y
        self.gate_color = (255, 255, 0)
        self.gate_speed = -1

        self.gate_sprite_1 = pygame.sprite.Sprite()
        self.gate_sprite_2 = pygame.sprite.Sprite()
        self.gate_sprite_1.rect = pygame.Rect(
            self.x, self.y, self.width, self.opening_y
        )
        self.gate_sprite_2.rect = pygame.Rect(
            self.x,
            self.opening_y + self.gap,
            self.width,
            620 - self.opening_y - self.gap,
        )

    def move_gate(self):
        self.x += self.gate_speed
        # if self.x < -100:
        #    self.x = 1700

    def get_x(self):
        return self.x

    def set_x(self, new_x):
        self.x = new_x

    def set_opening_y(self, new_opening_y):
        self.opening_y = new_opening_y

    def draw_gate(self, screen):
        gate_rect_1 = (0, 0, self.width, self.opening_y)
        gate_rect_2 = (0, 0, self.width, 620 - self.opening_y - self.gap)
        self.gate_sprite_1.rect = pygame.Rect(
            self.x, self.y, self.width, self.opening_y
        )
        self.gate_sprite_2.rect = pygame.Rect(
            self.x,
            self.opening_y + self.gap,
            self.width,
            620 - self.opening_y - self.gap,
        )

        surf_1 = pygame.Surface((self.width, self.opening_y))
        surf_1.set_alpha(255)
        # surf_1.fill((0,0,0,255))
        pygame.draw.rect(surf_1, self.gate_color, gate_rect_1)

        surf_2 = pygame.Surface((self.width, 620 - self.opening_y - self.gap))
        surf_2.set_alpha(255)
        # surf_2.fill((0,0,0,255))
        pygame.draw.rect(surf_2, self.gate_color, gate_rect_2)

        self.gate_sprite_1.image = surf_1
        self.gate_sprite_2.image = surf_2
        screen.blit(self.gate_sprite_1.image, self.gate_sprite_1.rect)
        screen.blit(self.gate_sprite_2.image, self.gate_sprite_2.rect)

    def has_ship_collided(self, ship_sprite):
        self.gate_sprite_1.mask = pygame.mask.from_surface(self.gate_sprite_1.image)
        self.gate_sprite_2.mask = pygame.mask.from_surface(self.gate_sprite_2.image)

        self.gate_sprite_1.mask.fill()
        self.gate_sprite_2.mask.fill()

        if (
            pygame.sprite.collide_mask(ship_sprite, self.gate_sprite_1) != None
            or pygame.sprite.collide_mask(ship_sprite, self.gate_sprite_2) != None
        ):
            return True
        else:
            return False


class Ship(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.running = True
        self.clock = pygame.time.Clock()
        self.size = self.width, self.height = 800, 620
        self.speed = [0.0, 0.0]
        self.black = (0, 0, 0)
        self.initial_speed = -2.5
        self.screen = pygame.display.set_mode(self.size)
        self.image = pygame.image.load("spaceship.png")
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect = self.rect.move(300, 100)
        self.g = -2.82
        self.t = 0.0
        self.fy = 0.8
        self.fx = 0.99
        self.time_1 = 0.0
        self.time_2 = 0.0
        self.time_diff = self.time_2 - self.time_1
        self.whistle_force = False

        self.add_speed_x = 1.5
        self.add_speed_y_up = 1.0
        self.add_speed_y_down = 1.7

    def is_running(self):
        return self.running

    def get_force(self):
        return self.whistle_force

    def whistle_up(self):
        # extra_speed = -diff*self.add_speed_y-self.speed[1]
        # self.initial_speed += extra_speed
        self.whistle_force = True

    def whistle_down(self):
        self.whistle_force = False

    def whistle_right(self):
        self.speed[0] += self.add_speed_x

    def move_ship(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    extra_speed = self.add_speed_y - self.speed[1]
                    self.initial_speed += extra_speed
                elif event.key == pygame.K_RIGHT:
                    self.speed[0] += self.add_speed_x
                elif event.key == pygame.K_LEFT:
                    self.speed[0] -= self.add_speed_x
        self.rect = self.rect.move(self.speed)

        if self.rect.top < 0:
            self.initial_speed = -self.speed[1] * self.fy
            self.t = 0.0
            self.rect.top = 0

        if self.rect.bottom > self.height:
            self.initial_speed = -self.speed[1] * self.fy
            self.speed[1] = 0.0
            self.t = 0.0
            self.rect.bottom = self.height

        if self.rect.left < 0:
            self.speed[0] = -self.speed[0] * self.fx
            self.rect.left = 0
        if self.rect.right > self.width:
            self.speed[0] = -self.speed[0] * self.fx
            self.rect.right = self.width
        if self.rect.bottom == self.height:
            self.speed[0] = self.speed[0] * self.fx
        if self.time_1 == 0.0 and self.time_2 == 0.0:
            self.time_2 = time.time()
            self.time_1 = time.time()
        else:
            self.time_2 = time.time()
            self.time_diff = self.time_2 - self.time_1
            self.time_1 = self.time_2
        self.t += self.time_diff
        # self.speed[1] = self.initial_speed+self.g*self.t
        if self.whistle_force == True:
            self.speed[1] = self.add_speed_y_down
        else:
            self.speed[1] = -self.add_speed_y_up

    def get_image(self):
        return self.image

    def get_rect(self):
        return self.rect


def stream():
    global chunks, inStream, bufferSize
    while True:
        chunks.append(inStream.read(bufferSize))


def record():
    global w, inStream, p, bufferSize
    inStream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sampleRate,
        input=True,
        frames_per_buffer=bufferSize,
    )
    threading.Thread(target=stream).start()


def downSample(fftx, ffty, degree=10):
    x, y = [], []
    for i in range(len(ffty) / degree - 1):
        x.append(fftx[i * degree + degree / 2])
        y.append(sum(ffty[i * degree : (i + 1) * degree]) / degree)
    return [x, y]


def smoothWindow(fftx, ffty, degree=10):
    lx, ly = fftx[degree:-degree], []
    for i in range(degree, len(ffty) - degree):
        ly.append(sum(ffty[i - degree : i + degree]))
    return [lx, ly]


def smoothMemory(ffty, degree=3):
    global ffts
    ffts = ffts + [ffty]
    if len(ffts) <= degree:
        return ffty
    ffts = ffts[1:]
    return scipy.average(numpy.array(ffts), 0)


def detrend(fftx, ffty, degree=10):
    lx, ly = fftx[degree:-degree], []
    for i in range(degree, len(ffty) - degree):
        ly.append(ffty[i] - sum(ffty[i - degree : i + degree]) / (degree * 2))
    return [lx, ly]


def whistle_check(wb):
    global chunks, bufferSize, fftx, ffty, w, max_spectrum_diff, max_spectrum_diff_index, prev_max_spectrum_diff_index, prev_whistle_status
    if len(chunks) > 0:
        data = chunks.pop(0)
        data = numpy.array(struct.unpack("%dB" % (bufferSize * 2), data))
        ffty = scipy.fftpack.fft(data)
        fftx = scipy.fftpack.rfftfreq(bufferSize * 2, 1.0 / sampleRate)

        fftx = fftx[0 : len(fftx) // 4]
        ffty = abs(ffty[0 : len(ffty) // 2]) / 1000
        ffty1 = ffty[: len(ffty) // 2]
        ffty2 = ffty[len(ffty) // 2 : :] + 2
        ffty2 = ffty2[::-1]
        ffty = ffty1 + ffty2
        ffty = scipy.log(ffty) - 2
        ffty_abs = []
        for f in ffty:
            ffty_abs.append(abs(f))
        whistle_spectrum = ffty_abs[250:]
        whistle_spectrum.sort()
        whistle_spectrum.reverse()

        median_index = int(len(whistle_spectrum) / 2)
        max_median_spectrum_diff = (
            max(whistle_spectrum) / whistle_spectrum[median_index]
        )

        max_spectrum_diff = max(ffty_abs[250:]) / scipy.mean(ffty_abs[250:])
        spectrum_var = scipy.var(ffty_abs[250:])
        max_spectrum_diff_index = ffty_abs[250:].index(max(ffty_abs[250:]))
        pitch_threshold = 6.0
        if max_median_spectrum_diff > pitch_threshold:
            if prev_whistle_status[0] == False and prev_whistle_status[1] == False:
                if wb.get_force() == True:
                    wb.whistle_down()
                else:
                    wb.whistle_up()
            prev_max_spectrum_diff_index = max_spectrum_diff_index
            prev_whistle_status[0] = prev_whistle_status[1]
            prev_whistle_status[1] = True
        else:
            prev_whistle_status[0] = prev_whistle_status[1]
            prev_whistle_status[1] = False

    if len(chunks) > 20:
        print("falling behind...", len(chunks))


# Game screen
pygame.init()
size = width, height = 800, 620
screen = pygame.display.set_mode(size)
black = (0, 0, 0)
clock = pygame.time.Clock()

# ADJUST THIS TO CHANGE SPEED/SIZE OF FFT
bufferSize = 2 ** 11

# ADJUST THIS TO CHANGE SPEED/SIZE OF FFT
sampleRate = 16000
p = pyaudio.PyAudio()
chunks = []
ffts = []

global w, fftx, ffty, max_spectrum_diff, max_spectrum_diff_index, prev_max_spectrum_diff_index, prev_whistle_status
max_spectrum_diff = 0.0
record_thread = threading.Thread(target=record)
record_thread.daemon = True
record_thread.start()
whistle_ship = Ship()

gates = []
one_gate = Gate(800, 150)
two_gate = Gate(1100, 50)
three_gate = Gate(1400, 250)
four_gate = Gate(1700, 200)
gates.append(one_gate)
gates.append(two_gate)
gates.append(three_gate)
gates.append(four_gate)
has_collided = False
prev_whistle_status = [False, False]
while whistle_ship.is_running() == True and has_collided == False:
    # Move ship and gates
    whistle_ship.move_ship()
    x_pos = []
    for gate in gates:
        gate.move_gate()
        x_pos.append(gate.get_x())
    max_x = max(x_pos)
    for gate in gates:
        if gate.get_x() < -100:
            gate.set_x(max_x + 300)
            gate.set_opening_y(random.randint(50, 200))
    whistle_check(whistle_ship)
    clock.tick(150)
    screen.fill(black)
    # Draw ship
    screen.blit(whistle_ship.get_image(), whistle_ship.get_rect())
    # Draw gates
    for gate in gates:
        gate.draw_gate(screen)

    # Check for collisions between ship and gates
    for gate in gates:
        if gate.has_ship_collided(whistle_ship) == True:
            has_collided = True
            break
    pygame.display.flip()

sys.exit()
