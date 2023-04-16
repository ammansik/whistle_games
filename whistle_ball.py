#!/usr/bin/python
import sys, pygame
import time

import pyaudio
import scipy
import numpy
import struct

import threading
import time, datetime
import math


class Ball:
    def __init__(self):
        pygame.init()
        self.running = True
        self.clock = pygame.time.Clock()
        self.size = self.width, self.height = 800, 620
        self.speed = [0.0, 0.0]
        self.black = (0, 0, 0)
        self.initial_speed = 0.0
        self.screen = pygame.display.set_mode(self.size)
        self.ball = pygame.image.load("ball.gif")
        self.ballrect = self.ball.get_rect()
        self.ballrect = self.ballrect.move(300, 100)
        self.g = 9.81
        self.t = 0.0
        self.fy = 0.8
        self.fx = 0.99
        self.time_1 = 0.0
        self.time_2 = 0.0
        self.time_diff = self.time_2 - self.time_1

        self.add_speed_x = 1.5
        self.add_speed_y = -3.5

    def is_running(self):
        return self.running

    def whistle_up(self, diff):
        extra_speed = diff * self.add_speed_y - self.speed[1]
        self.initial_speed += extra_speed

    def whistle_right(self):
        self.speed[0] += self.add_speed_x

    def move_ball(self):
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
        self.ballrect = self.ballrect.move(self.speed)
        if self.ballrect.top < 0:
            self.initial_speed = -self.speed[1] * self.fy
            self.t = 0.0
            self.ballrect.top = 0
        if self.ballrect.bottom > self.height:
            self.initial_speed = -self.speed[1] * self.fy
            self.speed[1] = 0.0
            self.t = 0.0
            self.ballrect.bottom = self.height

        if self.ballrect.left < 0:
            self.speed[0] = -self.speed[0] * self.fx
            self.ballrect.left = 0
        if self.ballrect.right > self.width:
            self.speed[0] = -self.speed[0] * self.fx
            self.ballrect.right = self.width
        if self.ballrect.bottom == self.height:
            self.speed[0] = self.speed[0] * self.fx
        if self.time_1 == 0.0 and self.time_2 == 0.0:
            self.time_2 = time.time()
            self.time_1 = time.time()
        else:
            self.time_2 = time.time()
            self.time_diff = self.time_2 - self.time_1
            self.time_1 = self.time_2
        self.t += self.time_diff
        self.speed[1] = self.initial_speed + self.g * self.t
        self.clock.tick(150)
        self.screen.fill(self.black)
        self.screen.blit(self.ball, self.ballrect)
        pygame.display.flip()


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


def graph(wb):
    global chunks, bufferSize, fftx, ffty, w, max_spectrum_diff, max_spectrum_diff_index, prev_max_spectrum_diff_index
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
        ffty = numpy.lib.scimath.log(ffty) - 2
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
        max_spectrum_diff = max(ffty_abs[250:]) // numpy.mean(ffty_abs[250:])
        spectrum_var = numpy.var(ffty_abs[250:])
        max_spectrum_diff_index = ffty_abs[250:].index(max(ffty_abs[250:]))
        pitch_threshold = 6.0
        if max_median_spectrum_diff > pitch_threshold:
            wb.whistle_up(max_spectrum_diff / pitch_threshold)
            try:
                pitch_diff = max_spectrum_diff_index - prev_max_spectrum_diff_index
                if pitch_diff > 50:
                    pass
            except:
                pass
            prev_max_spectrum_diff_index = max_spectrum_diff_index
    if len(chunks) > 20:
        print("falling behind...", len(chunks))


# ADJUST THIS TO CHANGE SPEED/SIZE OF FFT
bufferSize = 2 ** 11

# ADJUST THIS TO CHANGE SPEED/SIZE OF FFT
sampleRate = 16000
p = pyaudio.PyAudio()
chunks = []
ffts = []

global w, fftx, ffty, max_spectrum_diff, max_spectrum_diff_index, prev_max_spectrum_diff_index
max_spectrum_diff = 0.0
record_thread = threading.Thread(target=record)
record_thread.daemon = True
record_thread.start()
whistle_ball = Ball()
while whistle_ball.is_running():
    whistle_ball.move_ball()
    graph(whistle_ball)

sys.exit()
