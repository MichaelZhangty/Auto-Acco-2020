#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import pretty_midi
# import scikits.audiolab
import pyaudio
# import analyse
import time
import copy
from scipy.integrate import quad
import wave
import numpy as np
import math
import sys
import pretty_midi
import matplotlib.pyplot as plt
from madmom.features.onsets import CNNOnsetProcessor
import os
import librosa
import statsmodels.api as sm
from score_following_utilities import *
import time
import threading
from auto_accompany_utilities import *
import fluidsynth
from scipy import stats


