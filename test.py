import pandas as pd
import numpy as np
import estimation

m_start = {0:{0: 0.1, 1: 0.9}, 1:{0: 0.1, 1: 0.9}}
u_start = {0:{0: 0.9, 1: 0.1}, 1:{0: 0.9, 1: 0.1}}
p_start = 0.1

y = pd.DataFrame([{0:0, 1:1},{0:1, 1:0},{0:0, 1:0}])
print y

est = estimation.ECMEstimate(y, m_start, u_start, p_start)
print est.summary()
print est.comparison_space
print est.p

est.estimate()
print est.g
print est.summary()
print est.m
print est.u
print est.p
