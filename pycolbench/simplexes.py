import hppfcl
import numpy as np

def newVertex(w0: np.array, w1: np.array, w: np.array, i0=0, i1=0):
    v = hppfcl.SimplexV()
    v.w0 = w0
    v.w1 = w1
    v.w = w
    v.index_w0 = i0
    v.index_w1 = i1
    return v

def newSimplex():
    s = hppfcl.Simplex()
    for i in range(4):
        zero_vec = np.zeros(3)
        v = newVertex(zero_vec, zero_vec, zero_vec)
        s.setVertex(v, i)
        s.rank = 0
    return s

def copySimplex(s):
    news = hppfcl.Simplex()
    for i in range(4):
        news.setVertex(s.getVertex(i), i)
        news.rank = s.rank
    return news
