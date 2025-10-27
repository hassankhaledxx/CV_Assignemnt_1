import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {path}")
    return img.astype(np.float64)

def CalculateIntegral(img):
    x,y = img.shape
    s  = np.zeros((x, y), dtype=np.float64)
    ii = np.zeros((x, y), dtype=np.float64)

    for i in range(x):
        row_sum = 0.0
        for j in range(y):
            row_sum += img[i, j]           # s(i,j)
            s[i, j] = row_sum
            ii[i, j] = s[i, j] + (ii[i-1, j] if i > 0 else 0.0)  # ii(i,j)=s(i,j)+ii(i-1,j)
    return ii

def _safe(ii, x, y):
    # returns 0 if index is outside image (to allow formula near borders)
    h, w = ii.shape
    if x < 0 or y < 0: return 0.0
    if x >= h: x = h-1
    if y >= w: y = w-1
    return ii[x, y]

def CalculateLocalSum(ii, p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    # ensure ordered corners
    x0, x1 = int(min(x0, x1)), int(max(x0, x1))
    y0, y1 = int(min(y0, y1)), int(max(y0, y1))
    A = _safe(ii, x1,   y1)
    B = _safe(ii, x0-1, y1)
    C = _safe(ii, x1,   y0-1)
    D = _safe(ii, x0-1, y0-1)
    return A - B - C + D

def _rc(center_i, center_j, dy, dx):
    # dy,dx are in pixels relative to kernel center
    return int(round(center_i + dy)), int(round(center_j + dx))

def _rect(ii, ci, cj, y_top, x_left, y_bot, x_right):
    # corners given as offsets; returns local sum via integral image
    p0 = _rc(ci, cj, y_top,  x_left)   # top-left
    p1 = _rc(ci, cj, y_bot,  x_right)  # bottom-right
    return CalculateLocalSum(ii, p0, p1)

def _kernel_sums(ii, ci, cj, n):
    m = 0.15 * n
    # Horizontal extents (from points): ±0.5n, ±0.325n, ±0.225n, ±0.1n, ±0.05n
    # Vertical extents: 0, 0.5m, 0.833m, 2m
    # Build 7 rectangular bands as in the figure (top→bottom):
    # LS1 (forehead-left)   : x in [-0.5n, -0.05n],  y in [-0.5m, 0]
    # LS2 (forehead-right)  : x in [ 0.05n,  0.5n],  y in [-0.5m, 0]
    # LS3 (brow-left)       : x in [-0.5n, -0.05n],  y in [0, 0.5m]
    # LS4 (brow-right)      : x in [ 0.05n,  0.5n],  y in [0, 0.5m]
    # LS5 (under-brow wide) : x in [-0.325n, 0.325n], y in [0.833m, 2m]
    # LS6 (under-brow midL) : x in [-0.225n,-0.1n],   y in [0.833m, 2m]
    # LS7 (under-brow midR) : x in [ 0.1n,  0.225n],  y in [0.833m, 2m]
    # (These follow the P1..P14 grid; do not overlap.)
    def sx(a): return a * n
    def sy(b): return b * m

    LS1 = _rect(ii, ci, cj, sy(-0.5), sx(-0.5),  sy(0.0),  sx(-0.05))
    LS2 = _rect(ii, ci, cj, sy(-0.5), sx( 0.05), sy(0.0),  sx( 0.5))
    LS3 = _rect(ii, ci, cj, sy( 0.0), sx(-0.5),  sy(0.5),  sx(-0.05))
    LS4 = _rect(ii, ci, cj, sy( 0.0), sx( 0.05), sy(0.5),  sx( 0.5))
    LS5 = _rect(ii, ci, cj, sy(0.833), sx(-0.325), sy(2.0), sx( 0.325))
    LS6 = _rect(ii, ci, cj, sy(0.833), sx(-0.225), sy(2.0), sx(-0.1))
    LS7 = _rect(ii, ci, cj, sy(0.833), sx( 0.1),   sy(2.0), sx( 0.225))

    return LS1, LS2, LS3, LS4, LS5, LS6, LS7


def DetectEye(ii, n):
    h, w = ii.shape
    m = int(round(0.15 * n))

    # scan limits so all rectangles stay inside; keep a small margin
    top  = int(math.ceil(2.0*m))         # to allow LS5..7 bottom
    left = int(math.ceil(0.5*n))
    bottom = h - 1
    right  = w - 1

    best_score = -1e18
    best_pos = (top, left)

    for i in range(top, bottom):
        for j in range(left, right):
            LS1, LS2, LS3, LS4, LS5, LS6, LS7 = _kernel_sums(ii, i, j, n)
            # White (+) vs Black (−) per figure:
            # Assume: forehead white, brow black, under-brow white (tune if inverted).
            score = (LS1 + LS2) - (LS3 + LS4) + (LS5) - (LS6 + LS7)
            if score > best_score:
                best_score = score
                best_pos = (i, j)
    return best_pos, best_score

def ExtractDetectedEye(img, max_pos, n):
    m = int(round(0.15 * n))
    ci, cj = max_pos
    top  = int(round(ci - 0.5 * m))
    bot  = int(round(ci + 2.0 * m))   # down to 2m (covers forehead→under-brow stack)
    left = int(round(cj - 0.5 * n))
    right= int(round(cj + 0.5 * n))
    top  = max(0, top); left = max(0, left)
    bot  = min(img.shape[0]-1, bot); right = min(img.shape[1]-1, right)
    return img[top:bot+1, left:right+1]

img1 = load_gray("D:\Bachelor\Sem 9\CV\Assignment 1\(_DMET901_) 2 - Assignment 1 (Assignment ) (1)\Image 1.jpeg")
ii1  = CalculateIntegral(img1)
pos1, s1 = DetectEye(ii1, n=330)
eye1 = ExtractDetectedEye(img1, pos1, n=330)

img2 = load_gray("D:\Bachelor\Sem 9\CV\Assignment 1\(_DMET901_) 2 - Assignment 1 (Assignment ) (1)\Image 2.jpeg")
ii2  = CalculateIntegral(img2)
pos2, s2 = DetectEye(ii2, n=150)
eye2 = ExtractDetectedEye(img2, pos2, n=150)

img3 = load_gray("D:\Bachelor\Sem 9\CV\Assignment 1\(_DMET901_) 2 - Assignment 1 (Assignment ) (1)\Image 3.jpeg")
ii3  = CalculateIntegral(img3)
pos3, s3 = DetectEye(ii3, n=250)
eye3 = ExtractDetectedEye(img3, pos3, n=250)

plt.figure(); plt.imshow(eye1, cmap='gray'); plt.title("Image 1 – n=330")
plt.figure(); plt.imshow(eye2, cmap='gray'); plt.title("Image 2 – n=150")
plt.figure(); plt.imshow(eye3, cmap='gray'); plt.title("Image 3 – n=250")
plt.show()
