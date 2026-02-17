import json
import tkinter as tk
from urllib.request import urlopen
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. GPS-координати маршруту
# ===============================
coords = [
    (48.164214, 24.536044),
    (48.164983, 24.534836),
    (48.165605, 24.534068),
    (48.166228, 24.532915),
    (48.166777, 24.531927),
    (48.167326, 24.530884),
    (48.167011, 24.530061),
    (48.166053, 24.528039),
    (48.166655, 24.526064),
    (48.166497, 24.523574),
    (48.166128, 24.520214),
    (48.165416, 24.517170),
    (48.164546, 24.514640),
    (48.163412, 24.512980),
    (48.162331, 24.511715),
    (48.162015, 24.509462),
    (48.162147, 24.506932),
    (48.161751, 24.504244),
    (48.161197, 24.501793),
    (48.160580, 24.500537),
    (48.160250, 24.500106)
]

# ===============================
# 2. Висоти (Open-Elevation)
# ===============================
locations = "|".join([f"{lat},{lon}" for lat, lon in coords])
url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

with urlopen(url) as response:
    data = json.load(response)

heights = [p["elevation"] for p in data["results"]]

# ===============================
# 3. Кумулятивна відстань (гаверсин)
# ===============================
def haversine(a, b, c, d):
    R = 6371.0
    a, b, c, d = map(radians, [a, b, c, d])
    return 2 * R * atan2(
        sqrt(sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2),
        sqrt(1 - (sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2))
    )

dist = [0.0]
for i in range(1, len(coords)):
    dist.append(dist[-1] + haversine(*coords[i-1], *coords[i]))

dist = np.array(dist)
heights = np.array(heights)

# ===============================
# 4. Метод прогонки
# ===============================
def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    for i in range(1, n):
        denom = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/denom if i < n-1 else 0
        dp[i] = (d[i] - a[i]*dp[i-1]) / denom

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]

    return x

# ===============================
# 5. Кубічні сплайни
# ===============================
def cubic_spline_coeffs(x, y):
    n = len(x) - 1
    h = x[1:] - x[:-1]

    A = np.zeros(n+1)
    B = np.ones(n+1)
    C = np.zeros(n+1)
    D = np.zeros(n+1)

    for i in range(1, n):
        A[i] = h[i-1]
        B[i] = 2*(h[i-1] + h[i])
        C[i] = h[i]
        D[i] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    c = thomas(A, B, C, D)

    a = y[:-1]
    b = []
    d = []

    for i in range(n):
        bi = (y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        di = (c[i+1]-c[i])/(3*h[i])
        b.append(bi)
        d.append(di)

    return a, np.array(b), c[:-1], np.array(d)

def spline_eval(x, xi, a, b, c, d):
    for i in range(len(a)):
        if xi[i] <= x <= xi[i+1]:
            dx = x - xi[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

# ===============================
# 6. Графіки для 10 / 15 / 20 вузлів
# ===============================
plt.figure(figsize=(10,5))
plt.plot(dist, heights, 'k.', label="Реальні дані")

for k in [10, 15, 20]:
    idx = np.linspace(0, len(dist)-1, k, dtype=int)
    xk = dist[idx]
    yk = heights[idx]

    a,b,c,d = cubic_spline_coeffs(xk, yk)

    xs = np.linspace(xk[0], xk[-1], 500)
    ys = [spline_eval(x, xk, a, b, c, d) for x in xs]

    plt.plot(xs, ys, label=f"{k} вузлів")

plt.legend()
plt.grid()
plt.title("Кубічна сплайн-апроксимація профілю")
plt.xlabel("Відстань, км")
plt.ylabel("Висота, м")
plt.show()

# ===============================
# 7. Аналіз маршруту
# ===============================
yy_full = np.array([spline_eval(x, xk, a, b, c, d) for x in dist])
grad = np.gradient(yy_full, dist) * 100

total_ascent = sum(max(heights[i]-heights[i-1],0) for i in range(1,len(heights)))
total_descent = sum(max(heights[i-1]-heights[i],0) for i in range(1,len(heights)))

print("Довжина маршруту (км):", dist[-1])
print("Набір висоти (м):", total_ascent)
print("Спуск (м):", total_descent)
print("Макс підйом (%):", np.max(grad))
print("Макс спуск (%):", np.min(grad))
print("Середній градієнт (%):", np.mean(np.abs(grad)))

mass = 80
energy = mass * 9.81 * total_ascent
print("Механічна енергія підйому (Дж):", energy)

# ===============================
# 8. Профіль у tkinter
# ===============================
root = tk.Tk()
root.title("Профіль рельєфу маршруту")

W, H, L = 900, 450, 60
canvas = tk.Canvas(root, width=W, height=H, bg="white")
canvas.pack()

xmin, xmax = dist.min(), dist.max()
ymin, ymax = heights.min(), heights.max()

def X(x): return L + (x - xmin)*(W-2*L)/(xmax-xmin)
def Y(y): return H-L - (y - ymin)*(H-2*L)/(ymax-ymin)

pts = []
for x,y in zip(dist, heights):
    pts += [X(x), Y(y)]

canvas.create_line(pts, fill="green", width=2)
root.mainloop()
