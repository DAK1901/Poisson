import numpy as np
import matplotlib.pyplot as plt

Lx = 1
Ly = 1
n = 10
m = 10

hx = Lx/n
hy = Ly/m

def u(x, y):
  return - (x*(x - 5*Lx) + y*(y - 4*Ly))                                               # 1 Параболоид
  return x + y                                                                         # 2 Плоскость
  return np.sin(2*np.pi*x/5) + np.cos(2*np.pi*y/4)                                     # 3 Тригонометрическая функция

def f(x, y):
  return 4                                                                             # 1 Параболоид
  return 0                                                                             # 2 Плоскость
  return (2*np.pi/5)**2*np.sin(2*np.pi*x/5) + (2*np.pi/4)**2*np.cos(2*np.pi*y/4)       # 3 Тригонометрическая функция

def Draw(U, args_x=(0, Lx), args_y=(0, Ly), title = ""):
  tx = np.linspace(*args_x, len(U))
  ty = np.linspace(*args_y, len(U[0]))

  Tx, Ty = np.meshgrid(tx, ty)

  fig = plt.figure()
  ax = plt.subplot(111, projection="3d")
  surf = ax.plot_surface(Tx, Ty, np.transpose(U), rstride=1, cstride=1, cmap='coolwarm')
  fig.colorbar(surf, shrink=0.5, aspect=5)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  ax.set_title(title)

def JacobyRect():
  eps = 10**-3
  #Точное решение
  U = np.zeros((n + 1, m + 1))
  for i in range(n + 1):
    for j in range(m + 1):
      U[i][j] = u(i*hx, j*hy)

  Uh = np.zeros((n + 1, m + 1))
  # Границы области
  Uh[:, 0] = np.array([u(i*hx, 0) for i in range(n + 1)])
  Uh[:, -1] = np.array([u(i*hx, Ly) for i in range(n + 1)])
  Uh[0] = np.array([u(0, j*hy) for j in range(m + 1)])
  Uh[-1] = np.array([u(Lx, j*hy) for j in range(m + 1)])

  Uh_ = Uh.copy()
  #Внутри области
  while True:
    for i in range(1, n):
      for j in range(1, m):
        Uh_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f(i*hx, j *hy) + hy**2*(Uh[i - 1][j] + Uh[i + 1][j]) + hx**2*(Uh[i][j - 1] + Uh[i][j + 1]))
    Uh = Uh_.copy()

    if np.linalg.norm(U - Uh) < eps: break

  Draw(Uh, (0, Lx), (0, Ly), "Метод Якоби в прямоугольной области")
  plt.show()

def JacobyArea():
  eps = 10**-2
  U_draw = np.zeros((5*n + 1, 4*m + 1))
  #Точное решение
  # U = np.zeros((5*n + 1, 4*m + 1))
  # for i in range(5*n + 1):
  #   for j in range(4*m + 1):
  #     U[i][j] = u(i*hx, j*hy)

  U1 = np.zeros((2*n + 1, m + 1))
  U2 = np.zeros((2*n + 1, m + 1))
  U3 = np.zeros((3*n + 1, m + 1))
  U4 = np.zeros((n + 1, m + 1))
  U5 = np.zeros((5*n + 1, m + 1))

  #---------- Граница ---------
  # ОБЛАСТЬ 1
  U1[0] = np.array([u(0, j*hy) for j in range(m + 1)])
  U1[-1] = np.array([u(2*Lx, j*hy) for j in range(m + 1)])
  U1[:, 0] = np.array([u(i*hx, 0) for i in range(2*n + 1)])
  U1[: n + 1, -1] = np.array([u(i*hx, Ly) for i in range(n + 1)])
  U1_ = U1.copy()

  # ОБЛАСТЬ 2
  U2[0] = np.array([u(3*Lx, j*hy) for j in range(m + 1)])
  U2[-1] = np.array([u(5*Lx, j*hy) for j in range(m + 1)])
  U2[:, 0] = np.array([u((3*n + i)*hx, 0) for i in range(2*n + 1)])
  U2[n :, -1] = np.array([u((4*n + i)*hx, Ly) for i in range(n + 1)])
  U2_ = U2.copy()

  # ОБЛАСТЬ 3
  U3[0] = np.array([u(Lx, (m + j)*hy) for j in range(m + 1)])
  U3[-1] = np.array([u(4*Lx, (m + j)*hy) for j in range(m + 1)])
  U3[n: 2*n + 1, 0] = np.array([u((2*n + i)*hx, Ly) for i in range(n + 1)])
  U3[: n + 1, -1] = np.array([u((n + i)*hx, 2*Ly) for i in range(n + 1)])
  U3[2*n: 3*n + 1, -1] = np.array([u((3*n + i)*hx, 2*Ly) for i in range(n + 1)])
  U3_ = U3.copy()

  # ОБЛАСТЬ 4
  U4[0] = np.array([u(2*Lx, (2*m + j)*hy) for j in range(m + 1)])
  U4[-1] = np.array([u(3*Lx, (2*m + j)*hy) for j in range(m + 1)])
  U4_ = U4.copy()

  # ОБЛАСТЬ 5
  U5[0] = np.array([u(0, (3*m + j)*hy) for j in range(m + 1)])
  U5[-1] = np.array([u(5*Lx, (3*m + j)*hy) for j in range(m + 1)])
  U5[: 2*n + 1, 0] = np.array([u(i*hx, 3*Ly) for i in range(2*n + 1)])
  U5[3*n: 5*n + 1, 0] = np.array([u((3*n + i)*hx, 3*Ly) for i in range(2*n + 1)])
  U5[:, -1] = np.array([u(i*hx, 4*Ly) for i in range(5*n + 1)])
  U5_ = U5.copy()

  while True:
    error = 0
    #--------- Внутри области 1 ---------
    for i in range(1, 2*n):
      for j in range(1, m):
        U1_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f(i*hx, j*hy) + hy**2*(U1[i - 1][j] + U1[i + 1][j]) + hx**2*(U1[i][j - 1] + U1[i][j + 1]))
        error += (U1_[i][j] - u(i*hx, j *hy))**2
    
    # Пересечение областей 1 и 3
    j = m
    for i in range(n + 1, 2*n):
      U1_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f(i*hx, j*hy) + hy**2*(U1[i - 1][j] + U1[i + 1][j]) + hx**2*(U1[i][j - 1] + U3[i - n][1]))
      error += (U1_[i][j] - u(i*hx, j *hy))**2

    U1 = U1_.copy()


    #--------- Внутри области 2 ---------
    for i in range(1, 2*n):
      for j in range(1, m):
        U2_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f((3*n + i)*hx, j *hy) + hy**2*(U2[i - 1][j] + U2[i + 1][j]) + hx**2*(U2[i][j - 1] + U2[i][j + 1]))
        error += (U2_[i][j] - u((3*n + i)*hx, j *hy))**2
    
    # Пересечение областей 2 и 3
    j = m
    for i in range(1, n):
      U2_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f((3*n + i)*hx, j *hy) + hy**2*(U2[i - 1][j] + U2[i + 1][j]) + hx**2*(U2[i][j - 1] + U3[2*n + i][1]))
      error += (U2_[i][j] - u((3*n + i)*hx, j *hy))**2

    U2 = U2_.copy()


    #--------- Внутри области 3 ---------
    for i in range(1, 3*n):
      for j in range(1, m):
        U3_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f((n + i)*hx, (m + j)*hy) + hy**2*(U3[i - 1][j] + U3[i + 1][j]) + hx**2*(U3[i][j - 1] + U3[i][j + 1]))
        error += (U3_[i][j] - u((n + i)*hx, (m + j)*hy))**2
    
    # Пересечение областей 3 и 4
    j = m
    for i in range(n + 1, 2*n):
      U3_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f((n + i)*hx, (m + j)*hy) + hy**2*(U3[i - 1][j] + U3[i + 1][j]) + hx**2*(U3[i][j - 1] + U4[i - n][1]))
      error += (U3_[i][j] - u((n + i)*hx, (m + j)*hy))**2

    U3_[1 : n, 0] = U1[n + 1: 2*n, m]
    U3_[2*n + 1: 3*n, 0] = U2[1 : n, m]

    U3 = U3_.copy()


    #--------- Внутри области 4 ---------
    for i in range(1, n):
      for j in range(1, m):
        U4_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f((2*n + i)*hx, (2*m + j)*hy) + hy**2*(U4[i - 1][j] + U4[i + 1][j]) + hx**2*(U4[i][j - 1] + U4[i][j + 1]))
        error += (U4_[i][j] - u((2*n + i)*hx, (2*m + j)*hy))**2
    
    # Пересечение областей 4 и 5
    j = m
    for i in range(1, n):
      U4_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f((2*n + i)*hx, (2*m + j)*hy) + hy**2*(U4[i - 1][j] + U4[i + 1][j]) + hx**2*(U4[i][j - 1] + U5[2*n + i][1]))
      error += (U4_[i][j] - u((2*n + i)*hx, (2*m + j)*hy))**2

    U4_[1 : n, 0] = U3[n + 1: 2*n, m]

    U4 = U4_.copy()


    #--------- Внутри области 5 ---------
    for i in range(1, 5*n):
      for j in range(1, m):
        U5_[i][j] = 0.5/(hx**2 + hy**2) * (hx**2*hy**2*f(i*hx, (3*m + j)*hy) + hy**2*(U5[i - 1][j] + U5[i + 1][j]) + hx**2*(U5[i][j - 1] + U5[i][j + 1]))
        error += (U5_[i][j] - u(i*hx, (3*m + j)*hy))**2

    U5_[2*n + 1: 3*n, 0] = U4[1: n, m]
    U5 = U5_.copy()

    if np.sqrt(error) < eps:
      break

  U_draw = np.zeros((5*n + 1, 4*m + 1))

  for i in range(5*n + 1):
    for j in range(m + 1):
      U_draw[i][3*m + j] = U5[i][j]

  for i in range(n + 1):
    for j in range(m + 1):
      U_draw[2*n + i][2*m + j] = U4[i][j]

  for i in range(3*n + 1):
    for j in range(m + 1):
      U_draw[n + i][m + j] = U3[i][j]

  for i in range(2*n + 1):
    for j in range(m + 1):
      U_draw[3*n + i][j] = U2[i][j]

  for i in range(2*n + 1):
    for j in range(m + 1):
      U_draw[i][j] = U1[i][j]

  Draw(U_draw, (0, 5*Lx), (0, 4*Ly), "Метод Якоби в сложной области")
  #Draw(U, (0, 5*Lx), (0, 4*Ly))
  plt.show()

def A(uh):
  uh_ = uh.copy()
  
  for i in range(1, n):
    for j in range(1, m):
      uh_[i][j] =  - ((uh[i - 1][j] - 2*uh[i][j] + uh[i + 1][j]) / hx**2 + (uh[i][j - 1] - 2*uh[i][j] + uh[i][j + 1]) / hy**2)

  return uh_

def IOMmRect():
  # MN = (n + 1) * (m + 1)
  MN = int ((n + 1) * (m + 1) / 2)
  H = np.zeros((MN + 1, MN + 1))

  Uh = np.zeros((n + 1, m + 1))
  # Границы области
  Uh[:, 0] = np.array([u(i*hx, 0) for i in range(n + 1)])
  Uh[:, -1] = np.array([u(i*hx, Ly) for i in range(n + 1)])
  Uh[0] = np.array([u(0, j*hy) for j in range(m + 1)])
  Uh[-1] = np.array([u(Lx, j*hy) for j in range(m + 1)])

  b_ = Uh.copy()

  for i in range(1, n):
    for j in range(1, m):
      b_[i][j] = f(i*hx, j*hy)

  while True:
    r0 = b_ - A(Uh)
    betha = np.linalg.norm(r0, ord='fro')

    if betha < 1e-5:
      break
      
    v = [ r0 / betha ]
    w = []

    k = int(MN / 2)
    for j in range(MN):
      w.append(A(v[j]))
      for i in range(max([j - k + 1, 0]), j + 1):
        H[i][j] = w[j].flatten() @ v[i].flatten()
        w[j] -= H[i][j] * v[i]
      H[j + 1][j] = np.linalg.norm(w[j], ord='fro')
      if H[j + 1][j] < 10**-12:
        MN = j + 1
        break
      v.append(w[j] / H[j + 1][j])

    g = np.zeros(MN)
    g[0] = betha

    H = H[:MN, :MN]

    for i in range(MN - 1):
      a = H[i][i]
      b = H[i + 1][i]

      c = a / np.sqrt(a**2 + b**2)
      s = b / np.sqrt(a**2 + b**2)

      H1 =   c * H[i] + s * H[i + 1]
      H2 = - s * H[i] + c * H[i + 1]

      H[i] = H1.copy()
      H[i + 1] = H2.copy()

      g[i + 1] = - s * g[i]
      g[i] *= c

    y = np.zeros(MN)

    for i in range(MN - 1, -1, -1):
      y[i] = (g[i] - y[i + 1:] @ H[i][i + 1:]) / H[i][i]

    for i in range(MN):
        Uh += y[i] * v[i]

  Draw(Uh, (0, Lx), (0, Ly), "Метод IOM(m) в прямоугольной области")
  plt.show()

  return Uh

def Dh2(U1, U2, U3, U4, U5):

  # 1 область
  U1_ = U1.copy()
  for i in range(1, 2*n):
    for j in range(1, m):
      U1_[i][j] = - ((U1[i - 1][j] - 2*U1[i][j] + U1[i + 1][j]) / hx**2 + (U1[i][j - 1] - 2*U1[i][j] + U1[i][j + 1]) / hy**2)

  # На границе 1 и 3 области
  for i in range(n + 1, 2*n):
    U1_[i][m] = - ((U1[i - 1][m] - 2*U1[i][m] + U1[i + 1][m]) / hx**2 + (U1[i][m - 1] - 2*U1[i][m] + U3[i - n][1]) / hy**2)

  # 2 область
  U2_ = U2.copy()
  for i in range(1, 2*n):
    for j in range(1, m):
      U2_[i][j] = - ((U2[i - 1][j] - 2*U2[i][j] + U2[i + 1][j]) / hx**2 + (U2[i][j - 1] - 2*U2[i][j] + U2[i][j + 1]) / hy**2)

  # На границе 2 и 3 области
  for i in range(1, n):
    U2_[i][m] = - ((U2[i - 1][m] - 2*U2[i][m] + U2[i + 1][m]) / hx**2 + (U2[i][m - 1] - 2*U2[i][m] + U3[i + 2*n][1]) / hy**2)

  # 3 область
  U3_ = U3.copy()
  for i in range(1, 3*n):
    for j in range(1, m):
      U3_[i][j] = - ((U3[i - 1][j] - 2*U3[i][j] + U3[i + 1][j]) / hx**2 + (U3[i][j - 1] - 2*U3[i][j] + U3[i][j + 1]) / hy**2)

  # 3 и 1 - копируем из U1_
  U3_[1 : n, 0] = U1_[n + 1 : -1, m]

  # 3 и 2 - копируем из U2_
  U3_[2*n + 1 : -1, 0] = U2_[1 : n, m]

  # На границе 3 и 4 области
  for i in range(n + 1, 2*n):
    U3_[i][m] = - ((U3[i - 1][m] - 2*U3[i][m] + U3[i + 1][m]) / hx**2 + (U3[i][m - 1] - 2*U3[i][m] + U4[i - n][1]) / hy**2)

  # 4 область
  U4_ = U4.copy()
  for i in range(1, n):
    for j in range(1, m):
      U4_[i][j] = - ((U4[i - 1][j] - 2*U4[i][j] + U4[i + 1][j]) / hx**2 + (U4[i][j - 1] - 2*U4[i][j] + U4[i][j + 1]) / hy**2)

  # 4 и 3 - копируем из U3_
  U4_[1 : -1, 0] = U3_[n + 1 : 2*n, m]

  # На границе 4 и 5 области
  for i in range(1, n):
    U4_[i][m] = - ((U4[i - 1][m] - 2*U4[i][m] + U4[i + 1][m]) / hx**2 + (U4[i][m - 1] - 2*U4[i][m] + U5[i + 2*n][1]) / hy**2)

  # 5 область
  U5_ = U5.copy()
  for i in range(1, 5*n):
    for j in range(1, m):
      U5_[i][j] = - ((U5[i - 1][j] - 2*U5[i][j] + U5[i + 1][j]) / hx**2 + (U5[i][j - 1] - 2*U5[i][j] + U5[i][j + 1]) / hy**2)

  # 5 и 4 - копируем из U4_
  U5_[2*n + 1 : 3*n, 0] = U4_[1 : -1, m]

  return U1_, U2_, U3_, U4_, U5_

def IOMmArea():

  U1 = np.zeros((2*n + 1, m + 1))
  U2 = np.zeros((2*n + 1, m + 1))
  U3 = np.zeros((3*n + 1, m + 1))
  U4 = np.zeros((n + 1, m + 1))
  U5 = np.zeros((5*n + 1, m + 1))

  #---------- Граница ---------
  # ОБЛАСТЬ 1
  U1[0] = np.array([u(0, j*hy) for j in range(m + 1)])
  U1[-1] = np.array([u(2*Lx, j*hy) for j in range(m + 1)])
  U1[:, 0] = np.array([u(i*hx, 0) for i in range(2*n + 1)])
  U1[: n + 1, -1] = np.array([u(i*hx, Ly) for i in range(n + 1)])

  B1 = U1.copy()
  for i in range(1, 2*n):
    for j in range(1, m):
      B1[i][j] = f(i*hx, j*hy)

  # Пересечение 1 и 3 области:
  B1[n + 1 : -1, m] = np.array([f((n + i)*hx, m*hy) for i in range(1, n)])

  # ОБЛАСТЬ 2
  U2[0] = np.array([u(3*Lx, j*hy) for j in range(m + 1)])
  U2[-1] = np.array([u(5*Lx, j*hy) for j in range(m + 1)])
  U2[:, 0] = np.array([u((3*n + i)*hx, 0) for i in range(2*n + 1)])
  U2[n :, -1] = np.array([u((4*n + i)*hx, Ly) for i in range(n + 1)])

  B2 = U2.copy()
  for i in range(1, 2*n):
    for j in range(1, m):
      B2[i][j] = f((3*n + i)*hx, j*hy)

  # Пересечение 2 и 3 области:
  B2[1 : n, m] = np.array([f((3*n + i)*hx, m*hy) for i in range(1, n)])

  # ОБЛАСТЬ 3
  U3[0] = np.array([u(Lx, (m + j)*hy) for j in range(m + 1)])
  U3[-1] = np.array([u(4*Lx, (m + j)*hy) for j in range(m + 1)])
  U3[n: 2*n + 1, 0] = np.array([u((2*n + i)*hx, Ly) for i in range(n + 1)])
  U3[: n + 1, -1] = np.array([u((n + i)*hx, 2*Ly) for i in range(n + 1)])
  U3[2*n: 3*n + 1, -1] = np.array([u((3*n + i)*hx, 2*Ly) for i in range(n + 1)])

  B3 = U3.copy()
  for i in range(1, 3*n):
    for j in range(1, m):
      B3[i][j] = f((n + i)*hx, (m + j)*hy)

  # Пересечение 3 и 1 области (копируем из B1):
  B3[1 : n, 0] = B1[n + 1 : -1, m]

  # Пересечение 3 и 2 области (копируем из B2):
  B3[2*n + 1 : -1, 0] = B2[1 : n, m]

  # Пересечение 3 и 4 области:
  B3[n + 1 : 2*n, m] = np.array([f((2*n + i)*hx, 2*m*hy) for i in range(1, n)])

  # ОБЛАСТЬ 4
  U4[0] = np.array([u(2*Lx, (2*m + j)*hy) for j in range(m + 1)])
  U4[-1] = np.array([u(3*Lx, (2*m + j)*hy) for j in range(m + 1)])

  B4 = U4.copy()
  for i in range(1, n):
    for j in range(1, m):
      B4[i][j] = f((2*n + i)*hx, (2*m + j)*hy)

  # Пересечение 4 и 3 области (копируем из B3):
  B4[1 : -1, 0] = B3[n + 1 : 2*n, m]

  # ОБЛАСТЬ 5
  U5[0] = np.array([u(0, (3*m + j)*hy) for j in range(m + 1)])
  U5[-1] = np.array([u(5*Lx, (3*m + j)*hy) for j in range(m + 1)])
  U5[: 2*n + 1, 0] = np.array([u(i*hx, 3*Ly) for i in range(2*n + 1)])
  U5[3*n: 5*n + 1, 0] = np.array([u((3*n + i)*hx, 3*Ly) for i in range(2*n + 1)])
  U5[:, -1] = np.array([u(i*hx, 4*Ly) for i in range(5*n + 1)])

  B5 = U5.copy()
  for i in range(1, 5*n):
    for j in range(1, m):
      B5[i][j] = f(i*hx, (3*m + j)*hy)

  # Пересечение 5 и 4 области (копируем из B4):
  B5[2*n + 1 : 3*n, 0] = B4[1 : -1, m]

  MN = int(((n + 1) * (m + 1) + 2* ((2*n + 1) * (m + 1)) + (3*n + 1) * (m + 1) + (5*n + 1) * (m + 1)) / 2)
  H = np.zeros((MN + 1, MN + 1))

  k = int(MN / 2)

  while True:
    A1, A2, A3, A4, A5 = Dh2(U1, U2, U3, U4, U5)
    r01 = B1 - A1
    r02 = B2 - A2
    r03 = B3 - A3
    r04 = B4 - A4
    r05 = B5 - A5

    betha = np.sqrt(np.linalg.norm(r01, ord='fro') ** 2 + \
                    np.linalg.norm(r02, ord='fro') ** 2 + \
                    np.linalg.norm(r03, ord='fro') ** 2 + \
                    np.linalg.norm(r04, ord='fro') ** 2 + \
                    np.linalg.norm(r05, ord='fro') ** 2)
    
    if betha < 1e-5:
      break
    
    v1 = [ r01 / betha ]
    v2 = [ r02 / betha ]
    v3 = [ r03 / betha ]
    v4 = [ r04 / betha ]
    v5 = [ r05 / betha ]

    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []

    for j in range(MN):
      Av1, Av2, Av3, Av4, Av5 = Dh2(v1[j], v2[j], v3[j], v4[j], v5[j])
      w1.append(Av1)
      w2.append(Av2)
      w3.append(Av3)
      w4.append(Av4)
      w5.append(Av5)
      for i in range(max([j - k + 1, 0]), j + 1):
        H[i][j] = w1[j].flatten() @ v1[i].flatten() + \
                  w2[j].flatten() @ v2[i].flatten() + \
                  w3[j].flatten() @ v3[i].flatten() + \
                  w4[j].flatten() @ v4[i].flatten() + \
                  w5[j].flatten() @ v5[i].flatten()
        
        w1[j] -= H[i][j] * v1[i]
        w2[j] -= H[i][j] * v2[i]
        w3[j] -= H[i][j] * v3[i]
        w4[j] -= H[i][j] * v4[i]
        w5[j] -= H[i][j] * v5[i]

      H[j + 1][j] = np.sqrt(np.linalg.norm(w1[j], ord='fro') ** 2 + \
                            np.linalg.norm(w2[j], ord='fro') ** 2 + \
                            np.linalg.norm(w3[j], ord='fro') ** 2 + \
                            np.linalg.norm(w4[j], ord='fro') ** 2 + \
                            np.linalg.norm(w5[j], ord='fro') ** 2)
      
      if H[j + 1][j] < 10**-12:
        MN = j + 1
        break

      v1.append(w1[j] / H[j + 1][j])
      v2.append(w2[j] / H[j + 1][j])
      v3.append(w3[j] / H[j + 1][j])
      v4.append(w4[j] / H[j + 1][j])
      v5.append(w5[j] / H[j + 1][j])

    g = np.zeros(MN)
    g[0] = betha

    H = H[:MN, :MN]

    for i in range(MN - 1):
      a = H[i][i]
      b = H[i + 1][i]

      c = a / np.sqrt(a**2 + b**2)
      s = b / np.sqrt(a**2 + b**2)

      H1 =   c * H[i] + s * H[i + 1]
      H2 = - s * H[i] + c * H[i + 1]

      H[i] = H1.copy()
      H[i + 1] = H2.copy()

      g[i + 1] = - s * g[i]
      g[i] *= c

    y = np.zeros(MN)

    for i in range(MN - 1, -1, -1):
      y[i] = (g[i] - y[i + 1:] @ H[i][i + 1:]) / H[i][i]

    for i in range(MN):
        U1 += y[i] * v1[i]
        U2 += y[i] * v2[i]
        U3 += y[i] * v3[i]
        U4 += y[i] * v4[i]
        U5 += y[i] * v5[i]

  U_draw = np.zeros((5*n + 1, 4*m + 1))

  for i in range(5*n + 1):
    for j in range(m + 1):
      U_draw[i][3*m + j] = U5[i][j]

  for i in range(n + 1):
    for j in range(m + 1):
      U_draw[2*n + i][2*m + j] = U4[i][j]

  for i in range(3*n + 1):
    for j in range(m + 1):
      U_draw[n + i][m + j] = U3[i][j]

  for i in range(2*n + 1):
    for j in range(m + 1):
      U_draw[3*n + i][j] = U2[i][j]

  for i in range(2*n + 1):
    for j in range(m + 1):
      U_draw[i][j] = U1[i][j]

  Draw(U_draw, (0, 5*Lx), (0, 4*Ly), "Метод IOM(m) в сложной области")
  plt.show()

JacobyRect()
JacobyArea()
IOMmRect()
IOMmArea()