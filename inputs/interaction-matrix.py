#!/usr/local/bin/python3

import matplotlib
import numpy as np
from numpy import loadtxt
# np.random.seed(21)

n_S = 10
n_s = 4
n_phis_cal = 1
n_phis_val = 0
n_times = 10
var = 0.001
#model inadequacy type: 1 = absolute value, 2 = squared, 3 = nonlinear diagonal, 4 = nonlinear mixed,
# 5 = state variables, 6 = diagonal state variables
#10 = memory, 11 = algebraic
inad_type = 6
filename = 'inputs/info.txt'
file = open(filename,'w')
file.write(str(n_S)+'\n'+str(n_s)+'\n'+str(n_phis_cal)+'\n'+str(n_phis_val)+'\n'+str(n_times)+'\n'+str(var)+'\n'+str(inad_type)+'\n')
file.close()

# choose if A is diagonally dominant or not:
# set to 0 for not, set to 1 to be dd
c = 1

# filename = 'inputs/matrix.txt'
# file = open(filename,'w')

# b = np.sqrt(n_S)*np.random.lognormal(0,1,n_S)
# b = n_S*[n_S]
B = np.random.lognormal(0,1,(n_S,n_S))
B = np.tril(B,-1)
B = B + B.T
offdiag = np.sum(B,axis=1)
# C = np.sqrt(n_S)*np.diag(np.random.lognormal(0,1,n_S))
if c == 0:
    C = np.diag(np.random.lognormal(0,n_S,n_S))
elif c == 1:
    # make A diagonally dominant with below
    C = np.diag(offdiag + np.random.lognormal(0,1,n_S))

# print(np.max(C))
A = -(B + C)
b = np.ones((n_S,1))
b = np.max(C)*b

# print(A)
# print(b)

seed_num = 13
np.random.seed(seed=seed_num)

def genereate_detailed_realization(varianceB, varianceC, size):
  """
  symmetric interaction matrix A with non-positive entries
  """
  np.random.seed(seed=seed_num)
  B = np.zeros((size, size))
  C = np.zeros((size, size))
  sampleB = np.random.lognormal(sigma=varianceB, size=size**2)
  sampleC = np.random.lognormal(sigma=varianceC, size=size)
  # sample B_ij ~ logN(0, varianceB), 1<= i< j<= S
  # B_ji = B_ij
  for i in range(size):
    for j in range(i+1, size):
      B[i][j] = sampleB[i+j-1]
      B[j][i] = sampleB[i+j-1]
  # sample C_ii ~ logN(0, varianceC) + sum_[k!=i}(B_ki), 1 <= i <= S
  for i in range(size):
    col_sum = 0.0
    for k in range(size):
      if k != i:
        col_sum += B[k][i]
    C[i][i] = sampleC[i] + col_sum
  A = -(B+C)

  r = np.ones((size,1))
  print(np.max(C))
  r = np.max(C)*r

  return A, r

def generate_no_symmetry_detailed_realization(varianceB, varianceC, size):
  """
  competitive asymmetric interaction matrix A with non-positive entries
  """
  np.random.seed(seed=seed_num)
  B = np.zeros((size, size))
  C = np.zeros((size, size))
  sampleB = np.random.lognormal(sigma=varianceB, size=size**2)
  sampleC = np.random.lognormal(sigma=varianceC, size=size)
  # sample B_ij ~ logN(0, varianceB), 1<= i< j<= S
  # B_ji = B_ij
  index = 0
  for i in range(size):
    for j in range(size):
      if i != j:
        B[i][j] = sampleB[index]
        index += 1
      # B[j][i] = sampleB[i+j-1]
  # sample C_ii ~ logN(0, varianceC) + sum_[k!=i}(B_ki), 1 <= i <= S
  for i in range(size):
    col_sum = 0.0
    for k in range(size):
      if k != i:
        col_sum += B[k][i]
    C[i][i] = sampleC[i] + col_sum
  A = -(B+C)

  r = np.ones((size,1))
  print(np.max(C))
  r = np.max(C)*r

  return A, r

def genereate_relaxed_detailed_realization(varianceB, varianceC, size):
  """
  symmetric interaction matrix A with diagonally dominance (pos and neg entries)
  """
  np.random.seed(seed=seed_num)
  B = np.zeros((size, size))
  C = np.zeros((size, size))
  # sampleB = np.random.normal(scale=varianceB, size=size**2)
  sampleB = np.random.lognormal(sigma=varianceB, size=size**2)
  sampleC = np.random.lognormal(sigma=varianceC, size=size)
  # sample B_ij ~ logN(0, varianceB), 1<= i< j<= S
  # B_ji = B_ij
  for i in range(size):
    for j in range(i+1, size):
      B[i][j] = sampleB[i+j-1]
      B[j][i] = sampleB[i+j-1]
  # sample C_ii ~ logN(0, varianceC) + sum_[k!=i}(B_ki), 1 <= i <= S
  for i in range(size):
    col_sum = 0.0
    for k in range(size):
      if k != i:
        col_sum += B[k][i]
    C[i][i] = sampleC[i] + col_sum
  # print("b\n", B)
  A = -(B+C)

  pos = np.random.randint(0, high=n_S, size=n_S)
  # print(pos)
  for index in range(0, len(pos)-1, 2):
    i = pos[index]
    j = pos[index+1]
    while i == j:
      j = np.random.randint(0, high=n_S)
    A[i][j] *= -1
    A[j][i] *= -1
    # print(A[i][j])

  r = np.ones((size,1))
  # print(np.max(C))
  r = np.max(C)*r

  return A, r

def generate_relaxed_no_symmetry_detailed_realization(varianceB, varianceC, size):
  """
  asymmetric interaction matrix A with diagonally dominance (pos and neg entries)
  """
  np.random.seed(seed=seed_num)
  B = np.zeros((size, size))
  C = np.zeros((size, size))
  sampleB = np.random.lognormal(sigma=varianceB, size=size**2)
  sampleC = np.random.lognormal(sigma=varianceC, size=size)
  # sample B_ij ~ logN(0, varianceB), 1<= i< j<= S
  # B_ji = B_ij
  index = 0
  for i in range(size):
    for j in range(size):
      if i != j:
        B[i][j] = sampleB[index]
        index += 1
      # B[j][i] = sampleB[i+j-1]
  # sample C_ii ~ logN(0, varianceC) + sum_[k!=i}(B_ki), 1 <= i <= S
  for i in range(size):
    col_sum = 0.0
    for k in range(size):
      if k != i:
        col_sum += B[k][i]
    C[i][i] = sampleC[i] + col_sum
  A = -(B+C)

  pos = np.random.randint(0, high=n_S, size=n_S*2)
  # print(pos)
  for index in range(0, len(pos)-1, 2):
    i = pos[index]
    j = pos[index+1]
    while i == j:
      j = np.random.randint(0, high=n_S)
    A[i][j] *= -1

  r = np.ones((size,1))
  print(np.max(C))
  r = np.max(C)*r

  return A, r

def gen_halfpos(varianceB, varianceC, c=1, symm=True):
  B = np.random.lognormal(0,1,(n_S,n_S))
  flips = np.random.randint(4, size= (n_S,n_S))
  # print(flips)
  for i in range(len(B)):
    for j in range(i+1, len(B[0])):
      # if i == j:
      #   B[i][j] = 0
      if flips[i][j] == 0:
        # print(i,j)
        B[i][j] *= -1
      if symm:
        B[j][i] = B[i][j]

  print(B)

  # B = np.tril(B,-1)
  # B = B + B.T
  offdiag = np.sum(B,axis=1)
  
  # offdiag = np.zeros(len(B))
  # for i in range(len(B)):
  #   for j in range(len(B[0])):
  #     offdiag[i] += abs(B[i][j])
  print(offdiag)
    

  # print(offdiag)
  # C = np.sqrt(n_S)*np.diag(np.random.lognormal(0,1,n_S))
  if c == 0:
      C = np.diag(np.random.lognormal(0,n_S,n_S))
  elif c == 1:
      # make A diagonally dominant with below
      C = np.diag(offdiag + np.random.lognormal(0,1,n_S))

  # print(np.max(C))
  A = -(B + C)
  # A = B - C
  b = np.ones((n_S,1))
  b = np.max(C)*b
  return A,b

# A, b = genereate_detailed_realization(1.0, 1.0, n_S)
# A, b = genereate_relaxed_detailed_realization(1.0, 1.0, n_S)
# A, b = generate_no_symmetry_detailed_realization(1.0, 1.0, n_S)
# A, b = generate_relaxed_no_symmetry_detailed_realization(1.0, 1.0, n_S)
A, b = gen_halfpos(1.0, 1.0, symm=False)
print(np.array(A))
print(b)

np.savetxt("inputs/matrix2.txt", A)
np.savetxt("inputs/growthrates2.txt", b)

# for i in range(n_S):
#   file.write(str(b[i][0])+'\n')
# for i in range(n_S):
#   for j in range(n_S):
#     file.write(str(A[i][j])+'\n')

# file.close()
