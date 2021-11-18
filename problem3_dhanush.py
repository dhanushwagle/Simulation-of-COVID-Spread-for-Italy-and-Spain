# %% import packages
import matplotlib.pyplot as plt
import numpy as np

# %% Model parameters
N = 60500000
kr = 1/7
ki0 = 0.25168
ke = 1          # α in original paper
c1 = 63.7       # χ2 in original paper
c2 = 0.135      # χ3 in original paper
kca = 1/7       # ν in original paper
n = 0.4         # f in original paper
kc = kca * n    # ν1 in original paper
ka = kca*(1-n)  # ν2 in original paper
Ti = 10.5       # day at which the mitigation policy starts

# %% simulation time settings
Ts = 1 # Time step [s]
t_start = 0 
t_stop = 300 
N_sim =int((t_stop - t_start)/Ts) + 1   # Number of Time_steps

# %% Preallocation of arrays for plotting :
S = np.zeros(N_sim)
E = np.zeros(N_sim)
X = np.zeros(N_sim)
I = np.zeros(N_sim)
C = np.zeros(N_sim)
CC = np.zeros(N_sim)
A = np.zeros(N_sim)
R = np.zeros(N_sim)
t= np.linspace(t_start,t_stop,N_sim)

# %% Initialization :
S[0] = N
E[0] = (c1 + kca) / ke
I[0]= (c1 * c2) / kc
C[0] = 1
CC[0] = 1
A[0] = (ka*I[0])/(kr+c1)
R[0] = 0
X[0] = 1

# %% Simulation loop :
for k in range(0, N_sim-1):
    # Values of U
    if 0<= k <=27:
        U = 0.9    
    elif 27<= k <=33:
        U = 0.85
    elif 33<= k <=57:
        U = 0.2
    elif 57<= k <=67:
        U = 0.25
    elif 67<= k <=95:
        U = 0.5
    elif 95<= k <=140:
        U = 0.85
    else:
        U = 0.9
    # Mitigation Model    
    dX_dt = (1/Ti)*(U-X[k])
    ki = ki0*X[k]
    
    # SEICAR Model
    dS_dt = -(ki/N)*(I[k]+A[k])*S[k]
    dE_dt = ((ki/N)*(I[k]+A[k])*S[k]) - ke*E[k]
    dI_dt = ke*E[k]-kc*I[k]-ka*I[k]
    dC_dt = kc*I[k]-kr*C[k]
    dCC_dt = kc*I[k] # for cumulative confirmed cases just considering compartment C
    dA_dt = ka*I[k]-kr*A[k]
    dR_dt = kr*C[k]+kr*A[k]  
    
    # State updates using the Euler method :
    S[k+1] = S[k] + dS_dt *Ts
    E[k+1] = E[k] + dE_dt *Ts
    I[k+1] = I[k] + dI_dt *Ts
    C[k+1] = C[k] + dC_dt *Ts
    CC[k+1] = CC[k] + dCC_dt *Ts
    A[k+1] = A[k] + dA_dt *Ts
    R[k+1] = R[k] + dR_dt *Ts
    X[k+1] = X[k] + dX_dt *Ts
    
# %% Plotting :
plt.close('all')
plt.figure(1)
plt.plot(t, CC, 'b')
plt.legend(labels=('Cumulative Confirmed for Italy'))
plt.grid()
plt.show()
