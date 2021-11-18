# %% import packages
import matplotlib.pyplot as plt
import numpy as np

# %% Model parameters
N = 60500000
kr = 1/7
ki = 0.25168
ke = 1          # α in original paper
c1 = 63.7       # χ2 in original paper
c2 = 0.135      # χ3 in original paper
kca = 1/6       # ν in original paper
n = 0.4         # f in original paper
kc = kca * n    # ν1 in original paper
ka = kca*(1-n)  # ν2 in original paper
Ti = 10.5       # day at which the mitigation policy starts

# %% simulation time settings
Ts = 1 # Time step [s]
t_start = 0 
t_stop = 250 
N_sim =int((t_stop - t_start)/Ts) + 1   # Number of Time_steps

# %% Preallocation of arrays for plotting :
S = np.zeros(N_sim)
V = np.zeros(N_sim)
E = np.zeros(N_sim)
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
V[0] = 0

# %% Simulation loop :
for k in range(0, N_sim-1):
    # Values of Vaccination Efficiency
    if 50<= k <=N_sim-1:
        p = 0.5 
    else:
        p = 0
    
    # Vaccination Model    
    dV_dt = p*S[k]
    
    # SEICAR Model
    dS_dt = -(ki/N)*(I[k]+A[k])*S[k] - p*S[k]
    dE_dt = ((ki/N)*(I[k]+A[k])*S[k]) - ke*E[k]
    dI_dt = ke*E[k]-kc*I[k]-ka*I[k]
    dC_dt = kc*I[k]-kr*C[k]
    dCC_dt = kc*I[k] # for cumulative confirmed cases just considering compartment C
    dA_dt = ka*I[k]-kr*A[k]
    dR_dt = kr*C[k]+kr*A[k] + p*S[k]
    
    # State updates using the Euler method :
    S[k+1] = S[k] + dS_dt *Ts
    E[k+1] = E[k] + dE_dt *Ts
    I[k+1] = I[k] + dI_dt *Ts
    C[k+1] = C[k] + dC_dt *Ts
    CC[k+1] = CC[k] + dCC_dt *Ts
    A[k+1] = A[k] + dA_dt *Ts
    R[k+1] = R[k] + dR_dt *Ts
    V[k+1] = V[k] + dV_dt *Ts
    
# %% Plotting :
plt.close('all')
plt.figure(1)
plt.plot(t, CC)
plt.legend(labels=('Cumulative Confirmed',))
plt.grid()
plt.show()
print(CC[249])