# %% import packages
import matplotlib.pyplot as plt
import numpy as np

# %% Model parameters
N = 763
kr = 0.3
ki = 0.0025 * N
Id = [25 , 75, 227 , 296 , 258 , 236 , 192 , 126 , 71, 28, 11, 7]
Td = np.linspace(3,14,12)

# %% simulation time settings
Ts = 0.1 # Time step [s]
t_start = 3  # [s]
t_stop = 14 # [s]
N_sim =int((t_stop - t_start)/Ts) + 1   # %% Number of Time_steps

# %% Preallocation of arrays for plotting :
S = np.zeros(N_sim)
I = np.zeros(N_sim)
R = np.zeros(N_sim)
t= np.linspace(3,14,N_sim)

# %% Initialization :
I[0] = 25
S[0] = N - I[0]
R[0] = 0 

# %% Simulation loop :
for k in range(0, N_sim-1):

    dS_dt = -(ki/N)*I[k]*S[k]
    dI_dt = ((ki/N)*S[k] - (kr))*I[k]
    dR_dt = kr*I[k]    
    # State updates using the Euler method :
    S[k+1] = S[k] + dS_dt *Ts
    I[k+1] = I[k] + dI_dt *Ts
    R[k+1] = R[k] + dR_dt *Ts
    
# %% Plotting :
plt.close('all')
plt.figure(1)
plt.plot(t, S, 'b')
plt.plot(t, I, 'r')
plt.plot(t, R, 'g')
plt.plot(Td, Id, 'ko')
plt.grid()
plt.xlabel('t [Day]')
plt.ylabel('Boys')
plt.legend(labels=('S','I', 'R','Id'))
plt.show()