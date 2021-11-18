# %% import packages
import matplotlib.pyplot as plt
import numpy as np

# %% Model parameters for Italy
N_I = 60500000
kr_I = 1/7
ki0_I = 0.25168
ke_I = 1          # α in original paper
c1_I = 63.7      # χ2 in original paper
c2_I = 0.135          # χ3 in original paper
kca_I = 1/7       # ν in original paper
n_I = 0.4         # f in original paper
kc_I = kca_I * n_I    # ν1 in original paper
ka_I = kca_I*(1-n_I)  # ν2 in original paper
Ti = 10.5 

# %% Model parameters for Spain
N_S = 46700000
kr_S = 1/7
ki0_S = 0.33204
ke_S = 1                # α in original paper
c1_S = 433.3             # χ2 in original paper
c2_S = 0.194            # χ3 in original paper
kca_S = 1/7             # ν in original paper
n_S = 0.4                # f in original paper
kc_S = kca_S * n_S      # ν1 in original paper
ka_S = kca_S * (1-n_S)  # ν2 in original paper
N = N_I + N_S
Ni = 100000              # immigration rate for both Italy and Spain
Ne = 100000             # emmigration rate for both Italy and Spain 

# %% simulation time settings
Ts = 1 # Time step [s]
t_start = 0 
t_stop = 300
N_sim =int((t_stop - t_start)/Ts) + 1   # %% Number of Time_steps

# %% Preallocation of arrays for plotting :
S_I = np.zeros(N_sim)
E_I = np.zeros(N_sim)
I_I = np.zeros(N_sim)
C_I = np.zeros(N_sim)
CC_I = np.zeros(N_sim)
A_I = np.zeros(N_sim)
R_I = np.zeros(N_sim)
X_I = np.zeros(N_sim)

t = np.linspace(t_start, t_stop, N_sim)

S_S = np.zeros(N_sim)
E_S = np.zeros(N_sim)
I_S = np.zeros(N_sim)
C_S = np.zeros(N_sim)
CC_S = np.zeros(N_sim)
A_S = np.zeros(N_sim)
R_S = np.zeros(N_sim)
X_S = np.zeros(N_sim)

# %% Initialization :

S_I[0] = N_I
E_I[0] = (c1_I + kca_I) / ke_I
I_I[0]= (c1_I * c2_I) / kc_I
C_I[0] = 1
CC_I[0] = 1
A_I[0] = (ka_I*I_I[0])/(kr_I+c1_I)
R_I[0] = 0
X_I[0] = 1

S_S[0] = N_S
E_S[0] = (c1_S + kca_S) / ke_S
I_S[0]= (c1_S * c2_S) / kc_S
C_S[0] = 1
CC_S[0] = 1
A_S[0] = (ka_S*I_S[0])/(kr_S+c1_S)
R_S[0] = 0
X_S[0] = 1

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
    dX_I_dt = (1/Ti)*(U-X_I[k])
    ki_I = ki0_I*X_I[k]
    
    dS_I_dt = Ni*S_S[k]/N_S - Ne*S_I[k]/N_I - (ki_I/N_I)*(I_I[k]+A_I[k])*S_I[k]
    dE_I_dt = Ni*E_S[k]/N_S - Ne*E_I[k]/N_I + ((ki_I/N_I)*(I_I[k]+A_I[k])*S_I[k]) - ke_I*E_I[k]
    dI_I_dt = Ni*I_S[k]/N_S - Ne*I_I[k]/N_I + ke_I*E_I[k]-kc_I*I_I[k]-ka_I*I_I[k]
    dC_I_dt = Ni*C_S[k]/N_S - Ne*C_I[k]/N_I + kc_I*I_I[k]-kr_I*C_I[k]
    dCC_I_dt = Ni*CC_S[k]/N_S - Ne*CC_I[k]/N_I + kc_I*I_I[k]
    dA_I_dt = Ni*A_S[k]/N_S - Ne*A_I[k]/N_I + ka_I*I_I[k]-kr_I*A_I[k]
    dR_I_dt = Ni*R_S[k]/N_S - Ne*R_I[k]/N_I + kr_I*C_I[k]+kr_I*A_I[k]    
    
    # State updates using the Euler method :
    S_I[k+1] = S_I[k] + dS_I_dt *Ts
    E_I[k+1] = E_I[k] + dE_I_dt *Ts
    I_I[k+1] = I_I[k] + dI_I_dt *Ts
    C_I[k+1] = C_I[k] + dC_I_dt *Ts
    A_I[k+1] = A_I[k] + dA_I_dt *Ts
    R_I[k+1] = R_I[k] + dR_I_dt *Ts
    CC_I[k+1] = CC_I[k] + dCC_I_dt *Ts
    X_I[k+1] = X_I[k] + dX_I_dt *Ts
    
    # Values of U
    if 0<= k <=40:
        U = 1
    if 40<= k <=50:
        U = 0.2    
    elif 40<= k <=70:
        U = 0.15
    elif 70<= k <=90:
        U = 0.25
    elif 90<= k <=120:
        U = 0.35
    elif 120<= k <=160:
        U = 0.7
    else:
        U = 0.9
    # Mitigation Model    
    dX_S_dt = (1/Ti)*(U-X_S[k])
    ki_S = ki0_S*X_S[k]
    
    dS_S_dt = Ni*S_I[k]/N_I - Ne*S_S[k]/N_S - (ki_S/N_S)*(I_S[k]+A_S[k])*S_S[k]
    dE_S_dt = Ni*E_I[k]/N_I - Ne*E_S[k]/N_S + ((ki_S/N_S)*(I_S[k]+A_S[k])*S_S[k]) - ke_S*E_S[k]
    dI_S_dt = Ni*I_I[k]/N_I - Ne*I_S[k]/N_S + ke_S*E_S[k]-kc_S*I_S[k]-ka_S*I_S[k]
    dC_S_dt = Ni*C_I[k]/N_I - Ne*C_S[k]/N_S + kc_S*I_S[k]-kr_S*C_S[k]
    dCC_S_dt = Ni*CC_I[k]/N_I - Ne*CC_S[k]/N_S + kc_S*I_S[k]
    dA_S_dt = Ni*A_I[k]/N_I - Ne*A_S[k]/N_S + ka_S*I_S[k]-kr_S*A_S[k]
    dR_S_dt = Ni*R_I[k]/N_I - Ne*R_S[k]/N_S + kr_S*C_S[k]+kr_S*A_S[k]    
    
    # State updates using the Euler method :
    S_S[k+1] = S_S[k] + dS_S_dt *Ts
    E_S[k+1] = E_S[k] + dE_S_dt *Ts
    I_S[k+1] = I_S[k] + dI_S_dt *Ts
    C_S[k+1] = C_S[k] + dC_S_dt *Ts
    A_S[k+1] = A_S[k] + dA_S_dt *Ts
    R_S[k+1] = R_S[k] + dR_S_dt *Ts
    CC_S[k+1] = CC_S[k] + dCC_S_dt *Ts
    X_S[k+1] = X_S[k] + dX_S_dt *Ts
    
# %% Plotting :

plt.close('all')
plt.figure(1)
plt.plot(t, CC_I, 'b-')
plt.plot(t, CC_S, 'g-')
plt.legend(labels=('Italy CC', 'Spain CC'))
plt.grid()
plt.show()