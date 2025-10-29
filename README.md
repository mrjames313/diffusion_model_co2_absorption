# diffusion_model_co2_absorption
Model of the chemical process for absorption of gaseous co2 using a solution of CaO in a CaCl2 solution using a learned diffusion model.  This is a relatively simple dynamical system which is used to sample from the ODE governing the chemical reactions.  These samples are the input to the ML learning of the diffusion model.

This notebook was developed using both AI-assisted (GPT 5) and traditional coding.

This notebook should be able to be run top-to-bottom with any changes to the top-cell parameters.  These specify:
 - use_full_x: Specifies whether to use just the concentrations of CO2, CaO, and CaCO3 in solution (which is a partially observable system), or to also include the randomly sampled reaction parameters (k_abs - the absorption rate of CO2, k_react - the reaction rate of CO2 and CaO, and C_gas - the concentration of CO2 in gaseous form, which makes it a fully observable system.
 - use_time: whether to also encode time as input to the NN.  This is important when we model a time-varying concentration of gaseous CO2, but can also lead to overfitting in the partially observable system.
 - use_time_varying_c_gas: add a periodic variation to the concentration of the CO2 gas.  This is a more interesting problem and takes longer to learn a good model.
 - A, f, phi: these are parameters governing the CO2 oscillation

The notebook works by first defining a traditional (ODE-based) model of the system including dissolving and chemical reactions.  This is accomplished through numerical approximation using Runge-Kutta (4).  This model is sampled for a number of trajectories of the system dynamics, which are initiated using randomly selected (from a reasonable range) reaction parameters - see above for details.  Each timestep of the trajectory contains both the state (input to the MLP) and the derivatives for each of the concentrations (target of the MLP).  These trajectories are then subsampled in order to create a training dataset, which is used to train the MLP.  

The MLP is a Torch model using alternating linear and SiLU layers.  We use the typical training functions, with a few tweaks as can be seen in the code. 

