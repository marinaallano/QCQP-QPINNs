import torch
import pennylane as qml
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.integrate import solve_ivp
import time
start_time = time.perf_counter()

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})


torch.manual_seed(42)
torch.set_num_threads(30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Constants

# Domain Parameter
X_COLLOC_POINTS = 100
BOUNDARY_SCALE = 10e1
X_END = 1.0 

HIDDEN_LAYER_FNN = 1
NEURONS_FNN = 5

##  Generate Domain
# Generate Collocation Points
x = torch.linspace(0.0, X_END, X_COLLOC_POINTS, device=device, requires_grad=True)

# Directory to store result plots
RESULTS_DIR = "results_layers_plots"
os.makedirs(RESULTS_DIR, exist_ok=True)


# EDOs to solve
def derivatives_fnc(EDO, x, u):
    if EDO == 1 :
        du_dx = 4*u - 6*u**2 + math.sin(50*x) + u*math.cos(25*x) - 0.5
    elif EDO == 2: 
        lamb = 8
        kappa = 0.1
        du_dx = -lamb*u*(kappa+torch.tan(lamb*x)) 
    elif EDO == 3:
        lamb = 20
        kappa = 0.1
        du_dx = -lamb*u*(kappa+torch.tan(lamb*x)) 

    return du_dx


## Create the Model
# Define QPINN
def circuit(x, basis=None):

    # Quantum Feature Map Encoding: 
    if map == "feature_map":
        for i in range(N_WIRES):
            qml.RY(torch.arcsin(x), wires=i)
    
    if map == "Chebyschev_sparse":
        for i in range(N_WIRES):
            qml.RY(2 * torch.arccos(x), wires=i)

    if map == "Chebyschev_tower":
        for i in range(N_WIRES):
            qml.RY(2 * (i+1) * torch.arccos(x), wires=i)
    
    if map == "fnn_map":
        for i in range(N_WIRES):
            qml.RY(basis[i] * x, wires=i)


    # Variational Quantum Circuit: Hardware Efficient Ansatz (paper **)
    for i in range(N_LAYERS):
        for j in range(N_WIRES):
            # layer of Rz-Rx-Rz rotations
            qml.RZ(theta[i,j,0], wires=j)
            qml.RX(theta[i,j,1], wires=j)
            qml.RZ(theta[i,j,2], wires=j)

        for j in range(N_WIRES // 2):
            qml.CNOT(wires=[2*j,2*j+1]) # first layer: even wires

        for j in range((N_WIRES - 1) // 2):
            qml.CNOT(wires=[2*j+1,2*j+2]) # second layer: odd wires


        # for j in range(N_WIRES - 1):
        #     # entangling layer
        #     qml.CNOT(wires=[j, j + 1])

    # Variational Quantum Circuit: Hardware Efficient Ansatz (paper Burger)
        # for i in range(N_LAYERS):
        #     for j in range(N_WIRES):
        #         qml.RX(theta[i,j,0], wires=j)
        #         qml.RY(theta[i,j,1], wires=j)
        #         qml.RZ(theta[i,j,2], wires=j)
        
        #     for j in range(N_WIRES - 1):
        #         qml.CNOT(wires=[j, j + 1])
 
    ## Cost Function
    ## Z-Magnetization as cost function
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_WIRES)]))  # output of the circuit is f(x)=<C>


class FNNBasisNet(torch.nn.Module):
    def __init__(self, n_hidden_layers, branch_width):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.branch_width = branch_width
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(1, branch_width))
        for i in range(n_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(branch_width, branch_width))
        self.layers.append(torch.nn.Linear(branch_width, N_WIRES))
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[self.n_hidden_layers](x)
        return x

def model(x):
    # Rescale input to [-0.95, 0.95]       
    x_rescaled = 0.95 * 2*x - 0.95
    
    if map == "fnn_map":
        return circuit_qnode(x_rescaled.T, basisNet(x_rescaled.unsqueeze(1)).T)
    else:
        return circuit_qnode(x_rescaled.T)


def rescale(x):
    # Rescale input to [-0.95, 0.95] (arcsin domain)       
    x_rescaled = 0.95 * 2*x - 0.95  
    # Avoid using .T on non-2D tensors (deprecated). If the tensor is 1D,
    # pass it unchanged; otherwise reverse dimensions explicitly.
    if x_rescaled.ndim == 1:
        inputs = x_rescaled
    else:
        inputs = x_rescaled.permute(*torch.arange(x_rescaled.ndim - 1, -1, -1))

    return circuit_qnode(inputs)  ## Compute the reference solution


def compute_MSE(EDO, save_pred=True):
    prediction = rescale(x)

    if EDO == 1:
        reference_solution = torch.tensor(
            solve_ivp(lambda t, y: derivatives_fnc(1, t, y), [0.0, X_END + 0.000001], [0.75], t_eval=x.detach().cpu()).y,
            device=device,
        )
    elif EDO == 2:
        lamb = 8
        kappa = 0.1
        reference_solution = torch.exp(-lamb*kappa*x)*torch.cos(lamb*x)  # u0 = 1.0

    elif EDO == 3:
        lamb = 20
        kappa = 0.1
        reference_solution = torch.exp(-lamb*kappa*x)*torch.cos(lamb*x)  # u0 = 1.0

    x_np = x.detach().cpu().numpy()
    pred_np = prediction.detach().cpu().numpy()
    ref_np = reference_solution.detach().cpu().squeeze().numpy()

    mse = torch.mean((prediction - reference_solution)**2).detach().cpu().item()

    # Save prediction arrays for later analysis
    if save_pred:
        pred_fname = os.path.join(RESULTS_DIR, f"pred_L{N_LAYERS}_Q{N_WIRES}_{map}_edo{EDO}.npz")
        np.savez(pred_fname, x=x_np, pred=pred_np, ref=ref_np)

    fig, ax = plt.subplots(figsize=(8, 4))
    #ax.plot(x_np, pred_np, linestyle='--', label="Prediction")
    #ax.plot(x_np, ref_np, label="Reference")
    ax.plot(x_np, ref_np, label="Reference", linewidth=3, zorder=1)
    ax.plot(x_np, pred_np, linestyle='--', label="Prediction", linewidth=2, zorder=2)
    ax.legend()
    ax.grid(True)
    ax.set_title(f"QPINN Prediction vs Reference Solution. MSE = {mse:.2E}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    plt_fname = os.path.join(RESULTS_DIR, f"L{N_LAYERS}_Q{N_WIRES}_{map}_mse_edo{EDO}.png")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.pause(3)
    plt.close(fig)
    
    return mse  ## Define the problem

def loss_diff_fnc():
    u = rescale(x) 
    u_at_0 = u[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    if EDO == 1:
        res = du_dx - (4*u - 6*u**2 + torch.sin(50*x) + u*torch.cos(25*x) - 0.5)
        boundary_loss = (u_at_0 - 0.75)**2
    elif EDO == 2:
        lamb = 8
        kappa = 0.1
        res = du_dx + lamb*u*(kappa + torch.tan(lamb*x))
        boundary_loss = (u_at_0 - 1)**2
    elif EDO == 3:
        lamb = 20
        kappa = 0.1
        res = du_dx + lamb*u*(kappa + torch.tan(lamb*x))
        boundary_loss = (u_at_0 - 1)**2
    

    return torch.mean(res**2), boundary_loss

"""def loss_boundary_fnc():
    u_0 = rescale(torch.zeros_like(x))
    if EDO == 1:
        boundary_condition = 0.75
    if EDO == 2 or EDO == 3:
        boundary_condition = 1
    return torch.mean((u_0 - boundary_condition)**2)"""

def loss_fnc():

    loss_diff, loss_boundary = loss_diff_fnc()
    #loss_boundary = loss_boundary_fnc()

    return BOUNDARY_SCALE*loss_boundary + loss_diff

def closure():
    opt.zero_grad()
    l = loss_fnc()
    l.backward()
    return l  



data = np.zeros((5,4,2)) # layer, qubits, (loss, MSE_re)


# for k,N_LAYERS in enumerate([1,3,5,7,10]):
#     for l,N_WIRES in enumerate([2, 4, 6, 8]):
#         print(f"\t Layers: {N_LAYERS} \t Qubits: {N_WIRES}")
        
tmp_loss = []
tmp_mse_ref = []
timing_results = []

# =========================
# Benchmark configuration
# =========================
EDO_LIST = [1,2,3]
MAP_LIST = ["FNN_map"]

N_LAYERS = 5        # fijo por ahora
N_WIRES  = 6        # fijo por ahora
MAX_EPOCHS = 500

LAYERS_LIST = [1,3,6,12,24,36,48]
        

for EDO in EDO_LIST:
    for map in MAP_LIST:

# for N_LAYERS in LAYERS_LIST:
#     for map in MAP_LIST:
        
        print("\n" + "="*60)
        print(f"Training QPINN — EDO {EDO}, MAP {map}, L={N_LAYERS}, Q={N_WIRES}")
        print("="*60)

        circuit_qnode = qml.QNode(circuit, device=qml.device("default.qubit", wires=N_WIRES))
        theta = torch.rand(N_LAYERS, N_WIRES, 3, device=device, requires_grad=True)

        if map == "fnn_map":
            basisNet = FNNBasisNet(HIDDEN_LAYER_FNN, NEURONS_FNN).to(device)
            opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")
        else:
            opt = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")

        # Plot and save the circuit diagram for inspection
        # fig, ax = qml.draw_mpl(circuit_qnode)(torch.tensor(0.0))
        # fig.suptitle(f"Quantum circuit — Layers={N_LAYERS}, Qubits={N_WIRES}")
        # plt_fname = os.path.join(RESULTS_DIR, f"circuit_L{N_LAYERS}_Q{N_WIRES}_{map}.png")
        # fig.savefig(plt_fname, bbox_inches="tight")
        # plt.show()
        # plt.close(fig)

        previous_loss = float('inf')
        loss_history = []
        patience = 0
        for epoch in range(500):
            opt.step(closure)   #  closure that reevaluates the model and returns the loss 
            current_loss = loss_fnc().item()
            loss_history.append(current_loss)
            print(f"Epoch {epoch}, Loss: {current_loss:.2E}", end="\r")

            if abs(previous_loss - current_loss)< 1e-12:
                patience += 1
            
            if patience > 8:
                break 

            previous_loss = current_loss


        final_loss = loss_fnc().item()
        tmp_loss.append(final_loss)
        # Plot and save loss history
        loss_fname = os.path.join(RESULTS_DIR,f"loss_L{N_LAYERS}_Q{N_WIRES}_{map}_edo{EDO}.png")
        loss_data_fname = os.path.join(RESULTS_DIR,f"loss_L{N_LAYERS}_Q{N_WIRES}_{map}_edo{EDO}.npy")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(range(len(loss_history)), loss_history, marker='o')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Loss history — L{N_LAYERS} Q{N_WIRES} {map} EDO{EDO}')
        fig2.savefig(loss_fname, bbox_inches='tight')
        np.save(loss_data_fname, np.array(loss_history))
        plt.close(fig2)

        tmp_mse_ref.append(compute_MSE(EDO=EDO, save_pred=True))  
                
            # data[k,l,0] = np.mean(tmp_loss)
            # data[k,l,1] = np.mean(tmp_mse_ref)

        print(f"Final Loss: {loss_fnc().item():.2E} \t Layers: {N_LAYERS} \t Qubits: {N_WIRES} \t MSE_ref {compute_MSE(EDO):.2E}")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        timing_results.append({"EDO": EDO,"MAP": map,"L": N_LAYERS,"Q": N_WIRES,"time_sec": elapsed_time})