import torch
import pennylane as qml
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import solve_ivp

torch.manual_seed(42)
torch.set_num_threads(30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Constants

# Domain Parameter
X_COLLOC_POINTS = 100
BOUNDARY_SCALE = 10e1
X_END = 1.0 

##  Generate Domain
# Generate Collocation Points
x = torch.linspace(0.0, X_END, X_COLLOC_POINTS, device=device, requires_grad=True)


## Create the Model
# Define QPINN
def circuit(x, basis=None):

    # Quantum Feature Map Encoding: Product Feature Map
    for i in range(N_WIRES):
        qml.RY(torch.arcsin(x), wires=i)

    # Variational Quantum Circuit: Hardware Efficient Ansatz
    for i in range(N_LAYERS):
        for j in range(N_WIRES):
            # layer of Rz-Rx-Rz rotations
            qml.RX(theta[i,j,0], wires=j)
            qml.RY(theta[i,j,1], wires=j)
            qml.RZ(theta[i,j,2], wires=j)
    
        for j in range(N_WIRES - 1):
            # entangling layer
            qml.CNOT(wires=[j, j + 1])
 
    ## Cost Function
    ## Z-Magnetization as cost function
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_WIRES)]))


# Define FNN for the basis
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
    
    if EMBEDDING == "FNN_BASIS":
        return circuit_qnode(x_rescaled.T, basisNet(x_rescaled.unsqueeze(1)).T)
    else:
        return circuit_qnode(x_rescaled.T)  ## Compute the reference solution

def derivatives_fnc(x, u):
    du_dx = 4*u - 6*u**2 + math.sin(50*x) + u*math.cos(25*x) - 0.5
    return du_dx

reference_solution = torch.tensor(solve_ivp(derivatives_fnc, [0.0,X_END+0.000001], [0.75], t_eval=x.detach().cpu()).y, device=device)

def compute_MSE_ref():
    prediction = model(x)
    return torch.mean((prediction-reference_solution)**2).detach().cpu().item()  ## Define the problem

def loss_diff_fnc():
    u = model(x) 
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    res = du_dx - (4*u - 6*u**2 + torch.sin(50*x) + u*torch.cos(25*x) - 0.5)

    return torch.mean(res**2)

def loss_boundary_fnc():
    u_0 = model(torch.zeros_like(x))
    return torch.mean((u_0 - 0.75)**2)

def loss_fnc():

    loss_diff     = loss_diff_fnc()
    loss_boundary = loss_boundary_fnc()

    return BOUNDARY_SCALE*loss_boundary + loss_diff

def closure():
    opt.zero_grad()
    l = loss_fnc()
    l.backward()
    return l  ## Benchmark different configurations

EMBEDDING_LIST = ["FNN_BASIS", "TOWER_CHEBYSHEV", "CHEBYSHEV" ]

data = np.zeros((5,4,2)) # layer, qubits, (loss, MSE_re)

for EMBEDDING in EMBEDDING_LIST:
    
    for k,N_LAYERS in enumerate([1,3,5,7,10]):
        for l,N_WIRES in enumerate([2, 4, 6, 8]):
            print(f"Embedding: {EMBEDDING} \t Layers: {N_LAYERS} \t Qubits: {N_WIRES}")
            
            tmp_loss = []
            tmp_mse_ref = []
            
            for i in range(3):
                
                circuit_qnode = qml.QNode(circuit, device=qml.device("default.qubit", wires=N_WIRES))
                theta = torch.rand(N_LAYERS, N_WIRES, 3, device=device, requires_grad=True)

                if EMBEDDING == "FNN_BASIS":
                    basisNet = FNNBasisNet(HIDDEN_LAYER_FNN, NEURONS_FNN).to(device)
                    opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")
                else:
                    opt = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")

                previous_loss = float('inf')
                for i in range(500):
                    opt.step(closure)
                    print(f"Epoch {i}, Loss: {loss_fnc().item():.2E}", end="\r")

                    if previous_loss == loss_fnc().item():
                        break
                    previous_loss = loss_fnc().item()
                
                tmp_loss.append(loss_fnc().item())
                tmp_mse_ref.append(compute_MSE_ref())
                
            data[k,l,0] = np.mean(tmp_loss)
            data[k,l,1] = np.mean(tmp_mse_ref)

            print(f"Final Loss: {loss_fnc().item():.2E} \t Iteration: {i} \t Embedding: {EMBEDDING} \t Layers: {N_LAYERS} \t Qubits: {N_WIRES} \t Iterations: {i} \t MSE_ref {compute_MSE_ref():.2E}")