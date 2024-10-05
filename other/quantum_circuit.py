import pennylane as qml
from pennylane import numpy as np

# Create a quantum device
dev = qml.device('default.qubit', wires=2)

# Define a quantum circuit
@qml.qnode(dev)
def quantum_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Define a cost function (sum of expectation values)
def cost(params):
    return np.sum(quantum_circuit(params))

# Set initial parameters
params = np.array([0.1, 0.2], requires_grad=True)

# Evaluate the circuit
result = quantum_circuit(params)
print("Circuit output:", result)

# Calculate the gradient using jacobian
grad = qml.jacobian(cost)(params)
print("Gradient:", grad)

# Optimize the circuit
opt = qml.GradientDescentOptimizer(stepsize=0.1)

steps = 100
for i in range(steps):
    params = opt.step(cost, params)
    if (i + 1) % 20 == 0:
        print(f"Step {i+1}: params = {params}, output = {quantum_circuit(params)}")