import pennylane as qml
from pennylane import numpy as np

# Define the number of qubits (now 12)
num_qubits = 12

# Create a quantum device
dev = qml.device('default.qubit', wires=num_qubits)

# Define the Hamiltonian (extended for 12 qubits)
def extended_hamiltonian():
    # Coefficients (extended for demonstration)
    coeffs = np.array([0.2252] + [0.3435] * 12 + [0.1809] * 11)
    
    # Observables
    obs = [qml.PauliZ(0) @ qml.PauliZ(1)]  # ZZ interaction
    obs += [qml.PauliZ(i) for i in range(num_qubits)]  # Z on each qubit
    obs += [qml.PauliX(i) @ qml.PauliX(i+1) for i in range(num_qubits-1)]  # XX interactions
    
    return qml.Hamiltonian(coeffs, obs)

H = extended_hamiltonian()

# Define a simple ansatz for state preparation
@qml.qnode(dev)
def circuit(params):
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
    for i in range(0, num_qubits-1, 2):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(H)

# Function to find the ground state
def find_ground_state(init_params):
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    params = init_params
    
    for i in range(100):
        params = opt.step(circuit, params)
    
    return params, circuit(params)

# Function to find the first excited state
def find_excited_state(ground_state_params):
    @qml.qnode(dev)
    def excited_circuit(params):
        # Prepare ground state
        for i in range(num_qubits):
            qml.RY(ground_state_params[i], wires=i)
        for i in range(0, num_qubits-1, 2):
            qml.CNOT(wires=[i, i+1])
        
        # Additional rotations for excited state
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for i in range(0, num_qubits-1, 2):
            qml.CNOT(wires=[i, i+1])
        
        # Orthogonalize to ground state
        qml.PauliX(0)
        
        return qml.expval(H)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    params = np.random.random(num_qubits)
    
    for i in range(100):
        params = opt.step(excited_circuit, params)
    
    return params, excited_circuit(params)

# Main execution
if __name__ == "__main__":
    initial_params = np.random.random(num_qubits)
    
    print("Finding ground state...")
    ground_params, ground_energy = find_ground_state(initial_params)
    print(f"Ground state energy: {ground_energy:.6f}")
    
    print("\nFinding first excited state...")
    excited_params, excited_energy = find_excited_state(ground_params)
    print(f"First excited state energy: {excited_energy:.6f}")
    
    print(f"\nExcitation energy: {excited_energy - ground_energy:.6f}")