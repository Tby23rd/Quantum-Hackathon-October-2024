import pennylane as qml
from pennylane import numpy as np

# Define the number of qubits
num_qubits = 10  # Simplified model for C2H4

# Create a quantum device
dev = qml.device('default.qubit', wires=num_qubits)

# Define the C2H4 Hamiltonian (simplified model)
def c2h4_hamiltonian():
    coeffs = np.array([
        0.8, 0.8,  # ZZ interactions for C-C double bond
        *[0.4] * num_qubits,  # Z on each qubit
        *[0.3] * (num_qubits - 1),  # XX interactions
        *[0.25] * (num_qubits - 1),  # YY interactions
        0.2, 0.2, 0.2, 0.2,  # ZZZ interactions for C-H bonds
        0.15  # XXXX interaction for carbon atoms
    ])
    
    obs = [
        qml.PauliZ(0) @ qml.PauliZ(1),  # ZZ interaction for C-C double bond
        qml.PauliZ(2) @ qml.PauliZ(3),  # ZZ interaction for C-C double bond
    ]
    obs += [qml.PauliZ(i) for i in range(num_qubits)]  # Z on each qubit
    obs += [qml.PauliX(i) @ qml.PauliX(i+1) for i in range(num_qubits-1)]  # XX interactions
    obs += [qml.PauliY(i) @ qml.PauliY(i+1) for i in range(num_qubits-1)]  # YY interactions
    obs += [
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2),  # ZZZ interaction for C-H bond
        qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5),  # ZZZ interaction for C-H bond
        qml.PauliZ(6) @ qml.PauliZ(7) @ qml.PauliZ(8),  # ZZZ interaction for C-H bond
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(9),  # ZZZ interaction for C-H bond
    ]
    obs += [qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3)]  # XXXX interaction for carbon atoms
    
    return qml.Hamiltonian(coeffs, obs)

H = c2h4_hamiltonian()

# Define a circuit for C2H4 state preparation
@qml.qnode(dev)
def circuit(params):
    # Prepare initial state
    qml.StatePrep(params, wires=range(num_qubits), pad_with=0.0)
    
    # Entangling layers
    for layer in range(3):
        for i in range(0, num_qubits - 1, 2):
            qml.CNOT(wires=[i, i+1])
        for i in range(1, num_qubits - 1, 2):
            qml.CNOT(wires=[i, i+1])
    
    return qml.expval(H)

# Function to find the ground state
def find_ground_state(init_params):
    opt = qml.AdamOptimizer(stepsize=0.1)
    params = init_params
    
    for i in range(200):
        params = opt.step(circuit, params)
        if (i + 1) % 40 == 0:
            print(f"Step {i+1}: Energy = {circuit(params):.6f}")
    
    return params, circuit(params)

# Function to find the first excited state
def find_excited_state(ground_state_params):
    @qml.qnode(dev)
    def excited_circuit(params):
        # Prepare ground state
        qml.StatePrep(ground_state_params, wires=range(num_qubits), pad_with=0.0)
        
        # Additional operations for excited state
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        
        # Entangling layers
        for layer in range(3):
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i+1])
            for i in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i+1])
        
        # Orthogonalize to ground state
        qml.PauliX(0)
        
        return qml.expval(H)
    
    opt = qml.AdamOptimizer(stepsize=0.05)
    params = np.random.random(num_qubits)
    
    for i in range(200):
        params = opt.step(excited_circuit, params)
        if (i + 1) % 40 == 0:
            print(f"Step {i+1}: Energy = {excited_circuit(params):.6f}")
    
    return params, excited_circuit(params)

# Main execution
if __name__ == "__main__":
    # Generate initial state vector (1024 elements for 10 qubits)
    initial_state = np.random.random(1024) + 1j * np.random.random(1024)
    initial_state = initial_state / np.linalg.norm(initial_state)
    
    print("Finding ground state...")
    ground_params, ground_energy = find_ground_state(initial_state)
    print(f"Ground state energy: {ground_energy:.6f}")
    
    print("\nFinding first excited state...")
    excited_params, excited_energy = find_excited_state(ground_params)
    print(f"First excited state energy: {excited_energy:.6f}")
    
    print(f"\nExcitation energy: {excited_energy - ground_energy:.6f}")