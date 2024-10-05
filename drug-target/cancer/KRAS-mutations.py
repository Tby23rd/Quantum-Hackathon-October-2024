import pennylane as qml
from pennylane import numpy as np

# Always use 12 qubits
num_qubits = 12

# Create a quantum device
dev = qml.device('default.qubit', wires=num_qubits)

# Define a quantum circuit for simulating drug-KRAS interaction
@qml.qnode(dev)
def kras_drug_interaction(params, drug_features, kras_features):
    # Encode drug features
    for i in range(6):
        qml.RY(drug_features[i], wires=i)
    
    # Encode KRAS protein features
    for i in range(6, 12):
        qml.RY(kras_features[i-6], wires=i)
    
    # Apply parameterized gates
    for layer in range(2):
        for i in range(num_qubits):
            qml.RX(params[layer * num_qubits + i], wires=i)
            qml.RY(params[(layer + 2) * num_qubits + i], wires=i)
        
        # Apply entangling gates
        for i in range(num_qubits - 1):
            qml.CZ(wires=[i, i+1])
        qml.CZ(wires=[num_qubits-1, 0])
    
    # Measure the expectation value of Z on all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# Function to calculate binding affinity (lower is better)
def binding_affinity(params, drug_features, kras_features):
    interactions = kras_drug_interaction(params, drug_features, kras_features)
    return qml.math.mean(qml.math.stack(interactions))

# Optimization loop
def optimize_drug(initial_params, kras_features, steps=100):
    opt = qml.AdamOptimizer(stepsize=0.1)
    params = initial_params

    print("Starting drug optimization...")
    for i in range(steps):
        drug_features = np.random.random(6) * np.pi  # Simulate different drug candidates
        params = opt.step(lambda p: binding_affinity(p, drug_features, kras_features), params)
        
        if (i + 1) % 10 == 0:
            affinity = binding_affinity(params, drug_features, kras_features)
            print(f"Step {i+1}: binding affinity = {affinity:.6f}")

    return params, drug_features

# Main execution
if __name__ == "__main__":
    print("KRAS Inhibitor Drug Discovery Simulation")
    
    # Initialize parameters
    num_params = 4 * num_qubits
    initial_params = np.random.random(num_params)
    
    # Simulate KRAS protein features (e.g., for G12V mutation)
    kras_features = np.array([0.5, 1.2, 0.8, 1.5, 0.3, 0.9]) * np.pi
    
    # Run optimization
    optimal_params, best_drug_features = optimize_drug(initial_params, kras_features)
    
    # Evaluate final binding affinity
    final_affinity = binding_affinity(optimal_params, best_drug_features, kras_features)
    print(f"\nFinal binding affinity: {final_affinity:.6f}")
    
    # Interpret the result
    if final_affinity < -0.5:
        print("Potential strong KRAS inhibitor discovered!")
    elif final_affinity < -0.2:
        print("Moderate KRAS inhibition potential")
    else:
        print("Weak KRAS inhibition, further optimization needed")

    print("\nOptimized drug features:")
    print(best_drug_features)

    # Optional: Visualize the optimization process
    try:
        import matplotlib.pyplot as plt

        # Collect data during optimization
        affinities = []
        steps = 100
        for i in range(steps):
            drug_features = np.random.random(6) * np.pi
            affinity = binding_affinity(optimal_params, drug_features, kras_features)
            affinities.append(affinity)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, steps + 1), affinities)
        plt.title("KRAS Inhibitor Optimization Process")
        plt.xlabel("Optimization Step")
        plt.ylabel("Binding Affinity")
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization.")