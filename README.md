# Repair of Deep Neural Networks

Project regarding formal verification and repair of deep neural networks at IIT Delhi under Professor Priyanka Golia and Professor Kumar Madhukar 

**Author:** Yash Bansal   
**Supervisors:** Prof. Priyanka Golia, Prof. Kumar Madhukar

### Overview

This project addresses the problem of **Deep Neural Network (DNN) repair** for safety-critical applications. When a trained DNN violates required safety properties on certain inputs, this work develops a gradient-based repair approach that minimally modifies network weights to satisfy the violated properties while preserving performance on other inputs.

### Approach

- Uses **Marabou** for neural network verification and counterexample generation
- Employs **Gurobi** for constrained optimization to find minimal network modifications
- Implements iterative repair by adding synthetic counterexamples to training data
- Exports repaired models to ONNX format for validation

### Key Components

- **Manners_DB/** - Neural network training with constraint-based repair loop
- **repair/** - Verification, counterexample generation, and repair implementations
- **Report.pdf** - Detailed methodology and results
- **Presentation.pdf** - Project overview slides