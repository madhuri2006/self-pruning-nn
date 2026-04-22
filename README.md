# 🧠 Self-Pruning Neural Network

A neural network that performs dynamic pruning during training using learnable gating parameters and L1 regularization.

---

## ⚙️ How It Works

Each weight in the network is associated with a learnable gate parameter.

W_effective = W × sigmoid(G)

- W = original weight  
- G = learnable gate score  
- sigmoid(G) maps values between 0 and 1  

If a gate value approaches 0, the corresponding weight is effectively pruned during the forward pass.

---

## 📉 Loss Function Trade-off

The two loss components create a trade-off during training:

- Classification loss encourages gates to remain active (values closer to 1) to preserve model accuracy  
- Sparsity loss encourages gates to move toward zero, effectively removing connections  

The network learns an optimal balance between performance and sparsity.

---

## 🧩 Why L1 Regularization Creates True Sparsity

A key reason this approach works is the behavior of L1 regularization:

| Regularization | Effect on Parameters | Outcome |
|----------------|---------------------|---------|
| L2 (sum of squares) | Gradually shrinks values | Parameters become small but rarely exactly zero |
| L1 (sum of absolute values) | Applies constant pressure toward zero | Many parameters become exactly zero |

The gradient of L1 regularization is constant (±1), meaning even small values experience the same push toward zero.  
As gate scores become sufficiently negative, the sigmoid function maps them close to zero, effectively pruning those weights.

This leads to a **bimodal distribution**:
- Values near 0 → pruned weights  
- Values away from 0 → important connections  

Very few values remain in between.

---

## 📉 Gate Value Distribution

The final distribution of gate values typically shows:

- A strong spike near 0 (pruned weights)  
- A separate cluster away from 0 (active weights)  

![Gate Distribution](gate_distributions.png)

*This reflects effective sparsity, where the model clearly separates useful and unnecessary connections.*

---

## 📈 Results

| λ (Sparsity Strength) | Test Accuracy | Sparsity (%) |
|----------------------|--------------|--------------|
| 0.0 (baseline)       | 47.33%       | 0%           |
| 0.0001               | 51.96%       | 64.2%        |
| 0.001                | 46.93%       | 99.3%        |
| 0.01                 | 42.18%       | 100%         |

---

## 📊 Observations

- **λ = 0.0001**  
  ~64% of weights pruned with improved accuracy  
  → Indicates regularization improves generalization  

- **λ = 0.001**  
  ~99% sparsity with minimal accuracy drop  
  → Suggests high redundancy in the network  

- **λ = 0.01**  
  Excessive pruning leads to performance degradation  

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python self_pruning_network.py
```

---

## 📌 Key Takeaways

- Dynamic pruning can be integrated directly into training  
- L1 regularization effectively enforces sparsity  
- Significant model compression is achievable with minimal loss in accuracy  
- There exists a clear trade-off between sparsity and performance  

---

## 📁 Files

- self_pruning_network.py → Implementation  
- requirements.txt → Dependencies  
- README.md → Documentation  
