# Quantifying Uncertainty in Predictions with Bayesian Neural Networks

This project demonstrates how to estimate **prediction uncertainty** using a **Bayesian Neural Network (BNN)**. It explores two main methods—**prediction variance** and **entropy**—to provide better insight into the reliability of neural network predictions.

---

## 📌 Objectives

- Implement and interpret uncertainty measures in BNNs
- Distinguish between **epistemic** and **aleatoric** uncertainty
- Use prediction variance and entropy to improve decision-making in machine learning models

---

## 🧠 Methods

### 1. **Prediction Variance**
- Applies to regression and classification
- Captures **epistemic uncertainty** (due to model parameters)
- Estimated by sampling from the posterior distribution:
  
  \[
  \text{Var}(y) = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2
  \]

- **Interpretation:**
  - Low variance → confident prediction
  - High variance → uncertainty due to lack of knowledge

---

### 2. **Entropy of Predictive Distribution**
- Applies to classification tasks
- Measures **total uncertainty** over predicted class probabilities using Shannon entropy:

  \[
  H(p) = -\sum_{i=1}^C p_i \log p_i
  \]

- **Interpretation:**
  - Low entropy → one class dominates → confident prediction
  - High entropy → prediction is uncertain across multiple classes

---

## 📊 Combined Uncertainty
Using both methods:
- For **regression** → use prediction mean ± variance
- For **classification** → assess both entropy and prediction confidence (e.g., top-1 probability)


---

## 🛠️ Tools Used

- Python 3
- PyTorch / TensorFlow (BNN implementation assumed)
- `matplotlib`, `numpy`, `scipy`, `torch.nn.functional`
- Jupyter Notebooks

---

## 👨‍💻 Author

- [Mihai Dilirici](mailto:mihai.dilirici@s.unibuc.ro)

---

Feel free to explore the code, review the findings, and contribute to improving dialect-aware language models.
