# Bayesian Neural Networks – Predictive Uncertainty Modeling

This project demonstrates how to model **predictive uncertainty** using **Bayesian Neural Networks (BNNs)** in Python. It applies two core techniques to quantify model confidence in both regression and classification tasks:

- 📈 **Prediction Variance** (Task 2) — Estimates **epistemic uncertainty** by analyzing output variability across multiple forward passes.
- 🤔 **Entropy of Class Probabilities** (Task 3) — Measures uncertainty in classification predictions using **Shannon entropy**.

---

## 🧠 Why It Matters

Uncertainty quantification is critical in high-stakes AI systems where confident decisions are essential — from healthcare to autonomous systems. This project showcases how to:

- Improve trust in ML models by quantifying confidence
- Use **MC Dropout** for Bayesian inference without altering standard neural architectures
- Interpret model behavior beyond accuracy using **entropy**, **variance**, and **F1 Score**

---

## 🛠️ Tools Used

- Python, PyTorch, NumPy
- Jupyter Notebooks
- `torch.nn.functional`, `scikit-learn`, `matplotlib`

---

## 👨‍💻 Author

- **Mihai Dilirici** — [Email](mailto:mihai.dilirici@s.unibuc.ro)

---

Feel free to explore the code, review the findings, and contribute to improving dialect-aware language models.
