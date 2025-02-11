# ELBO-KL-Bayesian-VAE-GAN-HW2

## 📌 Table of Contents

- [Introduction](#introduction)
- [Course Information](#course-information)
- [Assignment Details](#assignment-details)
- [Sections Overview](#sections-overview)
  - [Flow-Based Generative Models](#flow-based-generative-models)
  - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
- [Implementation Details](#implementation-details)
- [Mathematical Derivations](#mathematical-derivations)
- [Training and Experimentation](#training-and-experimentation)
- [Submission Guidelines](#submission-guidelines)
- [Academic Integrity Policy](#academic-integrity-policy)
- [License](#license)

---

## 📝 Introduction

This repository contains **Homework 2** for the **Deep Generative Models** course at the **University of Tehran**. This assignment covers **advanced generative models**, including:

- **Flow-based models** (e.g., RealNVP, Residual Flows)
- **Generative Adversarial Networks (GANs)**
- **Wasserstein GAN (WGAN)** and improvements over standard GANs
- **Likelihood-based out-of-distribution (OOD) detection**
- **Mathematical derivations and deep learning implementations**

By completing this assignment, students will gain hands-on experience with **modern generative models**, understand their strengths and weaknesses, and implement them in Python.

---

## 🎓 Course Information

- **University**: University of Tehran
- **Department**: Electrical and Computer Engineering
- **Course**: Deep Generative Models
- **Instructor**: Dr. Mostafa Tavasoli
- **Term**: Fall 1403

---

## 🏆 Assignment Details

This assignment consists of two **major sections**:

### 🔹 **1. Flow-Based Generative Models**:

- **Understanding Normalizing Flows** (e.g., RealNVP, Residual Flows)
- **Implementing RealNVP on FashionMNIST**
- **Likelihood-based Out-of-Distribution Detection**
- **Jacobian determinant estimation in residual flows**

### 🔹 **2. Generative Adversarial Networks (GANs)**:

- **Mathematical derivations of GAN training**
- **Implementation of GAN and WGAN**
- **FID (Frechet Inception Distance) for quality evaluation**
- **Analyzing convergence and stability of GAN training**

---

## 📂 Sections Overview

### 🔥 **Flow-Based Generative Models**

Flow-based models provide a **reversible transformation** between a simple distribution (e.g., Gaussian) and complex data distributions.

#### ✅ **Tasks:**

1. **Understanding RealNVP (Normalizing Flow Model)**:

   - Study the **original RealNVP paper**.
   - Understand how **log-likelihood estimation** works in normalizing flows.
2. **Implementing RealNVP on FashionMNIST**:

   - Implement **coupling transformations** for RealNVP.
   - Train the model on **FashionMNIST dataset**.
   - Evaluate the model by generating **out-of-distribution samples**.
3. **Likelihood-Based OOD Detection**:

   - Analyze how flow models fail in detecting **out-of-distribution samples**.
   - Experiment with **FashionMNIST and another dataset**.
4. **Jacobian Determinant Estimation in Residual Flows**:

   - Study how **Hutchinson trace estimator** is used for Jacobian computation.

---

### 🔥 **Generative Adversarial Networks (GANs)**

GANs are **powerful generative models** trained using an adversarial framework.

#### ✅ **Tasks:**

1. **Mathematical Analysis of GANs**:

   - Prove why **vanishing gradients** occur in standard GAN loss.
   - Derive **optimal discriminator function**.
2. **Implementing GAN from Scratch**:

   - Define **generator** and **discriminator** models.
   - Train a **FashionMNIST GAN** and visualize generated samples.
3. **Understanding Wasserstein GAN (WGAN)**:

   - Explain why **WGAN improves over standard GAN**.
   - Study the **role of Lipschitz constraint** and **critic loss**.
4. **Implementing WGAN**:

   - Modify the standard GAN to **WGAN framework**.
   - Train the model and compare **convergence properties**.
5. **Evaluating GANs using FID**:

   - Compute **Frechet Inception Distance (FID)**.
   - Compare FID scores across **different training epochs**.

---

## ⚙️ Implementation Details

### **🔹 Dataset**

- The dataset used is **FashionMNIST**.
- **80/20** train-test split.
- **Preprocessing**:
  - Normalize pixel values to `[0,1]`.

### **🔹 GAN Model Architecture**

| **Layer Type** | **Generator** | **Discriminator** |
| -------------------- | ------------------- | ----------------------- |
| Input                | Noise vector (100D) | 28×28 Image            |
| Convolutional        | 4×4 Transpose Conv | 4×4 Conv               |
| Activation           | ReLU                | LeakyReLU               |
| BatchNorm            | Yes                 | Yes                     |
| Output               | 28×28 Image        | Sigmoid                 |

### **🔹 Training Parameters**

| Parameter     | Value  |
| ------------- | ------ |
| Latent Dim    | 100    |
| Learning Rate | 0.0002 |
| Batch Size    | 64     |
| Epochs        | 100    |

### **🔹 Loss Functions**

- **GAN Loss**:
- $$
  L_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]
  $$

  $$
  L_G = -\mathbb{E}[\log D(G(z))]
  $$
- **WGAN Loss**:
- $$
  L_D = -\mathbb{E}[D(x)] + \mathbb{E}[D(G(z))]
  $$
- $$
  L_G = -\mathbb{E}[D(G(z))]
  $$

---

## 📊 Mathematical Derivations

### **1️⃣ Why Do GANs Suffer from Vanishing Gradients?**

- When \( D(G(z)) \approx 0 \), gradients vanish.

### **2️⃣ Why Does WGAN Work Better?**

- WGAN **removes sigmoid activation** and uses **Wasserstein distance** instead of KL/JS divergence.

### **3️⃣ Computing FID Score**

- FID measures **feature space similarity** between real and generated samples:
  $$
  FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
  $$

---

## 🚀 Training and Experimentation

1. **Train GAN and WGAN** and monitor **loss curves**.
2. **Compare standard GAN and WGAN** for stability.
3. **Compute FID scores** and analyze improvements.

---
