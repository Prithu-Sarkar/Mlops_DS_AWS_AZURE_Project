# 🎓 Student Performance Predictor — End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![AWS](https://img.shields.io/badge/AWS-Elastic%20Beanstalk-FF9900?logo=amazonaws)
![Azure](https://img.shields.io/badge/Azure-Web%20App-0078D4?logo=microsoftazure)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)

A production-ready, end-to-end machine learning project that predicts student math scores based on demographic and academic factors. The project covers the full MLOps lifecycle — from exploratory data analysis and model training to containerized deployment on both **AWS Elastic Beanstalk** and **Azure Web App** via a **CI/CD pipeline using GitHub Actions**.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Models Evaluated](#-models-evaluated)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Author](#-author)

---

## 🎯 Problem Statement

Predict a student's **math score** based on the following input features:

| Feature | Description |
|---|---|
| `gender` | Student's gender |
| `race/ethnicity` | Ethnic group (A–E) |
| `parental_level_of_education` | Highest education attained by parent |
| `lunch` | Standard or free/reduced lunch |
| `test_preparation_course` | Whether the student completed a prep course |
| `reading_score` | Score in reading (out of 100) |
| `writing_score` | Score in writing (out of 100) |

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **ML Libraries** | Scikit-learn, XGBoost, CatBoost |
| **Data & EDA** | Pandas, NumPy, Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Containerization** | Docker |
| **Cloud (AWS)** | Elastic Beanstalk, ECR, AWS CLI |
| **Cloud (Azure)** | Azure Web App, Azure Container Registry |
| **CI/CD** | GitHub Actions |
| **Package Management** | pip, setup.py |

---

## 🏗 Project Architecture


User Input (Web Form)
│
▼
Flask Web App (app.py)
│
▼
Prediction Pipeline
┌─────────────────────┐
│  Data Transformation│  ← preprocessor.pkl
│  Model Inference    │  ← model.pkl
└─────────────────────┘
│
▼
Predicted Math Score
│
┌─────────────┐
│  Deployment │
│  AWS EB     │  ← via Docker + GitHub Actions
│  Azure App  │
└─────────────┘

---

## 📁 Project Structure
Mlops_DS_AWS_AZURE_Project/
│
├── .ebextensions/          # AWS Elastic Beanstalk configuration
├── .github/workflows/      # GitHub Actions CI/CD pipelines
│
├── artifacts/              # Saved model and preprocessor artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── catboost_info/          # CatBoost training logs
├── logs/                   # Application logs
├── notebook/               # EDA and model training notebooks
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── app.py                  # Flask application (local)
├── application.py          # Flask application (AWS EB entry point)
├── Dockerfile              # Docker container definition
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md

---

## ⚙️ ML Pipeline

The project implements a modular, object-oriented ML pipeline:

1. **Data Ingestion** — Reads raw data, splits into train/test sets, and saves to `artifacts/`.
2. **Data Transformation** — Applies `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical features. Saves `preprocessor.pkl`.
3. **Model Training** — Trains and evaluates multiple regression models using `GridSearchCV`. Saves the best model as `model.pkl`.
4. **Prediction Pipeline** — Loads saved artifacts and generates predictions from new input data via the Flask web interface.

---

## 🤖 Models Evaluated

The following regression models were trained and compared using **R² score**:

- Linear Regression
- Ridge & Lasso Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- **XGBoost**
- **CatBoost** ✅ *(Best Performer)*
- AdaBoost

The best model is automatically selected and serialized to `artifacts/model.pkl`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker (for containerized deployment)
- AWS CLI / Azure CLI (for cloud deployment)

### 1. Clone the repository

```bash
git clone https://github.com/Prithu-Sarkar/Mlops_DS_AWS_AZURE_Project.git
cd Mlops_DS_AWS_AZURE_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the training pipeline

```bash
python src/pipeline/train_pipeline.py
```

### 4. Launch the Flask app locally

```bash
python app.py
```

Then open your browser at `http://localhost:5000`.

### 5. Run with Docker

```bash
docker build -t student-performance-app .
docker run -p 5000:5000 student-performance-app
```

---

## ☁️ Deployment

### AWS Elastic Beanstalk

The app is configured for AWS EB deployment via `.ebextensions/`. The `application.py` file serves as the EB entry point.

1. Build and push your Docker image to **Amazon ECR**.
2. Configure your EB environment to pull from ECR.
3. The GitHub Actions workflow automates this process on every push to `main`.

### Azure Web App

1. Build and push your Docker image to **Azure Container Registry (ACR)**.
2. Configure the Azure Web App to pull from ACR.
3. The GitHub Actions workflow handles this automatically.

---

## 🔄 CI/CD Pipeline

Two separate GitHub Actions workflows are defined under `.github/workflows/`:

**AWS Pipeline:**
- Triggers on push to `main`
- Builds Docker image → pushes to ECR → deploys to Elastic Beanstalk

**Azure Pipeline:**
- Triggers on push to `main`
- Builds Docker image → pushes to ACR → deploys to Azure Web App

---

## 👤 Author

**Prithu Sarkar**
- GitHub: [@Prithu-Sarkar](https://github.com/Prithu-Sarkar)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
