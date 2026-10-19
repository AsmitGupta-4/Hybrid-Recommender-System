#  Hybrid Movie & Product Recommender System

A Python-based hybrid recommendation engine built with Pandas, Cosine Similarity, and Streamlit. This application combines content-based filtering and collaborative filtering techniques to deliver personalized recommendations.

---

##  Features

* **Hybrid Recommendation Engine:** Leverages both item features and user rating patterns.
* **Interactive UI:** Simple, interactive web dashboard powered by **Streamlit**.
* **Large Dataset Handling:** Integrated with **Git LFS** to manage large dataset files (`ratings.csv`).

---

##  Tech Stack & Tools

* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-learn
* **Web Framework:** Streamlit
* **Version Control:** Git & Git LFS, GitHub

---

##  Repository Structure

```text
├── app.py              # Main Streamlit application
├── movies.csv          # Movie metadata dataset
├── ratings.csv         # Ratings dataset (tracked via Git LFS)
├── .gitattributes      # Git LFS configuration file
└── README.md           # Project documentation