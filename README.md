# World-Water-Quality-Estimate
Water quality prediction with machine learning (Turkcell Future Makers Project)
# World Water Quality Prediction
This project aims to classify whether water is **potable (safe to drink)** or **non-potable** using **machine learning algorithms**. It was developed as part of the [Turkcell Geleceği Yazanlar - Application-Based Machine Learning](https://gelecegiyazanlar.turkcell.com.tr/egitimler/uygulama-tabanli-makine-ogrenimi/uygulama-tabanli-makine-ogrenimi?lesson=4983) training.
## Project Structure
- World_Water_Quality_Estimate.py → main project file  
- water_potability.csv → dataset  
- Requirements.txt → required libraries  
- README.md → project description
## Overview
The dataset includes physical and chemical properties of water such as pH, hardness, solids, chloramines, sulfate, and turbidity. The goal is to predict the **Potability** label, where 1 = Potable and 0 = Not Potable. The project covers exploratory data analysis, preprocessing, modeling, and hyperparameter tuning.
## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Missingno
## Project Steps
### 1. Exploratory Data Analysis (EDA)
- Analyzed missing values and summary statistics  
- Visualized feature distributions and correlations  
- Created a pie chart showing potable vs non-potable ratio
### 2. Data Preprocessing
- Filled missing values with mean  
- Split data: 70% training, 30% testing  
- Applied min-max normalization
### 3. Modeling
- Trained Decision Tree and Random Forest models  
- Evaluated with precision score and confusion matrix  
- Visualized Decision Tree structure
### 4. Hyperparameter Tuning
Used RandomizedSearchCV on Random Forest with these best parameters:  
n_estimators = 100, max_features = 'log2', max_depth = 13
## Results
- Best model: Random Forest  
- Average cross-validation score: ~0.67  
- Generated visuals: potability_pie_chart.html, confusion matrices, feature plots
## How to Run
```bash
git clone https://github.com/<Eren Erpay>/world-water-quality.git
cd world-water-quality
pip install -r Requirements.txt
python World_Water_Quality_Estimate.py
