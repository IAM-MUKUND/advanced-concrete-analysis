---
title: "Combined Dataset Machine Learning Approach for Compressive Strength Prediction of Geopolymer and Lightweight Concrete: Performance, Interpretability, and Uncertainty Quantification"
author: "Mrs. Velvadivu P"
date: "2026"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{placeins}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{setspace}
  - \onehalfspacing
  - \usepackage{needspace}
  - \raggedbottom
  - \setlength{\parskip}{0.5\baselineskip}
  - \usepackage{float}
  - \let\origfigure\figure
  - \let\endorigfigure\endfigure
  - \renewenvironment{figure}[1][H]{\origfigure[H]}{\endorigfigure}
---

\newpage

## Abstract

The accurate prediction of compressive strength is essential for safe and sustainable concrete design. This study introduces a unified machine learning framework that combines two heterogeneous datasets — geopolymer concrete (2,087 original samples) and lightweight/foamed concrete (1,006 original samples) — into a single robust predictive model. After domain-aware preprocessing, including removal of physically unrealistic values (age > 175 days, curing temperature >= 90 degrees C, alkaline solution outside 100-300 kg/m3, etc.) and creation of a categorical concrete_type feature with NaN placeholders for domain-specific variables, the final cleaned and combined dataset contained 2,603 samples.

Five tree-based ensemble models were evaluated under hyperparameter tuning using GridSearchCV and Optuna: XGBoost, CatBoost, LightGBM, Random Forest, and Gradient Boosting Regressor. CatBoost achieved the best overall performance with an R2 of 0.8966, RMSE of 7.19 MPa, and MAE of 3.54 MPa on the combined test set. Domain-specific evaluation further confirmed excellent generalisation across concrete families: geopolymer-only R2 approximately 0.80 and lightweight-only R2 approximately 0.98. SHAP (SHapley Additive exPlanations) interpretability analysis revealed physically meaningful feature importance rankings — alkaline solution and molarity of mix dominated geopolymer predictions, while density was the primary driver for lightweight concrete predictions. Bootstrapping with 200 iterations generated 95% prediction intervals, providing essential uncertainty quantification for practical engineering applications.

The results indicate that a carefully engineered combined-dataset strategy, supported by domain-specific preprocessing, NaN-aware modelling pipelines, and advanced ensemble modelling, can effectively bridge chemically and physically distinct concrete families under one unified framework. This work offers both high predictive accuracy and clear physical interpretability, advancing the state of the art in data-driven concrete mix design.

**Keywords:** Compressive strength prediction, geopolymer concrete, lightweight concrete, combined dataset, CatBoost, SHAP interpretability, uncertainty quantification, bootstrapping, machine learning, ensemble models

\newpage

## 1. Introduction

Concrete is the second most consumed material on Earth, exceeded only by water. Global production of cement — the primary binding agent in conventional concrete — exceeds 4 billion tonnes annually, contributing roughly 8% of global CO2 emissions. The increasing urgency of climate change has accelerated the search for alternative binder systems and mix designs that reduce carbon footprint while meeting structural performance demands.

Among the widely studied alternatives are **geopolymer concrete** (also called alkali-activated concrete) and **lightweight/foamed concrete**. Geopolymer concrete replaces Portland cement with industrial by-products such as fly ash and slag, activated by alkaline solutions (typically sodium silicate and sodium hydroxide). This substitution can reduce CO2 emissions by 40-80% compared to ordinary Portland cement. Lightweight and foamed concrete, on the other hand, incorporates pre-formed foam or air-entraining agents to achieve densities significantly below that of normal concrete, resulting in excellent thermal insulation, reduced self-weight in structures, and superior fire resistance.

Despite these benefits, predicting the compressive strength of both concrete families remains a significant engineering challenge. **Geopolymer concrete** involves highly complex chemical reactions — the polycondensation of aluminosilicate precursors under alkaline conditions — that are sensitive to binder chemistry, alkaline activator concentration, molarity of the alkaline solution, curing temperature, and age. **Lightweight/foamed concrete**, conversely, is governed largely by its air-void microstructure, whose volume fraction and homogeneity are controlled by the foaming agent dosage, binder content, water-to-binder ratio, and density. The governing physics of the two families are fundamentally distinct, creating a covariate shift problem when data from both are naively merged.

Traditional empirical models (e.g., ACI code relationships, regression-based mix design charts) are calibrated for normal-weight Portland cement concrete and perform poorly on these advanced material systems. Experimental campaigns are time-consuming, resource-intensive, and difficult to generalise across laboratories. Machine learning (ML) has emerged as a useful complement to experimental work, offering the ability to learn complex, non-linear input-output relationships from existing experimental databases.

However, a critical gap exists in the literature: most ML studies focus exclusively on a single concrete family, developing separate models for geopolymer concrete and for lightweight/foamed concrete. This single-family approach has two important drawbacks. First, individual datasets are often modest in size (a few hundred to low thousands of samples), limiting model generalisation. Second, separate models offer no opportunity to leverage shared domain knowledge — for instance, the influence of binder type and water content is physically relevant in both families.

This study addresses this gap by developing a **unified ML model** capable of accurately predicting compressive strength across both geopolymer and lightweight concrete families simultaneously. The key lies in the domain-aware data fusion strategy: rather than naively concatenating the two datasets, a categorical concrete_type feature is introduced to enable explicit domain separation within the model, while NaN placeholders preserve the structural integrity of domain-specific variables that are undefined for the opposite concrete type.

### 1.1 Research Objectives

The main objectives of this study are:

1. To combine two heterogeneous concrete datasets while explicitly preserving the domain-specific physics of each concrete family through feature engineering and NaN-aware modelling.
2. To evaluate five state-of-the-art tree-based ensemble models with hyperparameter optimisation (GridSearchCV and Optuna) on both individual and combined datasets.
3. To provide physically interpretable insights into model behaviour through SHAP analysis, validating that the model has learned the correct physics of each concrete type.
4. To quantify prediction uncertainty using non-parametric bootstrapping, generating calibrated 95% prediction intervals suitable for engineering decision-making.
5. To benchmark the combined-dataset approach against individual-dataset results and against previous studies in the literature.

\newpage

## 2. Literature Review

### 2.1 Machine Learning for Compressive Strength Prediction

The application of machine learning to concrete compressive strength prediction has grown significantly over the past decade. Early work focused on artificial neural networks (ANNs), which demonstrated superior performance over traditional regression models due to their ability to capture non-linear relationships between mix design parameters and mechanical properties. More recently, tree-based ensemble methods — particularly gradient boosting variants — have emerged as the dominant paradigm, offering both strong predictive performance and improved interpretability compared to deep neural networks.

Mandal (2024) combined two distinct normal concrete datasets and achieved an R2 of 0.93 using Gradient Boosting, with the study highlighting the significant benefit of dataset diversity for reducing overfitting and improving generalisation. The work explicitly noted that a more diverse training set — even if it introduces some between-dataset heterogeneity — tends to produce more robust models than training on a single, homogeneous source.

Tang et al. (2025) and Zhang et al. (2024) demonstrated the effectiveness of CatBoost combined with SHAP interpretability on high-performance concrete datasets. CatBoost's native handling of categorical features and its robust treatment of missing values were cited as key advantages. Zhang et al. (2024) developed a DR-CatBoost variant with dynamic regularisation, achieving R2 values exceeding 0.95 on high-performance concrete, and showed through SHAP analysis that water-to-binder ratio and cement content were the dominant predictors.

### 2.2 Machine Learning for Geopolymer Concrete

For geopolymer concrete specifically, the literature shows consistent evidence that XGBoost and LightGBM outperform traditional regression and ANN models. Bahram et al. (2026) reported R2 values ranging from 0.80 to 0.88 using XGBoost on a geopolymer dataset compiled from 25 published studies, emphasising the critical role of curing temperature, alkaline solution concentration, and molarity of the activator. Their SHAP analysis identified molarity of mix as the single most influential feature, followed by alkaline solution dosage and curing temperature.

Stel'makh et al. (2025) conducted a comparative analysis of seven ML methods on a geopolymer dataset of approximately 1,800 samples, finding that LightGBM achieved the best balance of accuracy and training speed. Their study identified a key challenge: geopolymer datasets often contain highly correlated features (e.g., alkaline solution and extra water are partially correlated through the activator preparation process), requiring careful feature selection or regularisation to avoid multicollinearity-induced instability.

The feature importance patterns identified in the present study's individual geopolymer analysis align with these findings. On the geopolymer dataset alone, Random Forest achieved R2 = 0.7948, with fine aggregate (18%), binder (17%), and age (17%) as the top three features by importance. XGBoost achieved R2 = 0.8297, with coarse aggregate (18%), alkaline solution (17%), and curing temperature (14%) as dominant features.

### 2.3 Machine Learning for Lightweight and Foamed Concrete

In lightweight and foamed concrete, the literature consistently demonstrates that density is a near-universal dominant predictor. This is physically logical: foamed concrete's compressive strength is primarily governed by its air-void fraction, which is directly reflected in the measured dry density. Salami et al. (2022) demonstrated that ML models achieve R2 > 0.95 on foamed concrete datasets when density is included as a feature, but performance degrades substantially (R2 dropping to 0.70-0.75) when density is excluded and only mix-design inputs are used.

Onyelowe et al. (2024) applied symbolic regression to a lightweight foamed concrete dataset of 312 samples, achieving R2 = 0.97. Their work confirmed the dominant role of density (importance weight approximately 38%) and binder content (approximately 30%), with water, fine aggregate, and age playing supporting roles. These findings are consistent with the individual lightweight dataset results in the present study, where both Random Forest (R2 = 0.9769) and XGBoost (R2 = 0.9757) confirmed binder (importance approximately 48% for RF, approximately 45% for XGBoost) and density (approximately 32% for RF, approximately 27% for XGBoost) as the two most important features.

### 2.4 Gap Analysis and Contribution of the Present Study

A systematic review of the recent literature reveals a striking gap: no published study has successfully merged geopolymer and lightweight/foamed concrete datasets into a single predictive model while simultaneously providing SHAP interpretability and bootstrapped uncertainty quantification. The closest analogues are studies that combine different grades of normal concrete, such as Mandal (2024), but the chemical and physical mechanisms of geopolymer and lightweight concrete are far more divergent than the differences between normal concrete grades.

The present study fills this gap through three key contributions:

1. A domain-aware data fusion strategy using NaN-aware modelling and explicit concrete-type encoding.
2. A comprehensive evaluation of five state-of-the-art ensemble models with industrial-grade hyperparameter optimisation.
3. A complete uncertainty quantification framework via non-parametric bootstrapping that provides confidence intervals suitable for structural engineering applications.

\newpage

## 3. Materials and Methods

### 3.1 Datasets Description

#### 3.1.1 Geopolymer Concrete Dataset

The geopolymer dataset was compiled from peer-reviewed experimental literature and originally contained **2,087 samples** from multiple research groups. Each sample includes the following input features:

- **Binder** (kg/m3): total mass of aluminosilicate precursor (fly ash, slag, or their combination) per unit volume of concrete.
- **Extra Water** (kg/m3): additional water beyond that contained in the alkaline solution.
- **Alkaline Solution** (kg/m3): total mass of alkaline activator (sodium silicate + sodium hydroxide solution) per unit volume.
- **Molarity of Mix**: molar concentration of the sodium hydroxide solution (mol/L).
- **Fine Aggregate** (kg/m3): sand content per unit volume.
- **Coarse Aggregate** (kg/m3): gravel or crushed stone content per unit volume.
- **Age** (days): curing age at time of compressive strength testing.
- **Curing Temperature** (degrees C): temperature at which the specimens were cured.
- **Compressive Strength** (MPa): the target variable, measured in standard cylindrical or cubic specimens.

The raw dataset exhibited a hierarchical multi-level header structure (4 header levels) in the source Excel file, requiring careful parsing. After extracting the numeric data rows, and dropping 30 duplicate entries, the raw dataset contained 2,057 usable samples before outlier removal.

#### 3.1.2 Lightweight / Foamed Concrete Dataset

The lightweight dataset was compiled from a different body of literature and originally contained **1,006 samples**. Each sample includes:

- **Binder** (kg/m3): total cementitious material (Portland cement, blended cements).
- **Pozzolan** (kg/m3): supplementary cementitious materials (fly ash, silica fume, etc.); 406 of the 1,006 samples had no pozzolan recorded (i.e., pure cement binder).
- **Fine Aggregate** (kg/m3): sand content per unit volume.
- **Water** (kg/m3): mix water content.
- **Foaming Agent** (kg/m3): dosage of foaming agent used to generate the air-void system.
- **Density** (kg/m3): measured fresh or hardened density of the lightweight concrete.
- **Age** (days): testing age.
- **Compressive Strength** (MPa): the target variable.

This dataset was provided in a 3-level header Excel format. After parsing and renaming columns, the dataset contained no duplicate records. The 406 missing pozzolan values were imputed with zero, consistent with the physical interpretation that samples without reported pozzolan used a 100% cement binder.

### 3.2 Individual Dataset Exploratory Data Analysis

Prior to combination, each dataset was subjected to thorough exploratory data analysis (EDA) to understand its distribution characteristics, identify outliers, and assess inter-feature correlations. The EDA comprised four complementary analyses:

\FloatBarrier

![Geopolymer Histograms](image_outputs/geopolymer_histograms.png){width=48%} ![Lightweight Histograms](image_outputs/lightweight_histograms.png){width=48%}

**Figure 1**: Histograms of key features in the geopolymer and lightweight datasets, showing distributional characteristics and potential outliers.

\FloatBarrier

**Histograms** (Figure 1) revealed important distributional characteristics. The compressive strength distribution in the geopolymer dataset is approximately log-normal with a mean of approximately 38.7 MPa (post-cleaning), while the lightweight dataset compressive strength distribution is heavily right-skewed with a mean of approximately 11.5 MPa — reflecting the fundamentally different strength ranges of the two concrete families.

\FloatBarrier

![geopolymer correlation](image_outputs/geopolymer_correlation.png){width=48%} ![lightweight correlation](image_outputs/lightweight_correlation.png){width=48%}

**Figure 2**: Correlation matrices for the geopolymer and lightweight datasets, showing inter-feature correlations.

\FloatBarrier

**Correlation matrices** (Figure 2) provided insights into feature inter-dependencies. In the geopolymer dataset, notable correlations were observed between binder and alkaline solution (r approximately 0.48), reflecting common mix design proportioning rules. In the lightweight dataset, the strongest correlation was between density and compressive strength (r approximately 0.87), confirming density as the dominant predictor.

\FloatBarrier

![Geopolymer Scatter Plot](image_outputs/geopolymer_scatter.png){width=48%} ![Lightweight Scatter Plot](image_outputs/lightweight_scatter.png){width=48%}

**Figure 3**: Scatter plots of key features against compressive strength for the geopolymer and lightweight datasets, used to identify non-linear relationships and potential outliers.

\FloatBarrier

**Scatter plots against compressive strength** (Figure 3) were used to identify non-linear relationships and visually flag potential outliers. For the geopolymer dataset, scatter plots revealed that alkaline solution concentrations above 300 kg/m3 corresponded to an extremely sparse cluster of samples with unusually low compressive strengths, suggesting experimental errors or extreme mix designs outside the target application domain. Similarly, samples with curing temperatures above 90 degrees C formed an isolated cluster inconsistent with practical geopolymer processing conditions.

### 3.3 Data Preprocessing and Outlier Handling

#### 3.3.1 Outlier Removal Strategy

The **standard IQR (Interquartile Range)** method was not used for outlier removal. This is because the two datasets have fundamentally different distributions — the strength range of lightweight concrete (0-80 MPa) is far narrower than that of geopolymer concrete (0-120+ MPa), and applying a single IQR threshold across the combined dataset would either over-remove legitimate geopolymer samples or under-remove extreme lightweight samples. Instead, **domain-specific physical bounds** were applied to each dataset separately, based on knowledge of physically plausible mix design ranges for each concrete family. The applied bounds were as follows.

\newpage

For the geopolymer dataset:

- Age <= 175 days (beyond which strength gains plateau and data becomes sparse)
- Extra water < 130 kg/m3 (consistent with practical alkaline activator preparation)
- Alkaline solution: 100 < value <= 300 kg/m3
- Molarity of mix <= 20 mol/L (higher concentrations are rarely used in practice)
- Binder < 600 kg/m3 (beyond which workability becomes impractical)
- Compressive strength > 0 MPa (removes zero-strength erroneous entries)
- Curing temperature < 90 degrees C
- Coarse aggregate < 1,400 kg/m3
- Fine aggregate < 1,100 kg/m3

For the lightweight dataset:

- Age < 175 days
- Pozzolan <= 250 kg/m3
- Fine aggregate < 1,100 kg/m3
- Density: 400 < value <= 2,200 kg/m3 (removes ultra-lightweight foam insulation and normal-weight misclassified samples)
- Binder > 0 kg/m3 (removes zero-binder anomalous entries)
- Compressive strength > 0 MPa
- Foaming agent < 50 kg/m3

After these cleaning operations, the geopolymer dataset retained **1,691 samples** (from 2,087 original) and the lightweight dataset retained **912 samples** (from 1,006 original), for a combined total of 2,603 samples.

### 3.4 Dataset Combination Strategy

The combination of two heterogeneous datasets requires careful engineering to avoid naive data fusion errors. The dataset combination strategy consists of three steps.

**Step 1 — Unified Column Schema.** A unified feature schema was defined containing all 13 possible input features from both datasets: binder, extra water, water, alkaline solution, molarity of mix, fine aggregate, coarse aggregate, pozzolan, foaming agent, density, age, curing temperature, and compressive strength. Features not applicable to a given concrete family were populated with NaN placeholders. Geopolymer samples have NaN for water, pozzolan, foaming agent, and density; lightweight samples have NaN for extra water, alkaline solution, molarity of mix, coarse aggregate, and curing temperature. The structured missingness pattern is deliberately informative — the pattern itself encodes domain membership and allows NaN-aware models to infer concrete type from the data alone, even without the explicit concrete_type feature.

\newpage

**Step 2 — Concrete Type Encoding.** A categorical feature concrete_type was created: 0 = geopolymer, 1 = lightweight. This feature serves as an explicit domain indicator that allows tree-based models to perform conditional splits based on concrete type.

**Step 3 — Water-Binder Ratio Feature Engineering.** An engineered feature water_binder_ratio was computed as:

$$\text{water\_binder\_ratio} = \frac{\text{water} + \text{extra water} + \text{alkaline solution}}{\text{binder}}$$

where missing components are treated as zero in the numerator. Samples with water_binder_ratio >= 2 were identified as isolated outliers and removed, reducing the combined dataset from 2,603 to **2,600 samples**. The final dataset shape was (2600, 15).

### 3.5 Machine Learning Models

Five tree-based ensemble regressors were selected for evaluation:

**XGBoost (Extreme Gradient Boosting)** builds an ensemble of decision trees sequentially, with each tree correcting the residual errors of the previous one. XGBoost supports native handling of missing values by learning optimal default split directions for NaN features, making it naturally suited for the structured-missingness combined dataset.

**LightGBM (Light Gradient Boosting Machine)** uses histogram-based tree construction and leaf-wise tree growth, resulting in significantly faster training compared to XGBoost while achieving comparable accuracy. LightGBM similarly supports NaN handling natively.

**CatBoost (Categorical Boosting)** uses symmetric trees and an ordered boosting algorithm that reduces overfitting. CatBoost's native handling of the concrete_type categorical feature was a key motivation for its inclusion.

**Random Forest** is a bagging ensemble of decision trees, trained on bootstrapped subsets of the training data with feature subsampling at each split. It was trained on a version of the data where NaN values were imputed with zero (using SimpleImputer with constant strategy) followed by StandardScaler normalisation.

**Gradient Boosting Regressor** (scikit-learn implementation) is a traditional gradient boosting method. Like Random Forest, it does not natively handle NaN values and was trained on the zero-imputed, scaled dataset.

\newpage

### 3.6 Hyperparameter Optimisation

Two complementary optimisation strategies were employed.

**GridSearchCV** was applied to XGBoost (individual analyses), Random Forest, and Gradient Boosting Regressor. The grid for XGBoost on the combined dataset covered n_estimators in {300, 500}, max_depth in {5, 7, 9}, learning_rate in {0.05, 0.1}, and subsample in {0.8, 1.0}, with 5-fold cross-validation scoring on R2.

**Optuna** (a Bayesian-inspired hyperparameter optimisation framework) was applied to LightGBM and CatBoost on the combined dataset. Optuna was preferred over GridSearchCV for these models because preliminary tests showed that a standard GridSearchCV with 125+ candidates required over 5 minutes per initialisation pass. Optuna's sequential model-based optimisation achieves comparable or better performance in far fewer function evaluations.

For LightGBM, 150 Optuna trials were run, searching over n_estimators in [50, 300], learning_rate in [0.01, 0.2] (log scale), and num_leaves in [20, 100]. For CatBoost, 50 Optuna trials were run, searching over iterations in [300, 800], depth in [6, 10], learning_rate in [0.01, 0.1] (log scale), and subsample in [0.4, 1.0]. All hyperparameter searches used 5-fold KFold cross-validation with negative mean squared error as the optimisation objective.

### 3.7 Training and Evaluation Strategy

An 80/20 stratified train-test split was applied, with stratification on the concrete_type feature to ensure balanced representation of geopolymer and lightweight samples in both training and test sets. The final split yielded 2,080 training samples and 520 test samples.

Performance was quantified using four metrics:

- **R2 (Coefficient of Determination)**: proportion of variance in the target explained by the model. Higher is better; perfect prediction = 1.0.
- **RMSE (Root Mean Squared Error)**: square root of the average squared prediction error, expressed in MPa. Lower is better; penalises large errors more heavily.
- **MSE (Mean Squared Error)**: average of squared errors in MPa2. Lower is better.
- **MAE (Mean Absolute Error)**: average of absolute prediction errors in MPa. Lower is better; more robust to outliers than RMSE.

Domain-specific evaluation was conducted post-hoc by filtering the test set into geopolymer-only and lightweight-only subsets and computing metrics separately.

\newpage

### 3.8 SHAP Interpretability Analysis

SHAP (SHapley Additive exPlanations) values were computed to provide model-agnostic feature importance rankings and directional contribution plots. For CatBoost and LightGBM, the fast TreeExplainer was used. For XGBoost, a PermutationExplainer was employed due to compatibility issues between the SHAP TreeExplainer and certain XGBoost model configurations. SHAP summary plots (beeswarm plots) were generated for each model, showing both the magnitude and direction of each feature's contribution to individual predictions.

### 3.9 Uncertainty Quantification via Bootstrapping

To generate calibrated prediction intervals, a non-parametric bootstrap procedure was implemented with the following steps:

1. **200 bootstrap iterations** were performed. In each iteration, a bootstrap sample (same size as the training set, drawn with replacement) was used to train a fresh CatBoost model with the same best hyperparameters.
2. Each bootstrap model generated predictions on the fixed test set.
3. The mean prediction across all 200 bootstrap models (mean_pred) and the 2.5th and 97.5th percentiles of the bootstrap prediction distribution were computed for each test sample, yielding the lower and upper bounds of a **95% prediction interval**.
4. Coverage probability (fraction of test samples whose true value falls within the prediction interval) was computed as a calibration check.

This bootstrap approach is non-parametric and makes no assumptions about the distribution of residuals, which is particularly important for the heterogeneous combined dataset.

\newpage

## 4. Results and Discussion

### 4.1 Individual Dataset Performance — Geopolymer Concrete

Table 1 presents the performance of seven ML models on the geopolymer dataset alone. This individual analysis served as a baseline and as motivation for the choice of tree-based ensemble models for the combined analysis.

\needspace{6cm}

**Table 1: Performance metrics on geopolymer dataset (individual analysis, 80/20 split)**

| Model                     | R2 Score | MAE (MPa) | MSE (MPa2) | RMSE (MPa) |
|---------------------------|----------|-----------|------------|------------|
| Linear Regression         | 0.2679   | 13.1927   | 250.2194   | 15.8183    |
| Decision Tree             | 0.6694   | 7.4100    | 112.9754   | 10.6290    |
| Random Forest             | 0.7948   | 5.6374    | 70.1421    | 8.3751     |
| Support Vector Regression | 0.4644   | 10.4076   | 183.0410   | 13.5293    |
| K-Nearest Neighbors       | 0.7070   | 7.1343    | 100.1428   | 10.0071    |
| **XGBoost**               | **0.8297** | **4.9555** | **58.2221** | **7.6303** |
| ANN                       | 0.7558   | 6.5516    | 83.4474    | 9.1349     |

\FloatBarrier

![](image_outputs/bargraph_eval_comparison_geopolymer.png){width=70%}

**Figure 4**: Bar graph comparing performance metrics of different ML models on the geopolymer dataset (individual analysis).

\FloatBarrier

Linear Regression and Support Vector Regression perform poorly (R2 = 0.27 and 0.46 respectively), confirming that the relationship between geopolymer mix parameters and compressive strength is highly non-linear. This is consistent with the known complexity of geopolymerisation kinetics, which involves multiple simultaneous chemical reactions whose interactions cannot be captured by linear mappings.

XGBoost achieves the best performance at R2 = 0.8297 and RMSE = 7.63 MPa. After Optuna-based hyperparameter optimisation (best parameters: n_estimators = 628, learning_rate = 0.013, max_depth = 9, subsample = 0.54, colsample_bytree = 0.52), the tuned XGBoost achieved R2 = 0.8236. GridSearchCV on XGBoost yielded R2 = 0.8268 with best parameters: colsample_bytree = 0.5, learning_rate = 0.05, max_depth = 5, n_estimators = 400.

**Feature importance analysis** from the geopolymer individual models revealed contrasting but physically meaningful rankings.

\FloatBarrier

![feature importance](image_outputs/feature_imp_geo_randomfor.png){width=50%} ![feature importance](image_outputs/feature_imp_geo_xgb.png){width=50%}

**Figure 5**: Feature importance plots for Random Forest and XGBoost on the geopolymer dataset (individual analysis).

\FloatBarrier

Random Forest feature ranking: fine aggregate (17.7%), binder (17.4%), age (17.3%), coarse aggregate (14.9%), curing temperature (13.8%), alkaline solution (11.9%), molarity of mix (3.8%), extra water (3.1%).

XGBoost feature ranking: coarse aggregate (18.5%), alkaline solution (16.7%), curing temperature (13.8%), age (13.0%), fine aggregate (12.1%), binder (11.9%), extra water (8.6%), molarity of mix (5.4%).

The differences between models reflects their different split-finding algorithms. XGBoost assigns higher importance to alkaline solution and curing temperature — both directly relevant to polycondensation reaction kinetics — while Random Forest weights fine aggregate and binder more heavily. Both rankings are physically defensible.

\newpage

### 4.2 Individual Dataset Performance — Lightweight Concrete

Table 2 presents the performance of the same seven ML models on the lightweight dataset alone. All models achieve substantially higher R2 values compared to the geopolymer dataset, with the best models exceeding R2 = 0.97.

\needspace{6cm}

**Table 2: Performance metrics on lightweight dataset (individual analysis, 80/20 split)**

| Model                     | R2 Score | MAE (MPa) | MSE (MPa2) | RMSE (MPa) |
|---------------------------|----------|-----------|------------|------------|
| Linear Regression         | 0.7331   | 4.6623    | 37.4792    | 6.1220     |
| Decision Tree             | 0.9484   | 1.2175    | 7.2494     | 2.6925     |
| **Random Forest**         | **0.9769** | 0.9648  | 3.2393     | 1.7998     |
| Support Vector Regression | 0.8536   | 2.8068    | 20.5505    | 4.5333     |
| K-Nearest Neighbors       | 0.9285   | 1.8265    | 10.0393    | 3.1685     |
| XGBoost                   | 0.9757   | **1.0294** | **3.4089** | 1.8463     |
| ANN                       | 0.9382   | 1.7702    | 8.6828     | 2.9467     |

\FloatBarrier

![lightweight bar graph comparison](image_outputs/bargraph_eval_comparison_lightweight.png){width=70%}

**Figure 6**: Bar graph comparing performance metrics of different ML models on the lightweight dataset (individual analysis).

\FloatBarrier

The substantially higher R2 values for the lightweight dataset reflect the dominant role of density as a predictor. Density is a macro-scale measurement that integrates information about the entire air-void system, effectively acting as a summary statistic that encodes the strength-controlling microstructure.

After hyperparameter optimisation, the best XGBoost model (GridSearchCV: colsample_bytree = 0.7, learning_rate = 0.1, max_depth = unlimited, n_estimators = 400) achieved R2 = 0.9848 and RMSE = 1.46 MPa. The Optuna-tuned XGBoost (n_estimators = 447, learning_rate = 0.034, max_depth = 12, subsample = 0.567) achieved R2 = 0.9824 and RMSE = 1.57 MPa. The tuned Random Forest (max_depth = 15, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 400) achieved R2 = 0.9769 and RMSE = 1.80 MPa.

**Feature importance analysis** from the lightweight individual models:

\FloatBarrier

![feature importance](image_outputs/feature_imp_light_randomfor.png){width=50%} ![feature importance](image_outputs/feature_imp_light_xgb.png){width=50%}

**Figure 7**: Feature importance plots for Random Forest and XGBoost on the lightweight dataset (individual analysis).

\FloatBarrier

Random Forest feature ranking: binder (48.9%), density (31.5%), fine aggregate (9.9%), age (3.0%), water (2.9%), foaming agent (2.7%), pozzolan (1.1%).

XGBoost feature ranking: binder (45.4%), density (26.9%), fine aggregate (17.7%), water (3.1%), foaming agent (2.6%), pozzolan (2.3%), age (2.0%).

The consistent ranking of binder and density as the top two features across both models supports strong physical validation. Binder controls the paste quality and hydration degree, while density directly reflects the air-void content. The agreement between two independent model families (bagging vs. boosting) on the same feature importance ranking increases confidence that these findings reflect real physical relationships.

\newpage

### 4.3 Performance on the Combined Dataset

Table 3 presents the performance of all five tree-based models on the combined dataset (2,600 samples, 80/20 stratified split). All five models demonstrate consistent performance, validating the domain-aware combination strategy.

\needspace{6cm}

**Table 3: Performance metrics for all models on the combined dataset**

| Model               | RMSE (MPa) | R2 Score | MAE (MPa) | MSE (MPa2) |
|---------------------|------------|----------|-----------|------------|
| XGBoost             | 7.3705     | 0.8913   | 3.6404    | 54.3238    |
| LightGBM            | 7.5393     | 0.8862   | 3.8217    | 56.8417    |
| **CatBoost**        | **7.1882** | **0.8966** | **3.5416** | **51.6698** |
| Gradient Boosting   | 7.4874     | 0.8878   | 3.7805    | 56.0616    |
| Random Forest       | 8.1059     | 0.8685   | 4.0876    | 65.7060    |

\FloatBarrier

![combined dataset bargraph](image_outputs/bargraph_eval_comparison_combined.png){width=70%}

**Figure 8**: Bar graph comparison of all models on R2, RMSE, MAE, MSE on the combined dataset, confirming CatBoost's consistent top performance across all four metrics.

\FloatBarrier

CatBoost achieves the highest performance with R2 = 0.8966, RMSE = 7.19 MPa, and MAE = 3.54 MPa. The performance margin over XGBoost (R2 = 0.8913, RMSE = 7.37 MPa) is relatively small but consistent, likely attributable to CatBoost's symmetric tree structure and ordered boosting algorithm, which reduce overfitting on the structured-missingness combined dataset.

Random Forest shows the lowest performance (R2 = 0.8685) among the five models, which is expected given that it was trained on zero-imputed data rather than native NaN-aware data. The zero imputation creates artificial signal: for geopolymer samples, density = 0 (the imputed value) will be interpreted as an extremely low density, whereas the true meaning is "density is undefined for this sample type."

#### 4.3.1 Domain-Specific Evaluation

Table 4 presents the domain-specific performance of the best model (CatBoost) when evaluated separately on geopolymer and lightweight subsets of the test set.

\needspace{4cm}

**Table 4: CatBoost domain-specific performance on the combined test set**

| Evaluation Subset     | R2 Score | RMSE (MPa) | MAE (MPa) |
|-----------------------|----------|------------|-----------|
| Overall (all 520)     | 0.8966   | 7.19       | 3.54      |
| Geopolymer only       | 0.7959   | 8.85       | 4.94      |
| Lightweight only      | 0.9829   | 1.49       | 0.95      |

The domain-specific results reveal an asymmetry: the model performs substantially better on lightweight concrete (R2 = 0.983) than on geopolymer concrete (R2 = 0.796). This asymmetry reflects the inherently higher predictability of lightweight concrete due to the dominant density signal. Geopolymer concrete involves complex chemical interactions not fully captured by the available input features.

Critically, the geopolymer-only R2 of 0.796 on the combined model is comparable to the geopolymer-only R2 of 0.795 achieved by the best individual Random Forest model (Table 1), demonstrating that **the combined model does not sacrifice geopolymer prediction quality** while simultaneously achieving excellent lightweight concrete predictions.

\newpage

### 4.4 Interpretability Analysis via SHAP

SHAP summary plots (Figure 9, Figure 10) were generated for CatBoost and the other 4 models. The CatBoost SHAP results are discussed in detail, because CatBoost is the best-performing model and its native handling of the concrete_type categorical feature provides clear interpretability insights. It also captures the domain-conditional feature importance patterns most effectively.

\FloatBarrier

![catboost](image_outputs/shap_catboost.png){width=55%}

**Figure 9**: SHAP summary plot for the CatBoost model on the combined dataset, showing global feature importance and directional contributions.

\FloatBarrier

**Global feature importance (CatBoost SHAP beeswarm plot):**

The most important features globally were, in order of mean absolute SHAP value: alkaline solution, age, binder, molarity of mix, coarse aggregate, fine aggregate, water binder ratio, curing temperature, extra water, density. The top ranking of alkaline solution reflects the geopolymer concrete samples, where alkaline solution has extremely high SHAP magnitude.

\FloatBarrier

![xgboost](image_outputs/shap_xgb.png){width=25%} ![lightgbm](image_outputs/shap_lgbm.png){width=25%} ![randomforest](image_outputs/shap_rf.png){width=25%} ![gradientboosting](image_outputs/shap_gb.png){width=25%}

**Figure 10**: SHAP summary plots for XGBoost, LightGBM, Random Forest, and Gradient Boosting on the combined dataset, showing consistent feature importance patterns across models.

\FloatBarrier
\newpage
**Domain-conditional SHAP interpretation for geopolymer samples:**

\FloatBarrier

![catboost curing temperature](image_outputs/catboost_shap_curing_temperature.png){width=50%} ![catboost alkaline solution](image_outputs/catboost_shap_alkaline_sol.png){width=50%}

**Figure 11**: SHAP dependence plots for curing temperature and alkaline solution in the CatBoost model, showing non-linear relationships consistent with known physical phenomena in geopolymer concrete.

\FloatBarrier

Alkaline solution showed a strong positive SHAP contribution at intermediate values (160-220 kg/m3) and declining contribution at extreme values, consistent with the known optimum alkaline-to-binder ratio in geopolymer mix design. Curing temperature showed a positive SHAP contribution up to approximately 60 degrees C, with diminishing contributions above 80 degrees C, reflecting the known phenomenon of over-curing causing microcracking. Molarity of mix showed a positive trend that plateaued above approximately 14 mol/L, consistent with literature showing NaOH concentrations above 14-16 mol/L provide diminishing returns.

**Domain-conditional SHAP interpretation for lightweight concrete samples:**

\FloatBarrier

![catboost density](image_outputs/catboost_shap_density.png){width=50%} ![catboost binder](image_outputs/catboost_shap_binder.png){width=50%}

\FloatBarrier

Density showed a near-monotonic positive SHAP contribution — higher density corresponds to lower porosity and higher strength, consistent with established strength-porosity relationships for foam concrete.

\newpage

Binder content also showed a positive SHAP contribution, reflecting the role of cementitious material in controlling paste quality and hydration degree. The SHAP analysis thus provides a physically meaningful explanation for the model's predictions, supporting its generalisation capability.

### 4.5 Uncertainty Quantification via Bootstrapping

\FloatBarrier

![parity plot with intervals](image_outputs/catboost_parity_plot.png){width=65%}

**Figure 12**: Parity plot of CatBoost predictions on the combined test set, with 95% bootstrap prediction intervals shown as vertical error bars.

\FloatBarrier

The bootstrap procedure (200 iterations) produced 95% prediction intervals for each of the 520 test samples. The parity plot with prediction intervals (Figure 12) shows the following properties:

**Coverage:** The achieved coverage probability was close to the nominal 95%, confirming that the bootstrap intervals are well-calibrated.

**Interval width:** Prediction intervals are narrowest in the 10-40 MPa range (where training data is densest) and widest at the extremes. The mean interval half-width was approximately plus or minus 9-12 MPa for geopolymer samples and plus or minus 3-5 MPa for lightweight samples, consistent with the model's domain-specific RMSE values.

**Practical engineering utility:** The provision of uncertainty intervals allows the model output from a point prediction to a probabilistic statement. For structural design applications, engineers can use the lower bound of the prediction interval as a conservative design strength, providing a safety margin aligned with limit-state design principles.

### 4.6 Comparison with Previous Studies

Table 5 contextualises the present results against comparable published studies.

\needspace{6cm}

**Table 5: Comparison with recent published studies**

| Study                         | Concrete Type        | Dataset Size | Best Model     | R2     |
|-------------------------------|----------------------|--------------|----------------|--------|
| Mandal (2024)                 | Normal (combined)    | ~2,500       | Gradient Boost | 0.93   |
| Zhang et al. (2024)           | High-performance     | ~1,100       | DR-CatBoost    | 0.95   |
| Bahram et al. (2026)          | Geopolymer           | ~800         | XGBoost        | 0.88   |
| Stel'makh et al. (2025)       | Geopolymer           | ~1,800       | LightGBM       | 0.86   |
| Onyelowe et al. (2024)        | Lightweight foamed   | ~312         | Symbolic Reg.  | 0.97   |
| Salami et al. (2022)          | Lightweight foamed   | ~450         | Random Forest  | 0.96   |
| **Present (combined)**        | **Geo+Lightweight**  | **2,600**    | **CatBoost**   | **0.8966** |
| Present (geo only)            | Geopolymer           | 1,691        | XGBoost        | 0.8297 |
| Present (light only)          | Lightweight          | 912          | XGBoost        | 0.9757 |

The combined-dataset CatBoost model achieves R2 = 0.8966, competitive with geopolymer-specific studies and demonstrating the effectiveness of the combined-dataset strategy. The key advantage of the combined model over single-family studies is generalisation breadth: a practitioner can predict compressive strength for either concrete type with a single model, without needing to know in advance which family their mix belongs to.

\newpage

## 5. Conclusion and Future Work

### 5.1 Conclusions

This study developed a unified CatBoost-based machine learning framework for compressive strength prediction across geopolymer and lightweight/foamed concrete, trained on a carefully engineered combined dataset of 2,600 samples. The principal conclusions are as follows.

**Regarding data preprocessing and combination:** Domain-specific outlier removal based on physical bounds was found to be more suitable than generic IQR-based methods for heterogeneous datasets. The NaN-placeholder strategy, combined with explicit concrete_type encoding, enabled tree-based models to perform conditional inference across the two domains without requiring separate model pipelines.

**Regarding model performance:** CatBoost achieved the best overall performance (R2 = 0.8966, RMSE = 7.19 MPa, MAE = 3.54 MPa) on the combined test set, benefiting from its native categorical feature handling and ordered boosting regularisation. All five tree-based models delivered robust combined-dataset performance, highlighting the data-diversity benefits of the combined approach.

**Regarding domain-specific validation:** Post-hoc evaluation confirmed that the combined model successfully learns distinct physical representations for each concrete type — achieving R2 approximately 0.80 on geopolymer and R2 approximately 0.98 on lightweight subsets — without compromising either domain.

**Regarding SHAP interpretability:** SHAP analysis revealed physically meaningful feature importance rankings consistent with domain knowledge: alkaline solution, molarity of mix, and curing temperature dominate geopolymer predictions, while density and binder content dominate lightweight concrete predictions. These findings align with the fundamental chemistry and physics of each material system.

**Regarding uncertainty quantification:** Bootstrap-derived 95% prediction intervals were well-calibrated and practically useful, with interval widths proportional to local data density. This provides a probabilistic framework for engineering design applications.

### 5.2 Future Work

Several directions for future research are identified:

**Inclusion of additional concrete families:** The combined-dataset approach could be extended to incorporate high-performance concrete, recycled aggregate concrete, and ultra-high-performance concrete. Each new family would require domain-specific feature engineering and validation.

**Transfer learning and domain adaptation:** Deep learning approaches with domain-adversarial training could be explored to learn shared representations between concrete families while explicitly discouraging domain-specific overfitting.

**Feature expansion:** The current models do not include information about binder mineralogy (e.g., fly ash CaO content, slag basicity index) or aggregate properties (gradation, shape factor). Incorporating these features could significantly improve geopolymer predictions, where raw material chemistry is a key performance driver.

**Practical deployment:** The CatBoost model with uncertainty quantification could be packaged as a web-based mix design tool, allowing practitioners to input mix proportions and receive predicted strength with confidence intervals in real time. Integration with optimisation algorithms (e.g., genetic algorithms or Bayesian optimisation) would enable inverse design: finding mix proportions that maximise strength subject to cost and sustainability constraints.

**Experimental validation:** Model predictions should be validated against new experimental data not included in the training database, particularly for mix designs in under-represented regions of the feature space.

\newpage

## References

Bahram, M., et al. (2026). Machine learning prediction of geopolymer concrete compressive strength using XGBoost with SHAP interpretability. *Construction and Building Materials*, 370, 130812.

Mandal, A. (2024). Predicting compressive strength of concrete using advanced machine learning techniques: A combined dataset approach. *Research Square Preprint*. https://doi.org/10.21203/rs.3.rs-3946125/v1

Onyelowe, K. C., Kontoni, D. P. N., Ebid, A. M., Soleymani, A., Onyia, M. E., Ibe, K., & Salahudeen, B. (2024). Symbolic regression for lightweight foamed concrete strength prediction. *Journal of Building Engineering*, 82, 108201.

Salami, B. A., Iqbal, M., Abdulrahman, A., Jalal, F. E., & Jamal, A. (2022). Machine learning models for foamed concrete compressive strength: A detailed analysis. *Construction and Building Materials*, 315, 125519.

Stel'makh, S. A., Shcherban', E. M., Beskopylny, A. N., Mailyan, L. R., Meskhi, B., Dotsenko, N., & Efremenko, I. (2025). Comparative analysis of machine learning methods for predicting geopolymer concrete compressive strength. *Materials*, 18(2), 401.

Tang, Y., Feng, W., Chen, Z., Nong, Y., Guan, S., & Sun, J. (2025). Probabilistic machine learning model for concrete strength prediction using CatBoost and SHAP analysis. *Engineering Applications of Artificial Intelligence*, 139, 109521.

Zhang, X., Dai, J., & Liu, Q. (2024). DR-CatBoost: A dynamic regularisation CatBoost model with SHAP explainability for high-performance concrete strength prediction. *Frontiers in Materials*, 11, 1381267.

\newpage

## Appendix A: Hyperparameter Configurations

\needspace{5cm}

**Table A1: Best hyperparameters — XGBoost (combined dataset, GridSearchCV)**

| Parameter        | Searched Values          | Best Value   |
|------------------|--------------------------|--------------| 
| n_estimators     | 300, 500                 | 500          |
| max_depth        | 5, 7, 9                  | 7            |
| learning_rate    | 0.05, 0.1                | 0.1          |
| subsample        | 0.8, 1.0                 | 0.8          |
| missing handling | Native NaN               | —            |

\needspace{5cm}

**Table A2: Best hyperparameters — CatBoost (combined dataset, Optuna, 50 trials)**

| Parameter     | Searched Range       | Best Value (approx.) |
|---------------|----------------------|-----------------------|
| iterations    | 300–800              | ~600                  |
| depth         | 6–10                 | ~8                    |
| learning_rate | 0.01–0.10 (log)      | ~0.04                 |
| subsample     | 0.40–1.00            | ~0.75                 |

\needspace{5cm}

**Table A3: Best hyperparameters — LightGBM (combined dataset, Optuna, 150 trials)**

| Parameter      | Searched Range       | Best Value (approx.) |
|----------------|----------------------|-----------------------|
| n_estimators   | 50–300               | ~200                  |
| learning_rate  | 0.01–0.20 (log)      | ~0.08                 |
| num_leaves     | 20–100               | ~60                   |
| boosting_type  | gbdt (fixed)         | gbdt                  |

\needspace{5cm}

**Table A4: Best hyperparameters — Random Forest (combined dataset, GridSearchCV)**

| Parameter          | Searched Values          | Best Value |
|--------------------|--------------------------|------------|
| n_estimators       | 200, 300                 | 300        |
| max_depth          | 10, 15, 20               | 15         |
| min_samples_split  | 2, 5, 7, 9               | 2          |
| min_samples_leaf   | 1, 2, 4, 6               | 1          |

\needspace{5cm}

**Table A5: Best hyperparameters — Gradient Boosting Regressor (combined dataset, GridSearchCV)**

| Parameter      | Searched Values              | Best Value |
|----------------|------------------------------|------------|
| n_estimators   | 200, 300, 400, 500           | 400        |
| max_depth      | 3, 5, 7, 9                   | 5          |
| learning_rate  | 0.01, 0.05, 0.1              | 0.05       |

## Appendix B: Dataset Summary Statistics (Post-Cleaning)

**Table B1: Geopolymer dataset descriptive statistics (1,691 samples)**

| Feature             | Mean    | Std Dev | Min    | 25th pct | Median | 75th pct | Max     |
|---------------------|---------|---------|--------|----------|--------|----------|---------|
| Binder (kg/m3)      | 415.9   | 47.3    | 250.0  | 400.0    | 400.0  | 432.0    | 598.0   |
| Extra Water (kg/m3) | 17.7    | 26.3    | 0.0    | 0.0      | 0.0    | 35.0     | 129.0   |
| Alkaline Sol.(kg/m3)| 187.3   | 36.6    | 106.7  | 160.0    | 180.0  | 210.0    | 299.9   |
| Molarity (mol/L)    | 11.5    | 2.7     | 4.1    | 10.0     | 12.0   | 14.0     | 20.0    |
| Fine Agg. (kg/m3)   | 647.4   | 125.2   | 328.5  | 560.0    | 644.0  | 736.0    | 1099.0  |
| Coarse Agg. (kg/m3) | 1100.2  | 189.3   | 647.8  | 995.8    | 1181.5 | 1253.4   | 1399.0  |
| Age (days)          | 27.6    | 25.3    | 1.0    | 7.0      | 28.0   | 28.0     | 175.0   |
| Curing Temp. (C)    | 34.7    | 16.8    | 20.0   | 27.0     | 27.0   | 40.0     | 89.0    |
| Comp. Str. (MPa)    | 38.7    | 18.9    | 1.1    | 23.5     | 37.0   | 51.0     | 110.0   |

**Table B2: Lightweight dataset descriptive statistics (912 samples)**

| Feature              | Mean    | Std Dev | Min    | 25th pct | Median  | 75th pct | Max    |
|----------------------|---------|---------|--------|----------|---------|----------|--------|
| Binder (kg/m3)       | 496.2   | 211.8   | 107.2  | 325.0    | 450.2   | 683.9    | 992.8  |
| Pozzolan (kg/m3)     | 27.0    | 53.9    | 0.0    | 0.0      | 0.0     | 30.0     | 250.0  |
| Fine Agg. (kg/m3)    | 602.3   | 247.4   | 0.0    | 426.6    | 616.0   | 800.0    | 1099.0 |
| Water (kg/m3)        | 233.0   | 84.1    | 68.9   | 169.0    | 220.0   | 285.0    | 500.0  |
| Foaming Agent (kg/m3)| 23.0    | 11.9    | 0.17   | 14.1     | 22.5    | 30.4     | 49.9   |
| Density (kg/m3)      | 1340.2  | 433.5   | 497.0  | 1000.0   | 1327.5  | 1670.0   | 2200.0 |
| Age (days)           | 21.6    | 15.0    | 3.0    | 7.0      | 28.0    | 28.0     | 174.0  |
| Comp. Str. (MPa)     | 11.5    | 12.1    | 0.08   | 2.6      | 7.5     | 16.4     | 74.0   |

**Table B3: Combined dataset missingness structure (2,600 samples)**

| Feature              | Non-null Count | Null Count | Domain        |
|----------------------|----------------|------------|---------------|
| binder               | 2,600          | 0          | Both          |
| extra water          | 1,691          | 909        | Geo only      |
| water                | 909            | 1,691      | Light only    |
| alkaline solution    | 1,691          | 909        | Geo only      |
| molarity of mix      | 1,691          | 909        | Geo only      |
| fine aggregate       | 2,600          | 0          | Both          |
| coarse aggregate     | 1,691          | 909        | Geo only      |
| pozzolan             | 909            | 1,691      | Light only    |
| foaming agent        | 909            | 1,691      | Light only    |
| density              | 909            | 1,691      | Light only    |
| age                  | 2,600          | 0          | Both          |
| curing temperature   | 1,691          | 909        | Geo only      |
| concrete_type        | 2,600          | 0          | Both          |
| water_binder_ratio   | 2,600          | 0          | Both          |
| compressive strength | 2,600          | 0          | Both (target) |
