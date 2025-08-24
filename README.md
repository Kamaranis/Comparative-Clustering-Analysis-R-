# A Comparative Analysis of Unsupervised Clustering Methods in R

## 📄 Project Goal

This project provides an in-depth exploration of unsupervised machine learning techniques, focusing on a comparative analysis of different clustering algorithms. Using the well-known "Hawks" dataset, the goal is not just to apply models, but to understand their strengths and weaknesses in a real-world scenario with noisy, variable data.

The analysis follows a logical progression:
1.  Intensive data preparation to create a clean, reliable dataset.
2.  Application of a centroid-based method (**K-Means**).
3.  Application of density-based methods (**DBSCAN & OPTICS**) to address the limitations found in K-Means.
4.  A final, critical comparison of the results using statistical metrics and visualizations.

## 🛠️ Part I: Data Preparation & Exploratory Data Analysis

Before any modeling, a rigorous data cleaning and preparation pipeline was executed to ensure the quality of the input data. This critical phase is often the most time-consuming in a data science project.

*   **Methodical NA Imputation:** Missing values in key biometric variables (`Wing`, `Weight`, `Culmen`, `Hallux`) were imputed using a conditional mean based on the hawk's `Species` and `Age`, preserving the underlying data structure.
*   **Outlier Detection and Handling:** The Interquartile Range (IQR) method was used to identify statistical outliers. Instead of removing them, a robust imputation strategy was applied, replacing extreme values with the conditional mean to avoid data loss while correcting anomalies.
*   **Data Integrity Checks:** The final cleaned dataset was thoroughly reviewed using visualizations (histograms, boxplots) to ensure data quality before moving to the clustering phase.

## 🤖 Part II: K-Means Clustering

K-Means was initially applied to segment the data. The analysis explored two hypotheses:
*   **k=3:** To cluster the three known hawk species.
*   **k=6:** To cluster by both species and age (adult/juvenile).

The `clusplot` visualization, which uses PCA to map the clusters, showed that K-Means could create a reasonable separation but struggled with significant overlap and was unable to capture the natural, non-spherical shapes of the data groups. This highlighted the limitations of centroid-based methods for this specific dataset.

## 🔬 Part III: Density-Based Clustering (DBSCAN & OPTICS)

To overcome the challenges faced by K-Means, more advanced density-based algorithms were employed.

*   **OPTICS (Ordering Points To Identify the Clustering Structure):** Used to explore the data's density and identify a suitable number of clusters, revealing a potential for 3 to 5 distinct groups of varying densities.
*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Applied with parameters informed by the OPTICS analysis. DBSCAN proved superior in:
    *   Identifying non-spherical clusters.
    *   Effectively isolating noise points that did not belong to any group.
    *   Providing a more natural segmentation of the hawk species.

## 📊 Part IV: Comparative Analysis & Conclusion

A statistical comparison using the `cluster.stats` function provided quantitative validation for the visual findings.

| Metric | K-Means (k=3) | DBSCAN | Interpretation |
| :--- | :---: | :---: | :--- |
| **Avg. Silhouette Width** | **0.71** | **0.73** | Both models group points well, with a slight edge to DBSCAN. |
| **Dunn Index** | 0.004 | **0.24** | DBSCAN creates much more compact and well-separated clusters. |
| **Noise Handling** | No | **Yes** | DBSCAN correctly identified and excluded outlier data points. |
| **Shape Flexibility** | Low (Spherical) | **High (Arbitrary)** | DBSCAN was better suited for the irregular shapes of the natural data. |

**Conclusion:** For this dataset, characterized by variable density and noise, **DBSCAN provided a superior and more realistic clustering solution** compared to K-Means.

## 💻 Technologies Used

*   **Language:** R
*   **Core Libraries:**
    *   `Stat2Data` (for the dataset)
    *   `tidyverse` (`dplyr`, `ggplot2`)
    *   `dbscan` (for DBSCAN and OPTICS)
    *   `fpc` (for cluster statistics)
    *   `ggbiplot` (for PCA visualization)

## 🚀 Getting Started

To reproduce this analysis:
1.  Clone this repository.
2.  Make sure R and RStudio are installed.
3.  The R Markdown file (`.Rmd`) will automatically prompt to install the required packages.
4.  Run the notebook cells sequentially to follow the entire workflow from data cleaning to the final model comparison.

## 👤 Author

**Antonio Barrera Mora**

*   **LinkedIn:** https://www.linkedin.com/in/anbamo/
*   **GitHub:** @Kamaranis
