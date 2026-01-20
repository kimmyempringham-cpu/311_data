# San José 311 Service Request Analysis  
**Service Delays, Equity, and Decision Support**

## Overview

This project analyzes San José’s 311 service request data to identify patterns in service delays and evaluate whether delays vary systematically across request types and neighborhoods. By integrating administrative service data with demographic and weather information, the analysis demonstrates how cities can use existing data to support operational planning and equity-focused evaluation.

A full technical report is linked at the end of this README.

---

## Motivation

Municipal governments must manage thousands of service requests under limited staffing and budget constraints. Without data-driven insight, service delays are often addressed reactively rather than proactively.

This project focuses on two core questions:

- What factors are associated with longer service resolution times?
- Are service delays distributed equitably across neighborhoods?

Rather than maximizing predictive accuracy, the analysis emphasizes interpretability and alignment with real-world government performance metrics.

---

## Data Sources

All data used in this project is publicly available:

- **San José 311 Service Requests (2019–2024)**  
  Source: City of San José 311 Database  
  https://data.sanjoseca.gov/dataset/311-service-request-data

- **Weather Data**  
  Source: National Oceanic and Atmospheric Administration (NOAA)  
  https://www.ncei.noaa.gov/cdo-web/api/v2/data

- **Demographic Data**  
  Source: American Community Survey (ACS), U.S. Census Bureau  
  https://www.census.gov/programs-surveys/acs.html

- **Geographic Reference Data**  
  ZCTA boundaries and city district shapefiles  
  https://catalog.data.gov/dataset/tiger-line-shapefile-current-nation-u-s-2020-census-5-digit-zip-code-tabulation-area-zcta5
  https://data.sanjoseca.gov/dataset/council-district

---

## Approach (High-Level)

- Data cleaning and integration using SQL and Python  
- Exploratory analysis of resolution times and geographic variation  
- Predictive modeling focused on identifying requests at risk of exceeding service thresholds  

Predicting exact resolution times proved ineffective due to the inherent noise in government service data. Reframing the problem as a threshold-based classification task produced more actionable results aligned with service-level agreements.

---

## Key Takeaways

- Service delays vary systematically by request type and location  
- Threshold-based models are more operationally useful than exact time predictions  
- Administrative data can support both efficiency and equity analyses  

---

## Real-World Applications

In a municipal setting, this framework could be used to:

- Flag incoming service requests that are at high risk of delay  
- Support proactive staffing and workload planning  
- Identify neighborhoods experiencing persistent service delays  
- Inform equity-focused performance reporting and policy discussions  

---

## Repository Structure

```text
.
├── analysis/
│   ├── eda_charts/          # Exploratory analysis visualizations
│   ├── modeling_charts/     # Model evaluation plots
│   ├── eda.py               # EDA and visualization code
│   └── modeling.py          # Classification modeling and evaluation
│
├── data/
│   └── Council_District.geojson   # City council district boundaries
│
├── data_prep/
│   ├── acs.py                # ACS demographic data processing
│   ├── noaa.py               # Weather data ingestion and cleaning
│   ├── reformat_dates.py     # Date standardization utilities
│   ├── resolution_hours.py  # Service request resolution time calculation
│   └── zcta.py               # Geographic assignment to ZCTAs
│
├── .idea/                    # IDE configuration (not required for analysis)
└── README.md

---

## Full Report

A detailed report covering data preparation, exploratory analysis, modeling, and results is available here:

**Full Report:** https://www.linkedin.com/in/kimberly-empringham/details/projects/1516868963/multiple-media-viewer/?profileId=ACoAAEmks78Bg-8vupZjyLI4lN5gbGAWb0X3z54&treasuryMediaId=1768895911493

---

## Data Availability

Datasets required for exploratory analysis and modeling are provided in this repository and via an external link due to file size constraints.

**Modeling dataset:**  
https://drive.google.com/file/d/1IjTNK8qEFGF72s4rSVjfETDrfLk4BFyL/view?usp=sharing

Initial SQL queries used for early data preparation are not fully preserved; however, all EDA and modeling results are reproducible using the provided datasets.
