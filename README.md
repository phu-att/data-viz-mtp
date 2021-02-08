# Data Visualization - Brexit impact on stock markets

## Scope of the Project
+   to analyze the performance of British, publicly-listed companies in the
aftermath of the 2016 Brexit Referendum. A group of publicly listed companies
based in France and Germany will offer the counterfactual data to estimate how
British companies could have performed in case of no-leave;
+   to visually display the insights emerging from the analysis.

## Directory Structure
``` bash 
data-viz-mtp
├── LICENSE
├── README.md
├── requirements.txt
├── companion_document.pdf
├── datasets
│   ├── companies.csv
│   ├── financials__long_term.csv
│   └── financials__short_term.csv
└── scripts
    ├── 00_prep&analysis.ipynb
    └── output
        └── 00_viz.pdf
```
## File Description
* ```requirements.txt``` -> Required Python libraries 
* ```companion_document.pdf``` -> Document that explains rational and analysis of the project
### datasets
* ```companies.csv``` -> The list of the company involved in the analysis
* ```financials__long_term.csv``` -> Monthly financial data spanning the years 2014 - 2018
* ```financials__short_term.csv```-> Daily financial data in the vicinity of the Brexit Referendum
### scripts
* ``` 00_prep&analysis.ipynb``` -> The code contains data exploration,processing processes & conducted analysis, including all outputs shown in the companion document
### output
* ```00_viz.pdf``` -> The combined visualization that contains all visualizations in the code
