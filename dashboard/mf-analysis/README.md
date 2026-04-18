# Mutual Fund Analysis Project

This project provides a comprehensive analysis of mutual funds using various financial ratios and visualizations. It includes a Streamlit application for user-friendly interaction and result display.

## Project Structure

```
mf-analysis
├── src
│   ├── main.py               # Entry point of the application
│   ├── macd_functions.py     # Functions related to MACD calculations
│   ├── mf_analysis.py        # Main logic for mutual fund analysis
│   ├── ratios.py             # Functions to calculate financial ratios
│   └── streamlit_app.py      # Streamlit application for displaying results
├── data
│   └── mf_ga_amfi.csv        # Mutual fund data for analysis
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd mf-analysis
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`. After activating your environment, run:
   ```
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Ensure that the `data/mf_ga_amfi.csv` file is present in the `data` directory. This file contains the mutual fund data required for analysis.

## Usage

1. **Run the analysis**:
   To execute the mutual fund analysis, run the following command:
   ```
   python src/main.py
   ```

2. **Launch the Streamlit application**:
   To view the results in a web application, run:
   ```
   streamlit run src/streamlit_app.py
   ```

## Analysis Overview

The project performs the following analyses on mutual funds:

- Calculation of financial ratios such as Sharpe Ratio, Alpha, Beta, and Information Ratio.
- Visualization of results using Streamlit for better user interaction.
- Comparison of mutual fund performance against benchmarks.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.