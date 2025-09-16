# Legal Clause Extraction with Transformers (Phase 1)

This project contains the code for Phase 1 of the "AI-Powered Contract Portfolio Analysis" project. It involves fine-tuning a DistilBERT model on the CUAD dataset to perform Named Entity Recognition (NER) for extracting specific clauses from legal contracts.

## Project Structure
- `src/`: Contains the main Python scripts.
  - `preprocess.py`: Handles all data loading and preprocessing.
  - `train.py`: The main script to execute the model training.
- `notebooks/`: Contains the original Jupyter Notebook used for exploration.
- `data/`: (Not in repo) Should contain the CUAD dataset.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HariviswanathB/Automated-Risk-And-Anomaly-Detection-in-legal-portfolios-using-Graph-AI
    cd Legal_AI_Project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    Manually download or clone the CUAD dataset into the `data/` folder.

## How to Run
To start the training process for the full dataset, run the following command from the main project directory:
```bash
python src/train.py
