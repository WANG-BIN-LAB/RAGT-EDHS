# RAGT-EDHS
# Main Experimental Results
Dataset	ACC	SEN	SPE	F1	AUC
ABIDE I	75.02±2.36	74.11±8.74	74.54±9.46	74.62±5.34	78.94±2.34
ABIDE II	73.10±4.56	78.78±9.91	64.35±9.16	77.05±5.09	71.19±5.52
# Dependencies
- python==3.11.10
- torch==2.1.0+cu118
- numpy==1.26.4
- tqdm==4.67.1
- pandas==2.2.3
- networkx==3.4.2
- argparse==1.4.0
- scipy==1.31.1
- scikit-learn==1.5.2
##  🛠 Installation
Run the following command to create and configure the environment:

```bash
# Create environment
conda create --name RAGT-EDHS python=3.11.10

# Activate environment
conda activate RAGT-EDHS

# Install dependency packages
pip install -r requirements.txt
```
## 🚀 Usage

Run the following command to start the main program:

```bash
python main.py
