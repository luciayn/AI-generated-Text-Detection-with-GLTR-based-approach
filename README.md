# AI-Generated Text Detection with a GLTR-based Approach
This study aims to explore various ways to improve GLTR's effectiveness for detecting AI-generated texts within the context of the [IberLef-AuTexTification 2023 shared task](https://arxiv.org/pdf/2309.11285), in both English and Spanish languages.

## Pre-requisites for running app locally
The deployed app can be found [here](https://ai-generated-text-detection-with-gltr-based-approach.streamlit.app).
1. Create a python virtualenv & activate it
2. Install dependencies in requirements.txt (pip install -r app/requirements.txt)
3. Run the app locally:
```bash
$ streamlit run app/app.py --logger.level WARNING
```  

## Project Structure
Overview of the project organization:
```
<root directory>
├── app/                   # Application code and assests
│   ├── images/            # Label display images
│   ├── app.py             # Main application script
│   ├── backend.py         # Backend functionality code
│   ├── ui.py              # User interface logic
│   └── requirements.txt   # Post-processing fairness methods
├── en/                    
│   ├── data/              # English dataset files
│   ├── data_visualization.ipynb        # Data visualization
│   ├── init_transformers.ipynb         # Data preprocessing
│   └── topk_gpt.ipynb     # Study of GPT models and results
├── es/                    
│   ├── data/              # Spanish dataset files
│   ├── es_data_visualization.ipynb     # Data visualization
│   ├── es_init_transformers.ipynb      # Data preprocessing
│   └── es_topk_gpt.ipynb  # Study of GPT models and results
├── LICENSE                # License file
└── README.md              # Project documentation (this file)
```