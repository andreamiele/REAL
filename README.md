# Project Title: AI Tutor for EPFL Course Content

## Team Members
- Andrea Miele [andrea.miele@epfl.ch](mailto:andrea.miele@epfl.ch)
- Luca Mouchel
- Elia Mounier-Poulat
- Frederik Gerard de Vries

## Introduction
Welcome to our AI tutor project designed to assist with EPFL course content! This project aims to build a real-world language model for educational assistance by generating training data, training a generator model, and extending it in innovative ways. This README will guide you through the structure of our project repository and provide links to essential resources.

## Repository Structure
```plaintext
final-version/
│
├── model/
│   ├── ...
│   └── README.md
│
├── pdfs/
│   └── REAL.pdf
│
├── milestone-1/
│   ├── _docs/
│   ├── data/
│   ├── milestone1/
│   ├── pdfs/
│   ├── README.md
│   ├── t.py
│   └── test.py
│
├── milestone-2/
│   ├── data/
│   ├── model/
│   ├── pdfs/
│   └── README.md
│
└── README.md
```

### Folder Descriptions

1. **final-version/**:
   - Contains the final version of the project including the trained models, necessary documentation, and relevant PDFs.

2. **model/**:
   - Houses the trained model files. Refer to `README.md` in this folder for detailed instructions on model usage.

3. **pdfs/**:
   - Includes reports throughout the project.

4. **milestone-1/**:
   - Contains all files and documentation related to the first project milestone. Includes initial data collection scripts and preliminary model files.

5. **milestone-2/**:
   - Contains all files and documentation related to the second project milestone. Includes further data collection, model improvement scripts, and additional documentation.

## Steps to Reproduce

### Step 1: Collecting Preference Data
- In the first stage, preference data was collected by distilling demonstrations from a large language model (LLM) such as ChatGPT.
- Data is stored in the `milestone-1/data/` directory.

### Step 2: Training the Generator Model
- Using the Direct Preference Optimization (DPO) method, the generator model was trained.
- Training scripts and initial model files are located in the `milestone-1/` directory.
- Final model files can be found in the `final-version/model/` directory.

### Step 3: Improving the Model
- The generator model was enhanced using techniques such as Retrieval Augmented Generation (RAG) and Quantization.
- Scripts and documentation for these improvements are in the `milestone-2/` directory.

## Model and Dataset Links
- Base model repository: [HuggingFace Base Model]()
- Final trained model repository: [HuggingFace Final Model]()
- Preference data: [Google Drive Link]()
- Additional datasets: [EPFL Dataset]()

## Usage Instructions
1. **Setup Environment**:
   - Ensure you have Python 3.8+ installed.
   - Install the required packages using `pip install -r requirements.txt`.

2. **Running the Model**:
   - Navigate to the `final-version/model/` directory.
   - Follow the instructions in `README.md` to run the model.

3. **Training the Model**:
   - For retraining or improving the model, use the scripts provided in the `milestone-1/` and `milestone-2/` directories.
   - Detailed instructions are provided within the respective README files.

## Contributing
We welcome contributions from the community. Please follow the guidelines below:
- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Make your changes and commit them (`git commit -m 'Add new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
We would like to thank our course instructors and mentors for their guidance and support throughout this project. Special thanks to the NLP lab at EPFL for providing the necessary resources and infrastructure.

---

For any questions or issues, please contact [andrea.miele@epfl.ch](mailto:andrea.miele@epfl.ch). Happy learning!

---

This README provides an overview of our project and guides you through the repository structure, usage instructions, and contribution guidelines. Please refer to the individual README files in each directory for more detailed information.
