# My Python Project

## Overview
This project is designed to process and analyze metadata from arXiv papers. It includes functionalities for data processing, embedding generation, vector storage, and search capabilities. The project also provides a web interface for user interaction and a command-line interface for script execution.

## Project Structure
```
my-python-project
├── requirements.txt
├── config.py
├── data
│   └── arxiv-metadata-oai-snapshot.json
├── src
│   ├── __init__.py
│   ├── data_processor.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── search_engine.py
│   └── utils.py
├── notebooks
│   └── exploration.ipynb
├── app.py
├── main.py
└── README.md
```

## Installation
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Configuration
Configuration settings such as API keys and file paths can be found in `config.py`. Make sure to update this file with your specific settings before running the project.

## Usage
### Running the Application
To start the web application, run:

```
streamlit run app.py
```

### Command-Line Interface
To use the command-line interface, execute:

```
python main.py
```

## Notebooks
The `notebooks/exploration.ipynb` file contains exploratory data analysis and experimentation with the data and models. You can open this notebook using Jupyter Notebook or Jupyter Lab.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.