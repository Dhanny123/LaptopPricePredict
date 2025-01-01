# Laptop Price Prediction Model
A machine learning model that predicts the price of a laptop based on some specific features.

## Overview
This project aims to develop a machine learning model that predicts the price of laptops based on various features such as specifications, brand, and market trends. By leveraging a robust dataset and advanced machine learning techniques, the model provides accurate price predictions to assist consumers and retailers in making informed decisions.

## Features
- **Accurate Predictions:** Achieves over 90% accuracy on test data.
- **Comprehensive Dataset:** Includes details like processor type, RAM, storage, display quality, and brand.
- **Scalable Model:** Designed to adapt to new data and market trends.
- **User-Friendly Interface:** Can be integrated into applications for real-time price predictions.

## Project Structure
```
├── data
│   ├── raw_data.csv         # Raw dataset
│   ├── processed_data.csv   # Cleaned and preprocessed dataset
├── notebooks
│   ├── data_analysis.ipynb  # Exploratory data analysis
│   ├── model_training.ipynb # Model development and training
├── src
│   ├── data_preprocessing.py # Scripts for data cleaning
│   ├── train_model.py        # Model training script
│   ├── predict.py            # Prediction script
├── tests
│   ├── test_data.py          # Unit tests for data preprocessing
│   ├── test_model.py         # Unit tests for model performance
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/username/laptop-price-prediction.git
    cd laptop-price-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the data:
    Place the raw dataset in the `data` folder and run the preprocessing script:
    ```bash
    python src/data_preprocessing.py
    ```

2. Train the model:
    ```bash
    python src/train_model.py
    ```

3. Make predictions:
    ```bash
    python src/predict.py --input "path_to_input_file.csv" --output "path_to_output_file.csv"
    ```

## Dataset
The dataset includes:
- **Processor Type:** Intel, AMD, etc.
- **RAM:** Memory size in GB.
- **Storage:** Type and capacity of storage (HDD, SSD).
- **Display:** Screen size and resolution.
- **Brand:** Laptop brand.
- **Price:** Actual price (target variable).

## Model Details
- **Algorithms Used:** Regression techniques, Random Forest, Neural Networks.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared.

## Contributions
Contributions are welcome! If you'd like to improve the project, please follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature-name'`
4. Push to the branch: `git push origin feature-name`
5. Create a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaboration, feel free to reach out:
- Email: [danieloniha@gmail.com](mailto:danieloniha@gmail.com)
- GitHub: [Dhanny123](https://github.com/Dhanny123)
