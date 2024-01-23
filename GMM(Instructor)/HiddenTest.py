import os
import importlib
import torch
import numpy as np

# Define the directory where student solution files are located
STUDENT_SOLUTION_DIR = os.getcwd()  # Use the current working directory


def generate_data():
    age = torch.tensor(
        [35, 45, 30, 28, 40, 55, 32, 48, 50, 42, 29, 37, 52, 31]
    ).unsqueeze(1)
    income = torch.tensor(
        [
            60000,
            75000,
            50000,
            55000,
            70000,
            80000,
            45000,
            90000,
            95000,
            60000,
            48000,
            62000,
            80000,
            55000,
        ]
    ).unsqueeze(1)
    purchase_frequency = torch.tensor(
        [2, 4, 1, 3, 5, 6, 2, 4, 3, 2, 1, 5, 4, 3]
    ).unsqueeze(1)
    dataset = torch.cat((age, income, purchase_frequency), dim=1)

    return dataset


def test_case(student_solution, srn):
    # Generate customer segmentation data
    data = generate_data()

    try:
        # Load the student's submission
        student_solution_module = student_solution.replace(".py", "")
        mymodule = importlib.import_module(student_solution_module)
        GMMModel = mymodule.GMMModel

        # Load the standard solution GMM
        standard_solution = importlib.import_module("GMM_Standard_Solution")

        # Test GMM fitting
        gmm = GMMModel(n_components=3)
        gmm.fit(data)

        # Test GMM prediction
        labels = gmm.predict(data)
        if labels.shape == (data.shape[0],):
            gmm_fit_prediction_result = "PASSED"
        else:
            gmm_fit_prediction_result = "FAILED"

        # Test getting cluster means
        cluster_means = gmm.get_cluster_means()
        if cluster_means.shape == (3, data.shape[1]):
            cluster_means_result = "PASSED"
        else:
            cluster_means_result = "FAILED"

        # Test getting cluster covariances
        cluster_covariances = gmm.get_cluster_covariances()
        if cluster_covariances.shape == (3, data.shape[1], data.shape[1]):
            cluster_covariances_result = "PASSED"
        else:
            cluster_covariances_result = "FAILED"

        # Test GMM prediction for a new sample
        new_sample = torch.tensor([[35, 60000, 2]], dtype=torch.float32)
        predicted_label = gmm.predict(new_sample)
        if predicted_label.shape == (1,) and 0 <= predicted_label[0] < 3:
            gmm_prediction_result = "PASSED"
        else:
            gmm_prediction_result = "FAILED"

        print(f"Test Case for {srn}:")
        print(f"Test case 1: {gmm_fit_prediction_result}")
        print(f"Test case 2: {cluster_means_result}")
        print(f"Test case 3: {cluster_covariances_result}")
        print(f"Test case 4: {gmm_prediction_result}")
    except Exception as e:
        print(f"Test Case for {srn} FAILED [ERROR]: {str(e)}")


if __name__ == "__main__":
    import os

    STUDENT_SOLUTION_PATH = os.getcwd()
    section = None
    LabNo = None
    for studentSolution in os.listdir(STUDENT_SOLUTION_PATH):
        if "PES" in studentSolution:
            try:
                _, section, srn, LabNo = studentSolution.split("_")
                test_case(studentSolution, srn)
            except Exception as e:
                print(e)
                continue
