import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()


def Extract(lst, idx):
    return [item[idx] for item in lst]

def conditional_probability(breast_cancer):
    KL = {}
    for idx, feature_name in enumerate(breast_cancer.feature_names):
               
        feature_data = np.array(Extract(breast_cancer.data, idx))
        target_0 = []
        target_1 = []
        for i, j in zip(feature_data, breast_cancer.target):
            if j == 0:
                target_0.append(i)
            else:
                target_1.append(i)

        
        target_0 = (target_0 - np.min(target_0)) / np.ptp(target_0)
        target_1 = (target_1 - np.min(target_1)) / np.ptp(target_1)

        mean_0 = np.mean(target_0)
        std_0 = np.std(target_0)
        mean_1 = np.mean(target_1)
        std_1 = np.std(target_1)

        x_values_0 = np.sort(np.array(target_0))
        y_values_0 = norm(mean_0, std_0)

        x_values_1 = np.sort(np.array(target_1))
        y_values_1 = norm(mean_1, std_1)

        pdf_0 = y_values_0.pdf(x_values_0)
        pdf_1 = y_values_1.pdf(x_values_1)


        plt.title(f"feature name: {feature_name}")
        plt.plot(x_values_0, pdf_0, label='target 0')
        plt.plot(x_values_1, pdf_1, label='target 1')
        plt.legend()
        plt.show()
       

        # drawing from normal distribution with mean and std of target 0
        x_values_0 = np.sort(np.random.normal(mean_0, std_0, 5000))
        y_values_0 = norm(mean_0, std_0)

        # drawing from normal distribution with mean and std of target 1
        x_values_1 = np.sort(np.random.normal(mean_1, std_1, 5000))
        y_values_1 = norm(mean_1, std_1)

        pdf_sample_0 = y_values_0.pdf(x_values_0)
        pdf_sample_1 = y_values_1.pdf(x_values_1)

        print(f"KL divergence of feature {feature_name}: {abs(np.sum(pdf_sample_0 * np.log(pdf_sample_0 / pdf_sample_1)))}")
        KL[feature_name] = float(abs(np.sum(pdf_sample_0 * np.log(pdf_sample_0 / pdf_sample_1))))

    # Print the feature name with the highest KL divergence 
    print(f"Feature with highest KL divergence: {max(KL, key=KL.get)}, with KL divergence: {max(KL.values())}")
