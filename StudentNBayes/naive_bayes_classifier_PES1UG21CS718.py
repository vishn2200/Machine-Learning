import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class NaiveBayesClassifier:

    @staticmethod
    def fit(features, labels):
        class_counts = {}
        class_probabilities = {}
        word_probabilities = {}
        total_samples = len(labels)

        unique_classes = np.unique(labels)

        for label in unique_classes:
            class_count = np.sum(np.array(labels) == label)
            class_counts[label] = class_count
            class_probabilities[label] = class_count / total_samples

            class_samples = features[np.array(labels) == label]
            word_probabilities[label] = (np.sum(class_samples, axis=0) + 1) / (np.sum(class_samples) + len(class_samples[0]))

        return class_probabilities, word_probabilities

    @staticmethod
    def predict(features, class_probabilities, word_probabilities, unique_classes):
        predictions = []

        for sample in features:
            max_log_prob = float('-inf')
            predicted_class = None

            for label in unique_classes:
                log_prob = np.log(class_probabilities[label])
                for index, count in enumerate(sample):
                    log_prob += count * np.log(word_probabilities[label][index])

                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    predicted_class = label

            predictions.append(predicted_class)

        return predictions
