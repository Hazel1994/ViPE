import json
from datasets import load_dataset
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Generating images from ViPE and Chatgpt prompts")

    parser.add_argument(
        "--model_name", type=str, default='vipe', help="which model's prompts to use? [pass chatgpt or vipe']"
    )
    args = parser.parse_args()
    return args

def main():
    args=parse_args()

    dataset = load_dataset('dair-ai/emotion')
    valid_dataset = dataset['test']


    model_name=args.model_name
    prompts_path = './prompts/vipe/' if model_name == 'vipe' else './prompts/chatgpt/'

    with open(prompts_path + 'pred_test_best_model_pred_{}'.format(model_name)) as file:
        visual_pred = json.load(file)

    true_labels=valid_dataset['label']

    # Define class labels
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    # Example predicted labels
    predicted = visual_pred
    # Example true labels
    true = true_labels

    # Compute confusion matrix
    cm = confusion_matrix(true, predicted)

    # Calculate percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add percentage values to the plot
    thresh = cm_percentage.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm_percentage[i, j]*100:.1f}%",
                 horizontalalignment="center",
                 color="white" if cm_percentage[i, j] > thresh else "black")

    plt.savefig('CM_{}.png'.format(model_name), bbox_inches='tight')

if __name__ == "__main__":
    main()
