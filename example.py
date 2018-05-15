from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from utils.improve import Baseline1
from utils.improve import Baseline2

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline2(language)

    baseline.train(data.trainset,data.testset)

    predictions = baseline.test(data.trainset,data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')

