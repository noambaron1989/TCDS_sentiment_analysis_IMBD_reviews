import re
import csv
import os
import ssl
import sys
from pathlib import Path
import fasttext
import matplotlib.pyplot as plt
from matplotlib import pylab
import nltk
import numpy as np
import pandas as pd
from flair.datasets import ClassificationCorpus
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, ELMoEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from typing import List, Union, Optional
from transformers import pipeline


hyper_params = {
    "lr": 0.35,         # Learning rate
    "epoch": 100,       # Number of training epochs to train for
    "wordNgrams": 3,    # Number of word n-grams to consider during training
    "dim": 155,         # Size of word vectors
    "ws": 5,            # Size of the context window for CBOW or skip-gram
    "minn": 2,          # Min length of char ngram
    "maxn": 5,          # Max length of char ngram
    "bucket": 2014846,  # Number of buckets
}


class FieldsEnum:
    SENTIMENT = 'sentiment'
    SENTIMENT_LABEL = 'sentiment_label'
    PRED = 'pred'
    REVIEW = 'review'
    SCORE = 'score'
    LABEL = 'label'
    TEXT = 'text'
    POLARITY = 'polarity'
    SUBJECTIVITY = 'subjectivity'
    NEG = 'negative'
    NEU = 'neutral'
    POS = 'positive'
    COMPOUND = 'compound'
    ID = 'id'
    POLARITY_VECTOR = 'polarity_vector'
    POLARITY_MAX = 'MaxPolarity'
    POLARITY_MIN = 'MinPolarity'
    POLARITY_AVG = 'AVGPolarity'
    POLARITY_STD = 'STDPolarity'
    NUM_POSITIVE = 'numPositive'
    NUM_NEGATIVE = 'numNegative'


def plot_confusion_matrix(y_true, y_pred, name, normalize=False):
    classes = [0, 1]
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', origin='lower', cmap='YlOrBr')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig = pylab.gcf()
    fig.canvas.set_window_title(name)
    # plt.show()
    plt.savefig(os.path.join(__results_folder__, '{}.png'.format(name)))


def extract_sub_datasets(df):
    # type: (pd.DataFrame) -> List[pd.DataFrame]
    sub_datasets = []
    positive_df = df[df[FieldsEnum.SENTIMENT] == 1]
    negative_df = df[df[FieldsEnum.SENTIMENT] == 0]
    while len(positive_df) > 0:
        try:
            _, pos_subset = train_test_split(positive_df, test_size=500 / len(positive_df))
        except ValueError:
            pos_subset = positive_df.copy()
        positive_df = positive_df[~positive_df.index.isin(pos_subset.index)]
        try:
            _, neg_subset = train_test_split(negative_df, test_size=500 / len(negative_df))
        except ValueError:
            neg_subset = negative_df.copy()
        negative_df = negative_df[~negative_df.index.isin(neg_subset.index)]
        sub_datasets.append(pd.concat([pos_subset, neg_subset]).reset_index(drop=True))

    return sub_datasets


def split_data_to_folders(path_to_csv):
    # type: (str) -> str
    base_folder = os.path.join(os.path.dirname(path_to_csv), 'split')
    df = pd.read_csv(path_to_csv)  # type: pd.DataFrame
    sub_datasets = extract_sub_datasets(df)
    for idx, mini_df in enumerate(sub_datasets):
        path_to_data_folder = os.path.join(base_folder, str(idx))
        if os.path.isdir(path_to_data_folder):
            continue

        os.makedirs(path_to_data_folder)
        mini_df.to_csv(os.path.join(path_to_data_folder, 'dataset.csv'), index=False)

    # change last folder to be test folder
    os.rename(os.path.join(base_folder, '24'), os.path.join(base_folder, 'test'))
    return base_folder


def get_metrics(df):
    # type: (pd.DataFrame) -> tuple
    acc = accuracy_score(df[FieldsEnum.SENTIMENT], df[FieldsEnum.PRED]) * 100
    f1 = f1_score(df[FieldsEnum.SENTIMENT], df[FieldsEnum.PRED], average='macro')
    precision = precision_score(df[FieldsEnum.SENTIMENT], df[FieldsEnum.PRED])
    recall = recall_score(df[FieldsEnum.SENTIMENT], df[FieldsEnum.PRED])
    roc = roc_auc_score(df[FieldsEnum.SENTIMENT], df[FieldsEnum.PRED])
    return acc, f1, precision, recall, roc


class Base:
    @staticmethod
    def _read_data(path_to_data, lower_case):
        # type: (str, bool) -> pd.DataFrame
        df = pd.read_csv(path_to_data)
        if lower_case:
            df[FieldsEnum.REVIEW] = df[FieldsEnum.REVIEW].str.lower()
        return df

    def __init__(self, name):
        # type: (str) -> None
        self.name = name
        self.train_df = None  # type: Optional[pd.DataFrame]
        self.test_df = None
        self.output_csv = os.path.join(__path_to_base__, '{}.csv'.format(self.name))
        self.acc = 0
        self.f1 = 0
        self.precision = 0
        self.recall = 0
        self.roc = 0
        self.model = None
        self.summary_results = None  # type: Optional[pd.DataFrame]

    def _calc_accuracy(self):
        # type: () -> None
        self.acc, self.f1, self.precision, self.recall, self.roc = get_metrics(self.test_df)

    def _dump_predictions(self, save_to_csv):
        # type: (bool) -> None
        self.summary_results = self.test_df[[FieldsEnum.PRED, FieldsEnum.SENTIMENT]].copy()
        if save_to_csv:
            self.summary_results.to_csv(self.output_csv, index=False)

    def _print_accuracy(self):
        # type: () -> None
        self._calc_accuracy()
        res = "Accuracy: {}\n" \
              "Macro F1-score: {}\n" \
              "Precision: {}\n" \
              "Recall: {}\n" \
              "ROC: {}".format(self.acc, self.f1, self.precision, self.recall, self.roc)
        print('In model {}, Found:\n{}'.format(self.name, res))
        with open(os.path.join(__path_to_base__, '{}.txt'.format(self.name)), 'w') as out_file:
            out_file.write(res)

    def _predict(self):
        # type: () -> None
        raise NotImplementedError()

    def _generate_model(self):
        # type: () -> None
        raise NotImplementedError()

    def _get_data(self, lower_case):
        # type: (bool) -> None
        self.train_df = self._read_data(os.path.join(__path_to_base__, 'dataset.csv'), lower_case)
        self.test_df = self._read_data(os.path.join(os.path.dirname(__path_to_base__), 'test/dataset.csv'), lower_case)

    def run(self, save_to_csv, lower_case=False):
        # type: (bool, Optional[bool]) -> None
        print('Getting data')
        self._get_data(lower_case)
        print('Generating model')
        self._generate_model()
        print('Start predicting')
        self._predict()
        # print('Calculating accuracy')
        self._print_accuracy()
        print('Dumping results')
        self._dump_predictions(save_to_csv)
        # print('Ploting confusion matrix')
        # plot_confusion_matrix(self.test_df[FieldsEnum.SENTIMENT], self.test_df[FieldsEnum.PRED], self.name)


class RuleBasedClass(Base):
    def __init__(self, name):
        # type: (str) -> None
        super().__init__(name)

    @staticmethod
    def _score(row):
        # type: (pd.Series) -> None
        raise NotImplementedError()

    def _generate_model(self):
        # type: () -> None
        pass

    def _predict(self):
        # type: () -> None
        self.test_df[FieldsEnum.SCORE] = self.test_df.apply(self._score, axis=1)
        self.test_df[FieldsEnum.PRED] = pd.cut(self.test_df[FieldsEnum.SCORE], bins=2, labels=[0, 1])


class TextBlobSentiment(RuleBasedClass):
    def __init__(self):
        super().__init__('TextBlob')

    @staticmethod
    def _score(row):
        # type: (pd.Series) -> float
        return TextBlob(row[FieldsEnum.REVIEW]).sentiment.polarity


class VaderSentiment(RuleBasedClass):
    def __init__(self):
        super().__init__('Vader')

    def _generate_model(self):
        # type: () -> None
        self.model = SentimentIntensityAnalyzer()

    def _score(self, row):
        # type: (pd.Series) -> float
        return self.model.polarity_scores(row[FieldsEnum.REVIEW])['compound']


class FeatureBasedClass(Base):
    def __init__(
            self,
            clf,  # type: (Union[SGDClassifier, MultinomialNB, LinearSVC, RandomForestClassifier, LogisticRegression])
            name  # type: str
    ):
        # type: (...) -> None
        super().__init__(name)
        self.pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', clf)])

    def _generate_model(self):
        # type: () -> None
        self.model = self.pipeline.fit(self.train_df[FieldsEnum.REVIEW], self.train_df[FieldsEnum.SENTIMENT])

    def _predict(self):
        # type: () -> None
        self.test_df[FieldsEnum.PRED] = pd.Series(self.model.predict(self.test_df[FieldsEnum.REVIEW]))
        self.test_df.fillna(0, inplace=True)


class LogisticRegressionSentiment(FeatureBasedClass):
    def __init__(self):
        super().__init__(LogisticRegression(solver='liblinear', multi_class='auto'), 'LogisticRegression')


class RandomForestClassifierSentiment(FeatureBasedClass):
    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
                         'RandomForestClassifier')


class LinearSVCSentiment(FeatureBasedClass):
    def __init__(self):
        super().__init__(LinearSVC(), 'LinearSVC')


class MultinomialNBsentiment(FeatureBasedClass):
    def __init__(self):
        super().__init__(MultinomialNB(), 'MultinomialNB')


class SVMSentiment(FeatureBasedClass):
    def __init__(self):
        super().__init__(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=100,
                                       learning_rate='optimal', tol=None), 'SVM')


class PolarityFeatureBasedClass(Base):
    def __init__(
            self,
            clf,  # type: (Union[SGDClassifier, MultinomialNB, LinearSVC, RandomForestClassifier, LogisticRegression])
            name  # type: str
    ):
        # type: (...) -> None
        super().__init__(name)
        self.model = clf

    def _prepare_data(self):
        # type: () -> None
        for dataset in (self.train_df, self.test_df):
            dataset[[FieldsEnum.POLARITY, FieldsEnum.SUBJECTIVITY]] = pd.DataFrame(
                dataset[FieldsEnum.REVIEW].apply(lambda row: TextBlob(row).sentiment).to_list())
            dataset[[FieldsEnum.NEG, FieldsEnum.NEU, FieldsEnum.POS, FieldsEnum.COMPOUND]] = pd.DataFrame(
                dataset[FieldsEnum.REVIEW].apply(
                    lambda row: SentimentIntensityAnalyzer().polarity_scores(row)).to_list())

            dataset[FieldsEnum.POLARITY_VECTOR] = dataset[FieldsEnum.REVIEW].apply(
                lambda row: np.array([TextBlob(sentence).sentiment.polarity for sentence in re.split(r'\?|\.', row)]))
            dataset[FieldsEnum.POLARITY_MAX] = dataset[FieldsEnum.POLARITY_VECTOR].apply(np.max)
            dataset[FieldsEnum.POLARITY_MIN] = dataset[FieldsEnum.POLARITY_VECTOR].apply(np.min)
            dataset[FieldsEnum.POLARITY_AVG] = dataset[FieldsEnum.POLARITY_VECTOR].apply(np.average)
            dataset[FieldsEnum.POLARITY_STD] = dataset[FieldsEnum.POLARITY_VECTOR].apply(np.std)
            dataset[FieldsEnum.NUM_NEGATIVE] = dataset[FieldsEnum.POLARITY_VECTOR].apply(lambda x: sum(x > 0))
            dataset[FieldsEnum.NUM_POSITIVE] = dataset[FieldsEnum.POLARITY_VECTOR].apply(lambda x: sum(x < 0))

            dataset.drop(columns=[FieldsEnum.POLARITY_VECTOR, FieldsEnum.REVIEW, FieldsEnum.ID], axis=1, inplace=True)

    def _generate_model(self):
        # type: () -> None
        self._prepare_data()
        self.model.fit(self.train_df.drop(FieldsEnum.SENTIMENT, axis=1), self.train_df[FieldsEnum.SENTIMENT])

    def _predict(self):
        # type: () -> None
        self.test_df[FieldsEnum.PRED] = self.model.predict(self.test_df.drop(FieldsEnum.SENTIMENT, axis=1))


class PolarityLogisticRegressionSentiment(PolarityFeatureBasedClass):
    def __init__(self):
        super().__init__(LogisticRegression(solver='liblinear', multi_class='auto'), 'PolarityLogisticRegression')


class PolarityRandomForestClassifierSentiment(PolarityFeatureBasedClass):
    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
                         'PolarityRandomForestClassifier')


class PolarityLinearSVCSentimentSentiment(PolarityFeatureBasedClass):
    def __init__(self):
        super().__init__(LinearSVC(), 'PolarityLinearSVC')


class PolaritySVMSentiment(PolarityFeatureBasedClass):
    def __init__(self):
        super().__init__(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=100,
                                       learning_rate='optimal', tol=None), 'PolaritySVM')


class FastTextSentiment(Base):

    def __init__(self):
        super().__init__('FastText')

    def _prepare_data(self, train_labeled_data):
        # type:(str) -> None
        data = self.train_df[[FieldsEnum.SENTIMENT, FieldsEnum.REVIEW]]
        data[FieldsEnum.SENTIMENT_LABEL] = ['__label__' + str(s) for s in data[FieldsEnum.SENTIMENT]]
        data[FieldsEnum.REVIEW] = data[FieldsEnum.REVIEW].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True)
        data[[FieldsEnum.SENTIMENT_LABEL, FieldsEnum.REVIEW]].to_csv(train_labeled_data, index=False, sep=' ',
                                                                     header=False, quoting=csv.QUOTE_NONE,
                                                                     quotechar="", escapechar=" ")

    def _generate_model(self):
        # type: () -> None
        labeled_trained_data = os.path.join(__path_to_base__, 'dataset_with_labels.txt')
        self._prepare_data(labeled_trained_data)
        self.model = fasttext.train_supervised(input=labeled_trained_data, **hyper_params)

    def _score(self, text):
        # type: (pd.Series) -> int
        labels, probabilities = self.model.predict(text[FieldsEnum.REVIEW])
        pred = int(labels[0][-1])
        return pred

    def _predict(self):
        # type: () -> None
        self.test_df[FieldsEnum.PRED] = self.test_df.apply(self._score, axis=1)


class FlairSentiment(Base):

    def __init__(self):
        # type: () -> None
        super().__init__('Flair')
        self.path_to_train = os.path.join(__path_to_base__, 'train.csv')
        self.path_to_test = os.path.join(__path_to_base__, 'test.csv')
        self.path_to_dev = os.path.join(__path_to_base__, 'dev.csv')

    def _train_model(self):
        # type: () -> None
        corpus = ClassificationCorpus(Path(__path_to_base__),
                                      test_file=os.path.basename(self.path_to_test),
                                      dev_file=os.path.basename(self.path_to_dev),
                                      train_file=os.path.basename(self.path_to_train))
        word_embeddings = [ELMoEmbeddings('original'),
                           FlairEmbeddings('news-forward-fast'),
                           FlairEmbeddings('news-backward-fast')]
        document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                    reproject_words_dimension=256)
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False)
        trainer = ModelTrainer(classifier, corpus)
        trainer.train(__path_to_base__, max_epochs=10)

    def _prepare_data(self):
        # type:() -> None
        data = self.train_df.sample(frac=1).drop_duplicates()
        data = data[[FieldsEnum.SENTIMENT, FieldsEnum.REVIEW]].rename(
            columns={FieldsEnum.SENTIMENT: FieldsEnum.LABEL, FieldsEnum.REVIEW: FieldsEnum.TEXT})
        data[FieldsEnum.LABEL] = ['__label__' + str(s) for s in data[FieldsEnum.LABEL]]
        data.iloc[0:int(len(data) * 0.8)].to_csv(self.path_to_train, sep='\t', index=False, header=False)
        data.iloc[int(len(data) * 0.8):int(len(data) * 0.9)].to_csv(self.path_to_test, sep='\t', index=False,
                                                                    header=False)
        data.iloc[int(len(data) * 0.9):].to_csv(self.path_to_dev, sep='\t', index=False, header=False)

    def _generate_model(self):
        # type: () -> None
        self._prepare_data()
        self._train_model()
        self.model = TextClassifier.load(os.path.join(__path_to_base__, 'best-model.pt'))

    def _score(self, text):
        # type: (pd.Series) -> int
        doc = Sentence(text[FieldsEnum.REVIEW])
        self.model.predict(doc)
        pred = int(doc.labels[0].value)
        return pred

    def _predict(self):
        # type: () -> None
        self.test_df[FieldsEnum.PRED] = self.test_df.apply(self._score, axis=1)


class TransformersPipelines:

    def _init_(self):
        self.name = 'TransformersPipelines'
        self.data = pd.read_csv(sys.argv[1])
        self.output_csv = os.path.join(os.path.dirname(sys.argv[1]), 'results', '{}.csv'.format(self.name))
        # this is the transformers model
        self.model = pipeline('sentiment-analysis')

    def _dump_results(self):
        # type: () -> None
        self.data[[FieldsEnum.SENTIMENT, FieldsEnum.PRED]].to_csv(self.output_csv, index=False)

    def _score(self, row):
        # type: (pd.Series) -> float
        if row.name % 100 == 0:
            print(row.name)
        return 0 if self.model(row[FieldsEnum.REVIEW])[0]['label'] == 'NEGATIVE' else 1

    def _predict(self):
        self.data[FieldsEnum.PRED] = self.data.apply(self._score, axis=1)

    def run(self):
        self._predict()
        self._dump_results()


def main():
    global __path_to_base__
    for obj in [TextBlobSentiment(), VaderSentiment(), LogisticRegressionSentiment(), RandomForestClassifierSentiment(),
                LinearSVCSentiment(), MultinomialNBsentiment(), SVMSentiment(), PolarityLogisticRegressionSentiment(),
                PolarityRandomForestClassifierSentiment(), PolarityLinearSVCSentimentSentiment(),
                PolaritySVMSentiment(), FastTextSentiment(), TransformersPipelines()]:
        print('In {} model'.format(obj.name))
        results_csv = os.path.join(__results_folder__, '{}.csv'.format(obj.name))
        if os.path.exists(results_csv):
            continue

        results = pd.DataFrame()
        for idx, train_folder in enumerate(os.listdir(__path_to_base__)):
            print('In {}/{}'.format(idx, len(os.listdir(__path_to_base__))))
            try:
                int(train_folder)
            except ValueError:
                continue

            obj.__init__()
            __path_to_base__ = os.path.join(__path_to_base__, train_folder)
            print('In folder: {}'.format(__path_to_base__))
            obj.run(save_to_csv=True)
            obj.summary_results.rename(columns={FieldsEnum.PRED: '{}_{}'.format(FieldsEnum.PRED, train_folder)},
                                       inplace=True)
            results = pd.concat([results, obj.summary_results.drop(columns=[FieldsEnum.SENTIMENT])], axis=1)

            __path_to_base__ = os.path.dirname(__path_to_base__)

        results[FieldsEnum.PRED] = (results.apply(sum, axis=1) / len(results.keys()) > 0.5).astype(int)
        results = pd.concat([results, obj.summary_results[FieldsEnum.SENTIMENT]], axis=1)
        results[[FieldsEnum.PRED, FieldsEnum.SENTIMENT]].to_csv(results_csv)
        with open(os.path.join(__results_folder__, '{}.txt'.format(obj.name)), 'w') as out_file:
            acc, f1, precision, recall, roc = get_metrics(results)
            res = "Accuracy: {}\n" \
                  "Macro F1-score: {}\n" \
                  "Precision: {}\n" \
                  "Recall: {}\n" \
                  "ROC: {}".format(acc, f1, precision, recall, roc)
            out_file.write(res)

        plot_confusion_matrix(results[FieldsEnum.SENTIMENT], results[FieldsEnum.PRED], obj.name)


class MetricsEnum:
    P = 'P'
    N = 'N'
    TP = 'TP'
    TN = 'TN'
    FP = 'FP'
    FN = 'FN'
    SENSITIVITY = 'Sensitivity'
    SPECIFICITY = 'Specificity'
    PRECISION = 'Precision'
    ACCURACY = 'Accuracy'
    F1 = 'F1 Score'


def extract_metrics(path_to_csv):
    # type: (str) -> pd.DataFrame
    df = pd.read_csv(path_to_csv, index_col=0)
    data = {MetricsEnum.P: [sum(df[FieldsEnum.SENTIMENT] == 1)],
            MetricsEnum.N: [sum(df[FieldsEnum.SENTIMENT] == 0)],
            MetricsEnum.TP: [sum((df[FieldsEnum.SENTIMENT] == 1) & (df[FieldsEnum.PRED] == 1))],
            MetricsEnum.TN: [sum((df[FieldsEnum.SENTIMENT] == 0) & (df[FieldsEnum.PRED] == 0))],
            MetricsEnum.FP: [sum((df[FieldsEnum.SENTIMENT] == 0) & (df[FieldsEnum.PRED] == 1))],
            MetricsEnum.FN: [sum((df[FieldsEnum.SENTIMENT] == 1) & (df[FieldsEnum.PRED] == 0))]}
    return pd.DataFrame(data)


def aggregate_results():
    # type: () -> None
    results = pd.DataFrame()
    for csv_file in os.listdir(__results_folder__):
        if csv_file.endswith('csv'):
            metrics = extract_metrics(os.path.join(__results_folder__, csv_file))
            metrics['Model'] = csv_file.split('.')[0]
            results = pd.concat([results, metrics])

    path_to_models_results = os.path.join(os.path.dirname(__results_folder__), 'models_summary.csv')
    results.to_csv(path_to_models_results, index=False)


def generate_polarities():
    for idx, train_folder in enumerate(os.listdir(__path_to_base__)):
        print('In {}/{}'.format(idx, len(os.listdir(__path_to_base__))))
        try:
            int(train_folder)
        except ValueError:
            continue

        dataset = pd.read_csv(os.path.join(__path_to_base__, train_folder, 'dataset.csv'))
        dataset[[FieldsEnum.POLARITY, FieldsEnum.SUBJECTIVITY]] = pd.DataFrame(
            dataset[FieldsEnum.REVIEW].apply(lambda row: TextBlob(row).sentiment).to_list())
        dataset[[FieldsEnum.NEG, FieldsEnum.NEU, FieldsEnum.POS, FieldsEnum.COMPOUND]] = pd.DataFrame(
            dataset[FieldsEnum.REVIEW].apply(
                lambda row: SentimentIntensityAnalyzer().polarity_scores(row)).to_list())

        path_to_save = os.path.join(__polarity_folder__, '{}.csv'.format(train_folder))
        dataset[[FieldsEnum.POLARITY, FieldsEnum.SUBJECTIVITY, FieldsEnum.NEG, FieldsEnum.NEU, FieldsEnum.POS,
                 FieldsEnum.COMPOUND, FieldsEnum.SENTIMENT]].to_csv(path_to_save, index=False)


if __name__ == '__main__':
    __path_to_base__ = split_data_to_folders(sys.argv[1])
    __results_folder__ = os.path.join(os.path.dirname(__path_to_base__), 'results')
    __polarity_folder__ = os.path.join(os.path.dirname(__path_to_base__), 'polarity')
    # generate_polarities()
    # ssl._create_default_https_context = ssl._create_unverified_context
    # nltk.download('vader_lexicon', quiet=True)
    main()
