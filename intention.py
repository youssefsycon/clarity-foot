from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import joblib

TRAIN_PATH = "data_intention.tsv"


def load_snips_file(file_path):
    list_pair = []
    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split('\t')
            pair = split_line[0], split_line[1].replace('\n', '')
            list_pair.append(pair)
    return list_pair


def train_model():
    train = load_snips_file(TRAIN_PATH)
    train_utterances, train_intents = zip(*train)
    vect = TfidfVectorizer()
    x_train = vect.fit_transform(train_utterances)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, train_intents, test_size=0.2, random_state=42
    )
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    confusion_mat = confusion_matrix(y_val, y_pred)
    unique_classes = set(y_val)  # Assuming y_true contains the true labels

    return clf, vect, precision, recall, accuracy, f1, confusion_mat,unique_classes


print("Training model...")
model, vectorizer, precision, recall, accuracy, f1, confusion_mat,unique = train_model()
print("Model trained!")
print("precision: ", round(precision * 100, 2), "%")
print("recall: ", round(recall * 100, 2), "%")
print("accuracy: ", round(accuracy * 100, 2), "%")
print("f1: ", round(f1 * 100, 2), "%")
print( confusion_mat)
print("unique classes: ", unique)
joblib.dump(model, 'saved/model.pkl')
joblib.dump(vectorizer, 'saved/vectorizer.pkl')
svm=[precision,recall,accuracy,f1,confusion_mat]