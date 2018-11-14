import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import scipy.stats
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import svm
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn import manifold
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


def define_labels(var):
    labels = pd.read_csv('y_train.txt', header=None)
    labels = labels.as_matrix()
    labels = np.squeeze(labels)
    labels_test = pd.read_csv('y_test.txt', header=None)
    labels_test = labels_test.as_matrix()
    labels_test = np.squeeze(labels_test)
    prof_clf = 'multi'
    labels_names = np.array(['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying'])
    labels_names = labels_names.transpose()
    if var == 1:
        prof_clf = 'bin'
        labels_bin = np.zeros(len(labels))

        for i in range(0, len(labels)):
            if labels[i] < 4:
                labels_bin[i] = 1
            else:
                labels[i] = 0

        labels_bin_test = np.zeros(len(labels_test))

        for i in range(0, len(labels_test)):
            if labels_test[i] < 4:
                labels_bin_test[i] = 1
            else:
                labels_bin_test[i] = 0
        labels = labels_bin
        labels_test = labels_bin_test
        labels_names = np.array(['Not Walking', 'Walking'])
        labels_names = labels_names.transpose()
    return labels, labels_test, labels_names, prof_clf


def inicialize_data():
    labels = pd.read_csv('y_train.txt', header=None)  # multi
    labels = labels.as_matrix()
    labels = np.squeeze(labels)
    features_names = pd.read_csv('features.txt', delim_whitespace=True, header=None)
    features_names = features_names.as_matrix()[:, 1]
    dataset = pd.read_csv('X_train.txt', delim_whitespace=True,
                          header=None)  # o data set é 7352 linhas por 561 colunas
    dataset = dataset.as_matrix()

    dataset_test = pd.read_csv('X_test.txt', delim_whitespace=True, header=None)
    dataset_test.as_matrix()  # o data set teste tem 2947 linhas por 561 colunas

    labels_test = pd.read_csv('y_test.txt', header=None)
    labels_test = labels_test.as_matrix()
    labels_test = np.squeeze(labels_test)

    return dataset, dataset_test, labels, labels_test, features_names


def pre_processamento(dataset1, dataset_test1):
    scaler = preprocessing.StandardScaler().fit(dataset1)
    dataset = scaler.transform(dataset1)
    dataset_test = scaler.transform(dataset_test1)
    return dataset, dataset_test


def features_reduction(dataset_scaled, dataset_test_scaled, labels, option, comp):
    if option == 1:
        # PCA
        pca = decomposition.PCA(n_components=comp)
        pca.fit(dataset_scaled)
        # Variance (% cumulative) explained by the principal components
        # print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100))

        # pca.components_ The components are sorted by explained_variance_
        # a 1a componente é aquela que tem mais variance_ratio
        data_reduced = pca.transform(dataset_scaled)
        data_test_reduced = pca.transform(dataset_test_scaled)
        n = len(data_reduced)
        kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)
        regr = LinearRegression()
        mse = []

        # começo por fazer este passo para obter a interceção, isto é, número de componentes é 0
        score = -1 * cross_validation.cross_val_score(regr, np.ones((n, 1)), labels.ravel(), cv=kf_10,
                                                      scoring='mean_squared_error').mean()
        mse.append(score)

        # faço uma regressão linear entre cada componente e os labels e calculo o MSE
        for i in np.arange(1, comp + 1):
            score = -1 * cross_validation.cross_val_score(regr, data_reduced[:, :i], labels.ravel(), cv=kf_10,
                                                          scoring='mean_squared_error').mean()
            mse.append(score)
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(mse, '-v')
        ax2.plot(np.arange(1, comp + 1), mse[1:comp + 1], '-v')
        ax2.set_title('Intercept excluded from plot')

        for ax in fig.axes:
            ax.set_xlabel('Number of principal components in regression')
            ax.set_ylabel('MSE')
            ax.set_xlim((-0.2, comp + 0.2))

    elif option == 2:
        mds = manifold.MDS(n_components=comp, metric=True, dissimilarity="euclidean")
        data_reduced = mds.fit_transform(dataset_scaled)
        data_test_reduced = mds.fit_transform(dataset_test_scaled)
    elif option == 3:
        # LDA
        lda = LinearDiscriminantAnalysis(n_components=comp)
        lda.fit(dataset_scaled, labels)
        data_reduced = lda.transform(dataset_scaled)
        data_test_reduced = lda.transform(dataset_test_scaled)

    return data_reduced, data_test_reduced


def features_selection(dataset_scaled, dataset_scaled_test, features_names, labels, option, feat, opt_clf):
    if option == 1:
        # Matriz de correlação
        # corrcoef() usa estrutura de dados em que as features estão por linhas
        matrix_corr = np.corrcoef(dataset_scaled.transpose())
        mean_corr = np.zeros(len(matrix_corr[:, 0]))

        for i in range(0, len(matrix_corr[:, 0])):
            mean_corr[i] = np.mean(matrix_corr[i, :])

        mean_corr = np.absolute(mean_corr)

        mean_corr_cm = []
        all_indexs = []

        for j in range(0, len(mean_corr)):
            if mean_corr[j] < 0.01:
                mean_corr_cm.append(mean_corr[j])
                all_indexs.append(j)

        # reduced_data_cm dados que resultaram da redução por interpretação dos coeficientes de correlação
        features_reduce = dataset_scaled[:, all_indexs]
        features_reduce_test = dataset_scaled_test[:, all_indexs]
        features_name_reduce = features_names[all_indexs]

    elif option == 2:
        # Feature Importance using Extra Trees Classifier
        labels = np.squeeze(labels)
        model = ExtraTreesClassifier()
        model.fit(dataset_scaled, labels)
        features_importance = model.feature_importances_
        ind = np.argsort(features_importance)[::-1]
        # RETORNA OS INDICES DAS FEATURES PELA ORDEM DE IMPORTANCIA DECRESCENTE
        features_reduce = dataset_scaled[:, ind[0:feat]]  # ESCOLHI AS MELHORES FEATURES
        features_name_reduce = features_names[ind[0:feat]]  # NOMES DESSAS FEATURES ESCOLHIDAS
        features_reduce_test = dataset_scaled_test[:, ind[0:feat]]

    elif option == 3:
        # LASSO
        if opt_clf == 'bin':
            thresh = 0.02
        else:
            thresh = 0.15

        clf = LassoCV(max_iter=50000)
        sfm = SelectFromModel(clf, threshold=thresh)
        sfm.fit(dataset_scaled, labels)
        features_reduce = sfm.transform(dataset_scaled)
        features_reduce_test = sfm.transform(dataset_scaled_test)
        features_name_reduce = features_names[sfm.get_support()]

    elif option == 4:
        # kruskal wallis
        score_kw = []
        pvalue_kw = []
        for i in range(0, (len(dataset_scaled[1, :]))):
            data = dataset_scaled[:, i]

            hstat, pval = scipy.stats.mstats.kruskalwallis(data, labels)

            score_kw.append(hstat)
            pvalue_kw.append(pval)

        scores_kw = np.asarray(score_kw)
        # pvalue_kw = np.asarray(pvalue_kw)
        # ordem das features
        ranks_scores = scores_kw.argsort()

        # RETORNA OS INDICES DAS FEATURES PELA ORDEM DE IMPORTANCIA DECRESCENTE
        features_reduce = dataset_scaled[:, ranks_scores[0:feat]]  # ESCOLHI POR EXEMPLO AS 40 MELHORES FEATURES
        features_name_reduce = features_names[ranks_scores[0:feat]]  # NOMES DESSAS 40 FEATURES ESCOLHIDAS
        features_reduce_test = dataset_scaled_test[:, ranks_scores[0:feat]]
    elif option == 5:
        # Correlação entre cada feature e o label
        arrayCoefs = []
        all_indexs = []
        for i in range(0, len(dataset_scaled[1, :])):
            coef = np.corrcoef(dataset_scaled[:, i], labels)
            arrayCoefs.append(np.absolute(coef[0, 1]))
            arrayCoefSelec = []

        for j in range(0, len(arrayCoefs)):
            if arrayCoefs[j] > 0.811:
                all_indexs.append(j)
                arrayCoefSelec.append(arrayCoefs[j])
        # print(len(mean_corr_cm))
        # print(mean_corr_cm.shape)
        # print(features_names_cm.shape)

        # reduced_data_cm dados que resultaram da redução por interpretação dos coeficientes de correlação
        features_reduce = dataset_scaled[:, all_indexs]
        features_reduce_test = dataset_scaled_test[:, all_indexs]
        features_name_reduce = features_names[all_indexs]

    elif option == 6:
        # ROC AUC
        scores = np.empty(0)
        for i in range(0, len(dataset_scaled[0])):
            fpr, tpr, thresholds = metrics.roc_curve(labels, dataset_scaled[:, i], pos_label=1)
            scores = np.append(scores, metrics.roc_auc_score(labels, dataset_scaled[:, i]))

        ind = np.argsort(scores)[::-1]

        # print(scores[np.where(scores > 0.99)].shape)
        features_reduce = dataset_scaled[:, ind[0:feat]]
        features_name_reduce = features_names[ind[0:feat]]
        features_reduce_test = dataset_scaled_test[:, ind[0:feat]]

    elif option == 7:
        # RFE (Recursive Feature Elimination) SVM
        model = SVR(kernel="linear")
        rfe = RFE(model, 30)
        labels = np.squeeze(labels)
        rfeResult = rfe.fit(dataset_scaled, labels)
        ind = rfeResult.argsort()
        features_reduce = dataset_scaled[:, ind[0:feat]]  # ESCOLHI POR EXEMPLO AS 40 MELHORES FEATURES
        features_name_reduce = features_names[ind[0:feat]]
        features_reduce_test = dataset_scaled_test[:, ind[0:feat]]
    return features_name_reduce, features_reduce, features_reduce_test


def metricsClf(labelsPred, labelsTrue, target_names, probCLF):
    report = classification_report(labelsTrue, labelsPred, target_names=target_names)
    acc = accuracy_score(labelsTrue, labelsPred)
    matrix = confusion_matrix(labelsTrue, labelsPred)
    if probCLF == "bin":
        rocScore = roc_auc_score(labelsTrue, labelsPred)

        print(report)
        print("Accuracy: ", acc)
        print("AUC: ", rocScore)
        print("Confusion Matrix: \n", matrix)

        return report, acc, matrix, rocScore

    else:

        print(report)
        print("Accuracy: ", acc)
        print("Confusion Matrix: \n", matrix)

        return report, acc, matrix


def clfSVM(dataTrain, dataTest, labelsTrain, labelsTest, labelsNames, probCLF):
    kFold = 10
    x_train, x_val, y_train, y_val = train_test_split(dataTrain, labelsTrain, test_size=0.3)

    clf = svm.SVC(kernel='linear', C=1)
    model = clf.fit(x_train, y_train)
    scores = cross_validation.cross_val_score(model, x_val, y_val, cv=kFold)

    print("Scores - Validação cruzada 10-fold: ", scores)
    print("Média dos scores da validação cruzada: ", np.mean(scores))
    labelsPred = model.predict(dataTest)
    metricsClf(labelsPred, labelsTest, labelsNames, probCLF)

    return labelsPred


def clfKNN(dataTrain, dataTest, labelsTrain, labelsTest, labelsNames, probCLF):
    if probCLF == "bin":
        nn = 2
    else:
        nn = 6
    kFold = 10
    x_train, x_val, y_train, y_val = train_test_split(dataTrain, labelsTrain, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=nn)
    model = knn.fit(x_train, y_train)
    scores = cross_validation.cross_val_score(model, x_val, y_val, cv=kFold)

    print("Scores - Validação cruzada 10-fold: ", scores)
    print("Média dos scores da validação cruzada: ", np.mean(scores))
    labelsPred = model.predict(dataTest)
    metricsClf(labelsPred, labelsTest, labelsNames, probCLF)

    return labelsPred


def clfLDA(dataTrain, dataTest, labelsTrain, labelsTest, labelsNames, probCLF):
    kFold = 10
    x_train, x_val, y_train, y_val = train_test_split(dataTrain, labelsTrain, test_size=0.3)

    lda = LinearDiscriminantAnalysis()
    model = lda.fit(x_train, y_train)
    scores = cross_validation.cross_val_score(model, x_val, y_val, cv=kFold)

    print("Scores - Validação cruzada 10-fold: ", scores)
    print("Média dos scores da validação cruzada: ", np.mean(scores))

    labelsPred = model.predict(dataTest)

    # Métricas do classificação
    metricsClf(labelsPred, labelsTest, labelsNames, probCLF)
    return labelsPred


def clfQDA(dataTrain, dataTest, labelsTrain, labelsTest, labelsNames, probCLF):
    kFold = 10
    x_train, x_val, y_train, y_val = train_test_split(dataTrain, labelsTrain, test_size=0.3)

    qda = QuadraticDiscriminantAnalysis()
    model = qda.fit(x_train, y_train)

    scores = cross_validation.cross_val_score(model, x_val, y_val, cv=kFold)

    print("Scores - Validação cruzada 10-fold: ", scores)
    print("Média dos scores da validação cruzada: ", np.mean(scores))

    labelsPred = model.predict(dataTest)

    # Métricas do modelo
    metricsClf(labelsPred, labelsTest, labelsNames, probCLF)
    return labelsPred


def clfNaiveBayes(dataTrain, dataTest, labelsTrain, labelsTest, labelsNames, probCLF):
    kFold = 10
    x_train, x_val, y_train, y_val = train_test_split(dataTrain, labelsTrain, test_size=0.3)
    # Gaussian Naive Bayes Implementation
    clfNB = GaussianNB()
    model = clfNB.fit(x_train, y_train)
    scores = cross_validation.cross_val_score(model, x_val, y_val, cv=kFold)

    print("Scores - Validação cruzada 10-fold: ", scores)
    print("Média dos scores da validação cruzada: ", np.mean(scores))
    labelsPred = model.predict(dataTest)

    # Métricas do modelo
    metricsClf(labelsPred, labelsTest, labelsNames, probCLF)
    return labelsPred


def clfRandomForest(dataTrain, dataTest, labelsTrain, labelsTest, labelsNames, probCLF):
    kFold = 10
    x_train, x_val, y_train, y_val = train_test_split(dataTrain, labelsTrain, test_size=0.3)
    # Gaussian Naive Bayes Implementation
    clfRF = RandomForestClassifier()
    model = clfRF.fit(x_train, y_train)

    scores = cross_validation.cross_val_score(model, x_val, y_val, cv=kFold)

    print("Scores - Validação cruzada 10-fold: ", scores)
    print("Média dos scores da validação cruzada: ", np.mean(scores))
    labelsPred = model.predict(dataTest)

    # Métricas do modelo
    metricsClf(labelsPred, labelsTest, labelsNames, probCLF)
    return labelsPred


def distance_min_classification_g(dataset_scaled, labels, dataset_test, type, distance):
    labels_vector = np.empty(0)
    if distance == 'euclidean':
        if type == 'bin':
            indices_0 = np.empty(0)
            indices_1 = np.empty(0)
            for i in range(0, len(labels)):
                if labels[i] == 0:
                    indices_0 = np.append(indices_0, i)
                else:
                    indices_1 = np.append(indices_1, i)

            indices_0 = indices_0.astype(int)
            indices_1 = indices_1.astype(int)

            data_0 = dataset_scaled[indices_0, :]
            data_1 = dataset_scaled[indices_1, :]

            mean_labels_1 = np.empty(0)
            mean_labels_0 = np.empty(0)

            for i in range(0, len(dataset_scaled[0])):
                mean_labels_0 = np.append(mean_labels_0, np.mean((data_0[:, i])))
                mean_labels_1 = np.append(mean_labels_1, np.mean((data_1[:, i])))
            for i in range(0, len(dataset_test[:, 0])):
                g_0 = np.dot(mean_labels_0.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_0), 2))
                g_1 = np.dot(mean_labels_1.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_1), 2))

                if g_0 < g_1:
                    labels_vector = np.append(labels_vector, [1], axis=0)
                else:
                    labels_vector = np.append(labels_vector, [0], axis=0)
        else:
            indices_1 = np.empty(0)
            indices_2 = np.empty(0)
            indices_3 = np.empty(0)
            indices_4 = np.empty(0)
            indices_5 = np.empty(0)
            indices_6 = np.empty(0)
            for i in range(0, len(labels)):
                if labels[i] == 1:
                    indices_1 = np.append(indices_1, i)
                elif labels[i] == 2:
                    indices_2 = np.append(indices_2, i)
                elif labels[i] == 3:
                    indices_3 = np.append(indices_3, i)
                elif labels[i] == 4:
                    indices_4 = np.append(indices_4, i)
                elif labels[i] == 5:
                    indices_5 = np.append(indices_5, i)
                else:
                    indices_6 = np.append(indices_6, i)

            indices_1 = indices_1.astype(int)
            indices_2 = indices_2.astype(int)
            indices_3 = indices_3.astype(int)
            indices_4 = indices_4.astype(int)
            indices_5 = indices_5.astype(int)
            indices_6 = indices_6.astype(int)

            data_1 = dataset_scaled[indices_1, :]
            data_2 = dataset_scaled[indices_2, :]
            data_3 = dataset_scaled[indices_3, :]
            data_4 = dataset_scaled[indices_4, :]
            data_5 = dataset_scaled[indices_5, :]
            data_6 = dataset_scaled[indices_6, :]

            mean_labels_1 = np.empty(0)
            mean_labels_2 = np.empty(0)
            mean_labels_3 = np.empty(0)
            mean_labels_4 = np.empty(0)
            mean_labels_5 = np.empty(0)
            mean_labels_6 = np.empty(0)

            for i in range(0, len(dataset_scaled[0])):
                mean_labels_1 = np.append(mean_labels_1, np.mean((data_1[:, i])))
                mean_labels_2 = np.append(mean_labels_2, np.mean((data_2[:, i])))
                mean_labels_3 = np.append(mean_labels_3, np.mean((data_3[:, i])))
                mean_labels_4 = np.append(mean_labels_4, np.mean((data_4[:, i])))
                mean_labels_5 = np.append(mean_labels_5, np.mean((data_5[:, i])))
                mean_labels_6 = np.append(mean_labels_6, np.mean((data_6[:, i])))

            g = np.empty(0);
            for i in range(0, len(dataset_test[:, 0])):
                g_1 = np.dot(mean_labels_1.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_1), 2));
                g_2 = np.dot(mean_labels_2.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_2), 2));
                g_3 = np.dot(mean_labels_3.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_3), 2));
                g_4 = np.dot(mean_labels_4.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_4), 2));
                g_5 = np.dot(mean_labels_5.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_5), 2));
                g_6 = np.dot(mean_labels_6.transpose(), dataset_test[i, :]) + np.multiply(1 / 2, np.power(
                    np.linalg.norm(mean_labels_6), 2));

                g = np.array([g_1, g_2, g_3, g_4, g_5, g_6])
                indexes = np.argsort(g)[::-1]
                labels_vector = np.append(labels_vector, [indexes[0] + 1], axis=0)
    else:
        if type == 'bin':
            indices_0 = np.empty(0)
            indices_1 = np.empty(0)
            for i in range(0, len(labels)):
                if labels[i] == 0:
                    indices_0 = np.append(indices_0, i)
                else:
                    indices_1 = np.append(indices_1, i)

            indices_0 = indices_0.astype(int)
            indices_1 = indices_1.astype(int)

            data_0 = dataset_scaled[indices_0, :]
            data_1 = dataset_scaled[indices_1, :]

            mean_labels_0 = [];
            mean_labels_1 = [];

            for i in range(0, len(dataset_scaled[0])):
                mean_labels_0.append(np.mean((data_0[:, i])))
                mean_labels_1.append(np.mean((data_1[:, i])))

            mean_labels_0 = np.reshape(mean_labels_0, (mean_labels_0.__len__(), 1))
            mean_labels_1 = np.reshape(mean_labels_1, (mean_labels_1.__len__(), 1))

            matrix = np.concatenate([mean_labels_0, mean_labels_1], axis=1)
            convariance_inverse = np.linalg.pinv(np.cov(matrix))

            for i in range(0, len(dataset_test[:, 0])):
                g_0 = np.dot(np.dot(convariance_inverse, mean_labels_0).transpose(), dataset_test[i, :]) + np.multiply(
                    -1 / 2, mean_labels_0.transpose().dot(convariance_inverse.dot(mean_labels_0)))
                g_1 = np.dot(np.dot(convariance_inverse, mean_labels_1).transpose(), dataset_test[i, :]) + np.multiply(
                    -1 / 2, mean_labels_1.transpose().dot(convariance_inverse.dot(mean_labels_1)))

                if g_0 < g_1:
                    labels_vector = np.append(labels_vector, [1], axis=0)
                else:
                    labels_vector = np.append(labels_vector, [0], axis=0)
        else:
            labels_vector = np.empty(0)

            indices_1 = np.empty(0)
            indices_2 = np.empty(0)
            indices_3 = np.empty(0)
            indices_4 = np.empty(0)
            indices_5 = np.empty(0)
            indices_6 = np.empty(0)

            for i in range(0, len(labels)):
                if labels[i] == 1:
                    indices_1 = np.append(indices_1, i)
                elif labels[i] == 2:
                    indices_2 = np.append(indices_2, i)
                elif labels[i] == 3:
                    indices_3 = np.append(indices_3, i)
                elif labels[i] == 4:
                    indices_4 = np.append(indices_4, i)
                elif labels[i] == 5:
                    indices_5 = np.append(indices_5, i)
                else:
                    indices_6 = np.append(indices_6, i)

            indices_1 = indices_1.astype(int)
            indices_2 = indices_2.astype(int)
            indices_3 = indices_3.astype(int)
            indices_4 = indices_4.astype(int)
            indices_5 = indices_5.astype(int)
            indices_6 = indices_6.astype(int)

            data_1 = dataset_scaled[indices_1, :]
            data_2 = dataset_scaled[indices_2, :]
            data_3 = dataset_scaled[indices_3, :]
            data_4 = dataset_scaled[indices_4, :]
            data_5 = dataset_scaled[indices_5, :]
            data_6 = dataset_scaled[indices_6, :]

            mean_labels_1 = []
            mean_labels_2 = []
            mean_labels_3 = []
            mean_labels_4 = []
            mean_labels_5 = []
            mean_labels_6 = []

            for i in range(0, len(dataset_scaled[0])):
                mean_labels_1.append(np.mean((data_1[:, i])))
                mean_labels_2.append(np.mean((data_2[:, i])))
                mean_labels_3.append(np.mean((data_3[:, i])))
                mean_labels_4.append(np.mean((data_4[:, i])))
                mean_labels_5.append(np.mean((data_5[:, i])))
                mean_labels_6.append(np.mean((data_6[:, i])))

            mean_labels_1 = np.reshape(mean_labels_1, (mean_labels_1.__len__(), 1))
            mean_labels_2 = np.reshape(mean_labels_2, (mean_labels_2.__len__(), 1))
            mean_labels_3 = np.reshape(mean_labels_3, (mean_labels_3.__len__(), 1))
            mean_labels_4 = np.reshape(mean_labels_4, (mean_labels_4.__len__(), 1))
            mean_labels_5 = np.reshape(mean_labels_5, (mean_labels_5.__len__(), 1))
            mean_labels_6 = np.reshape(mean_labels_6, (mean_labels_6.__len__(), 1))

            matrix = np.concatenate(
                [mean_labels_1, mean_labels_2, mean_labels_3, mean_labels_4, mean_labels_5, mean_labels_6], axis=1)
            convariance_inverse = np.linalg.pinv(np.cov(matrix))

            for i in range(0, len(dataset_test[:, 0])):
                g_1 = np.asscalar(
                    np.dot(np.dot(convariance_inverse, mean_labels_1).transpose(), dataset_test[i, :]) + np.multiply(
                        -1 / 2, mean_labels_1.transpose().dot(convariance_inverse.dot(mean_labels_1))))
                g_2 = np.asscalar(
                    np.dot(np.dot(convariance_inverse, mean_labels_2).transpose(), dataset_test[i, :]) + np.multiply(
                        -1 / 2, mean_labels_2.transpose().dot(convariance_inverse.dot(mean_labels_2))))
                g_3 = np.asscalar(
                    np.dot(np.dot(convariance_inverse, mean_labels_3).transpose(), dataset_test[i, :]) + np.multiply(
                        -1 / 2, mean_labels_3.transpose().dot(convariance_inverse.dot(mean_labels_3))))
                g_4 = np.asscalar(
                    np.dot(np.dot(convariance_inverse, mean_labels_4).transpose(), dataset_test[i, :]) + np.multiply(
                        -1 / 2, mean_labels_4.transpose().dot(convariance_inverse.dot(mean_labels_4))))
                g_5 = np.asscalar(
                    np.dot(np.dot(convariance_inverse, mean_labels_5).transpose(), dataset_test[i, :]) + np.multiply(
                        -1 / 2, mean_labels_5.transpose().dot(convariance_inverse.dot(mean_labels_5))))
                g_6 = np.asscalar(
                    np.dot(np.dot(convariance_inverse, mean_labels_6).transpose(), dataset_test[i, :]) + np.multiply(
                        -1 / 2, mean_labels_6.transpose().dot(convariance_inverse.dot(mean_labels_6))))

                g = np.array([g_1, g_2, g_3, g_4, g_5, g_6])
                indexes = np.argsort(g)[::-1]

                labels_vector = np.append(labels_vector, [indexes[0] + 1], axis=0)
    return labels_vector


def distMinClf_CV(dataset_scaled, labels, dataset_test, labels_test, probCLF, distance, labelsNames):
    kfold = 10
    accVal = []
    for i in range(1, kfold):
        x_train, x_val, y_train, y_val = train_test_split(dataset_scaled, labels, test_size=0.3)
        labelsPredVal = distance_min_classification_g(x_train, y_train, x_val, probCLF, distance)
        scoreACC = accuracy_score(y_val, labelsPredVal)
        accVal.append(scoreACC)

    print("Scores - Validação cruzada 10-fold: ", accVal)
    print("Média dos scores da validação cruzada: ", np.mean(accVal))

    labelsPredTeste = distance_min_classification_g(dataset_scaled, labels, dataset_test, probCLF, distance)
    metricsClf(labelsPredTeste, labels_test, labelsNames, probCLF)
    return labelsPredTeste


def ROC(n_classes, y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    '''
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
'''
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    return fpr, tpr, roc_auc
