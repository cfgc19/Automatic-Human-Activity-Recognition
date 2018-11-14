from tkinter import *

import functions_RP
import pandas as pd
import numpy as np
import matplotlib as plt
plt.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter.filedialog

global labels, dataset, dataset_test, labels_test , matriz_confusao, features_names, dataset_scaled, dataset_scaled_test, can_classify_2, can_classify_1, labels_names, probCLF, labels_test_result

labels_names = np.array(['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying'])
labels_names = labels_names.transpose()

can_classify_2 = 0
can_classify_1 = 0


def sel():
    if(var.get() ==2):
        roc.config(state=DISABLED)
        var2.set(0)
        var1.set(0)
    else:
        roc.config(state = NORMAL)


def show_entry_fields():
    global dataset, dataset_test, labels, labels_test, labels_names, probCLF, features_names, labels_test, labels_test_result

    dataset,dataset_test,labels,labels_test,features_names = functions_RP.inicialize_data()

    try:
        dataset, dataset_test,labels_test,labels,features_names
    except NameError:
        label_classificar.configure(text="Não pode Classificar. Tem de fazer load de tudo!")
    else:
        dataset1 = dataset
        dataset_test1= dataset_test
        labels1= labels
        labels_test1= labels_test
        features_names1 = features_names
        if(var.get() == 0 or var3.get() == 0):
            print("Nao pode classificar")
            label_classificar.configure(text="Dados insuficientes para classificar! ")
        else:
            label_classificar.configure(text="")
            if(var4.get() == 1):
                dataset1, dataset_test1 = functions_RP.pre_processamento(dataset1, dataset_test1)
            if(var.get() != 0):
                labels1, labels_test1, labels_names1, probCLF = functions_RP.define_labels(var.get())
            if(var1.get() != 0):
                features_names1, dataset1, dataset_test1 = functions_RP.features_selection(dataset1, dataset_test1, features_names1, labels1, var1.get(), 30, probCLF)
            if(var2.get() != 0):
                dataset1, dataset_test1 = functions_RP.features_reduction(dataset1, dataset_test1, labels1, var2.get(), 10)
            if(var3.get() == 1):
                labels_test_result = functions_RP.distMinClf_CV(dataset1, labels1, dataset_test1, labels_test1, probCLF, 'euclidean', labels_names)
            elif (var3.get() == 2):
                labels_test_result = functions_RP.distMinClf_CV(dataset1, labels1, dataset_test1, labels_test1, probCLF, 'mahalanobis', labels_names)
            elif(var3.get() == 3):
                labels_test_result = functions_RP.clfNaiveBayes(dataset1, dataset_test1, labels1, labels_test1, labels_names, probCLF)
            elif (var3.get() == 4):
                labels_test_result = functions_RP.clfSVM(dataset1, dataset_test1, labels1, labels_test1, labels_names, probCLF)
            elif(var3.get() == 5):
                labels_test_result = functions_RP.clfKNN(dataset1, dataset_test1, labels1, labels_test1, labels_names, probCLF)
            elif(var3.get() == 6):
                labels_test_result = functions_RP.clfRandomForest(dataset1, dataset_test1, labels1, labels_test1, labels_names, probCLF)
            elif (var3.get() == 7):
                labels_test_result = functions_RP.clfQDA(dataset1, dataset_test1, labels1, labels_test1, labels_names, probCLF)
            elif (var3.get() == 8):
                labels_test_result = functions_RP.clfLDA(dataset1, dataset_test1, labels1, labels_test1, labels_names, probCLF)
            if(probCLF == 'bin'):
                report, acc, matrix, rocScore = functions_RP.metricsClf(labels_test_result, labels_test1, labels_names, probCLF)
                label_report.configure(text=report)
                text_acc = "Accuracy: " + str(acc)
                label_aac.configure(text=text_acc)
                label_aac.grid_configure(row=7, column=2, columnspan=5)
                text_roc = "AUC: " + str(rocScore)
                label_roc_auc.configure(text=text_roc)
                text_matrix = "Matriz de Confusão: \n   0    1 \n0  "+ str(matrix[0,0]) + " " + str(matrix[0,1]) + "\n1  " + str(matrix[1,0]) + " " + str(matrix[1,1])
                label_matrix.configure(text=text_matrix)
                label_matrix.grid_configure(row=9, column=2, columnspan=5, rowspan=4, sticky=N)
                frp, tpr, roc_auc = functions_RP.ROC(1, labels_test1, labels_test_result)

                f.set_visible(True)
                f.clear()
                f.add_axes
                canvas = FigureCanvasTkAgg(f)
                lw = 2
                a = f.add_subplot(111)
                a.set_xlabel('True Positive Rate')
                a.set_ylabel('True Positive Rate')
                a.set_title('Receiver operating characteristic')
                a.axis((0.0,1.0,0.0,1.0))
                a.plot(frp, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                canvas.get_tk_widget().grid(row=17, column=2, rowspan=10, columnspan=6, padx=50, sticky=W, pady=20)

            else:
                report, acc, matrix = functions_RP.metricsClf(labels_test_result, labels_test1, labels_names, probCLF)
                f.clear()
                f.set_visible(False)
                canvas = FigureCanvasTkAgg(f)
                canvas.get_tk_widget().grid(row=17, column=2, rowspan=10, columnspan=6, padx=50, sticky=W, pady=20)
                canvas.close_event()
                report, acc, matrix = functions_RP.metricsClf(labels_test_result, labels_test1, labels_names, probCLF)
                label_report.configure(text=report)
                label_report.grid_configure(row=1, column=2, rowspan=9, columnspan=5, sticky=N)
                text_acc = "Accuracy: " + str(acc)
                label_aac.configure(text=text_acc)
                label_roc_auc.configure(text="")
                label_aac.grid_configure(row=10, column=2, columnspan = 5, sticky=N)
                text_matrix = "Matriz de Confusão: \n      1     2     3     4     5     6"
                for i in range(1,7):
                    text_matrix = text_matrix + "\n" + str(i) + "     "
                    for j in range(0,6):
                        text_matrix= text_matrix +  str(matrix[i-1,j]) + "   "
                print(text_matrix)
                label_matrix.configure(text=text_matrix)
                label_matrix.grid_configure(row=11, column=2, columnspan = 6, rowspan=8, sticky=N)


def funcao (var):
    global dataset,dataset_test,labels,labels_test,features_names
    file = open(tkinter.filedialog.askopenfilename(), 'r')
    if(var==1):
        dataset = pd.read_csv(file, delim_whitespace=True, header=None)  # o data set é 7352 linhas por 561 colunas
        dataset = dataset.as_matrix()
    elif(var == 2):
        dataset_test = pd.read_csv('X_test.txt', delim_whitespace=True, header=None)
        dataset_test.as_matrix()  # o data set teste tem 2947 linhas por 561 colunas
    elif(var == 3):
        features_names = pd.read_csv('features.txt', delim_whitespace=True, header=None)
        features_names = features_names.as_matrix()[:, 1]
    elif(var == 4):
        labels = pd.read_csv('y_train.txt', header=None)  # multi
        labels = labels.as_matrix()
        labels = np.squeeze(labels)
    elif(var == 5):
        labels_test = pd.read_csv('y_test.txt', header=None)
        labels_test = labels_test.as_matrix()
        labels_test = np.squeeze(labels_test)


root = Tk()
root.geometry("1000x600")
var = IntVar()
var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var4 = IntVar()

f = Figure(figsize=(4.5, 4), dpi=60)
canvas = FigureCanvasTkAgg(f)

Label(root, text="Tipo de Classificação", font = "Verdana 10 bold").grid(row=3, column=0, padx=10)
Label(root, text="Redução de Features", font = "Verdana 10 bold").grid(row=3, column=1, padx=70, sticky=W)
var = IntVar()

binario = Radiobutton(root, text="Binário", variable=var, value=1,
                  command=sel)
binario.grid(row=4, column=0, sticky=W, padx=20)

multi = Radiobutton(root, text="Multi", variable=var, value=2,
                  command=sel)
multi.grid(row=5, column=0, sticky=W,padx=20)

Label(root, text=" ", font = "Verdana 10 bold").grid(row=6, column=0, padx=20)

pre_processamento = Radiobutton(root, text="Pre-Processamento", variable=var4, value=1)
pre_processamento.grid(row=7, column=0, sticky=W, padx=10)

Label(root, text=" ", font = "Verdana 10 bold").grid(row=8, column=0, padx=20)
Label(root, text="Seleção de Features", font = "Verdana 10 bold").grid(row=9, column=0)

matriz_correlacao = Radiobutton(root, text="Matriz de correlação", variable=var1, value=1)
matriz_correlacao.grid(row=10, column=0, sticky=W, padx=20)

trees = Radiobutton(root, text="Feature Importance using Trees", variable=var1, value=2)
trees.grid(row=12, column=0, sticky=W, padx=20)

lassocv= Radiobutton(root, text="Lasso CV", variable=var1, value=3)

lassocv.grid(row=13, column=0, sticky=W, padx=20)
kruskal = Radiobutton(root, text="Kruskal Wallis", variable=var1, value=4)

kruskal.grid(row=14, column=0, sticky=W, padx=20)

correlacao_labels = Radiobutton(root, text="Correlação entre features e labels", variable=var1, value=5)
correlacao_labels.grid(row=15, column=0, sticky=W, padx=20)

roc = Radiobutton(root, text="ROC AUC", variable=var1, value=6)
roc.grid(row=16, column=0, sticky=W, padx=20)

pca = Radiobutton(root, text="PCA", variable=var2, value=1)
pca.grid(row=4, column=1, padx=90, sticky=W)

mds = Radiobutton(root, text="MDS", variable=var2, value=2)
mds.grid(row=5, column=1, padx=90, sticky=W)

Label(root, text=" ", font = "Verdana 10 bold").grid(row=6, column=1, padx=70)
Label(root, text=" ", font = "Verdana 10 bold").grid(row=7, column=1, padx=70)
Label(root, text="Classificadores",font = "Verdana 10 bold").grid(row=8, column=1, padx=70, sticky=W)

mdc_euclideana = Radiobutton(root, text="MDC - Euclideana", variable=var3, value=1)
mdc_euclideana.grid(row=9, column=1, padx=90, sticky=W)

mdc_mahalanobis = Radiobutton(root, text="MDC - Mahalanobis", variable=var3, value=2)
mdc_mahalanobis.grid(row=10, column=1, padx=90,sticky=W)

random = Radiobutton(root, text="RandomForest", variable=var3, value=3)
random.grid(row=12, column=1, padx=90, sticky=W)

svm = Radiobutton(root, text="SVM", variable=var3, value=4)
svm.grid(row=13, column=1, padx=90, sticky=W)

knn = Radiobutton(root, text="KNN", variable=var3, value=5)
knn.grid(row=14, column=1, padx=90, sticky=W)

naive = Radiobutton(root, text="NaiveBayes", variable=var3, value=6)
naive.grid(row=15, column=1, padx=90, sticky=W)

qda =  Radiobutton(root, text="QDA", variable=var3, value=7)
qda.grid(row=16, column=1, padx=90, sticky=W)

lda= Radiobutton(root, text="LDA", variable=var3, value=8)
lda.grid(row=18, column=1, padx=90, sticky=W)

button = Button(root, text='Classificar', command=show_entry_fields)
button.grid(row=19, column=0, sticky=W, pady=20, padx=200,columnspan=2)

label_metrics = Label(root, text="Métricas de Classificação", font="Verdana 15 bold")
label_metrics.grid(row=0, column=2, columnspan = 5)

label_report = Label(root, text="")
label_report.grid(row=1, column=2, rowspan = 7, columnspan = 5, sticky = N)

label_aac = Label(root, text = "")
label_aac.grid(row=7, column=2, columnspan = 5)

label_roc_auc = Label(root, text="")
label_roc_auc.grid(row=8, column=2, columnspan = 5)

label_matrix = Label(root, text="")
label_matrix.grid(row=9, column=2, columnspan = 5, rowspan=4, sticky=N)


label_classificar = Label(root, text="", foreground='red')
label_classificar.grid(row=20, column=0, sticky=NW, padx=120,columnspan=3)


menu = Menu(root)
menulabels = Menu(menu)
menudata = Menu(menu)
menudata.add_command(label="Train Data", command=lambda: funcao(1))
menudata.add_command(label="Test Data", command=lambda:funcao(2))
menudata.add_command(label="Features Names", command=lambda:funcao(3))
menulabels.add_command(label="Train Labels", command=lambda:funcao(4))
menulabels.add_command(label="Test Labels", command=lambda:funcao(5))
menu.add_cascade(label="Load Data", menu=menudata)
menu.add_cascade(label="Load Labels", menu=menulabels)


root.config(menu=menu)
root.mainloop()
