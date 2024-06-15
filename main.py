from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from keras.models import Sequential
from keras.layers import Dense, LSTM
import csv

instrukcja = "instrukcja.txt"
plik_instrukcja = open(instrukcja,'r',encoding='utf-8')
instrukcja_tekst = plik_instrukcja.read()

#przekształca szereg czasowy na format odpowiedni dla LSTM
def get_data(data, look_back):
  data_x, data_y = [],[]
  for i in range(len(data)-look_back-1):
    data_x.append(data[i:(i+look_back),0])
    data_y.append(data[i+look_back,0])
  return np.array(data_x) , np.array(data_y) #zwraca 2 tablice z danymi


def predict_for_lstm(data, dates, split_index): # dodanie parametru split_index
    scaler = MinMaxScaler()  # obiekt skalera
    data = scaler.fit_transform(data) # skaluje dane do przedzialu [0,1]
    train = data[:split_index] # używanie split_index zamiast stałej wartości
    test = data[split_index:] # --II--
    look_back = 1 # liczba probek wstecz
    x_train , y_train = get_data(train, look_back) # generuje dane treningowe 
    x_test , y_test = get_data(test, look_back) # --II-- testowe 
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # Przekształca dane treningowe do formatu wymaganego przez LSTM 
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) # --II-- testowe 
    n_features = x_train.shape[1] # okresla liczbe cech w danych treningowych
    model = Sequential() # inicializacja modelu sekwencyjnego Keras
    model.add(LSTM(100, activation='relu', input_shape=(1, 1))) # dodaje warstwę LSTM z 100 jednostkami i funkcją aktywacji ReLU.
    model.add(Dense(n_features)) # dodaje warste wyjsciowa
    model.compile(optimizer='adam', loss='mse') # kompiluje model z optymalizatorem Adam i funkcją straty MSE
    model.fit(x_train, y_train, epochs=5, batch_size=1) # trenuje model przez 5 epok z rozmiarem partii 1

    y_pred = model.predict(x_test) # przewiduje wartosci na podstawie danych testowych
    y_pred = scaler.inverse_transform(y_pred) # skaluje prognozowane wartości z powrotem do oryginalengo przedzialu

    y_test = np.array(y_test).reshape(-1, 1) # tak samo
    y_test = scaler.inverse_transform(y_test)

    return y_test, y_pred # zwraca wartosci testowe, przewidywane 
#zwraca wynik i wybiera końcową część tablicy dates odpowiadającą długości tablicy y_test

def predict_for_svr(data_set):
    features = data_set.drop(columns=[currency_country])#Usuwa kolumnę z kursem waluty, pozostawiając cechy
    target = data_set[currency_country].astype(float)#Wybiera kolumnę z kursem waluty jako cel.
    scaler_X = StandardScaler()#Inicjalizuje standardowy skaler dla cech.
    scaler_y = StandardScaler()#Inicjalizuje standardowy skaler dla celu.
    X_scaled = scaler_X.fit_transform(features)#Skaluje cechy.
    y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1)).ravel()#Skaluje cel.

    tscv = TimeSeriesSplit(n_splits=5)#Inicjalizuje podział na zestawy treningowe i testowe w sposób specyficzny dla szeregów czasowych
    
    #Siatka parametrów dla SVR
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    }

    svm = SVR(kernel='rbf')#Inicjalizuje model SVR z jądrem RBF.
    grid_search = GridSearchCV(svm, param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error')#Konfiguruje wyszukiwanie siatki dla SVR.
    grid_search.fit(X_scaled, y_scaled)#opasowuje model do skalowanych danych.

    best_svm = grid_search.best_estimator_#Wybiera najlepszy model na podstawie wyszukiwania siatki

    split_index = int(len(features) * 0.8) #Oblicza indeks do podziału danych na zestawy treningowe i testowe
    #często stosuje się podział danych na część treningową (np. 80% danych) i część testową (np. 20% danych).
    
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]#Dzieli dane cech na treningowe i testowe
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]# -- II -- celu

    best_svm.fit(X_train, y_train)#Dopasowuje najlepszy model do danych treningowych

    y_pred_scaled = best_svm.predict(X_test)#Dopasowuje najlepszy model do danych treningowych
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()#Skaluje prognozowane wartości z powrotem do oryginalnego przedziału.
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()#Skaluje rzeczywiste wartości z powrotem do oryginalnego przedziału.
#Zwraca rzeczywiste wartości, prognozowane wartości i odpowiadające im daty.
    return y_test, y_pred, data_set.index[split_index:]

   
def button_prediction():
    global currency_country
    currency_country = combobox.get()
    if currency_country == "":
        tk.messagebox.showerror(title="Błąd", message="Wybierz kraj!" )
        return

    data_set = pd.read_csv(filename, na_values='ND') 
    data_set = data_set.replace('ND', np.nan)#Zastępuje 'ND' wartościami NaN (na wypadek, gdyby były inne instancje).
    data_set.interpolate(inplace=True)# Interpoluje brakujące wartości (szacowanie)
    data_set['Time Serie'] = pd.to_datetime(data_set['Time Serie'])# Konwertuje kolumnę 'Time Serie' na datetime
    data_set.set_index('Time Serie', inplace=True)#Ustawia kolumnę 'Time Serie' jako indeks

    data_for_lstm = data_set[currency_country]
    data_for_lstm = np.array(data_for_lstm).reshape(-1, 1)
    data_for_svr = data_set

    split_index = int(len(data_set) * 0.8) # oblicz split_index na podstawie tego samego podziału co SVR

    y_test_svr, y_pred_svr, dates_svr = predict_for_svr(data_for_svr) #wywoluje metody predict
    y_pred_lstm, dates_lstm = predict_for_lstm(data_for_lstm, data_set.index, split_index)


    #Rysowanie wykresu o wymiarach 14x7 cali
    plt.figure(figsize=(14, 7))
    plt.plot(dates_svr, y_test_svr, label='Actual')
    plt.plot(dates_lstm, y_pred_lstm, label='Predicted for LSTM')
    plt.plot(dates_svr, y_pred_svr, label='Predicted for SVR')
    plt.title(f'Exchange Rate Prediction for {currency_country}')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.show()



    
#pozwala otworzyc plik i zapisac z niego nazwy walut do comoboxa  
def read_file():
    global filename
    filename = fd.askopenfilename(title='Open a file')
    plik_csv = open(filename, 'r', encoding='utf-8')
    czytnik_csv = csv.reader(plik_csv)
    pierwszy_wiersz = next(czytnik_csv)
    nazwy_walut = [nazwa.strip(",") for nazwa in pierwszy_wiersz[2:]] 
    combobox['values'] = nazwy_walut



################################################ GUI ###################################################################
root = tk.Tk()
root.title("LSTM, SVR - Forecasting exchange rates for US Dollar")
root.geometry("500x350")
root.configure(bg="#be9b7b")
root.iconbitmap("icon.ico")

#ramka na guziki
frame_buttons = tk.Frame(root, bg="#be9b7b")
frame_buttons.pack(pady=10)

#wybor pliku
wybor_pliku = tk.Label(frame_buttons, text="Wybierz plik:", bg="#be9b7b")
wybor_pliku.grid(row=0, column=0, padx=5, pady=5)
button_wczytaj = tk.Button(frame_buttons, text="Wczytaj plik", command=read_file, width=15)
button_wczytaj.grid(row=0, column=1, padx=5, pady=5)

#wybor kraju
wybor_kraju_label = tk.Label(frame_buttons, text="Wybierz kraj:", bg="#be9b7b")
wybor_kraju_label.grid(row=1, column=0, padx=5, pady=5)
combobox = ttk.Combobox(frame_buttons, width=27)
combobox.grid(row=1, column=1, padx=5, pady=5)

#przycisk do przewidywania
predict_button = tk.Button(root, text="Przewidywanie wartosci", command=button_prediction, width=40)
predict_button.pack(pady=5)



#ramka dla instrukcji
frame_label = tk.Frame(root, bg="#f0f0f0")
frame_label.pack(pady=10)

label = tk.Label(frame_label, text=instrukcja_tekst, bg="#fff4e6")
label.pack()

root.mainloop()
plik_instrukcja.close()