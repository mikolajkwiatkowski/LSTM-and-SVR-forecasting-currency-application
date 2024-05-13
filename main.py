from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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

def predict(nazwa_pliku):
    currency_country = combobox.get()
    data_set = pd.read_csv(nazwa_pliku,na_values='ND') #wczytanie danych i zamiana ND na NaN
    data_for_chosen_currency = data_set[currency_country]
    data_for_chosen_currency = np.array(data_for_chosen_currency).reshape(-1,1) #zamiana na tablice
    
def generuj_plik_z_nazwami_walut(nazwa_pliku):
    plik_csv = open(nazwa_pliku, 'r', encoding='utf-8')
    czytnik_csv = csv.reader(plik_csv)
    pierwszy_wiersz = next(czytnik_csv)
    nazwy_walut = [nazwa.strip(",") for nazwa in pierwszy_wiersz[2:]] 
    
    plik_tekstowy = open("currencies.txt", 'w', encoding='utf-8')
    for nazwa_waluty in nazwy_walut:
        plik_tekstowy.write(nazwa_waluty + '\n')
    
    plik_csv.close()
    plik_tekstowy.close()
    
    return nazwy_walut
    
def wczytaj_plik():
    nazwa_pliku = file_input.get()
    data_set = pd.read_csv(nazwa_pliku, na_values='ND')  # Wczytanie danych i zamiana ND na NaN
    nazwy_walut = generuj_plik_z_nazwami_walut(nazwa_pliku)
    combobox['values'] = nazwy_walut

#glowne okno
root = tk.Tk()
root.title("LSTM - Predicting exchange rates for US Dollar")
root.geometry("500x350")
root.configure(bg="#be9b7b")
root.iconbitmap("icon.ico")

#ramka na guziki
frame_buttons = tk.Frame(root, bg="#be9b7b")
frame_buttons.pack(pady=10)

#wybor pliku
wybor_pliku = tk.Label(frame_buttons, text="Nazwa pliku:", bg="#be9b7b")
wybor_pliku.grid(row=0, column=0, padx=5, pady=5)
file_input = tk.Entry(frame_buttons, width=30)
file_input.grid(row=0, column=1, padx=3, pady=5)
button_wczytaj = tk.Button(frame_buttons, text="Wczytaj plik", command=wczytaj_plik, width=15)
button_wczytaj.grid(row=0, column=2, padx=5, pady=5)

#wybor kraju
wybor_kraju_label = tk.Label(frame_buttons, text="Wybierz kraj:", bg="#be9b7b")
wybor_kraju_label.grid(row=1, column=0, padx=5, pady=5)
combobox = ttk.Combobox(frame_buttons, width=27)
combobox.grid(row=1, column=1, padx=5, pady=5)

#przycisk do przewidywania
predict_button = tk.Button(root, text="Przewidywanie wartosci", command=predict, width=20)
predict_button.pack(pady=5)

#ramka dla instrukcji
frame_label = tk.Frame(root, bg="#f0f0f0")
frame_label.pack(pady=10)
label = tk.Label(frame_label, text=instrukcja_tekst, bg="#fff4e6")
label.pack()

root.mainloop()
plik_instrukcja.close()