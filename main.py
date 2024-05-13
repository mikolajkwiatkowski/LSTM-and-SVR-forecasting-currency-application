from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from keras.models import Sequential
from keras.layers import Dense, LSTM
import csv

def funkcja1():
    print("Funkcja 1 została uruchomiona")

def funkcja2():
    print("Funkcja 2 została uruchomiona")

def funkcja3():
    print("Funkcja 3 została uruchomiona")
    
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
    
def wczytaj_plik():
    nazwa_pliku = file_input.get()
    data_set = pd.read_csv(nazwa_pliku, na_values='ND')  # Wczytanie danych i zamiana ND na NaN
    generuj_plik_z_nazwami_walut(nazwa_pliku)
    



#glowne okno
root = tk.Tk()
root.title("LSTM - Predicting exchange rates for US Dol")
root.geometry("400x350")  
root.configure(bg="#fff4e6")


frame_buttons = tk.Frame(root, bg="#fff4e6")
frame_buttons.pack(pady=10)

#pole tekstowe na plik
file_input = tk.Entry(frame_buttons, width=30)
file_input.pack(side=tk.LEFT, padx=5)



# Przycisk do wczytywania pliku
button_wczytaj = tk.Button(frame_buttons, text="Wczytaj plik", command=wczytaj_plik, width=15)
button_wczytaj.pack(side=tk.LEFT, padx=5)

#pole tekstowe na nazwe kraju z ktorego chcemy walute
country_input = tk.Entry(frame_buttons, width=30)
country_input.pack(side=tk.LEFT, pady=5)

# Przyciski z funkcjami
button1 = tk.Button(root, text="Przycisk 1", command=funkcja1, width=20)
button1.pack(pady=5)

button2 = tk.Button(root, text="Przycisk 2", command=funkcja2, width=20)
button2.pack(pady=5)

button3 = tk.Button(root, text="Przycisk 3", command=funkcja3, width=20)
button3.pack(pady=5)


frame_label = tk.Frame(root, bg="#f0f0f0")
frame_label.pack(pady=10)

label = tk.Label(frame_label, text="Instrukcja:", bg="#be9b7b")
label.pack()




#petla programu
root.mainloop()
