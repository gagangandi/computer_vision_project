import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def evaluate_accuracy(df):
    df['Absolute Error'] = abs(df['Predicted Distance (cm)'] - df['True Distance (cm)'])
    df['% Error'] = 100 * df['Absolute Error'] / df['True Distance (cm)']
    df['Accuracy (%)'] = 100 - df['% Error']

    mae = df['Absolute Error'].mean()
    mse = np.mean((df['Predicted Distance (cm)'] - df['True Distance (cm)'])**2)
    rmse = np.sqrt(mse)

    metrics = {
        'MAE (cm)': mae,
        'MSE (cm²)': mse,
        'RMSE (cm)': rmse,
        'Average Accuracy (%)': df['Accuracy (%)'].mean()
    }

    return df, metrics

def plot_comparison(df):
    fig, ax = plt.subplots()
    ax.plot(df['Image'], df['True Distance (cm)'], label='True', marker='o')
    ax.plot(df['Image'], df['Predicted Distance (cm)'], label='Predicted', marker='x')
    ax.set_xlabel('Image')
    ax.set_ylabel('Distance (cm)')
    ax.set_title('True vs Predicted Distance')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_error(df):
    fig, ax = plt.subplots()
    ax.bar(df['Image'], df['Absolute Error'], color='orange')
    ax.set_xlabel('Image')
    ax.set_ylabel('Absolute Error (cm)')
    ax.set_title('Absolute Errors per Image')
    ax.grid(True)
    st.pyplot(fig)
