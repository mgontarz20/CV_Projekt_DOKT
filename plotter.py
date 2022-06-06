import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv(r'C:\Users\mgont\PycharmProjects\CV_Projekt_DOKT\good_model\UNet_5_96x96_2022-06-03_15-22_fringes_MSE\UNet_5_96x96_2022-06-03_15-22_fringes_MSE.csv')

    # print(df.head)
    # print(df.columns)


    fig1, ax = plt.subplots(1,2, figsize = (20,10))

    p1a = ax[0].plot(df['epoch'],df['loss'], 'g--', linewidth = 1)
    p1b = ax[0].plot(df['epoch'],df['val_loss'], 'r-.', linewidth = 1.2)

    p2a = ax[1].plot(df['epoch'], df['SSIMMetric'], 'g--', linewidth = 1.1)
    p2b = ax[1].plot(df['epoch'], df['val_SSIMMetric'], 'r--', linewidth = 1.2)

    pb1 = ax[0].plot(np.argmin(df['val_loss']), np.min(df['val_loss']), marker = 'x', markersize = 10, color= 'r')
    ax[0].annotate(f'Best Validation Loss: {np.min(df["val_loss"])}\nFor Epoch: {np.argmin(df["val_loss"])} | Saved Model', (np.argmin(df['val_loss']), np.min(df['val_loss'])),

            xytext=(np.argmin(df['val_loss']),np.min(df["val_loss"]) + 0.002),
                   ha = 'center',
                   bbox=dict(
                    boxstyle="round",
                    fc=(1.0, 0.7, 0.7),
                    ec=(1., .5, .5),
                   color = 'r'),
                   arrowprops = dict(
                       arrowstyle = '-|>',
                   color = 'r')

                   )
    pb2 = ax[1].plot(df["epoch"][162], df["val_SSIMMetric"][162], marker = 'x', markersize = 10, color = 'r')
    ax[1].annotate(
        f'Validation SSIM Metric Value: {np.round(df["val_SSIMMetric"][162], 5)}\nFor Epoch: {df["epoch"][162]} | Saved Model',
        (df["epoch"][162], df["val_SSIMMetric"][162]),

        xytext=(df["epoch"][161], df["val_SSIMMetric"][161] - 0.04),
        ha='center',
        bbox=dict(
            boxstyle="round",
            fc=(1.0, 0.7, 0.7),
            ec=(1., .5, .5),
            color='r'),
        arrowprops=dict(
            arrowstyle='-|>',
            color='r')

        )

    ax[0].legend((p1a, p1b, pb1), ("Training Loss", "Validation Loss", "Best (Saved) Model"), loc = 'best')
    ax[1].legend((p2a, p2b, pb2), ("Training SSIM Metric", "Validation SSIM Metric", "Best (Saved) Model"), loc = 'best')
    plt.show()
