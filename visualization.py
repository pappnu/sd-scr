import math

import matplotlib.pyplot as plt
import numpy as np


def index_of_last_occurence(items, item):
    # the last index is the first index from the reversed list
    return len(items) - items[::-1].index(item)


def draw_spectrogram(images, labels, figsize=(10, 12)):
    columns = math.floor(math.sqrt(len(images)))
    rows = int(len(images) / columns)

    _, axes = plt.subplots(rows, columns, figsize=figsize)
    for i, _ in enumerate(images):
        row = i // columns
        col = i % columns
        ax = axes[row][col]
        ax.imshow(images[i], interpolation='none', origin='lower')
        ax.set_title(labels[i])


def draw_spectrogram_from_tensors(data, figsize=(10, 12)):
    images = [i[0].numpy().squeeze() for i in data]
    images = [np.swapaxes(i, 0, 1) for i in images]
    labels = [i[1].numpy().astype(str) for i in data]

    draw_spectrogram(images, labels, figsize=figsize)

    plt.show()


def plot_losses(losses, title=''):
    plt.figure()

    for loss in losses:
        min_val = min(loss[0])
        min_index = index_of_last_occurence(loss[0], min_val) - 1
        plt.plot(loss[0], label=loss[1])
        plt.plot(min_index, min_val, '.', label=loss[1] + ' minimum')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title(title)

    plt.show(block=False)


def plot_confusion_matrix(values, labels, title, colorbar=False, save_path=''):
    fig = plt.figure()
    ax = plt.gca()
    tick_n = np.arange(len(labels))

    pos = ax.matshow(values, cmap='Oranges')
    if colorbar:
        fig.colorbar(pos, ax=ax)

    ax.set_xticks(tick_n)
    ax.set_yticks(tick_n)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')

    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value:.2f}', va='center', ha='center')

    plt.xticks(rotation=-45)
    plt.xlabel('ennuste')
    plt.ylabel('oikea arvo')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def plot_false_alarm_and_miss_rates(
    false_alarms, miss_rates, thresholds, title='', legends=[], save_path=''
):
    plt.figure()
    ax = plt.gca()

    for false_alarm, miss_rate, threshold in zip(false_alarms, miss_rates, thresholds):
        plt.plot(false_alarm, miss_rate, '-*')

        for i, j, l in zip(false_alarm, miss_rate, threshold):
            ax.annotate(str(l), xy=(i, j), xytext=(5, 5), textcoords='offset points')

    plt.xlabel('väärät hälytykset (%)')
    plt.ylabel('hylätyt oikeat ennusteet (%)')
    plt.title(title)
    if legends:
        plt.legend(legends)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
