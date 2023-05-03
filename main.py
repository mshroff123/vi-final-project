import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import linregress
from collections import defaultdict


from dict import get_image_dict, get_scores_dict
from canvas import get_canvas


def get_crowd_data(file_path):
    total_fruits = 360
    each_fruit = 60
    general_accuracy = 0
    apple_accuracy = 0
    banana_accuracy = 0
    kiwi_accuracy = 0
    mango_accuracy = 0
    orange_accuracy = 0
    pear_accuracy = 0

    crowd_dict = defaultdict()
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        for column in headers[1:]:
            fruit_type, number = column.split('_')
            day = int((int(number) - 13) / 2)
            days_list = []
            score_list = []
            csv_file.seek(0) # move file pointer back to beginning of file
            next(csv_reader)
            for row in csv_reader:
                row = row[1:]
                days, score = row[headers.index(column)].split(' ')
                days_list.append(int(days))
                score_list.append(float(score))
                if day == int(days):
                    print("yes")
                    general_accuracy += 1
                    if fruit_type == "apple":
                        apple_accuracy += 1
                    elif fruit_type == "banana":
                        banana_accuracy += 1
                    elif fruit_type == "kiwi":
                        kiwi_accuracy += 1
                    elif fruit_type == "mango":
                        mango_accuracy += 1
                    elif fruit_type == "orange":
                        orange_accuracy += 1
                    else:
                        pear_accuracy += 1

            avg_score = round((sum(score_list))/6, 2)
            avg_day = round((sum(days_list))/6)
            key = str(days) + fruit_type + "3"
            crowd_dict[key] = {"days": avg_day, "ripeness": avg_score}

    # print(f'Banana Crowd Accuracy: {banana_accuracy/each_fruit:.4f}')
    # print(f'Mango Crowd Accuracy: {mango_accuracy/each_fruit:.4f}')
    # print(f'Apple Crowd Accuracy: {apple_accuracy/each_fruit:.4f}')
    # print(f'Pear Crowd Accuracy: {pear_accuracy/each_fruit:.4f}')
    # print(f'Orange Crowd Accuracy: {orange_accuracy/each_fruit:.4f}')
    # print(f'Kiwi Crowd Accuracy: {kiwi_accuracy/each_fruit:.4f}')
    # print(f'Overall Crowd Accuracy: {general_accuracy/total_fruits:.4f}')

    print(crowd_dict)
    return crowd_dict


def train(mock_dict):
    fruits = defaultdict(list)
    for key, value in mock_dict.items():
        if key[-1] != '3':
            # if leading number is 10
            if key[0:2] == '10':
                fruit = key[2:-1]
                fruits[fruit].append((key[0:2], value))
            else:
                fruit = key[1:-1]
                fruits[fruit].append((key[0], value))
    
    fig, ax = plt.subplots()
    
    fruit_lines = {}
    for fruit, data in fruits.items():
        days = [int(d) for d, _ in data]
        scores = [s for _, s in data]
        slope, intercept, _, _, _ = linregress(days, scores)
        if intercept < 0:
            intercept = 0
        line = [slope * day + intercept for day in days]
        ax.plot(days, line, label=fruit)
        fruit_lines[fruit] = line
        
    days = []
    scores = []
    for key, value in mock_dict.items():
        if key.startswith('10'):
            days.append(int(key[0:2]))
        else:
            days.append(int(key[0]))
        scores.append(value)
    slope, intercept, _, _, _ = linregress(days, scores)
    if intercept < 0:
        intercept = 0
    line = [slope * day + intercept for day in days]
    fruit_lines['average'] = (slope, intercept, line)
    ax.plot(days, line, label='average')

    ax.set_xlabel('Days')
    ax.set_ylabel('Ripeness')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend()
    plt.show()
    
    return fruit_lines

def test(fruit_lines, mock_dict):
    # Compute the average line
    # total_slope, total_intercept = 0, 0
    # for line in fruit_lines.values():
    #     slope, intercept = linregress(range(1, 11), line[:10])[:2]
    #     total_slope += slope
    #     total_intercept += intercept
    # avg_slope, avg_intercept = total_slope / len(fruit_lines), total_intercept / len(fruit_lines)

    avg_slope, avg_intercept, avg_line = fruit_lines['average']
    avg_line = [avg_slope * day + avg_intercept for day in range(1, 11)]
    
    # Compute the vertical distances and plot the graphs
    for fruit in set(''.join(filter(str.isalpha, key[1:])) for key in mock_dict.keys() if key.endswith('3')):
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), avg_line, label='average')
        total_distance = 0
        for key, value in mock_dict.items():
            if key.endswith(fruit+'3'):
                if key.startswith('10'):
                    day = int(key[0:2])
                else:
                    day = int(key[0])
                avg_value = avg_slope * day + avg_intercept
                dist = abs(value - (avg_slope * day + avg_intercept))
                total_distance += dist
                # if value is less than the average line, plot a green line
                if value < avg_slope * day + avg_intercept:
                    ax.axvline(x=day, ymin=value, ymax=value, color='red', linestyle='--')


                ax.axvline(x=day, ymin=0, ymax=value, color='red', linestyle='--')

        ax.set_xlabel('Days')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylabel('Ripeness')
        ax.set_title(fruit)
        ax.legend()
        plt.show()
        print(f'Total vertical distance for {fruit}: {total_distance}')



img_dict = get_image_dict()
score_dict = get_scores_dict(img_dict)
crowd_dict = get_crowd_data("fruitcsv.csv")


lines = train(score_dict)
total_distance = test(lines, score_dict)


