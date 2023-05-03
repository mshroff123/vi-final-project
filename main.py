import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import linregress
from collections import defaultdict


from dict import get_image_dict, get_scores_dict
from canvas import get_canvas


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
    line = [slope * day + intercept for day in days]
    fruit_lines['average'] = line
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
    total_slope, total_intercept = 0, 0
    for line in fruit_lines.values():
        slope, intercept = linregress(range(1, 11), line[:10])[:2]
        total_slope += slope
        total_intercept += intercept
    avg_slope, avg_intercept = total_slope / len(fruit_lines), total_intercept / len(fruit_lines)
    avg_line = [avg_slope * day + avg_intercept for day in range(1, 11)]
    
    # Compute the vertical distances and plot the graphs
    for fruit in set(key[1:-1] for key in mock_dict.keys() if key[-1] == '3'):
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), avg_line, label='average')
        total_distance = 0
        for key, value in mock_dict.items():
            if key.endswith(fruit+'3'):
                if key.startswith('10'):
                    day = int(key[0:2])
                else:
                    day = int(key[0])
                dist = abs(value - (avg_slope * day + avg_intercept))
                total_distance += dist
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
#score_dict = get_scores_dict(img_dict)




get_canvas(img_dict, 'mango')

# # Create an empty dictionary to store the updated entries
# score_dict = defaultdict(list)

# # Loop through each key-value pair in the original dictionary
# for key, value in score_dict_old.items():
#     # If the key starts with 0, update the key with 10 and add the entry to the new dictionary
#     if key.startswith('0'):
#         score_dict['10' + key[1:]] = value
#     # If the key does not start with 0, add the entry to the new dictionary as is
#     else:
#         score_dict[key] = value




mock_dict = {}

# for day in range(1, 11):
#     for number in range(1, 3):
#         for fruit in ["apple", "banana", "kiwi", "mango", "pear"]:
#             key = f"{day}{fruit}{number}"
#             base_value = day / 5  # closer to 1 as the day gets higher, closer to 0 as the day gets lower
#             random_offset = random.uniform(-0.05, 0.05)  # some level of randomness
#             value = max(0, min(1, base_value + random_offset))
#             mock_dict[key] = round(value, 2)



# lines = train(score_dict)
# total_distance = test(lines, score_dict)


# total_distance = calculate_vertical_distance(mock_dict, lines)
# print(f'Total vertical distance for each fruit across all days: {total_distance}')
